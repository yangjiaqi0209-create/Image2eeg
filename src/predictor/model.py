"""CLIP -> EEG [B, C, T]: Transformer decoder + dilated temporal refinement."""

from __future__ import annotations

from typing import Dict, Literal, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

GeneratorArch = Literal['full', 'no_dconv', 'tcn_only', 'no_self_attn']
GENERATOR_ARCH_CHOICES: tuple[str, ...] = (
    'full', 'no_dconv', 'tcn_only', 'no_self_attn',
)

DEFAULT_TCN_DILATIONS: tuple[int, ...] = (1, 2, 4, 8, 16)


def parse_tcn_dilations(value: str | Sequence[int] | None) -> tuple[int, ...]:
    if value is None:
        return DEFAULT_TCN_DILATIONS
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(',') if p.strip()]
        return tuple(int(p) for p in parts)
    return tuple(int(v) for v in value)


class AdaLN(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mod = nn.Linear(cond_dim, 2 * hidden)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden: int,
        cond_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        *,
        use_self_attn: bool = True,
    ):
        super().__init__()
        self.use_self_attn = use_self_attn
        self.adaln1 = AdaLN(hidden, cond_dim)
        self.self_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.adaln2 = AdaLN(hidden, cond_dim)
        self.cross_attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
        self.adaln3 = AdaLN(hidden, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, q: Tensor, cond_token: Tensor, cond: Tensor) -> Tensor:
        if self.use_self_attn:
            h = self.adaln1(q, cond)
            q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.adaln2(q, cond)
        q = q + self.cross_attn(h, cond_token, cond_token, need_weights=False)[0]
        q = q + self.ff(self.adaln3(q, cond))
        return q


class DilatedConvRefinement(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = DEFAULT_TCN_DILATIONS,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual
        self.blocks = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=d),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=d),
                    nn.BatchNorm1d(channels),
                )
            )
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            out = block(x)
            x = self.act(out + x) if self.residual else self.act(out)
        return x


class MLP_TCN_Generator(nn.Module):
    """
    Blur CLIP [B, D] -> cond MLP -> temporal queries [B, T, H]
    -> Transformer decoder -> dilated conv -> per-channel heads -> [B, C, T].

    arch:
      full         — Transformer (AdaLN) + dilated conv (default)
      no_dconv     — Transformer only
      tcn_only     — cond MLP -> broadcast -> dilated conv only (no Transformer)
      no_self_attn — Transformer without self-attention
    """

    def __init__(
        self,
        img_dim: int = 1024,
        n_channels: int = 63,
        seq_len: int = 250,
        mlp_hidden: int = 1024,
        hidden: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
        per_channel_heads: bool = True,
        arch: GeneratorArch = 'full',
        tcn_dilations: tuple[int, ...] | str | None = None,
        tcn_residual: bool = True,
        n_spatial: int = 0,
    ):
        super().__init__()
        if arch not in GENERATOR_ARCH_CHOICES:
            raise ValueError(f'Unknown generator arch: {arch!r}')
        self.arch = arch
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden = hidden
        self.n_layers = n_layers
        self.img_dim = img_dim
        self.per_channel_heads = per_channel_heads and n_spatial <= 0
        self.n_spatial = int(n_spatial) if n_spatial else 0
        self.tcn_dilations = parse_tcn_dilations(tcn_dilations)
        self.tcn_residual = tcn_residual
        self.use_transformer = arch != 'tcn_only'
        self.use_dconv = arch != 'no_dconv'
        self.use_self_attn = arch != 'no_self_attn'

        self.cond_encoder = nn.Sequential(
            nn.Linear(img_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        if self.use_transformer:
            self.query_embed = nn.Parameter(torch.zeros(1, seq_len, hidden))
            nn.init.normal_(self.query_embed, std=0.02)
            self.decoder_layers = nn.ModuleList(
                TransformerDecoderLayer(
                    hidden, hidden, n_heads, dropout,
                    use_self_attn=self.use_self_attn,
                )
                for _ in range(n_layers)
            )
        else:
            self.query_embed = None
            self.decoder_layers = nn.ModuleList()

        if self.use_dconv:
            self.temporal_refine = DilatedConvRefinement(
                hidden,
                dilations=self.tcn_dilations,
                dropout=dropout,
                residual=tcn_residual,
            )
        else:
            self.temporal_refine = None

        if self.n_spatial > 0:
            self.latent_proj = nn.Linear(hidden, self.n_spatial)
            self.spatial_maps = nn.Parameter(torch.empty(self.n_spatial, n_channels))
            nn.init.orthogonal_(self.spatial_maps)
            self.channel_heads = None
            self.out_proj = None
        elif per_channel_heads:
            self.latent_proj = None
            self.spatial_maps = None
            self.channel_heads = nn.ModuleList(
                nn.Linear(hidden, 1) for _ in range(n_channels)
            )
            self.out_proj = None
        else:
            self.latent_proj = None
            self.spatial_maps = None
            self.channel_heads = None
            self.out_proj = nn.Linear(hidden, n_channels)

    def config_dict(self) -> Dict[str, Union[int, float, str, bool, tuple[int, ...]]]:
        return {
            'hidden': self.hidden,
            'n_layers': self.n_layers,
            'img_dim': self.img_dim,
            'dropout': 0.1,
            'per_channel_heads': self.per_channel_heads,
            'n_spatial': self.n_spatial,
            'generator_arch': self.arch,
            'tcn_dilations': self.tcn_dilations,
            'tcn_residual': self.tcn_residual,
        }

    def _decode_channels(self, x: Tensor) -> Tensor:
        """x: [B, T, H] -> [B, C, T]"""
        if self.spatial_maps is not None:
            z = self.latent_proj(x)
            return torch.einsum('btk,kc->bct', z, self.spatial_maps)
        if self.channel_heads is not None:
            channels = [head(x).squeeze(-1) for head in self.channel_heads]
            return torch.stack(channels, dim=1)
        return self.out_proj(x).transpose(1, 2)

    def _apply_dconv(self, q: Tensor) -> Tensor:
        if self.temporal_refine is None:
            return q
        return self.temporal_refine(q.transpose(1, 2)).transpose(1, 2)

    def forward(self, img_feat: Tensor) -> Tensor:
        b = img_feat.shape[0]
        cond = self.cond_encoder(img_feat)

        if self.use_transformer:
            cond_token = cond.unsqueeze(1)
            q = self.query_embed.expand(b, -1, -1)
            for layer in self.decoder_layers:
                q = layer(q, cond_token, cond)
        else:
            q = cond.unsqueeze(1).expand(b, self.seq_len, -1)

        x = self._apply_dconv(q)
        return self._decode_channels(x)


@torch.no_grad()
def init_spatial_maps_pca(
    model: MLP_TCN_Generator,
    eeg: Tensor,
    max_rows: int = 80000,
) -> None:
    """Initialize spatial_maps (K, C) from PCA of train EEG [N, C, T]."""
    if model.spatial_maps is None:
        return
    n, c, t = eeg.shape
    x = eeg.permute(0, 2, 1).reshape(n * t, c)
    if x.shape[0] > max_rows:
        idx = torch.randperm(x.shape[0], device=x.device)[:max_rows]
        x = x[idx]
    x = x - x.mean(dim=0, keepdim=True)
    q = int(model.spatial_maps.shape[0])
    _, _, v = torch.pca_lowrank(x.float(), q=q, niter=4)
    model.spatial_maps.copy_(v.T.contiguous())


def build_generator(
    arch: str,
    *,
    img_dim: int = 1024,
    n_channels: int = 63,
    seq_len: int = 250,
    hidden: int = 256,
    n_layers: int = 4,
    per_channel_heads: bool = True,
    dropout: float = 0.1,
    tcn_dilations: tuple[int, ...] | str | None = None,
    tcn_residual: bool = True,
    n_spatial: int = 0,
) -> MLP_TCN_Generator:
    """Factory for manuscript generator architectures."""
    return MLP_TCN_Generator(
        img_dim=img_dim,
        n_channels=n_channels,
        seq_len=seq_len,
        hidden=hidden,
        n_layers=n_layers,
        dropout=dropout,
        per_channel_heads=per_channel_heads,
        arch=arch,  # type: ignore[arg-type]
        tcn_dilations=tcn_dilations,
        tcn_residual=tcn_residual,
        n_spatial=n_spatial,
    )


def load_any_generator_from_checkpoint(
    ckpt: dict,
    n_channels: int,
    seq_len: int,
) -> MLP_TCN_Generator:
    args = ckpt.get('args', {})
    model = build_generator(
        args.get('generator_arch', 'full'),
        img_dim=args.get('img_dim', args.get('z_dim', 1024)),
        n_channels=n_channels,
        seq_len=seq_len,
        hidden=args.get('hidden', 256),
        n_layers=args.get('n_layers', 4),
        dropout=args.get('dropout', 0.1),
        per_channel_heads=args.get('per_channel_heads', True),
        tcn_dilations=args.get('tcn_dilations'),
        tcn_residual=args.get('tcn_residual', True),
        n_spatial=int(args.get('n_spatial', 0) or 0),
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model
