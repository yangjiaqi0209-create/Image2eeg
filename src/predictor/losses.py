"""Predictor loss: MSE + spectral + bandpower + correlation + diversity + semantic."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from predictor.data import eeg_to_ubp_embedding

LAMBDA_TIME = 1.0
LAMBDA_FREQ = 0.2
LAMBDA_BAND = 0.2
LAMBDA_CORR = 0.8
LAMBDA_DIV = 0.1
LAMBDA_SEM = 0.0
LAMBDA_UBP = 0.05
LAMBDA_MARGIN = 0.002
LAMBDA_EEG_NCE = 0.1
LAMBDA_BAND_CORR = 0.0
LAMBDA_ENC = 0.0
SEMANTIC_TEMPERATURE = 0.07
SEMANTIC_MODES = ('clip', 'ubp_margin')
SFREQ = 250.0
GAMMA_FMAX = 45.0

EEG_BANDS: List[Tuple[str, float, float]] = [
    ('delta', 0.5, 4.0),
    ('theta', 4.0, 8.0),
    ('alpha', 8.0, 13.0),
    ('beta', 13.0, 30.0),
    ('gamma', 30.0, 45.0),
]

DEFAULT_BAND_WEIGHTS: Dict[str, float] = {
    'delta': 1.0,
    'theta': 1.0,
    'alpha': 1.0,
    'beta': 1.0,
    'gamma': 1.0,
}


def resolve_eeg_bands(gamma_fmax: float = GAMMA_FMAX) -> List[Tuple[str, float, float]]:
    """Return band edges; gamma upper bound is configurable for training."""
    bands = []
    for name, fmin, fmax in EEG_BANDS:
        if name == 'gamma':
            fmax = gamma_fmax
        bands.append((name, fmin, fmax))
    return bands


def _channel_region(ch: str) -> str:
    """10-20 EEG name (Oz) → frontal/central/temporal/parietal/occipital/other."""
    token = ch.split('-')[0]
    letters = ''.join(c for c in token if c.isalpha()).upper()
    if letters.startswith(('PO', 'O')) or letters in ('OZ', 'POZ'):
        return 'occipital'
    if letters.startswith(('TP', 'T', 'FT')) or 'T' in letters[:2]:
        return 'temporal'
    if letters.startswith(('CP', 'P')) or letters in ('PZ', 'CPZ'):
        return 'parietal'
    if letters.startswith(('FC', 'C')) or letters in ('CZ', 'FCZ'):
        return 'central'
    if letters.startswith(('FP', 'AF', 'F')) or letters in ('FZ', 'AFZ'):
        return 'frontal'
    return 'other'


def build_channel_weights(ch_names: List[str], spec: str) -> Optional[torch.Tensor]:
    """Parse 'occipital=2,temporal=1.5' into a (C,) weight tensor. None if empty."""
    if not spec or not ch_names:
        return None
    region_w = {'frontal': 1.0, 'central': 1.0, 'temporal': 1.0,
                'parietal': 1.0, 'occipital': 1.0, 'other': 1.0}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        name, val = part.split('=', 1)
        region_w[name.strip().lower()] = float(val)
    w = torch.tensor([region_w[_channel_region(ch)] for ch in ch_names], dtype=torch.float32)
    return w / w.mean().clamp_min(1e-8)


def build_time_weights(seq_len: int, sfreq: float, spec: str) -> Optional[torch.Tensor]:
    """Parse '80-400:2.5,0-80:0.5' into a (T,) weight tensor. Unspecified samples stay 1."""
    if not spec:
        return None
    w = torch.ones(seq_len, dtype=torch.float32)
    ms = torch.arange(seq_len, dtype=torch.float32) * (1000.0 / float(sfreq))
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        window, val = part.split(':', 1)
        t0_s, t1_s = window.split('-', 1)
        mask = (ms >= float(t0_s)) & (ms < float(t1_s))
        w[mask] = float(val)
    return w / w.mean().clamp_min(1e-8)


def _apply_st_weights(
    err: torch.Tensor,
    channel_weight: Optional[torch.Tensor],
    time_weight: Optional[torch.Tensor],
) -> torch.Tensor:
    """err (B, C, T) → weighted mean; weights already mean-normalized."""
    if channel_weight is not None:
        err = err * channel_weight.to(device=err.device, dtype=err.dtype).view(1, -1, 1)
    if time_weight is not None:
        err = err * time_weight.to(device=err.device, dtype=err.dtype).view(1, 1, -1)
    return err.mean()


def time_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
    time_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if channel_weight is None and time_weight is None:
        return F.mse_loss(y_hat, y)
    return _apply_st_weights((y_hat - y).square(), channel_weight, time_weight)


def frequency_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    sfreq: float = SFREQ,
    fmin: float = 1.0,
    fmax: float = 40.0,
) -> torch.Tensor:
    spec_hat = torch.fft.rfft(y_hat, dim=-1)
    spec = torch.fft.rfft(y, dim=-1)
    freqs = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq).to(y.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return (spec_hat.abs()[..., mask] - spec.abs()[..., mask]).abs().mean()


def bandpower_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    sfreq: float = SFREQ,
    bands: List[Tuple[str, float, float]] = EEG_BANDS,
    band_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    spec_hat = torch.fft.rfft(y_hat, dim=-1)
    spec = torch.fft.rfft(y, dim=-1)
    power_hat = spec_hat.abs().square()
    power = spec.abs().square()
    freqs = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq).to(y.device)
    weights = band_weights or DEFAULT_BAND_WEIGHTS

    losses = []
    weight_sum = 0.0
    for name, fmin, fmax in bands:
        mask = (freqs >= fmin) & (freqs < fmax)
        if not mask.any():
            continue
        w = float(weights.get(name, 1.0))
        log_bp_hat = torch.log1p(power_hat[..., mask].mean(dim=-1))
        log_bp = torch.log1p(power[..., mask].mean(dim=-1))
        losses.append(w * F.l1_loss(log_bp_hat, log_bp))
        weight_sum += w
    if not losses:
        return y_hat.new_tensor(0.0)
    return torch.stack(losses).sum() / max(weight_sum, 1e-8)


def correlation_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
    time_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """1 − Pearson r along time, then mean over batch × channels."""
    if time_weight is None:
        a = y_hat - y_hat.mean(dim=-1, keepdim=True)
        b = y - y.mean(dim=-1, keepdim=True)
        num = (a * b).sum(dim=-1)
        den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    else:
        w = time_weight.to(device=y_hat.device, dtype=y_hat.dtype).view(1, 1, -1)
        w_sum = w.sum().clamp_min(1e-8)
        a = y_hat - (y_hat * w).sum(dim=-1, keepdim=True) / w_sum
        b = y - (y * w).sum(dim=-1, keepdim=True) / w_sum
        num = (w * a * b).sum(dim=-1)
        den = torch.sqrt((w * a * a).sum(dim=-1) * (w * b * b).sum(dim=-1)) + 1e-8
    loss = 1.0 - (num / den)
    if channel_weight is None:
        return loss.mean()
    cw = channel_weight.to(device=loss.device, dtype=loss.dtype).view(1, -1)
    return (loss * cw).mean()


def encoding_correlation_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    channel_weight: Optional[torch.Tensor] = None,
    time_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """1 − Pearson r across the batch (images) at each channel × time.

    This is the training analogue of encoding / %NC (r over images).
    """
    if y_hat.shape[0] < 3:
        return y_hat.new_tensor(0.0)
    a = y_hat - y_hat.mean(dim=0, keepdim=True)
    b = y - y.mean(dim=0, keepdim=True)
    num = (a * b).sum(dim=0)
    den = a.norm(dim=0) * b.norm(dim=0) + 1e-8
    loss = 1.0 - (num / den)
    return _apply_st_weights(loss.unsqueeze(0), channel_weight, time_weight)


def _bandpass_fft_torch(
    y: torch.Tensor,
    fmin: float,
    fmax: float,
    sfreq: float = SFREQ,
) -> torch.Tensor:
    spec = torch.fft.rfft(y, dim=-1)
    freqs = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq).to(y.device)
    mask = (freqs >= fmin) & (freqs < fmax)
    filtered = torch.zeros_like(spec)
    filtered[..., mask] = spec[..., mask]
    return torch.fft.irfft(filtered, n=y.shape[-1], dim=-1)


def band_correlation_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    sfreq: float = SFREQ,
    bands: List[Tuple[str, float, float]] = EEG_BANDS,
    band_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Pearson correlation on band-limited waveforms (matches eval bandpower_correlation)."""
    weights = band_weights or DEFAULT_BAND_WEIGHTS
    losses = []
    weight_sum = 0.0
    for name, fmin, fmax in bands:
        w = float(weights.get(name, 1.0))
        yh = _bandpass_fft_torch(y_hat, fmin, fmax, sfreq)
        yt = _bandpass_fft_torch(y, fmin, fmax, sfreq)
        losses.append(w * correlation_loss(yh, yt))
        weight_sum += w
    if not losses:
        return y_hat.new_tensor(0.0)
    return torch.stack(losses).sum() / max(weight_sum, 1e-8)


def batch_diversity_loss(y_hat: torch.Tensor) -> torch.Tensor:
    if y_hat.shape[0] < 2:
        return y_hat.new_tensor(0.0)
    x = F.normalize(y_hat.flatten(1), dim=1)
    sim = x @ x.t()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return sim[mask].mean()


def semantic_margin_loss(
    y_hat: torch.Tensor,
    clip_feat: torch.Tensor,
    brain: nn.Module,
    *,
    temperature: float = SEMANTIC_TEMPERATURE,
    emb_g: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """In-batch InfoNCE: paired CLIP must rank above other CLIP in the batch."""
    if emb_g is None:
        eeg_emb = eeg_to_ubp_embedding(y_hat, brain)
    else:
        eeg_emb = emb_g
    clip_emb = F.normalize(clip_feat, dim=1)
    if eeg_emb.shape[0] < 2:
        return eeg_emb.new_tensor(0.0)
    logits = (eeg_emb @ clip_emb.T) / max(temperature, 1e-8)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def semantic_eeg_infonce_loss(
    emb_g: torch.Tensor,
    emb_t: torch.Tensor,
    *,
    temperature: float = SEMANTIC_TEMPERATURE,
) -> torch.Tensor:
    """In-batch EEG–EEG InfoNCE: brain(y_hat_i) matches brain(y_i) over other y_j."""
    if emb_g.shape[0] < 2:
        return emb_g.new_tensor(0.0)
    logits = (emb_g @ emb_t.T) / max(temperature, 1e-8)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


class GeneratorLoss(nn.Module):
    """L = MSE + spectral + bandpower + correlation + diversity + semantic."""

    def __init__(
        self,
        brain: Optional[nn.Module] = None,
        *,
        lambda_time: float = LAMBDA_TIME,
        lambda_freq: float = LAMBDA_FREQ,
        lambda_band: float = LAMBDA_BAND,
        lambda_corr: float = LAMBDA_CORR,
        lambda_div: float = LAMBDA_DIV,
        lambda_sem: float = LAMBDA_SEM,
        lambda_ubp: float = LAMBDA_UBP,
        lambda_margin: float = LAMBDA_MARGIN,
        lambda_eeg_nce: float = LAMBDA_EEG_NCE,
        lambda_band_corr: float = LAMBDA_BAND_CORR,
        lambda_enc: float = LAMBDA_ENC,
        semantic_mode: str = 'ubp_margin',
        semantic_temperature: float = SEMANTIC_TEMPERATURE,
        sfreq: float = SFREQ,
        gamma_fmax: float = GAMMA_FMAX,
        band_weights: Optional[Dict[str, float]] = None,
        channel_weight: Optional[torch.Tensor] = None,
        time_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if semantic_mode not in SEMANTIC_MODES:
            raise ValueError(f'semantic_mode must be one of {SEMANTIC_MODES}, got {semantic_mode!r}')
        self.semantic_mode = semantic_mode
        self.semantic_temperature = semantic_temperature
        self.lambda_time = lambda_time
        self.lambda_freq = lambda_freq
        self.lambda_band = lambda_band
        self.lambda_corr = lambda_corr
        self.lambda_div = lambda_div
        has_brain = brain is not None
        if semantic_mode == 'clip':
            self.lambda_sem = lambda_sem if has_brain else 0.0
            self.lambda_ubp = 0.0
            self.lambda_margin = 0.0
            self.lambda_eeg_nce = 0.0
        else:
            self.lambda_sem = 0.0
            self.lambda_ubp = lambda_ubp if has_brain else 0.0
            self.lambda_margin = lambda_margin if has_brain else 0.0
            self.lambda_eeg_nce = lambda_eeg_nce if has_brain else 0.0
        self.lambda_band_corr = lambda_band_corr
        self.lambda_enc = lambda_enc
        self.sfreq = sfreq
        self.gamma_fmax = gamma_fmax
        self.band_weights = band_weights
        self.brain = brain
        self.channel_weight = channel_weight
        self.time_weight = time_weight

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        clip_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cw, tw = self.channel_weight, self.time_weight
        lt = time_loss(y_hat, y, channel_weight=cw, time_weight=tw)
        lf = (
            frequency_loss(y_hat, y, sfreq=self.sfreq)
            if self.lambda_freq else y_hat.new_tensor(0.0)
        )
        bands = resolve_eeg_bands(self.gamma_fmax) if self.lambda_band else EEG_BANDS
        lb = (
            bandpower_loss(
                y_hat, y, sfreq=self.sfreq, bands=bands, band_weights=self.band_weights,
            )
            if self.lambda_band else y_hat.new_tensor(0.0)
        )
        lbc = (
            band_correlation_loss(
                y_hat, y, sfreq=self.sfreq,
                bands=resolve_eeg_bands(self.gamma_fmax),
                band_weights=self.band_weights,
            )
            if self.lambda_band_corr else y_hat.new_tensor(0.0)
        )
        lc = correlation_loss(y_hat, y, channel_weight=cw, time_weight=tw)
        le = (
            encoding_correlation_loss(y_hat, y, channel_weight=cw, time_weight=tw)
            if self.lambda_enc else y_hat.new_tensor(0.0)
        )
        ld = batch_diversity_loss(y_hat) if self.lambda_div else y_hat.new_tensor(0.0)
        lubp = y_hat.new_tensor(0.0)
        lmargin = y_hat.new_tensor(0.0)
        leeg_nce = y_hat.new_tensor(0.0)
        lsem = y_hat.new_tensor(0.0)
        if self.brain is not None:
            need_emb_g = (
                (self.semantic_mode == 'clip' and self.lambda_sem > 0)
                or (
                    self.semantic_mode == 'ubp_margin'
                    and (self.lambda_ubp > 0 or self.lambda_margin > 0 or self.lambda_eeg_nce > 0)
                )
            )
            emb_g = eeg_to_ubp_embedding(y_hat, self.brain) if need_emb_g else None
            emb_t = None
            if self.lambda_ubp > 0 or self.lambda_eeg_nce > 0:
                with torch.no_grad():
                    emb_t = eeg_to_ubp_embedding(y, self.brain)

            if self.semantic_mode == 'clip' and self.lambda_sem > 0 and emb_g is not None:
                clip_emb = F.normalize(clip_feat, dim=1)
                lsem = (1.0 - (emb_g * clip_emb).sum(dim=1)).mean()
            elif self.semantic_mode == 'ubp_margin' and emb_g is not None:
                if self.lambda_ubp > 0 and emb_t is not None:
                    lubp = (1.0 - (emb_g * emb_t).sum(dim=1)).mean()
                if self.lambda_eeg_nce > 0 and emb_t is not None:
                    leeg_nce = semantic_eeg_infonce_loss(
                        emb_g, emb_t, temperature=self.semantic_temperature,
                    )
                if self.lambda_margin > 0:
                    lmargin = semantic_margin_loss(
                        y_hat, clip_feat, self.brain,
                        temperature=self.semantic_temperature,
                        emb_g=emb_g,
                    )

        total = (
            self.lambda_time * lt
            + self.lambda_freq * lf
            + self.lambda_band * lb
            + self.lambda_band_corr * lbc
            + self.lambda_corr * lc
            + self.lambda_enc * le
            + self.lambda_div * ld
            + self.lambda_sem * lsem
            + self.lambda_ubp * lubp
            + self.lambda_eeg_nce * leeg_nce
            + self.lambda_margin * lmargin
        )
        details = {
            'loss': total.item(),
            'time': lt.item(),
            'freq': lf.item() if self.lambda_freq else 0.0,
            'band': lb.item() if self.lambda_band else 0.0,
            'band_corr': lbc.item() if self.lambda_band_corr else 0.0,
            'corr': lc.item(),
            'enc': le.item() if self.lambda_enc else 0.0,
            'diversity': ld.item() if self.lambda_div else 0.0,
            'semantic': lsem.item(),
            'ubp': lubp.item() if self.lambda_ubp else 0.0,
            'eeg_nce': leeg_nce.item() if self.lambda_eeg_nce else 0.0,
            'margin': lmargin.item() if self.lambda_margin else 0.0,
        }
        return total, details
