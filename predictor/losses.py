"""Generator loss: MSE + spectral + bandpower + correlation + diversity + semantic."""

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
LAMBDA_HF = 0.0
LAMBDA_BAND_CORR = 0.0
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


def time_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(y_hat, y)


def frequency_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    sfreq: float = SFREQ,
    fmin: float = 1.0,
    fmax: float = 40.0,
    low_freq_emphasis: float = 0.0,
) -> torch.Tensor:
    spec_hat = torch.fft.rfft(y_hat, dim=-1)
    spec = torch.fft.rfft(y, dim=-1)
    freqs = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq).to(y.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    err = (spec_hat.abs()[..., mask] - spec.abs()[..., mask]).abs()
    if low_freq_emphasis > 0:
        w = 1.0 + low_freq_emphasis / (freqs[mask] + 1.0)
        return (err * w).mean()
    return err.mean()


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


def high_frequency_excess_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    sfreq: float = SFREQ,
    fmin: float = 35.0,
    fmax: float = 45.0,
) -> torch.Tensor:
    """Penalize only excess |FFT| in a high-frequency band (reduces 40–45 Hz artifacts)."""
    spec_hat = torch.fft.rfft(y_hat, dim=-1)
    spec = torch.fft.rfft(y, dim=-1)
    freqs = torch.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq).to(y.device)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return y_hat.new_tensor(0.0)
    excess = (spec_hat.abs()[..., mask] - spec.abs()[..., mask]).clamp(min=0.0)
    return excess.mean()


def correlation_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    a = y_hat - y_hat.mean(dim=-1, keepdim=True)
    b = y - y.mean(dim=-1, keepdim=True)
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    r = num / den
    return (1.0 - r).mean()


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


def semantic_consistency_loss(
    y_hat: torch.Tensor,
    clip_feat: torch.Tensor,
    brain: nn.Module,
) -> torch.Tensor:
    """Positive-only alignment: maximize cos(brain(y_hat), clip) per sample."""
    eeg_emb = eeg_to_ubp_embedding(y_hat, brain)
    clip_emb = F.normalize(clip_feat, dim=1)
    return (1.0 - (eeg_emb * clip_emb).sum(dim=1)).mean()


def semantic_ubp_anchor_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    brain: nn.Module,
) -> torch.Tensor:
    """Align brain(y_hat) with brain(y); GT EEG is the semantic anchor."""
    emb_g = eeg_to_ubp_embedding(y_hat, brain)
    with torch.no_grad():
        emb_e = eeg_to_ubp_embedding(y, brain)
    return (1.0 - (emb_g * emb_e).sum(dim=1)).mean()


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
        lambda_hf: float = LAMBDA_HF,
        lambda_band_corr: float = LAMBDA_BAND_CORR,
        semantic_mode: str = 'ubp_margin',
        semantic_temperature: float = SEMANTIC_TEMPERATURE,
        sfreq: float = SFREQ,
        gamma_fmax: float = GAMMA_FMAX,
        band_weights: Optional[Dict[str, float]] = None,
        hf_fmin: float = 35.0,
        hf_fmax: float = 45.0,
        low_freq_emphasis: float = 0.0,
        waveform_only: bool = False,
    ):
        super().__init__()
        if semantic_mode not in SEMANTIC_MODES:
            raise ValueError(f'semantic_mode must be one of {SEMANTIC_MODES}, got {semantic_mode!r}')
        self.semantic_mode = semantic_mode
        self.semantic_temperature = semantic_temperature
        if waveform_only:
            self.lambda_time = lambda_time
            self.lambda_corr = lambda_corr
            self.lambda_freq = 0.0
            self.lambda_band = 0.0
            self.lambda_div = 0.0
            self.lambda_sem = 0.0
            self.lambda_ubp = 0.0
            self.lambda_margin = 0.0
            self.lambda_eeg_nce = 0.0
            self.lambda_hf = 0.0
            self.lambda_band_corr = 0.0
        else:
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
            self.lambda_hf = lambda_hf
            self.lambda_band_corr = lambda_band_corr
        self.sfreq = sfreq
        self.gamma_fmax = gamma_fmax
        self.band_weights = band_weights
        self.hf_fmin = hf_fmin
        self.hf_fmax = hf_fmax
        self.low_freq_emphasis = low_freq_emphasis
        self.brain = brain

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        clip_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        lt = time_loss(y_hat, y)
        lf = (
            frequency_loss(
                y_hat, y, sfreq=self.sfreq, low_freq_emphasis=self.low_freq_emphasis,
            )
            if self.lambda_freq else y_hat.new_tensor(0.0)
        )
        bands = resolve_eeg_bands(self.gamma_fmax) if self.lambda_band else EEG_BANDS
        lb = (
            bandpower_loss(
                y_hat, y, sfreq=self.sfreq, bands=bands, band_weights=self.band_weights,
            )
            if self.lambda_band else y_hat.new_tensor(0.0)
        )
        lhf = (
            high_frequency_excess_loss(
                y_hat, y, sfreq=self.sfreq, fmin=self.hf_fmin, fmax=self.hf_fmax,
            )
            if self.lambda_hf else y_hat.new_tensor(0.0)
        )
        lbc = (
            band_correlation_loss(
                y_hat, y, sfreq=self.sfreq,
                bands=resolve_eeg_bands(self.gamma_fmax),
                band_weights=self.band_weights,
            )
            if self.lambda_band_corr else y_hat.new_tensor(0.0)
        )
        lc = correlation_loss(y_hat, y)
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
            + self.lambda_hf * lhf
            + self.lambda_band_corr * lbc
            + self.lambda_corr * lc
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
            'hf': lhf.item() if self.lambda_hf else 0.0,
            'band_corr': lbc.item() if self.lambda_band_corr else 0.0,
            'corr': lc.item(),
            'diversity': ld.item() if self.lambda_div else 0.0,
            'semantic': lsem.item(),
            'ubp': lubp.item() if self.lambda_ubp else 0.0,
            'eeg_nce': leeg_nce.item() if self.lambda_eeg_nce else 0.0,
            'margin': lmargin.item() if self.lambda_margin else 0.0,
        }
        return total, details
