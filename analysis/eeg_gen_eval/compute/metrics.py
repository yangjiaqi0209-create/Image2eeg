"""Waveform and frequency metrics for predicted vs true EEG."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import welch

from analysis.eeg_gen_eval.config import (
    CHANNELS,
    CHANNELS_O,
    EEG_BANDS,
    ERP_COMPONENT_WINDOWS_MS,
    SFREQ,
    TIME_WINDOWS_MS,
)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float('nan')
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    if den < 1e-12:
        return float('nan')
    return float(np.dot(a, b) / den)


def ms_to_samples(ms: float) -> int:
    return int(round(ms * SFREQ / 1000.0))


def global_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def global_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(global_mse(y_pred, y_true)))


def global_nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """MSE normalized by var(y_true); NMSE=1 ⇒ error variance equals signal variance."""
    var = float(np.var(y_true))
    if var < 1e-12:
        return float('nan')
    return global_mse(y_pred, y_true) / var


def global_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def global_pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return _pearson(y_pred, y_true)


def r2_pooled_vs_mean(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Pooled R²: all samples/channels/times in one SS ratio (can be negative)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    baseline = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - baseline) ** 2)
    if ss_tot < 1e-12:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def per_sample_r2(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Per-image R² vs the test-set grand-mean EEG baseline (same baseline for all images).

    For image i:
        R²_i = 1 - ||y_i - ŷ_i||² / ||y_i - ȳ||²
    where ȳ is the mean EEG across all N test images (63 × T).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    baseline = np.mean(y_true, axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(1, 2))
    ss_tot = np.sum((y_true - baseline) ** 2, axis=(1, 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = 1.0 - ss_res / ss_tot
    r2[ss_tot < 1e-12] = np.nan
    return r2


def per_sample_r2_waveform(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Per-image waveform R²: mean over channels of squared Pearson r along time.

    Scale-invariant; measures how well temporal shape is recovered per image.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n, n_ch, _ = y_true.shape
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        rs = []
        for c in range(n_ch):
            r = _pearson(y_pred[i, c], y_true[i, c])
            if np.isfinite(r):
                rs.append(r * r)
        out[i] = float(np.mean(rs)) if rs else float('nan')
    return out


def per_sample_r2_summary(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    r2_mean = per_sample_r2(y_pred, y_true)
    r2_wave = per_sample_r2_waveform(y_pred, y_true)

    def _stats(vec: np.ndarray, prefix: str) -> Dict[str, float]:
        valid = vec[np.isfinite(vec)]
        if valid.size == 0:
            return {
                f'{prefix}_mean': float('nan'),
                f'{prefix}_std': float('nan'),
                f'{prefix}_median': float('nan'),
                f'{prefix}_positive_frac': float('nan'),
            }
        return {
            f'{prefix}_mean': float(np.mean(valid)),
            f'{prefix}_std': float(np.std(valid, ddof=0)),
            f'{prefix}_median': float(np.median(valid)),
            f'{prefix}_positive_frac': float(np.mean(valid > 0)),
        }

    stats = _stats(r2_mean, 'r2_per_sample_vs_mean')
    stats.update(_stats(r2_wave, 'r2_per_sample_waveform'))
    return stats


def per_channel_pearson(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Pearson r per channel, pooled over samples and time. Shape (C,)."""
    n_ch = y_true.shape[1]
    out = np.empty(n_ch, dtype=np.float64)
    for c in range(n_ch):
        out[c] = _pearson(y_pred[:, c, :], y_true[:, c, :])
    return out


def mean_per_channel_pearson(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Pearson r across channels (one r per channel over all trials × time)."""
    per_ch = per_channel_pearson(y_pred, y_true)
    valid = per_ch[np.isfinite(per_ch)]
    if valid.size == 0:
        return float('nan')
    return float(np.mean(valid))


def per_timepoint_pearson(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Pearson r at each time point, pooled over samples and channels. Shape (T,)."""
    n_time = y_true.shape[2]
    out = np.empty(n_time, dtype=np.float64)
    for t in range(n_time):
        out[t] = _pearson(y_pred[:, :, t], y_true[:, :, t])
    return out


def time_window_pearson(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    windows: List[Tuple[str, int, int]] | None = None,
) -> Dict[str, float]:
    windows = windows or TIME_WINDOWS_MS
    result = {}
    for name, t0_ms, t1_ms in windows:
        s0, s1 = ms_to_samples(t0_ms), ms_to_samples(t1_ms)
        s1 = max(s1, s0 + 1)
        result[name] = _pearson(y_pred[:, :, s0:s1], y_true[:, :, s0:s1])
    return result


def occipital_erp_component_pearson(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    o_indices: List[int] | None = None,
    windows: List[Tuple[str, int, int]] | None = None,
) -> Dict[str, float]:
    """Pearson r per occipital ERP component window (channel-mean O-region ERP)."""
    if o_indices is None:
        o_indices = [CHANNELS.index(ch) for ch in CHANNELS_O]
    windows = windows or ERP_COMPONENT_WINDOWS_MS
    pred_o = y_pred[:, o_indices, :].mean(axis=1)
    true_o = y_true[:, o_indices, :].mean(axis=1)
    result = {}
    for name, t0_ms, t1_ms in windows:
        s0, s1 = ms_to_samples(t0_ms), ms_to_samples(t1_ms)
        s1 = max(s1, s0 + 1)
        result[name] = _pearson(pred_o[:, s0:s1], true_o[:, s0:s1])
    return result


# Peak polarity within each ERP window (occipital mean ERP).
_ERP_PEAK_POLARITY = {
    'C1': 'neg',
    'P1': 'pos',
    'N1': 'neg',
}


def occipital_erp_component_fidelity(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    o_indices: List[int] | None = None,
    windows: List[Tuple[str, int, int]] | None = None,
    sfreq: float = SFREQ,
) -> Dict[str, Dict[str, float]]:
    """Occipital ERP component fidelity: Pearson, amplitude corr, latency.

    Latency score (per sample, then averaged):
        LatencyScore = 1 − |t_r − t_g| / T
    Amplitude correlation: Pearson r of mean occipital amplitude within each
    component window (across trials). Latency still uses peak latency.
    """
    if o_indices is None:
        o_indices = [CHANNELS.index(ch) for ch in CHANNELS_O]
    windows = windows or ERP_COMPONENT_WINDOWS_MS
    pred_o = y_pred[:, o_indices, :].mean(axis=1)  # (N, T)
    true_o = y_true[:, o_indices, :].mean(axis=1)
    ms_per_sample = 1000.0 / sfreq
    out: Dict[str, Dict[str, float]] = {}

    for name, t0_ms, t1_ms in windows:
        s0, s1 = ms_to_samples(t0_ms), ms_to_samples(t1_ms)
        s1 = max(s1, s0 + 1)
        win_w = float(t1_ms - t0_ms)
        pred_w = pred_o[:, s0:s1]
        true_w = true_o[:, s0:s1]
        # Mean amplitude within component window (per trial)
        amp_r = pred_w.mean(axis=1)
        amp_g = true_w.mean(axis=1)

        polarity = _ERP_PEAK_POLARITY.get(name, 'pos')
        if polarity == 'neg':
            idx_p = np.argmin(pred_w, axis=1)
            idx_t = np.argmin(true_w, axis=1)
        else:
            idx_p = np.argmax(pred_w, axis=1)
            idx_t = np.argmax(true_w, axis=1)

        lat_r_ms = t0_ms + idx_p.astype(np.float64) * ms_per_sample  # t_r (generated)
        lat_g_ms = t0_ms + idx_t.astype(np.float64) * ms_per_sample  # t_g (ground truth)
        t_window = win_w  # T (ms)
        delta_ms = np.abs(lat_r_ms - lat_g_ms)
        latency_score = float(np.clip(1.0 - np.mean(delta_ms / t_window), 0.0, 1.0))
        latency_mae_ms = float(np.mean(delta_ms))

        out[name] = {
            'pearson': _pearson(pred_w, true_w),
            'amplitude_corr': _pearson(amp_r, amp_g),
            'latency_mae_ms': latency_mae_ms,
            'latency_consistency': latency_score,
            'window_ms': t_window,
        }
    return out


def per_channel_erp_component_pearson(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    windows: List[Tuple[str, int, int]] | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Pearson r per (component, channel), pooled over test samples.

    Returns
    -------
    mat : ndarray, shape (n_components, n_channels)
    names : component names (e.g. C1, P1, N1)
    """
    windows = windows or ERP_COMPONENT_WINDOWS_MS
    n_ch = y_pred.shape[1]
    names = [w[0] for w in windows]
    mat = np.empty((len(windows), n_ch), dtype=np.float64)
    for i, (_, t0_ms, t1_ms) in enumerate(windows):
        s0, s1 = ms_to_samples(t0_ms), ms_to_samples(t1_ms)
        s1 = max(s1, s0 + 1)
        for c in range(n_ch):
            mat[i, c] = _pearson(y_pred[:, c, s0:s1], y_true[:, c, s0:s1])
    return mat, names


def fft_magnitude(y: np.ndarray, sfreq: float = SFREQ) -> Tuple[np.ndarray, np.ndarray]:
    """Mean |FFT| over batch and channels. Returns freqs (F,), mag (F,)."""
    spec = np.fft.rfft(y, axis=-1)
    mag = np.abs(spec).mean(axis=(0, 1))
    freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq)
    return freqs, mag


def fft_magnitude_error(y_pred: np.ndarray, y_true: np.ndarray, sfreq: float = SFREQ) -> Dict[str, np.ndarray]:
    freqs, mag_pred = fft_magnitude(y_pred, sfreq)
    _, mag_true = fft_magnitude(y_true, sfreq)
    abs_err = np.abs(mag_pred - mag_true)
    rel_err = abs_err / (mag_true + 1e-8)
    return {
        'freqs_hz': freqs,
        'mag_pred': mag_pred,
        'mag_true': mag_true,
        'abs_error': abs_err,
        'l1_error': float(np.mean(abs_err)),
        'mse_error': float(np.mean(abs_err ** 2)),
    }


def bandpower_cube(y: np.ndarray, sfreq: float = SFREQ) -> Tuple[np.ndarray, List[str]]:
    """Log bandpower per sample and channel. Shape (N, C, n_bands)."""
    spec = np.fft.rfft(y, axis=-1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq)
    bands = []
    cube = []
    for name, fmin, fmax in EEG_BANDS:
        mask = (freqs >= fmin) & (freqs < fmax)
        bp = np.log1p(power[..., mask].mean(axis=-1))
        cube.append(bp)
        bands.append(name)
    return np.stack(cube, axis=-1), bands


def _bandpass_fft(y: np.ndarray, fmin: float, fmax: float, sfreq: float = SFREQ) -> np.ndarray:
    """Band-limit along the time axis via FFT masking."""
    spec = np.fft.rfft(y, axis=-1)
    freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / sfreq)
    mask = (freqs >= fmin) & (freqs < fmax)
    filtered = np.zeros_like(spec)
    filtered[..., mask] = spec[..., mask]
    return np.fft.irfft(filtered, n=y.shape[-1], axis=-1)


def bandpower_channel_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Pearson r per (band, channel), pooled over test samples. Shape (n_bands, n_ch)."""
    bp_pred, band_names = bandpower_cube(y_pred, SFREQ)
    bp_true, _ = bandpower_cube(y_true, SFREQ)
    n_bands, n_ch = len(band_names), y_true.shape[1]
    out = np.empty((n_bands, n_ch), dtype=np.float64)
    for i in range(n_bands):
        for c in range(n_ch):
            out[i, c] = _pearson(bp_pred[:, c, i], bp_true[:, c, i])
    return out, band_names


def band_channel_waveform_correlation(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Pearson r per (band, channel) on band-limited waveforms, pooled over test samples."""
    band_names = [b[0] for b in EEG_BANDS]
    n_bands, n_ch = len(band_names), y_true.shape[1]
    out = np.empty((n_bands, n_ch), dtype=np.float64)
    for i, (_, fmin, fmax) in enumerate(EEG_BANDS):
        yp = _bandpass_fft(y_pred, fmin, fmax)
        yt = _bandpass_fft(y_true, fmin, fmax)
        for c in range(n_ch):
            out[i, c] = _pearson(yp[:, c, :].ravel(), yt[:, c, :].ravel())
    return out, band_names


def bandpower_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
    bp_pred, band_names = bandpower_cube(y_pred)
    bp_true, _ = bandpower_cube(y_true)
    abs_err = np.abs(bp_pred - bp_true)
    errors = {b: float(abs_err[..., i].mean()) for i, b in enumerate(band_names)}
    corrs = {b: _pearson(bp_pred[..., i], bp_true[..., i]) for i, b in enumerate(band_names)}
    return {
        'band_names': band_names,
        'bandpower_pred': bp_pred,
        'bandpower_true': bp_true,
        'bandpower_abs_error': errors,
        'bandpower_correlation': corrs,
    }


def bandpower_means(
    bandpower_correlation: Dict[str, float],
    bandpower_abs_error: Dict[str, float],
) -> Tuple[float, float]:
    """Mean bandpower corr / abs-error across named bands (ablation aggregates)."""
    corr = float(np.mean([float(v) for v in bandpower_correlation.values()]))
    err = float(np.mean([float(v) for v in bandpower_abs_error.values()]))
    return corr, err


def occipital_tfr_mean(
    y: np.ndarray,
    o_indices: Sequence[int],
    *,
    sfreq: float = SFREQ,
    nperseg: int = 64,
    noverlap: int = 48,
) -> Dict[str, np.ndarray]:
    """Mean STFT power of occipital ROI channel-mean waveforms.

    Returns freqs (Hz), times (s), and Sxx (F, T) averaged over samples.
    """
    from scipy.signal import spectrogram

    y = np.asarray(y, dtype=np.float64)
    sig = y[:, list(o_indices), :].mean(axis=1)  # (N, T)
    sig = sig - sig.mean(axis=-1, keepdims=True)
    specs: List[np.ndarray] = []
    freqs = times = None
    for i in range(sig.shape[0]):
        freqs, times, Sxx = spectrogram(
            sig[i],
            fs=sfreq,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            detrend='constant',
            scaling='density',
            mode='psd',
        )
        specs.append(Sxx)
    return {
        'freqs_hz': np.asarray(freqs, dtype=np.float64),
        'times_s': np.asarray(times, dtype=np.float64),
        'tfr_mean': np.mean(np.stack(specs, axis=0), axis=0),
    }


def psd_mean_sem(y: np.ndarray, sfreq: float = SFREQ) -> Dict[str, np.ndarray]:
    """Mean PSD across samples (avg over channels), with SEM across samples."""
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean(axis=-1, keepdims=True)
    freqs, psd = welch(
        y,
        fs=sfreq,
        window='hann',
        nperseg=128,
        noverlap=64,
        detrend='constant',
        axis=-1,
    )
    psd_ch = psd.mean(axis=1)
    if psd_ch.shape[0] > 1:
        psd_sem = psd_ch.std(axis=0, ddof=1) / np.sqrt(psd_ch.shape[0])
    else:
        psd_sem = np.zeros_like(psd_ch.mean(axis=0))
    return {
        'freqs_hz': freqs,
        'psd_mean': psd.mean(axis=(0, 1)),
        'psd_sem': psd_sem,
    }


def compute_all_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
    """Full metric dict + arrays for saving."""
    per_ch = per_channel_pearson(y_pred, y_true)
    per_t = per_timepoint_pearson(y_pred, y_true)
    tw = time_window_pearson(y_pred, y_true)
    fft = fft_magnitude_error(y_pred, y_true)
    bp = bandpower_metrics(y_pred, y_true)
    band_ch_corr, band_names = band_channel_waveform_correlation(y_pred, y_true)
    psd_pred = psd_mean_sem(y_pred)
    psd_true = psd_mean_sem(y_true)

    r2_vec = per_sample_r2(y_pred, y_true)
    r2_wave_vec = per_sample_r2_waveform(y_pred, y_true)
    r2_stats = per_sample_r2_summary(y_pred, y_true)

    summary = {
        'n_samples': int(y_pred.shape[0]),
        'n_channels': int(y_pred.shape[1]),
        'n_timepoints': int(y_pred.shape[2]),
        'sfreq_hz': SFREQ,
        'trial_avg': True,
        'n_trials_averaged': 80,
        'mse': global_mse(y_pred, y_true),
        'rmse': global_rmse(y_pred, y_true),
        'nmse': global_nmse(y_pred, y_true),
        'mae': global_mae(y_pred, y_true),
        'pearson_r': global_pearson(y_pred, y_true),
        **r2_stats,
        'r2_pooled_vs_mean': r2_pooled_vs_mean(y_pred, y_true),
        'r2_definition': {
            'r2_per_sample_vs_mean': (
                'Per image i: 1 - ||y_i - ŷ_i||² / ||y_i - ȳ||², '
                'ȳ = mean test EEG (all images); negative ⇒ worse than mean template'
            ),
            'r2_per_sample_waveform': (
                'Per image i: mean_c corr(ŷ_i,c, y_i,c)² along time; '
                'scale-invariant waveform fidelity'
            ),
        },
        'time_window_pearson': tw,
        'fft_l1_magnitude_error': fft['l1_error'],
        'fft_mse_magnitude_error': fft['mse_error'],
        'bandpower_abs_error': bp['bandpower_abs_error'],
        'bandpower_correlation': bp['bandpower_correlation'],
    }
    arrays = {
        'per_sample_r2': r2_vec,
        'per_sample_r2_waveform': r2_wave_vec,
        'per_channel_pearson': per_ch,
        'per_timepoint_pearson': per_t,
        'fft_freqs_hz': fft['freqs_hz'],
        'fft_mag_pred': fft['mag_pred'],
        'fft_mag_true': fft['mag_true'],
        'fft_mag_abs_error': fft['abs_error'],
        'bandpower_pred': bp['bandpower_pred'],
        'bandpower_true': bp['bandpower_true'],
        'band_channel_correlation': band_ch_corr,
        'band_names': np.array(band_names),
        'psd_freqs_hz': psd_pred['freqs_hz'],
        'psd_pred_mean': psd_pred['psd_mean'],
        'psd_pred_sem': psd_pred['psd_sem'],
        'psd_true_mean': psd_true['psd_mean'],
        'psd_true_sem': psd_true['psd_sem'],
    }
    return {'summary': summary, 'arrays': arrays}
