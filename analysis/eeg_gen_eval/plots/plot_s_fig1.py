"""Supplementary Figure S1: prediction-quality panels (Nature / Cell style).

Layout (two rows, double-column; reading order a–e):
  Row 1: a | b
  Row 2: c | d | e
"""

from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.eeg_gen_eval.config import CHANNELS, FIG_DIR, RAW_DIR, SFREQ
from analysis.eeg_gen_eval.figure_names import S_FIG1, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _despine,
    _ensure_rmse_stats,
    _load_summary,
    _plot_fig1_panel_c,
    _plot_psd_curves,
    _setup_nature_rc,
    add_panel_label,
    calculate_frequency_fidelity_statistics,
    load_frequency_fidelity_data,
    plot_frequency_fidelity,
)
from analysis.eeg_gen_eval.helpers.plot_visualizations import (
    _load_aggregate,
    _time_axis_ms,
)

# Nature double-column supp figure type scale
S_FS = {
    'panel_title': 9.0,
    'panel_letter': 11.5,
    'axis': 8.0,
    'tick': 7.2,
    'legend': 6.8,
    'annot': 6.8,
    'colorbar': 7.4,
    'sublabel': 8.0,
    'grey': 6.6,
}

_C_GT = '#0F4D92'
_C_PRED = '#DC2626'
_C_MEAN = '#B91C1C'
_SPINE = '#4B5563'
_BAND_SHORT = {
    'All': 'All', 'Delta': 'δ', 'Theta': 'θ',
    'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ',
}
_REGION_TICKS = [(0, 'Fp'), (14, 'F'), (30, 'C'), (54, 'P'), (62, 'O')]
_REGION_BOUNDS = (15, 35, 55)


def _setup_s_fig1_rc():
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': S_FS['tick'],
        'axes.labelsize': S_FS['axis'],
        'axes.titlesize': S_FS['panel_title'],
        'xtick.labelsize': S_FS['tick'],
        'ytick.labelsize': S_FS['tick'],
        'legend.fontsize': S_FS['legend'],
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.55,
        'ytick.major.width': 0.55,
        'xtick.major.size': 2.4,
        'ytick.major.size': 2.4,
    })


def _style_spines(ax, *, heatmap: bool = False):
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color(_SPINE)
    if not heatmap:
        _despine(ax)
    ax.tick_params(length=2.2, width=0.55, labelsize=S_FS['tick'], colors='#1F2937')


# Shared header baseline so titles align within each row
_HDR_Y = 0.55
_HDR_H_RATIO = 0.14  # title strip / plot height within each row


def _add_panel_letter(ax, letter: str, *, x: float = -0.02, y: float = 1.03):
    """Bold panel letter at the axes top-left corner."""
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=S_FS['panel_letter'], fontweight='bold', color='#111827',
        clip_on=False, zorder=20,
    )


def _title_strip(fig, spec, title: str, *, y: float = 0.25):
    """Centered title only (letter lives on the axes)."""
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(
        0.5, y, title,
        ha='center', va='center',
        fontsize=S_FS['panel_title'], fontweight='bold', color='#111827',
        transform=ax.transAxes, clip_on=False,
    )
    return ax


def _panel_header(fig, spec, title: str, letter: str, *, letter_x: float = 0.0):
    """Legacy header helper (letter + title). Prefer _add_panel_letter + _title_strip."""
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(
        letter_x, _HDR_Y, letter,
        ha='left', va='center',
        fontsize=S_FS['panel_letter'], fontweight='bold', color='#111827',
        transform=ax.transAxes, clip_on=False,
    )
    ax.text(
        0.5, _HDR_Y, title,
        ha='center', va='center',
        fontsize=S_FS['panel_title'], fontweight='bold', color='#111827',
        transform=ax.transAxes, clip_on=False,
    )
    return ax


def _load_wholebrain_psd():
    psd_freqs = None
    psd_true, psd_pred = [], []
    psd_true_sem, psd_pred_sem = [], []
    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        if not os.path.isfile(os.path.join(d, 'psd_freqs_hz.npy')):
            continue
        psd_freqs = np.load(os.path.join(d, 'psd_freqs_hz.npy'))
        psd_true.append(np.load(os.path.join(d, 'psd_true_mean.npy')))
        psd_pred.append(np.load(os.path.join(d, 'psd_pred_mean.npy')))
        psd_true_sem.append(np.load(os.path.join(d, 'psd_true_sem.npy')))
        psd_pred_sem.append(np.load(os.path.join(d, 'psd_pred_sem.npy')))
    return {
        'freqs': psd_freqs,
        'true': np.stack(psd_true).mean(axis=0),
        'pred': np.stack(psd_pred).mean(axis=0),
        'true_sem': np.stack(psd_true_sem).mean(axis=0),
        'pred_sem': np.stack(psd_pred_sem).mean(axis=0),
    }


def _subject_psd_correlations(metas: list[dict]) -> np.ndarray:
    """Subject-level log-PSD Pearson r over the analysed 0.5–45 Hz range."""
    values = []
    for meta in metas:
        d = os.path.join(RAW_DIR, str(meta['subject']))
        freqs = np.load(os.path.join(d, 'psd_freqs_hz.npy'))
        true_psd = np.load(os.path.join(d, 'psd_true_mean.npy'))
        pred_psd = np.load(os.path.join(d, 'psd_pred_mean.npy'))
        keep = (
            (freqs >= 0.5) & (freqs <= 45.0)
            & np.isfinite(true_psd) & np.isfinite(pred_psd)
            & (true_psd > 0) & (pred_psd > 0)
        )
        if np.count_nonzero(keep) < 2:
            values.append(np.nan)
            continue
        values.append(float(np.corrcoef(
            np.log10(true_psd[keep]),
            np.log10(pred_psd[keep]),
        )[0, 1]))
    return np.asarray(values, dtype=np.float64)


def _cosine_from_metrics_dict(meta: dict) -> float | None:
    """Extract Pred–CLIP semantic cosine from a metrics dict if present."""
    for key in ('test_semantic_cosine', 'semantic_cosine'):
        if key in meta and meta[key] is not None:
            return float(meta[key])
    return None


def _load_semantic_cosine_by_subject() -> dict[str, float]:
    """Pred–CLIP semantic cosine from cache, else per-subject metrics.json."""
    path = os.path.join(RAW_DIR, 'semantic_cosine_by_subject.json')
    out: dict[str, float] = {}
    if os.path.isfile(path):
        with open(path) as f:
            raw = json.load(f)
        for sub, val in raw.items():
            if not str(sub).startswith('sub-'):
                sub = f'sub-{int(sub):02d}'
            out[str(sub)] = float(val)
        if out:
            return out

    # Fallback: evaluate raw metrics.json
    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        mpath = os.path.join(RAW_DIR, sub_dir, 'metrics.json')
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            meta = json.load(f)
        val = _cosine_from_metrics_dict(meta)
        if val is not None:
            out[sub_dir] = val
    if out:
        return out

    # Fallback 2: generator training result_dir (dataset profile), if available.
    try:
        from analysis.eeg_gen_eval.config import DATASET_PROFILES, active_dataset
        results_dir = DATASET_PROFILES.get(active_dataset(), {}).get('results_dir')
    except Exception:
        results_dir = None
    if results_dir and os.path.isdir(results_dir):
        for sub_dir in sorted(os.listdir(results_dir)):
            if not sub_dir.startswith('sub-'):
                continue
            mpath = os.path.join(results_dir, sub_dir, 'metrics.json')
            if not os.path.isfile(mpath):
                continue
            with open(mpath) as f:
                meta = json.load(f)
            val = _cosine_from_metrics_dict(meta)
            if val is not None:
                out[sub_dir] = val
    return out


def _load_rsa_by_subject() -> dict[str, float]:
    """Per-subject GT–Pred RDM Spearman ρ from main RSA meta."""
    path = os.path.join(RAW_DIR, 'rsa', 'meta.json')
    with open(path) as f:
        meta = json.load(f)
    out: dict[str, float] = {}
    for row in meta.get('per_subject', []):
        out[str(row['subject'])] = float(row['rsa_spearman'])
    return out


def _plot_metrics_compact(ax, summary: dict, per_ch_all: np.ndarray | None):
    """Prediction-metric bars (same as fig13): Error / Waveform / Spectral / Repr."""
    metas = list(summary.get('per_subject', []))
    subjects = [str(m['subject']) for m in metas]
    cos_map = _load_semantic_cosine_by_subject()
    rsa_map = _load_rsa_by_subject()

    sub_rmse = np.array([float(m.get('rmse', np.sqrt(m['mse']))) for m in metas])
    sub_pr = np.array([float(m['pearson_r']) for m in metas])
    if per_ch_all is not None and per_ch_all.ndim == 2:
        sub_ch = per_ch_all.mean(axis=1)
    else:
        sub_ch = np.full(len(metas), np.nan)
    sub_psd = _subject_psd_correlations(metas)
    sub_bp = np.array([
        float(np.mean(list(m['bandpower_correlation'].values())))
        for m in metas
    ])
    sub_cos = np.array([cos_map.get(s, np.nan) for s in subjects], dtype=np.float64)
    sub_rsa = np.array([rsa_map.get(s, np.nan) for s in subjects], dtype=np.float64)

    labels = ['RMSE', 'Pool.', 'Ch.', 'PSD', 'BP', 'Cos', 'RSA']
    series = [sub_rmse, sub_pr, sub_ch, sub_psd, sub_bp, sub_cos, sub_rsa]
    # Compact spacing for a 1/3-width column with 7 bars
    xpos = np.array([0.0, 1.15, 2.15, 3.45, 4.45, 5.75, 6.75])
    colors = [
        '#7C8591', '#0F4D92', '#2B6CB0', '#287271', '#4D908E',
        '#8B5E9A', '#C06C84',
    ]
    err_c = [
        '#4B5563', '#0A3A6E', '#1A4F86', '#1D5553', '#356866',
        '#6B4578', '#9A4F66',
    ]

    means = np.array([float(np.nanmean(s)) for s in series])
    sems = np.array([
        float(np.nanstd(s, ddof=1) / np.sqrt(np.sum(np.isfinite(s))))
        if np.sum(np.isfinite(s)) > 1 else 0.0
        for s in series
    ])

    rng = np.random.default_rng(0)
    for x, s, c, ec, mu, se in zip(xpos, series, colors, err_c, means, sems):
        ax.bar(x, mu, width=0.70, color=c, edgecolor='none', zorder=2, alpha=0.92)
        ax.errorbar(
            x, mu, yerr=se, fmt='none', ecolor=ec,
            elinewidth=0.75, capsize=1.3, capthick=0.65, zorder=3,
        )
        ok = np.isfinite(s)
        if ok.any():
            jitter = rng.uniform(-0.09, 0.09, size=int(ok.sum()))
            ax.scatter(
                np.full(ok.sum(), x) + jitter, s[ok],
                s=6, color='#1F2937', alpha=0.32, linewidths=0, zorder=4,
            )
        ax.text(
            x, mu + se + 0.018, f'{mu:.3f}',
            ha='center', va='bottom', fontsize=5.2, color='#374151', zorder=5,
        )

    for xv in (0.58, 2.80, 5.10):
        ax.axvline(xv, color='#E5E7EB', lw=0.65, zorder=0)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=5.2, rotation=28, ha='right')
    ax.tick_params(axis='x', pad=1.2)
    # Group labels just below rotated ticks
    ax.text(
        0.06, -0.20, 'Error ↓', transform=ax.transAxes,
        ha='center', va='center', fontsize=5.2, color='#6B7280', clip_on=False,
    )
    ax.text(
        0.28, -0.20, 'Waveform ↑', transform=ax.transAxes,
        ha='center', va='center', fontsize=5.2, color='#6B7280', clip_on=False,
    )
    ax.text(
        0.54, -0.20, 'Spectral ↑', transform=ax.transAxes,
        ha='center', va='center', fontsize=5.2, color='#6B7280', clip_on=False,
    )
    ax.text(
        0.84, -0.20, 'Repr. ↑', transform=ax.transAxes,
        ha='center', va='center', fontsize=5.2, color='#6B7280', clip_on=False,
    )
    ax.set_xlim(-0.50, 7.25)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel('Value', fontsize=S_FS['axis'])
    _style_spines(ax)


def _plot_channel_r_bars(ax, per_ch: np.ndarray, per_ch_std: np.ndarray):
    """Sorted channel-r bars; thin SEM, clean gradient (Nature-style)."""
    order = np.argsort(per_ch)
    sorted_ch = [CHANNELS[i] for i in order]
    sorted_r = per_ch[order]
    sorted_std = per_ch_std[order]
    y = np.arange(len(CHANNELS))

    norm = mpl.colors.Normalize(vmin=0.0, vmax=0.70)
    colors = mpl.cm.RdBu_r(norm(sorted_r))

    ax.barh(
        y, sorted_r, xerr=sorted_std, height=0.78,
        color=colors, edgecolor='none', alpha=0.95,
        error_kw={'elinewidth': 0.28, 'capsize': 0, 'ecolor': '#CBD5E1', 'alpha': 0.85},
    )
    mean_r = float(np.mean(per_ch))
    ax.axvline(mean_r, color=_C_MEAN, ls=(0, (2.8, 1.8)), lw=0.85, zorder=4)
    ax.text(
        0.97, 0.03, f'mean r = {mean_r:.3f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=S_FS['annot'], color=_C_MEAN, zorder=5,
    )

    # Show ends + a few mid ticks only (avoid clutter)
    tick_idx = [0, 10, 20, 31, 42, 52, len(CHANNELS) - 1]
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([sorted_ch[i] for i in tick_idx], fontsize=S_FS['tick'] - 0.5)
    ax.set_xlabel('Pearson r', fontsize=S_FS['axis'])
    ax.set_ylabel('Channel (sorted)', fontsize=S_FS['axis'])
    ax.set_xlim(0.0, max(0.95, float(sorted_r.max()) + 0.06))
    ax.set_ylim(-0.7, len(CHANNELS) - 0.3)
    _style_spines(ax)


def _draw_channel_heatmap(
    ax, arr: np.ndarray, t_ms: np.ndarray, *,
    cmap: str, vmin=None, vmax=None, ylabel: str | None = 'Channel',
    yticklabels: bool = True,
):
    """Shared channel×time heatmap styling."""
    n_ch = arr.shape[0]
    kw = dict(
        aspect='auto', origin='lower', cmap=cmap,
        extent=[float(t_ms[0]), float(t_ms[-1]), -0.5, n_ch - 0.5],
        interpolation='nearest', rasterized=True,
    )
    if vmin is not None:
        kw['vmin'] = vmin
    if vmax is not None:
        kw['vmax'] = vmax
    im = ax.imshow(arr, **kw)
    for b in _REGION_BOUNDS:
        ax.axhline(b - 0.5, color='white', lw=0.5, alpha=0.75, zorder=2)
    ax.set_yticks([i for i, _ in _REGION_TICKS])
    if yticklabels:
        ax.set_yticklabels([lab for _, lab in _REGION_TICKS], fontsize=S_FS['tick'])
    else:
        ax.set_yticklabels([])
    ax.set_xlabel('Time (ms)', fontsize=S_FS['axis'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=S_FS['axis'], labelpad=1)
    ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
    ax.set_ylim(-0.5, n_ch - 0.5)
    _style_spines(ax, heatmap=True)
    return im


def _cbar_compact(cax, im, label: str):
    """Thin Nature-style colorbar with vertical side label."""
    cb = cax.figure.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=S_FS['colorbar'] - 1.0, labelpad=1)
    cb.ax.tick_params(labelsize=S_FS['tick'] - 1.0, length=1.2, width=0.4, pad=0.4)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(_SPINE)
    return cb


def _polish_time_r(ax):
    mean_r = None
    for txt in list(ax.texts):
        s = txt.get_text()
        if s.startswith('mean') or s == 'early visual':
            if s.startswith('mean'):
                try:
                    mean_r = float(s.split('=')[-1].strip())
                except ValueError:
                    pass
            txt.remove()
    # Drop early-visual highlight band
    for p in list(ax.patches):
        p.remove()
    if mean_r is None:
        for line in ax.lines:
            yd = np.asarray(line.get_ydata(), dtype=float)
            if yd.size and np.allclose(yd, yd[0], rtol=0, atol=1e-9):
                mean_r = float(yd[0])
                break
    if mean_r is not None:
        ax.text(
            0.97, 0.96, f'mean r = {mean_r:.3f}',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=S_FS['annot'], color=_C_MEAN, zorder=5,
        )
    for coll in ax.collections:
        coll.set_alpha(0.22)
    for line in ax.lines:
        if line.get_linestyle() not in ('--', (0, (3.5, 2.2)), (0, (2.8, 1.8))):
            line.set_color(_C_GT)
            line.set_linewidth(1.15)
        else:
            line.set_color(_C_MEAN)
            line.set_linewidth(0.85)
    ax.set_xlabel('Time (ms)', fontsize=S_FS['axis'])
    ax.set_ylabel('Pearson r', fontsize=S_FS['axis'])
    y1 = ax.get_ylim()[1]
    ax.set_ylim(0.1, max(y1, 0.75))
    _style_spines(ax)


def _polish_psd(ax):
    # Band Greek letters already in the header strip
    for txt in list(ax.texts):
        if txt.get_text() in {'δ', 'θ', 'α', 'β', 'γ'}:
            txt.remove()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    # Restyle by label / linestyle (do not match hex substrings)
    for line in ax.lines:
        lab = (line.get_label() or '').lower()
        ls = line.get_linestyle()
        is_pred = (
            'gen' in lab or 'pred' in lab
            or ls not in ('-', 'solid')
        )
        if is_pred:
            line.set_color(_C_PRED)
            line.set_linewidth(1.15)
            line.set_linestyle((0, (3.2, 1.4)))
        else:
            line.set_color(_C_GT)
            line.set_linewidth(1.25)
            line.set_linestyle('-')
    ax.legend(
        handles=[
            Line2D([0], [0], color=_C_GT, lw=1.3, label='GT'),
            Line2D([0], [0], color=_C_PRED, lw=1.15, ls=(0, (3.2, 1.4)), label='Pred'),
        ],
        loc='lower left', fontsize=S_FS['legend'], frameon=False,
        handlelength=1.25, handletextpad=0.28, borderaxespad=0.1,
    )
    ax.set_xlabel('Frequency (Hz)', fontsize=S_FS['axis'])
    ax.set_ylabel('PSD (a.u.)', fontsize=S_FS['axis'])
    _style_spines(ax)


def _polish_fidelity(ax, *, legend_in_axes: bool = False):
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels(
        [_BAND_SHORT.get(lab, lab) for lab in labels],
        fontsize=S_FS['tick'],
    )
    ax.set_xlabel('Band', fontsize=S_FS['axis'])
    ax.set_ylabel('Bandpower r', fontsize=S_FS['axis'])
    for txt in ax.texts:
        s = txt.get_text()
        if s in ('***', '**', '*'):
            txt.set_fontsize(7.0)
            txt.set_color('#111827')
        elif s == 'ns':
            txt.set_fontsize(5.5)
            txt.set_color('#6B7280')
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    if legend_in_axes:
        # Gamma bars are low — lower-right avoids star brackets / high bars.
        ax.legend(
            fontsize=S_FS['legend'] - 0.4, loc='lower right', frameon=False,
            handlelength=0.85, handletextpad=0.25, borderaxespad=0.15,
            labelspacing=0.15,
        )
    _style_spines(ax)


# Shared header baselines (axes coordinates within each row's title strip)
_ROW1_TITLE_Y = 0.62
_ROW2_TITLE_Y = 0.52        # closer to meta line (grey bands / legend)
_ROW2_META_Y = 0.14         # c grey band labels ↔ d legend


def _fidelity_legend_in_header(ax_hdr, *, y: float = _ROW2_META_Y):
    """Centered True / Shuffle legend on the shared meta baseline."""
    ax_hdr.legend(
        handles=[
            Patch(facecolor='#0F4D92', edgecolor='none', label='True'),
            Patch(facecolor='#93C5FD', edgecolor='none', label='Shuffle'),
        ],
        loc='center',
        bbox_to_anchor=(0.5, y),
        fontsize=S_FS['legend'] - 0.4,
        frameon=False,
        handlelength=0.85,
        handletextpad=0.28,
        borderaxespad=0.0,
        labelspacing=0.12,
        ncol=2,
        columnspacing=0.85,
    )


def _psd_band_labels_in_header(ax_hdr, *, y: float = _ROW2_META_Y):
    """Grey δ/θ/α/β/γ labels on the shared meta baseline (aligned with d legend)."""
    for f0, f1, lab in (
        (0.5, 4, 'δ'), (4, 8, 'θ'), (8, 13, 'α'), (13, 30, 'β'), (30, 45, 'γ'),
    ):
        ax_hdr.text(
            (0.5 * (f0 + f1)) / 45.0, y, lab,
            ha='center', va='center',
            fontsize=S_FS['grey'], color='#6B7280',
            transform=ax_hdr.transAxes, clip_on=False,
        )


def plot_s_fig1_prediction_quality_supp():
    """Assemble S_fig1 (a–e) in Nature / Cell multi-panel style."""
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)
    _setup_s_fig1_rc()

    summary = _load_summary()
    _ensure_rmse_stats(summary)
    per_t_mean = np.array(summary['per_timepoint_pearson']['mean'])
    per_t_std = np.array(summary['per_timepoint_pearson']['std'])
    time_ms = np.arange(len(per_t_mean)) * (1000.0 / SFREQ)
    per_ch_all_path = os.path.join(RAW_DIR, 'per_channel_pearson_all.npy')
    per_ch_all = np.load(per_ch_all_path) if os.path.isfile(per_ch_all_path) else None

    vis = _load_aggregate()
    t_ms = _time_axis_ms(vis['erp_true'].shape[1])

    psd = _load_wholebrain_psd()
    fidelity_occ = calculate_frequency_fidelity_statistics(
        load_frequency_fidelity_data(occipital=True),
    )
    tm = np.asarray(fidelity_occ['true_mean'], dtype=np.float64)
    sm = np.asarray(fidelity_occ['shuffle_mean'], dtype=np.float64)
    te = np.asarray(fidelity_occ.get('true_sem', np.zeros_like(tm)), dtype=np.float64)
    se = np.asarray(fidelity_occ.get('shuffle_sem', np.zeros_like(sm)), dtype=np.float64)
    tops = np.maximum(tm + te, sm + se)
    ts = fidelity_occ.get('true_by_subject')
    ss = fidelity_occ.get('shuffle_by_subject')
    if ts is not None:
        tops = np.maximum(tops, np.nanmax(np.asarray(ts), axis=0))
    if ss is not None:
        tops = np.maximum(tops, np.nanmax(np.asarray(ss), axis=0))
    fid_ylim = (0.0, min(1.12, float(np.max(tops)) * 1.18))

    # Nature double-column (~183 mm); letters follow reading order:
    #   Row 1: a | b
    #   Row 2: c | d | e
    # Shared header height keeps plots aligned within each row.
    fig = plt.figure(figsize=(7.2, 5.20), facecolor='white')
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.05],
        hspace=0.30,
        left=0.072, right=0.980, top=0.960, bottom=0.088,
    )

    # ── Row 1: a | b ──────────────────────────────────────────────
    # cols: a | gap | GT | Pred | cbar
    gs1 = outer[0].subgridspec(
        2, 5,
        height_ratios=[0.20, 1.0],
        width_ratios=[1.05, 0.14, 1.20, 1.20, 0.032],
        hspace=0.02, wspace=0.16,
    )
    _title_strip(fig, gs1[0, 0], 'Per-timepoint correlation', y=_ROW1_TITLE_Y)

    # Main title spans both heatmaps (GT/Pred sit on each axes, truly centered).
    _title_strip(fig, gs1[0, 2:4], 'Grand-average ERP', y=_ROW1_TITLE_Y)

    ax_a = fig.add_subplot(gs1[1, 0])
    _plot_fig1_panel_c(ax_a, time_ms, per_t_mean, per_t_std)
    ax_a.set_title('')
    _polish_time_r(ax_a)
    _add_panel_letter(ax_a, 'a', x=-0.18, y=1.02)

    erp_t = np.asarray(vis['erp_true'], dtype=np.float64)
    erp_p = np.asarray(vis['erp_pred'], dtype=np.float64)
    vmax = float(np.max(np.abs(np.concatenate([erp_t, erp_p], axis=0))))
    if vmax <= 0:
        vmax = 1e-6
    ax_gt = fig.add_subplot(gs1[1, 2])
    im_b = _draw_channel_heatmap(
        ax_gt, erp_t, t_ms, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        ylabel='Channel', yticklabels=True,
    )
    ax_gt.text(
        0.5, 1.015, 'GT',
        transform=ax_gt.transAxes, ha='center', va='bottom',
        fontsize=S_FS['sublabel'], fontweight='bold', color='#374151',
        clip_on=False,
    )
    _add_panel_letter(ax_gt, 'b', x=-0.14, y=1.02)
    ax_pred = fig.add_subplot(gs1[1, 3])
    _draw_channel_heatmap(
        ax_pred, erp_p, t_ms, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        ylabel=None, yticklabels=False,
    )
    ax_pred.text(
        0.5, 1.015, 'Pred',
        transform=ax_pred.transAxes, ha='center', va='bottom',
        fontsize=S_FS['sublabel'], fontweight='bold', color='#374151',
        clip_on=False,
    )
    cax_b = fig.add_subplot(gs1[1, 4])
    _cbar_compact(cax_b, im_b, 'Amp.')

    # ── Row 2: c | d | e ──────────────────────────────────────────
    # Shared strip: title line + meta line (c grey bands ↔ d legend).
    gs2 = outer[1].subgridspec(
        2, 3,
        height_ratios=[0.26, 1.0],
        width_ratios=[1.0, 1.08, 1.45],
        hspace=0.012, wspace=0.28,
    )

    ax_c_hdr = fig.add_subplot(gs2[0, 0])
    ax_c_hdr.set_axis_off()
    ax_c_hdr.text(
        0.5, _ROW2_TITLE_Y, 'Mean PSD',
        ha='center', va='center',
        fontsize=S_FS['panel_title'], fontweight='bold', color='#111827',
        transform=ax_c_hdr.transAxes, clip_on=False,
    )
    _psd_band_labels_in_header(ax_c_hdr, y=_ROW2_META_Y)

    ax_d_hdr = fig.add_subplot(gs2[0, 1])
    ax_d_hdr.set_axis_off()
    ax_d_hdr.text(
        0.5, _ROW2_TITLE_Y, 'True–Shuffle band fidelity (occ.)',
        ha='center', va='center',
        fontsize=S_FS['panel_title'] - 0.5, fontweight='bold', color='#111827',
        transform=ax_d_hdr.transAxes, clip_on=False,
    )
    _fidelity_legend_in_header(ax_d_hdr, y=_ROW2_META_Y)

    _title_strip(fig, gs2[0, 2], 'Prediction metrics', y=_ROW2_TITLE_Y)

    ax_c = fig.add_subplot(gs2[1, 0])
    _plot_psd_curves(
        ax_c, psd['freqs'], psd['true'], psd['pred'],
        psd['true_sem'], psd['pred_sem'],
        title='', show_legend=False,
    )
    _polish_psd(ax_c)
    _add_panel_letter(ax_c, 'c', x=-0.18, y=1.02)

    ax_d = fig.add_subplot(gs2[1, 1])
    plot_frequency_fidelity(
        ax_d, fidelity_occ,
        title='',
        ylim=fid_ylim, show_legend=False,
    )
    _polish_fidelity(ax_d, legend_in_axes=False)
    _add_panel_letter(ax_d, 'd', x=-0.18, y=1.02)

    ax_e = fig.add_subplot(gs2[1, 2])
    _plot_metrics_compact(ax_e, summary, per_ch_all)
    _add_panel_letter(ax_e, 'e', x=-0.14, y=1.02)

    stem = fig_path(S_FIG1)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


if __name__ == '__main__':
    plot_s_fig1_prediction_quality_supp()
