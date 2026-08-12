"""Nature-style figures for generator waveform & frequency evaluation."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.eeg_gen_eval.config import (
    CHANNELS,
    CHANNELS_O,
    EEG_BANDS,
    ERP_COMPONENT_WINDOWS_MS,
    FIG_DIR,
    RAW_DIR,
    SFREQ,
)
from analysis.eeg_gen_eval.figure_names import FIG1, FIG2, fig_path
from analysis.eeg_gen_eval.compute.metrics import (
    _bandpass_fft,
    occipital_erp_component_fidelity,
    occipital_erp_component_pearson,
    psd_mean_sem,
)

PALETTE = {
    'true': '#0F4D92',
    'pred': '#E53935',
    'neutral': '#767676',
    'accent': '#42949E',
    'fill_true': '#3775BA33',
    'fill_pred': '#E9A6A133',
}


def _setup_nature_rc():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
        'font.size': 7,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.right': True,
        'axes.spines.top': True,
        'axes.linewidth': 0.8,
        'legend.frameon': False,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'axes.titlelocation': 'center',
    })


def add_panel_label(
    ax,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.10,
    fontsize: float = 11,
):
    """Nature-style panel letter (a, b, c, …) on an axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight='bold',
        va='top',
        ha='left',
        clip_on=False,
    )


def label_axes(axes, start: str = 'a'):
    """Label a sequence of axes with consecutive lowercase letters."""
    letters = 'abcdefghijklmnopqrstuvwxyz'
    i0 = letters.index(start)
    for ax, letter in zip(np.atleast_1d(axes).flat, letters[i0:]):
        add_panel_label(ax, letter)


def save_pub(
    fig,
    stem: str,
    dpi: int = 600,
    *,
    svg_dpi: Optional[int] = None,
):
    """Save publication figures as SVG (vector) + PNG (raster) only."""
    out_dir = os.path.dirname(stem) or FIG_DIR
    os.makedirs(out_dir, exist_ok=True)
    svg_kwargs = {'dpi': svg_dpi} if svg_dpi is not None else {}
    fig.savefig(
        f'{stem}.svg', bbox_inches='tight', facecolor='white', **svg_kwargs,
    )
    fig.savefig(f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')


def _label_bar_tops(
    ax,
    xs,
    heights,
    err_heights,
    *,
    fmt: str = '{:.2f}',
    gap: float | None = None,
    fontsize: float = 6,
    color: str = '#333333',
):
    """Annotate bar means above error bars."""
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo if yhi > ylo else 1.0
    if gap is None:
        gap = 0.03 * span
    label_top = ylo
    for xi, h, eh in zip(xs, heights, err_heights):
        if h != h:
            continue
        eh_plot = eh if eh == eh else 0.0
        y_text = h + eh_plot + gap
        ax.text(
            xi, y_text, fmt.format(h),
            ha='center', va='bottom', fontsize=fontsize,
            color=color, fontweight='bold', zorder=5, clip_on=False,
        )
        label_top = max(label_top, y_text)
    return label_top


def _set_bar_ylim(
    ax,
    label_top: float,
    *,
    base: float = 0.0,
    pad_frac: float = 0.16,
    pad_abs: float = 0.0,
    min_top: float | None = None,
):
    """Expand y-axis so bar labels sit clear of the top spine."""
    span = max(label_top - base, 1e-6)
    ymax = label_top + max(pad_frac * span, pad_abs)
    if min_top is not None:
        ymax = max(ymax, min_top)
    ax.set_ylim(base, ymax)


def _load_occipital_erp_component_by_subject() -> tuple[List[str], List[str], np.ndarray]:
    """Occipital ERP component Pearson r: (comp_names, subject_ids, values [n_sub, n_comp])."""
    comp_names = [w[0] for w in ERP_COMPONENT_WINDOWS_MS]
    subject_ids: List[str] = []
    rows: List[List[float]] = []
    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        pred_path = os.path.join(d, 'y_pred.npy')
        true_path = os.path.join(d, 'y_true.npy')
        if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
            continue
        comps = occipital_erp_component_pearson(
            np.load(pred_path), np.load(true_path),
        )
        subject_ids.append(sub_dir)
        rows.append([comps[name] for name in comp_names])
    return comp_names, subject_ids, np.asarray(rows, dtype=np.float64)


def _load_summary() -> Dict:
    path = os.path.join(RAW_DIR, 'summary_all_subjects.json')
    with open(path) as f:
        return json.load(f)


def _ensure_rmse_stats(summary: Dict) -> None:
    """Backfill aggregate RMSE from per-subject MSE/RMSE when missing."""
    if 'rmse' in summary:
        return
    rmse_vals = []
    for meta in summary.get('per_subject', []):
        if 'rmse' in meta:
            rmse_vals.append(float(meta['rmse']))
        elif 'mse' in meta:
            rmse_vals.append(float(np.sqrt(meta['mse'])))
    if rmse_vals:
        summary['rmse'] = {
            'mean': float(np.mean(rmse_vals)),
            'std': float(np.std(rmse_vals, ddof=0)),
        }


def _load_erp_component_fidelity_mean() -> Dict[str, Dict[str, float]]:
    """Mean occipital ERP component fidelity across subjects."""
    rows: List[Dict[str, Dict[str, float]]] = []
    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        pred_path = os.path.join(d, 'y_pred.npy')
        true_path = os.path.join(d, 'y_true.npy')
        if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
            continue
        rows.append(occipital_erp_component_fidelity(
            np.load(pred_path), np.load(true_path),
        ))
    if not rows:
        raise FileNotFoundError(f'No subject y_pred/y_true under {RAW_DIR}')

    comp_names = [w[0] for w in ERP_COMPONENT_WINDOWS_MS]
    keys = ('pearson', 'amplitude_corr', 'latency_consistency', 'latency_mae_ms', 'window_ms')
    out: Dict[str, Dict[str, float]] = {}
    for name in comp_names:
        out[name] = {
            k: float(np.mean([r[name][k] for r in rows]))
            for k in keys
        }
    return out


def _load_erp_component_fidelity_arrays() -> tuple[List[str], Dict[str, np.ndarray]]:
    """Per-subject occipital ERP fidelity arrays: metric → (n_sub, n_comp)."""
    comp_names = [w[0] for w in ERP_COMPONENT_WINDOWS_MS]
    keys = ['pearson', 'amplitude_corr', 'latency_mae_ms']
    rows: Dict[str, List[List[float]]] = {k: [] for k in keys}
    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        pred_path = os.path.join(d, 'y_pred.npy')
        true_path = os.path.join(d, 'y_true.npy')
        if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
            continue
        fid = occipital_erp_component_fidelity(np.load(pred_path), np.load(true_path))
        for k in keys:
            rows[k].append([float(fid[c][k]) for c in comp_names])
    return comp_names, {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}


def _plot_fig1_panel_c(ax, time_ms: np.ndarray, per_t_mean: np.ndarray, per_t_std: np.ndarray):
    """Nature-style per-timepoint Pearson r curve."""
    color = '#0F4D92'
    fill = '#0F4D9228'
    mean_r = float(per_t_mean.mean())

    ax.axvspan(80, 180, color='#FEF3C7', alpha=0.55, lw=0, zorder=0)
    ax.text(
        130, 0.98, 'early visual',
        transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=5, color='#92400E', clip_on=False,
    )

    ax.fill_between(
        time_ms, per_t_mean - per_t_std, per_t_mean + per_t_std,
        color=fill, linewidth=0, zorder=1,
    )
    ax.plot(time_ms, per_t_mean, color=color, lw=1.35, zorder=2, solid_capstyle='round')
    ax.axhline(mean_r, color='#B91C1C', ls=(0, (3.5, 2.2)), lw=0.85, zorder=3)
    ax.text(
        float(time_ms[-1]) - 8, mean_r + 0.025, f'mean = {mean_r:.3f}',
        ha='right', va='bottom', fontsize=5.5, color='#B91C1C', zorder=4,
    )

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pearson r')
    ax.set_xlim(float(time_ms[0]), float(time_ms[-1]))
    y0 = float(max(0.0, (per_t_mean - per_t_std).min() - 0.04))
    y1 = float(min(1.0, (per_t_mean + per_t_std).max() + 0.08))
    ax.set_ylim(y0, y1)
    ax.set_title('Per-timepoint correlation', fontweight='bold')
    _despine(ax)
    ax.tick_params(length=2.5, width=0.6)


def _plot_erp_fidelity_dotplot(
    ax, fidelity_arrays: Dict[str, np.ndarray], components: List[str],
    *, compact: bool = False,
    legend_fontsize: float | None = None,
    axis_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    short_legend: bool = False,
):
    """Nature-style ERP fidelity: means ± SEM, dual axis, subject dots."""
    from matplotlib.lines import Line2D

    xs = np.arange(len(components), dtype=float)
    if short_legend:
        left_specs = [
            ('pearson', 'Pearson r', 'o', '#0F4D92', -0.16),
            ('amplitude_corr', 'Amplitude r', '^', '#C2410C', 0.0),
        ]
        lat_label = 'Latency err.'
    else:
        left_specs = [
            ('pearson', 'Pearson correlation', 'o', '#0F4D92', -0.16),
            ('amplitude_corr', 'Amplitude correlation', '^', '#C2410C', 0.0),
        ]
        lat_label = 'Peak latency error'
    lat_key = 'latency_mae_ms'
    lat_color = '#3F6212'
    lat_dx = 0.16
    legend_fs = (4.5 if compact else 5.0) if legend_fontsize is None else float(legend_fontsize)
    axis_fs = 7.0 if axis_fontsize is None else float(axis_fontsize)
    tick_fs = 6.0 if tick_fontsize is None else float(tick_fontsize)

    for x in xs:
        ax.axvline(x, color='#F3F4F6', lw=4.5, zorder=0)

    rng = np.random.default_rng(1)
    for key, _lab, marker, color, dx in left_specs:
        arr = fidelity_arrays[key]
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
        xp = xs + dx
        ax.errorbar(
            xp, mean, yerr=sem, fmt='none',
            ecolor=color, elinewidth=0.9, capsize=2.0, capthick=0.8, zorder=2,
        )
        ax.plot(xp, mean, color=color, lw=0.8, alpha=0.35, zorder=2)
        ax.scatter(
            xp, mean, s=42 if not compact else 34, marker=marker, color=color,
            edgecolors='white', linewidths=0.6, zorder=4,
        )
        for i in range(arr.shape[0]):
            jitter = rng.uniform(-0.04, 0.04, size=len(components))
            ax.scatter(
                xp + jitter, arr[i], s=8 if not compact else 6, marker=marker,
                color=color, alpha=0.28, linewidths=0, zorder=3,
            )
        if not compact:
            for x, y, e in zip(xp, mean, sem):
                ax.text(
                    x, y + e + 0.03, f'{y:.2f}',
                    ha='center', va='bottom', fontsize=5.5, color=color, zorder=5,
                )

    ax2 = ax.twinx()
    arr_l = fidelity_arrays[lat_key]
    mean_l = arr_l.mean(axis=0)
    sem_l = arr_l.std(axis=0, ddof=1) / np.sqrt(arr_l.shape[0])
    xp_l = xs + lat_dx
    ax2.errorbar(
        xp_l, mean_l, yerr=sem_l, fmt='none',
        ecolor=lat_color, elinewidth=0.9, capsize=2.0, capthick=0.8, zorder=2,
    )
    ax2.plot(xp_l, mean_l, color=lat_color, lw=0.8, alpha=0.35, zorder=2)
    ax2.scatter(
        xp_l, mean_l, s=42 if not compact else 34, marker='s', color=lat_color,
        edgecolors='white', linewidths=0.6, zorder=4,
    )
    for i in range(arr_l.shape[0]):
        jitter = rng.uniform(-0.04, 0.04, size=len(components))
        ax2.scatter(
            xp_l + jitter, arr_l[i], s=8 if not compact else 6, marker='s',
            color=lat_color, alpha=0.28, linewidths=0, zorder=3,
        )
    if not compact:
        for x, y, e in zip(xp_l, mean_l, sem_l):
            ax2.text(
                x, y + e + 0.55, f'{y:.1f}',
                ha='center', va='bottom', fontsize=5.5, color=lat_color, zorder=5,
            )

    ax.set_xticks(xs)
    ax.set_xticklabels(components, fontsize=tick_fs)
    ax.set_xlabel('ERP component', fontsize=axis_fs)
    ax.set_ylabel('Fidelity score', fontsize=axis_fs)
    ax.set_xlim(-0.55, len(components) - 0.45)
    ax.set_ylim(0.0, 1.05 if compact else 1.12)
    ax.set_title(
        'Occipital ERP component fidelity', fontweight='bold',
        pad=2 if compact else 6, loc='center',
    )
    _despine(ax, left=True, bottom=True)
    ax.tick_params(length=2.5, width=0.6, labelsize=tick_fs)

    y_max = float((mean_l + sem_l).max())
    ax2.set_ylim(0.0, max(16.0, y_max * (1.22 if compact else 1.35)))
    ax2.set_ylabel('Latency error (ms) ↓', fontsize=axis_fs, color=lat_color)
    ax2.tick_params(axis='y', labelsize=tick_fs, colors=lat_color, length=2.5, width=0.6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(lat_color)
    ax2.spines['right'].set_linewidth(0.9)

    legend_handles = [
        Line2D([0], [0], marker=m, color='w', markerfacecolor=c,
               markeredgecolor='white', markersize=6.0 if compact else 6.5, label=lab)
        for _, lab, m, c, _ in left_specs
    ]
    legend_handles.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor=lat_color,
               markeredgecolor='white', markersize=6.0 if compact else 6.5,
               label=lat_label),
    )
    ax.legend(
        handles=legend_handles,
        loc='lower right',
        fontsize=legend_fs,
        frameon=False, handletextpad=0.3, borderaxespad=0.12,
        labelspacing=0.25 if compact else 0.35,
    )



def _despine(ax, *, left=True, bottom=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)


def _plot_fig1_panel_a(ax, summary: Dict, per_ch_all: np.ndarray | None):
    """Nature-style global metrics: grouped bars + subject dots."""
    metas = list(summary.get('per_subject', []))
    # Per-subject scalars for overlay dots
    sub_rmse = np.array([float(m.get('rmse', np.sqrt(m['mse']))) for m in metas], dtype=np.float64)
    sub_mae = np.array([float(m['mae']) for m in metas], dtype=np.float64)
    sub_pr = np.array([float(m['pearson_r']) for m in metas], dtype=np.float64)
    sub_r2 = np.array([float(m['r2_per_sample_waveform_mean']) for m in metas], dtype=np.float64)
    if per_ch_all is not None and per_ch_all.ndim == 2:
        sub_ch = per_ch_all.mean(axis=1)
    else:
        sub_ch = np.full(len(metas), float('nan'))

    # Layout: error metrics | fidelity metrics (visual gap)
    # positions: 0,1 | 3,4,5
    labels = ['RMSE', 'MAE', 'Pearson r', 'Channel r', 'R²']
    series = [sub_rmse, sub_mae, sub_pr, sub_ch, sub_r2]
    xpos = np.array([0.0, 1.0, 2.6, 3.6, 4.6])
    colors = ['#9AA0A6', '#6B7280', '#0F4D92', '#2B6CB0', '#3D8B8C']
    err_colors = ['#6B7280', '#4B5563', '#0A3A6E', '#1A4F86', '#2A6566']

    means = np.array([float(np.nanmean(s)) for s in series])
    sems = np.array([
        float(np.nanstd(s, ddof=1) / np.sqrt(np.sum(np.isfinite(s))))
        if np.sum(np.isfinite(s)) > 1 else 0.0
        for s in series
    ])

    rng = np.random.default_rng(0)
    for i, (x, s, c) in enumerate(zip(xpos, series, colors)):
        ax.bar(
            x, means[i], width=0.72, color=c, edgecolor='none',
            zorder=2, alpha=0.92,
        )
        ax.errorbar(
            x, means[i], yerr=sems[i], fmt='none',
            ecolor=err_colors[i], elinewidth=1.0, capsize=2.2, capthick=0.9,
            zorder=3,
        )
        # Subject dots
        ok = np.isfinite(s)
        if ok.any():
            jitter = rng.uniform(-0.14, 0.14, size=int(ok.sum()))
            ax.scatter(
                np.full(ok.sum(), x) + jitter, s[ok],
                s=9, color='#1F2937', alpha=0.35, linewidths=0,
                zorder=4, clip_on=False,
            )
        ax.text(
            x, means[i] + sems[i] + 0.028, f'{means[i]:.3f}',
            ha='center', va='bottom', fontsize=5.5, color='#374151', zorder=5,
        )

    # Group separators / captions
    ax.axvline(1.8, color='#E5E7EB', lw=0.8, zorder=0)
    ax.text(0.5, -0.16, 'Error ↓', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=5.5, color='#6B7280', clip_on=False)
    ax.text(3.6, -0.16, 'Fidelity ↑', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=5.5, color='#6B7280', clip_on=False)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_xlim(-0.55, 5.15)
    ymax = float(np.nanmax([np.nanmax(s) for s in series if np.any(np.isfinite(s))]))
    ax.set_ylim(0.0, max(0.75, ymax * 1.18))
    ax.set_ylabel('Value')
    ax.set_title('Reconstruction metrics', fontweight='bold')
    _despine(ax)
    ax.tick_params(length=2.5, width=0.6)


def _plot_fig1_panel_b(ax, per_ch_mean: np.ndarray, per_ch_std: np.ndarray):
    """Nature-style per-channel r: anterior→posterior gradient + region bands."""
    n = len(CHANNELS)
    x = np.arange(n, dtype=float)
    mean_r = float(per_ch_mean.mean())

    # Soft anatomical region bands (by channel order on the montage list)
    regions = [
        (0, 15, '#F8F8F8', 'Frontal'),
        (15, 35, '#F5F7FF', 'Central'),
        (35, 55, '#F4FBF7', 'Parietal'),
        (55, n, '#FFF8EB', 'Occipital'),
    ]
    for x0, x1, color, _name in regions:
        ax.axvspan(x0 - 0.5, x1 - 0.5, color=color, lw=0, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'ch_r', ['#93C5FD', '#1E3A8A'],
    )
    colors = cmap(np.linspace(0.05, 0.95, n))

    ax.vlines(
        x, per_ch_mean - per_ch_std, per_ch_mean + per_ch_std,
        colors='#CBD5E1', linewidths=0.5, alpha=0.8, zorder=1,
    )
    ax.scatter(
        x, per_ch_mean, c=colors, s=12, linewidths=0, zorder=3,
    )

    ax.axhline(mean_r, color='#B91C1C', ls=(0, (3.5, 2.2)), lw=0.9, zorder=4)
    ax.text(
        1.0, mean_r + 0.04, f'mean = {mean_r:.3f}',
        ha='left', va='bottom', fontsize=5.5, color='#B91C1C', zorder=5,
    )

    for x0, x1, _color, name in regions:
        ax.text(
            (x0 + x1) / 2 - 0.5, 1.01, name,
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=5, color='#6B7280',
            clip_on=False,
        )

    tick_idx = [0, 14, 30, 54, 62]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([CHANNELS[i] for i in tick_idx], fontsize=6)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('Channel (anterior → posterior)')
    ax.set_title('Per-channel correlation', fontweight='bold')
    _despine(ax)
    ax.tick_params(length=2.5, width=0.6)


def plot_fig1_basic_quality(summary: Dict | None = None):
    """Panel figure: scalars, channel/time r, ERP component fidelity dots."""
    _setup_nature_rc()
    if summary is None:
        summary = _load_summary()
    _ensure_rmse_stats(summary)

    per_ch_mean = np.array(summary['per_channel_pearson']['mean'])
    per_ch_std = np.array(summary['per_channel_pearson']['std'])
    per_t_mean = np.array(summary['per_timepoint_pearson']['mean'])
    per_t_std = np.array(summary['per_timepoint_pearson']['std'])
    time_ms = np.arange(len(per_t_mean)) * (1000.0 / SFREQ)

    per_ch_all_path = os.path.join(RAW_DIR, 'per_channel_pearson_all.npy')
    per_ch_all = np.load(per_ch_all_path) if os.path.isfile(per_ch_all_path) else None

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_fig1_panel_a(ax_a, summary, per_ch_all)

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_fig1_panel_b(ax_b, per_ch_mean, per_ch_std)

    # c — per-timepoint Pearson
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_fig1_panel_c(ax_c, time_ms, per_t_mean, per_t_std)

    # d — ERP component fidelity
    ax_d = fig.add_subplot(gs[1, 1])
    comps, fid_arr = _load_erp_component_fidelity_arrays()
    _plot_erp_fidelity_dotplot(ax_d, fid_arr, comps)

    label_axes([ax_a, ax_b, ax_c, ax_d])

    stem = fig_path(FIG1)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


# ---------------------------------------------------------------------------
# Fig2 panels c/d — frequency-specific neural fidelity (True vs Shuffle)
# Subject-level n=10; relative gap on subject means:
#   rel = (True − Shuffle) / True
#   H0: rel <  δ0
#   H1: rel ≥  δ0
# δ0=0.03 means Shuffle is ≥3% below True (relative to True).
# ---------------------------------------------------------------------------

FREQ_FIDELITY_BANDS: Tuple[str, ...] = ('All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma')
_FREQ_FIDELITY_FORMULA = 'image_wilcoxon_greater'
_FREQ_FIDELITY_CACHE_NAME = 'bandpower_true_vs_shuffle.json'
_N_IMAGES_PER_SUBJECT = 200
# Relative threshold: (True − Shuffle) / True ≥ δ0
_FIDELITY_DELTA0 = 0.03

# True / Shuffle fill + darker error-bar tones
_NPG = {
    'true': '#016190',
    'true_err': '#014A6E',
    'shuffle': '#C5DEFA',
    'shuffle_err': '#7AADD8',
    'edge': '#4B5563',
}


def _waveform_band_specs() -> List[Tuple[str, float, float]]:
    return [('All', 0.5, 45.0)] + [
        (name.capitalize(), fmin, fmax) for name, fmin, fmax in EEG_BANDS
    ]


def _per_image_pearson(
    y_pred_b: np.ndarray,
    y_true_b: np.ndarray,
    perm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pearson r per image on band-limited (C×T) waveforms. Shape (N,)."""
    if perm is not None:
        y_pred_b = y_pred_b[perm]
    n = y_pred_b.shape[0]
    a = y_pred_b.reshape(n, -1)
    b = y_true_b.reshape(n, -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    r = np.full(n, np.nan, dtype=np.float64)
    ok = den > 1e-12
    r[ok] = (a[ok] * b[ok]).sum(axis=1) / den[ok]
    return r


def _p_to_star(p: float) -> str:
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'ns'


def load_frequency_fidelity_data(
    *,
    n_shuffles: int = 50,
    seed: int = 2023,
    force: bool = False,
    occipital: bool = False,
) -> Dict[str, object]:
    """Image-level True/Shuffle r (n=2000) for one-sided Wilcoxon.

    If ``occipital``, restrict waveforms to CHANNELS_O (PO+O).
    """
    cache_path = (
        os.path.join(RAW_DIR, 'bandpower_true_vs_shuffle_occipital.json')
        if occipital
        else os.path.join(RAW_DIR, _FREQ_FIDELITY_CACHE_NAME)
    )
    formula = (
        f'{_FREQ_FIDELITY_FORMULA}_occipital' if occipital else _FREQ_FIDELITY_FORMULA
    )
    if not force and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if (
            cached.get('n_shuffles') == n_shuffles
            and cached.get('seed') == seed
            and cached.get('formula') == formula
        ):
            return cached

    ch_idx = [CHANNELS.index(ch) for ch in CHANNELS_O] if occipital else None
    rng = np.random.default_rng(seed)
    true_all = {b: [] for b in FREQ_FIDELITY_BANDS}
    shuf_all = {b: [] for b in FREQ_FIDELITY_BANDS}
    n_subjects = 0

    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        pred_path = os.path.join(d, 'y_pred.npy')
        true_path = os.path.join(d, 'y_true.npy')
        if not (os.path.isfile(pred_path) and os.path.isfile(true_path)):
            continue
        y_pred = np.load(pred_path)
        y_true = np.load(true_path)
        if ch_idx is not None:
            y_pred = y_pred[:, ch_idx, :]
            y_true = y_true[:, ch_idx, :]
        n_img = y_pred.shape[0]
        pred_bands = {
            lab: _bandpass_fft(y_pred, f0, f1)
            for lab, f0, f1 in _waveform_band_specs()
        }
        true_bands = {
            lab: _bandpass_fft(y_true, f0, f1)
            for lab, f0, f1 in _waveform_band_specs()
        }
        for band in FREQ_FIDELITY_BANDS:
            r_true = _per_image_pearson(pred_bands[band], true_bands[band])
            r_shuf_acc = np.zeros(n_img, dtype=np.float64)
            for _ in range(n_shuffles):
                perm = rng.permutation(n_img)
                r_shuf_acc += _per_image_pearson(
                    pred_bands[band], true_bands[band], perm,
                )
            r_shuf = r_shuf_acc / float(n_shuffles)
            true_all[band].extend(r_true.tolist())
            shuf_all[band].extend(r_shuf.tolist())
        n_subjects += 1
        roi = 'occipital' if occipital else 'all-ch'
        print(f'  fidelity ({roi}) {sub_dir}: {n_img} images', flush=True)

    if n_subjects == 0:
        raise FileNotFoundError(f'No subject y_pred/y_true under {RAW_DIR}')

    n_images = len(true_all[FREQ_FIDELITY_BANDS[0]])
    payload = {
        'formula': formula,
        'unit': 'image',
        'roi': 'occipital' if occipital else 'all',
        'n_samples': n_images,
        'n_shuffles': n_shuffles,
        'seed': seed,
        'bands': list(FREQ_FIDELITY_BANDS),
        'n_subjects': n_subjects,
        'n_images': n_images,
        'true': true_all,
        'shuffle': shuf_all,
        'description': (
            'Per-image band-limited waveform Pearson r; pool 10×200=2000 images. '
            + ('Occipital ROI (PO+O). ' if occipital else '')
            + 'One-sided Wilcoxon signed-rank, H1: True > Shuffle.'
        ),
    }
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(payload, f, indent=2)
    return payload


def calculate_frequency_fidelity_statistics(data: Dict[str, object]) -> Dict[str, object]:
    """Subject-level (n=10) means ± SEM; relative-gap Wilcoxon.

    Per subject: mean True/Shuffle over images, then
    rel = (True − Shuffle) / True.
    H0: rel <  δ0 (= ``_FIDELITY_DELTA0``)
    H1: rel ≥  δ0
    """
    bands = list(data['bands'])
    if 'true' not in data or not isinstance(data['true'], dict):
        raise ValueError(
            f'Cache formula={data.get("formula")!r} missing image-level arrays; '
            'recompute with load_frequency_fidelity_data(force=True).'
        )

    n_subj = int(data.get('n_subjects', 10))
    n_img = int(data.get('n_images', len(data['true'][bands[0]])))
    per = _N_IMAGES_PER_SUBJECT
    if n_subj * per != n_img:
        if n_img % per != 0:
            raise ValueError(f'Cannot reshape {n_img} images into subjects×{per}')
        n_subj = n_img // per

    true_img = np.column_stack([np.asarray(data['true'][b], dtype=np.float64) for b in bands])
    shuf_img = np.column_stack([np.asarray(data['shuffle'][b], dtype=np.float64) for b in bands])
    true_mat = true_img.reshape(n_subj, per, -1).mean(axis=1)
    shuf_mat = shuf_img.reshape(n_subj, per, -1).mean(axis=1)
    n = true_mat.shape[0]
    sem_scale = np.sqrt(n) if n > 1 else 1.0
    delta0 = float(_FIDELITY_DELTA0)
    eps = 1e-8

    true_mean = true_mat.mean(axis=0)
    shuf_mean = shuf_mat.mean(axis=0)
    true_sem = true_mat.std(axis=0, ddof=1) / sem_scale if n > 1 else np.zeros_like(true_mean)
    shuf_sem = shuf_mat.std(axis=0, ddof=1) / sem_scale if n > 1 else np.zeros_like(shuf_mean)

    p_values: List[float] = []
    stars: List[str] = []
    deltas: List[float] = []
    rel_means: List[float] = []
    for i, _band in enumerate(bands):
        t_col = true_mat[:, i]
        s_col = shuf_mat[:, i]
        abs_delta = float((t_col - s_col).mean())
        deltas.append(abs_delta)

        ok = np.abs(t_col) > eps
        if int(ok.sum()) < 2:
            p = 1.0
            rel_mean = float('nan')
        else:
            rel = (t_col[ok] - s_col[ok]) / t_col[ok]
            rel_mean = float(rel.mean())
            shifted = rel - delta0
            if np.allclose(shifted, 0.0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(
                    shifted, alternative='greater', zero_method='wilcox',
                ).pvalue)
        rel_means.append(rel_mean)
        p_values.append(p)
        stars.append(_p_to_star(p))

    return {
        'bands': bands,
        'true_mean': true_mean,
        'true_sem': true_sem,
        'shuffle_mean': shuf_mean,
        'shuffle_sem': shuf_sem,
        'true_by_subject': true_mat,
        'shuffle_by_subject': shuf_mat,
        'p_values': p_values,
        'deltas': deltas,
        'rel_means': rel_means,
        'delta0': delta0,
        'stars': stars,
        'n_samples': n,
        'n_subjects': n_subj,
    }


def plot_frequency_fidelity(
    ax,
    stats_dict: Dict[str, object],
    *,
    true_err: Optional[Sequence[float]] = None,
    shuffle_err: Optional[Sequence[float]] = None,
    title: str = 'Frequency-specific neural fidelity',
    ylim: Optional[Tuple[float, float]] = None,
    show_legend: bool = True,
):
    """Nature-style True vs Shuffle bars with subject dots and significance."""
    bands = list(stats_dict['bands'])
    true_mean = np.asarray(stats_dict['true_mean'], dtype=np.float64)
    shuf_mean = np.asarray(stats_dict['shuffle_mean'], dtype=np.float64)
    if true_err is None:
        true_err = stats_dict.get('true_sem')
    if shuffle_err is None:
        shuffle_err = stats_dict.get('shuffle_sem')
    true_err = None if true_err is None else np.asarray(true_err, dtype=np.float64)
    shuf_err = None if shuffle_err is None else np.asarray(shuffle_err, dtype=np.float64)
    stars = list(stats_dict['stars'])
    true_sub = stats_dict.get('true_by_subject')
    shuf_sub = stats_dict.get('shuffle_by_subject')
    if true_sub is not None:
        true_sub = np.asarray(true_sub, dtype=np.float64)
    if shuf_sub is not None:
        shuf_sub = np.asarray(shuf_sub, dtype=np.float64)

    x = np.arange(len(bands), dtype=float)
    width = 0.34
    gap = 0.03
    x_true = x - width / 2 - gap / 2
    x_shuf = x + width / 2 + gap / 2
    color_true = '#0F4D92'
    color_shuf = '#93C5FD'
    err_true = '#0A3A6E'
    err_shuf = '#64748B'

    ax.bar(
        x_true, true_mean, width=width, color=color_true, edgecolor='none',
        label='True', zorder=2, alpha=0.92,
    )
    ax.bar(
        x_shuf, shuf_mean, width=width, color=color_shuf, edgecolor='none',
        label='Shuffle', zorder=2, alpha=0.95,
    )
    if true_err is not None:
        ax.errorbar(
            x_true, true_mean, yerr=true_err, fmt='none',
            ecolor=err_true, elinewidth=0.85, capsize=1.8, capthick=0.75, zorder=3,
        )
    if shuf_err is not None:
        ax.errorbar(
            x_shuf, shuf_mean, yerr=shuf_err, fmt='none',
            ecolor=err_shuf, elinewidth=0.85, capsize=1.8, capthick=0.75, zorder=3,
        )

    rng = np.random.default_rng(2)
    if true_sub is not None and shuf_sub is not None:
        for i in range(len(bands)):
            jt = rng.uniform(-0.08, 0.08, size=true_sub.shape[0])
            js = rng.uniform(-0.08, 0.08, size=shuf_sub.shape[0])
            ax.scatter(
                np.full(true_sub.shape[0], x_true[i]) + jt, true_sub[:, i],
                s=9, color='#111827', alpha=0.30, linewidths=0, zorder=4,
            )
            ax.scatter(
                np.full(shuf_sub.shape[0], x_shuf[i]) + js, shuf_sub[:, i],
                s=9, color='#111827', alpha=0.24, linewidths=0, zorder=4,
            )

    y_tops = []
    for i in range(len(bands)):
        t_top = true_mean[i] + (0.0 if true_err is None else float(true_err[i]))
        s_top = shuf_mean[i] + (0.0 if shuf_err is None else float(shuf_err[i]))
        if true_sub is not None:
            t_top = max(t_top, float(np.nanmax(true_sub[:, i])))
        if shuf_sub is not None:
            s_top = max(s_top, float(np.nanmax(shuf_sub[:, i])))
        y_tops.append(max(t_top, s_top))

    data_max = max(float(np.max(y_tops)), 1e-6)
    star_gap = data_max * 0.038
    bracket_h = data_max * 0.020
    if ylim is None:
        y_lim = max(data_max * 1.18, max(y_tops) + star_gap + bracket_h + data_max * 0.10)
        ax.set_ylim(0.0, min(1.18, y_lim))
    else:
        ax.set_ylim(*ylim)

    for i, star in enumerate(stars):
        y0 = y_tops[i] + star_gap
        x0, x1 = x_true[i], x_shuf[i]
        ax.plot(
            [x0, x0, x1, x1], [y0, y0 + bracket_h, y0 + bracket_h, y0],
            color='#6B7280', lw=0.65, clip_on=False, zorder=5,
        )
        ax.text(
            x[i], y0 + bracket_h + star_gap * 0.15, star,
            ha='center', va='bottom',
            fontsize=6.5 if star != 'ns' else 5.5,
            fontstyle='normal' if star != 'ns' else 'italic',
            color='#374151', clip_on=False, zorder=6,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(bands, fontsize=6.5)
    ax.set_xlabel('Frequency band')
    ax.set_ylabel('Bandpower correlation')
    ax.set_title(title, fontweight='bold', pad=4)
    if show_legend:
        ax.legend(
            fontsize=5.5, loc='upper right', frameon=False,
            handlelength=1.0, handletextpad=0.35, borderaxespad=0.15,
            labelspacing=0.25,
        )
    _despine(ax)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.6)
    ax.set_axisbelow(False)


def _plot_psd_curves(
    ax, freqs, true_mean, pred_mean, true_sem, pred_sem, *,
    title: str, show_legend: bool = True,
):
    """Nature-style GT vs Gen PSD (semilogy, ≤45 Hz)."""
    mask = np.asarray(freqs) <= 45
    f = np.asarray(freqs)[mask]
    gt = np.asarray(true_mean)[mask]
    gen = np.asarray(pred_mean)[mask]
    gt_sem = np.asarray(true_sem)[mask]
    gen_sem = np.asarray(pred_sem)[mask]

    # Soft canonical band guides + Greek labels (Nature-style)
    band_spans = [
        (0.5, 4, '#F8FAFC', 'δ'),
        (4, 8, '#EEF2F7', 'θ'),
        (8, 13, '#F8FAFC', 'α'),
        (13, 30, '#EEF2F7', 'β'),
        (30, 45, '#F8FAFC', 'γ'),
    ]
    for f0, f1, c, _lab in band_spans:
        ax.axvspan(f0, f1, color=c, lw=0, zorder=0)

    ax.fill_between(
        f, np.maximum(gt - gt_sem, 1e-12), gt + gt_sem,
        color='#0F4D922E', linewidth=0, zorder=1,
    )
    ax.fill_between(
        f, np.maximum(gen - gen_sem, 1e-12), gen + gen_sem,
        color='#DC262628', linewidth=0, zorder=1,
    )
    ax.semilogy(f, gt, color='#0F4D92', lw=1.55, label='Ground truth', zorder=3)
    ax.semilogy(
        f, gen, color='#DC2626', lw=1.35, ls=(0, (3.5, 1.4)),
        label='Generated', zorder=3,
    )

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (a.u.)')
    ax.set_xlim(0.0, 45.0)
    ax.set_title(title, fontweight='bold', pad=8)
    # Band labels in axes coords (above plot, Nature-style)
    for f0, f1, _c, lab in band_spans:
        x_ax = (0.5 * (f0 + f1)) / 45.0
        ax.text(
            x_ax, 1.015, lab, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=6.5, color='#6B7280',
            clip_on=False, zorder=4,
        )
    if show_legend:
        ax.legend(
            fontsize=5.5, loc='upper right', frameon=False,
            handlelength=1.5, handletextpad=0.35, borderaxespad=0.15,
            labelspacing=0.25,
        )
    _despine(ax)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.6)
    ax.grid(False)


def _fig2_topo_info():
    import mne
    if not hasattr(_fig2_topo_info, '_info'):
        info = mne.create_info(CHANNELS, SFREQ, ch_types='eeg')
        info.set_montage('standard_1020', match_case=False, on_missing='ignore')
        _fig2_topo_info._info = info
    return _fig2_topo_info._info


# Match fig4 topomap template (plot_visualizations.TOPO_*)
_FIG2_TOPO_SPHERE = (0, 0.0, 0, 0.105)
_FIG2_TOPO_CMAP = 'RdBu_r'


def _plot_fig2_topomap(vals, ax, *, vmin: float, vmax: float):
    """Fig4-style scalp map: RdBu_r, sensors, contours=4, head outline."""
    import mne

    if vmax <= vmin:
        vmax = vmin + 1e-6
    im, _ = mne.viz.plot_topomap(
        vals, _fig2_topo_info(), axes=ax, show=False,
        cmap=_FIG2_TOPO_CMAP, vlim=(vmin, vmax),
        sphere=_FIG2_TOPO_SPHERE,
        contours=4, sensors=True, outlines='head', extrapolate='head',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def _mean_log_bandpower_topo(y: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Channel-mean log bandpower over samples. y: (N, C, T) → (C,)."""
    y = np.asarray(y, dtype=np.float64)
    spec = np.fft.rfft(y, axis=-1)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / SFREQ)
    mask = (freqs >= fmin) & (freqs < fmax)
    bp = np.log1p(power[..., mask].mean(axis=-1))
    return bp.mean(axis=0)


def plot_fig2_frequency(summary: Dict | None = None):
    """Fig2 frequency-domain suite (5 panels).

    a Mean PSD (whole-brain)
    b Visual cortex PSD (occipital ROI)
    c Neural fidelity True vs Shuffle (all channels)
    d Neural fidelity True vs Shuffle (occipital ROI)
    e Topomap band power (All + bands)
    """
    _setup_nature_rc()
    if summary is None:
        summary = _load_summary()

    o_idx = [CHANNELS.index(ch) for ch in CHANNELS_O]
    # All + named EEG bands for topomap
    topo_band_specs = [('All', 0.5, 45.0)] + [
        (name.capitalize(), fmin, fmax) for name, fmin, fmax in EEG_BANDS
    ]
    topo_labels = [lab for lab, _, _ in topo_band_specs]

    psd_freqs = None
    psd_true, psd_pred = [], []
    psd_true_sem, psd_pred_sem = [], []
    occ_true, occ_pred = [], []
    occ_true_sem, occ_pred_sem = [], []
    occ_freqs = None
    # topo: list of (C, n_topo_bands) per subject
    topo_true_list, topo_pred_list = [], []

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

        y_true = np.load(os.path.join(d, 'y_true.npy'))
        y_pred = np.load(os.path.join(d, 'y_pred.npy'))

        pt = psd_mean_sem(y_true[:, o_idx, :])
        pp = psd_mean_sem(y_pred[:, o_idx, :])
        occ_freqs = pt['freqs_hz']
        occ_true.append(pt['psd_mean'])
        occ_pred.append(pp['psd_mean'])
        occ_true_sem.append(pt['psd_sem'])
        occ_pred_sem.append(pp['psd_sem'])

        bp_t = np.load(os.path.join(d, 'bandpower_true.npy')).mean(axis=0)  # (C, 5)
        bp_p = np.load(os.path.join(d, 'bandpower_pred.npy')).mean(axis=0)
        all_t = _mean_log_bandpower_topo(y_true, 0.5, 45.0)
        all_p = _mean_log_bandpower_topo(y_pred, 0.5, 45.0)
        topo_true_list.append(np.column_stack([all_t, bp_t]))
        topo_pred_list.append(np.column_stack([all_p, bp_p]))
        print(f'  fig2 extras {sub_dir}', flush=True)

    psd_true = np.stack(psd_true).mean(axis=0)
    psd_pred = np.stack(psd_pred).mean(axis=0)
    psd_true_sem = np.stack(psd_true_sem).mean(axis=0)
    psd_pred_sem = np.stack(psd_pred_sem).mean(axis=0)

    occ_true = np.stack(occ_true).mean(axis=0)
    occ_pred = np.stack(occ_pred).mean(axis=0)
    occ_true_sem = np.stack(occ_true_sem).mean(axis=0)
    occ_pred_sem = np.stack(occ_pred_sem).mean(axis=0)

    topo_true = np.stack(topo_true_list).mean(axis=0)  # (C, 6)
    topo_pred = np.stack(topo_pred_list).mean(axis=0)

    fidelity_all = calculate_frequency_fidelity_statistics(
        load_frequency_fidelity_data(occipital=False),
    )
    fidelity_occ = calculate_frequency_fidelity_statistics(
        load_frequency_fidelity_data(occipital=True),
    )

    # Shared y-lim for c/d so all-channel vs occipital are visually comparable
    def _fidelity_ymax(st: Dict[str, object]) -> float:
        tm = np.asarray(st['true_mean'], dtype=np.float64)
        sm = np.asarray(st['shuffle_mean'], dtype=np.float64)
        te = np.asarray(st.get('true_sem', np.zeros_like(tm)), dtype=np.float64)
        se = np.asarray(st.get('shuffle_sem', np.zeros_like(sm)), dtype=np.float64)
        tops = np.maximum(tm + te, sm + se)
        ts = st.get('true_by_subject')
        ss = st.get('shuffle_by_subject')
        if ts is not None:
            tops = np.maximum(tops, np.nanmax(np.asarray(ts), axis=0))
        if ss is not None:
            tops = np.maximum(tops, np.nanmax(np.asarray(ss), axis=0))
        data_max = float(np.max(tops))
        return min(1.18, data_max * 1.22)

    fid_ylim = (0.0, max(_fidelity_ymax(fidelity_all), _fidelity_ymax(fidelity_occ)))

    fig = plt.figure(figsize=(7.2, 8.8))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.05, 1.65],
        hspace=0.48,
        wspace=0.32,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_psd_curves(
        ax_a, psd_freqs, psd_true, psd_pred, psd_true_sem, psd_pred_sem,
        title='Mean power spectral density', show_legend=True,
    )

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_psd_curves(
        ax_b, occ_freqs, occ_true, occ_pred, occ_true_sem, occ_pred_sem,
        title='Visual cortex PSD (occipital ROI)', show_legend=False,
    )

    ax_c = fig.add_subplot(gs[1, 0])
    plot_frequency_fidelity(
        ax_c, fidelity_all, title='Neural fidelity (all channels)',
        ylim=fid_ylim, show_legend=True,
    )

    ax_d = fig.add_subplot(gs[1, 1])
    plot_frequency_fidelity(
        ax_d, fidelity_occ, title='Neural fidelity (occipital ROI)',
        ylim=fid_ylim, show_legend=False,
    )

    # e — fig4-style topomap band power (RdBu_r, sensors, contours)
    n_bands = len(topo_labels)
    topo_short = {
        'All': 'All', 'Delta': 'δ', 'Theta': 'θ',
        'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ',
    }
    # Shared absolute scale across GT/Gen × bands (same idea as fig4 shared vlim)
    vmin = float(min(topo_true.min(), topo_pred.min()))
    vmax = float(max(topo_true.max(), topo_pred.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Layout mirrors plot_visualizations._plot_topomap_panel
    outer_e = gs[2, :]
    width_ratios = [0.06] + [1.0] * n_bands + [0.22, 0.14]  # labels | maps | gap | cbar
    inner_e = outer_e.subgridspec(
        4, n_bands + 3,
        width_ratios=width_ratios,
        height_ratios=[0.12, 0.10, 1.0, 1.0],
        wspace=0.02,
        hspace=0.14,
    )

    ax_e_title = fig.add_subplot(inner_e[0, 1:1 + n_bands])
    ax_e_title.set_axis_off()
    ax_e_title.text(
        0.5, 0.35, 'Topomap band power',
        ha='center', va='center', fontsize=9, fontweight='bold',
        transform=ax_e_title.transAxes,
    )

    ax_lbl_t = fig.add_subplot(inner_e[2, 0])
    ax_lbl_t.set_axis_off()
    ax_lbl_t.text(
        1.0, 0.5, 'GT', ha='right', va='center', fontsize=7.5,
        fontweight='bold', transform=ax_lbl_t.transAxes,
    )
    ax_lbl_p = fig.add_subplot(inner_e[3, 0])
    ax_lbl_p.set_axis_off()
    ax_lbl_p.text(
        1.0, 0.5, 'Gen', ha='right', va='center', fontsize=7.5,
        fontweight='bold', transform=ax_lbl_p.transAxes,
    )
    ax_e_anchor = ax_lbl_t

    last_im = None
    for bi, name in enumerate(topo_labels):
        ax_col = fig.add_subplot(inner_e[1, bi + 1])
        ax_col.set_axis_off()
        ax_col.text(
            0.5, 0.15, topo_short.get(name, name),
            ha='center', va='bottom', fontsize=7.0,
            transform=ax_col.transAxes,
        )

        ax_t = fig.add_subplot(inner_e[2, bi + 1])
        ax_p = fig.add_subplot(inner_e[3, bi + 1])
        _plot_fig2_topomap(topo_true[:, bi], ax_t, vmin=vmin, vmax=vmax)
        last_im = _plot_fig2_topomap(topo_pred[:, bi], ax_p, vmin=vmin, vmax=vmax)

    # Invisible spacer (fig4 keeps gap before colorbar)
    ax_gap = fig.add_subplot(inner_e[2:4, n_bands + 1])
    ax_gap.set_axis_off()
    ax_gap.set_xlim(0, 1)
    ax_gap.set_ylim(0, 1)
    ax_gap.plot([0], [0], alpha=0.0)

    cax = fig.add_subplot(inner_e[2:4, n_bands + 2])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label('log BP', fontsize=7, labelpad=6)
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.5, pad=2)
    ticks = np.linspace(vmin, vmax, 5)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:.2f}' for t in ticks])
    cb.outline.set_linewidth(0.55)

    label_axes([ax_a, ax_b, ax_c, ax_d, ax_e_anchor])
    stem = fig_path(FIG2)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_all_figures():
    summary = _load_summary()
    plot_fig1_basic_quality(summary)
    plot_fig2_frequency(summary)
