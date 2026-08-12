"""Section III: visualization experiments (Nature-style)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.eeg_gen_eval.config import CHANNELS, CHANNELS_O, FIG_DIR, SFREQ
from analysis.eeg_gen_eval.figure_names import FIG3, FIG4, FIG5, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import (
    PALETTE,
    _despine,
    _setup_nature_rc,
    label_axes,
    save_pub,
)
from analysis.eeg_gen_eval.helpers.vis_data import KEY_TIMEPOINTS_MS, VIS_DIR, ms_to_idx

WAVEFORM_CHANNELS = ['Fp1', 'Cz', 'Pz', 'Oz']  # legacy
TOPO_TIMES_MS = [100, 150, 200]
TOPO_100MS_BINS = [(i, i + 100) for i in range(0, 1000, 100)]  # 0–100, …, 900–1000 ms
TOPO_SPHERE = (0, 0.0, 0, 0.105)
TOPO_CMAP = 'RdBu_r'
PANEL_A_N_CHANNELS = 6
PANEL_A_N_COLS = 3

# Fig4/topo helpers keep slightly larger type; fig3 main figure uses Nature rc (~7 pt).
FIG3_FONT = {
    'panel_title': 9,
    'axis_label': 7,
    'tick': 6.5,
    'legend': 5.5,
    'topo_time': 10,
    'topo_row': 9,
    'colorbar_label': 7,
    'colorbar_tick': 6.5,
    'channel_tag': 6.5,
}


BRAIN_REGION_ORDER = ['Frontal', 'Central', 'Temporal', 'Parietal', 'Occipital']
BRAIN_REGION_COLORS = [
    PALETTE['true'],
    '#6FA8DC',
    '#B279A2',
    PALETTE['accent'],
    PALETTE['pred'],
]


def _channel_region(ch: str) -> str:
    """Map each channel to one mutually exclusive scalp region.

    Border prefixes are assigned consistently for visualization:
    FC→Frontal, CP→Central, FT/TP→Temporal, PO→Occipital.
    """
    if ch.startswith('Fp') or ch.startswith('AF'):
        return 'Frontal'
    if ch.startswith('FT'):
        return 'Temporal'
    if ch.startswith('FC'):
        return 'Frontal'
    if ch.startswith('F'):
        return 'Frontal'
    if ch.startswith('TP'):
        return 'Temporal'
    if ch.startswith('T'):
        return 'Temporal'
    if ch.startswith('CP'):
        return 'Central'
    if ch.startswith('PO'):
        return 'Occipital'
    if ch.startswith('O'):
        return 'Occipital'
    if ch.startswith('P'):
        return 'Parietal'
    if ch.startswith('C'):
        return 'Central'
    raise ValueError(f'Unmapped channel: {ch}')


def _load_region_reconstruction_samples() -> tuple[Dict[str, np.ndarray], np.ndarray]:
    """Regional Pearson-r pools with one mutually exclusive region per channel."""
    region_samples = {r: [] for r in BRAIN_REGION_ORDER}
    ch_regions = [_channel_region(ch) for ch in CHANNELS]
    all_vals: List[float] = []
    for sub_dir in sorted(os.listdir(VIS_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        path = os.path.join(VIS_DIR, sub_dir, 'per_channel_pearson.npy')
        if not os.path.isfile(path):
            continue
        per_ch = np.load(path)
        all_vals.extend(float(x) for x in per_ch)
        for ci, region in enumerate(ch_regions):
            val = float(per_ch[ci])
            region_samples[region].append(val)
    return (
        {r: np.asarray(v, dtype=np.float64) for r, v in region_samples.items()},
        np.asarray(all_vals, dtype=np.float64),
    )


def _filter_region_display_outliers(values: np.ndarray) -> np.ndarray:
    """Hide Tukey 1.5×IQR outliers for display without changing statistics."""
    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = q3 - q1
    if iqr <= 0:
        return values
    return values[(values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)]


def _plot_panel_c_region_boxplot(
    ax, *, compact: bool = False, show_title: bool = True,
    annot_fontsize: float | None = None,
    axis_fontsize: float | None = None,
    tick_fontsize: float | None = None,
):
    """Panel c: regional reconstruction quality (per-channel Pearson r)."""
    samples, all_vals_unfiltered = _load_region_reconstruction_samples()
    active_regions = [
        r for r in BRAIN_REGION_ORDER
        if len(samples.get(r, np.asarray([]))) > 0
    ]
    active_colors = [
        BRAIN_REGION_COLORS[BRAIN_REGION_ORDER.index(r)] for r in active_regions
    ]
    plot_samples = {
        region: _filter_region_display_outliers(samples[region])
        for region in active_regions
    }
    data = [plot_samples[r] for r in active_regions]
    rng = np.random.default_rng(0)
    annot_fs = FIG3_FONT['legend'] if annot_fontsize is None else float(annot_fontsize)
    axis_fs = FIG3_FONT['axis_label'] if axis_fontsize is None else float(axis_fontsize)
    tick_fs = FIG3_FONT['tick'] if tick_fontsize is None else float(tick_fontsize)

    bp = ax.boxplot(
        data,
        tick_labels=active_regions,
        patch_artist=True,
        widths=0.52,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 1.2},
        boxprops={'linewidth': 0.8, 'edgecolor': 'black'},
        whiskerprops={'linewidth': 0.8, 'color': 'black'},
        capprops={'linewidth': 0.8, 'color': 'black'},
    )
    for patch, color in zip(bp['boxes'], active_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)

    for i, (region, color) in enumerate(zip(active_regions, active_colors), start=1):
        y = plot_samples[region]
        x = rng.normal(i, 0.07, size=len(y))
        ax.scatter(
            x, y, s=14 if not compact else 11, color=color, alpha=0.45,
            edgecolors='white', linewidths=0.25, zorder=3,
        )

    all_vals = np.concatenate(data)
    y_min = float(np.floor(all_vals.min() * 10) / 10)
    tick_step = 0.2
    tick_lo = tick_step * np.floor(y_min / tick_step)
    grand_mean = float(np.mean(all_vals_unfiltered))
    ax.axhline(grand_mean, color=PALETTE['neutral'], ls='--', lw=0.9, zorder=1)
    ax.axhline(0, color=PALETTE['neutral'], ls=':', lw=0.6, alpha=0.7)
    ax.text(
        0.03, 0.97, f'mean r = {grand_mean:.3f}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=annot_fs, color='#6B7280', zorder=5,
    )
    ax.set_xlabel('Brain region', fontsize=axis_fs)
    ax.set_ylabel('Pearson r', fontsize=axis_fs)
    if show_title:
        ax.set_title(
            'Regional reconstruction quality',
            fontweight='bold', fontsize=FIG3_FONT['panel_title'],
            pad=3 if compact else 6, loc='center',
        )
    ax.set_ylim(y_min - 0.05, 1.0)
    ax.set_yticks(np.arange(tick_lo, 1.0 + tick_step * 0.01, tick_step))
    ax.tick_params(axis='both', labelsize=tick_fs)
    if compact:
        ax.tick_params(axis='x', labelrotation=12)
        for label in ax.get_xticklabels():
            label.set_ha('center')
            label.set_rotation_mode('default')
    _despine(ax)


def _o_indices() -> List[int]:
    return [CHANNELS.index(ch) for ch in CHANNELS_O]


def _load_occipital_subject_erps() -> tuple[np.ndarray, np.ndarray]:
    """Per-subject O-region mean ERP. Returns (n_sub, T) for true and pred."""
    o_idx = _o_indices()
    erp_t, erp_p = [], []
    for sub_dir in sorted(os.listdir(VIS_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(VIS_DIR, sub_dir)
        erp_t.append(np.load(os.path.join(d, 'erp_true_mean.npy'))[o_idx].mean(axis=0))
        erp_p.append(np.load(os.path.join(d, 'erp_pred_mean.npy'))[o_idx].mean(axis=0))
    return np.stack(erp_t), np.stack(erp_p)


def _plot_panel_b_occipital_erp(
    ax, t_ms: np.ndarray, *, compact: bool = False,
    annot_fontsize: float | None = None,
    legend_fontsize: float | None = None,
    axis_fontsize: float | None = None,
    tick_fontsize: float | None = None,
    show_early_visual: bool = True,
):
    """Panel b: O-region grand-average ERP, mean ± SEM (Nature style)."""
    sub_t, sub_p = _load_occipital_subject_erps()
    n_sub = sub_t.shape[0]
    annot_fs = (5.0 if compact else 5.5) if annot_fontsize is None else float(annot_fontsize)
    legend_fs = (5.0 if compact else 5.5) if legend_fontsize is None else float(legend_fontsize)
    axis_fs = FIG3_FONT['axis_label'] if axis_fontsize is None else float(axis_fontsize)
    tick_fs = FIG3_FONT['tick'] if tick_fontsize is None else float(tick_fontsize)

    if show_early_visual:
        # Soft early-visual window (C1/P1/N1-ish range)
        ax.axvspan(80, 180, color='#FFF7ED', lw=0, zorder=0)
        ax.text(
            130, 0.96 if compact else 0.98, 'early visual',
            transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=annot_fs,
            color='#9A3412', zorder=5,
        )

    for sub_arr, color, fill, label, ls in (
        (sub_t, '#0F4D92', '#0F4D9228', 'GT', '-'),
        (sub_p, '#DC2626', '#DC262622', 'Pred', (0, (3.5, 1.4))),
    ):
        mean = sub_arr.mean(axis=0)
        sem = sub_arr.std(axis=0, ddof=1) / np.sqrt(n_sub)
        ax.fill_between(t_ms, mean - sem, mean + sem, color=fill, linewidth=0, zorder=1)
        ax.plot(t_ms, mean, color=color, lw=1.35, ls=ls, label=label, zorder=3)

    ax.axhline(0, color='#9CA3AF', lw=0.5, ls=':', zorder=2)
    # Data ends near 996 ms at 250 Hz; force 0–1000 axis so 1000 ms is labeled.
    ax.set_xlim(0.0, 1000.0)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlabel('Time (ms)', fontsize=axis_fs)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=axis_fs)
    ax.set_title(
        'Occipital ERP comparison', fontweight='bold',
        pad=2 if compact else 4, loc='center',
    )
    ax.legend(
        fontsize=legend_fs,
        loc='lower left', frameon=False,
        handlelength=1.5, handletextpad=0.35, borderaxespad=0.15,
    )
    _despine(ax)
    ax.tick_params(length=2.5, width=0.6, labelsize=tick_fs)


def _setup_fig3_rc():
    _setup_nature_rc()
    mpl.rcParams.update({
        'font.size': FIG3_FONT['tick'],
        'axes.titlesize': FIG3_FONT['panel_title'],
        'axes.labelsize': FIG3_FONT['axis_label'],
        'xtick.labelsize': FIG3_FONT['tick'],
        'ytick.labelsize': FIG3_FONT['tick'],
        'legend.fontsize': FIG3_FONT['legend'],
    })


def _time_axis_ms(n_time: int) -> np.ndarray:
    return np.arange(n_time) * (1000.0 / SFREQ)


def _load_aggregate() -> Dict:
    d = os.path.join(VIS_DIR, 'aggregate')
    with open(os.path.join(d, 'meta.json')) as f:
        meta = json.load(f)
    erp_true = np.load(os.path.join(d, 'erp_true_mean.npy'))
    if os.path.isfile(os.path.join(d, 'panel_a_erp_true.npy')):
        panel_a_erp_true = np.load(os.path.join(d, 'panel_a_erp_true.npy'))
        panel_a_erp_pred = np.load(os.path.join(d, 'panel_a_erp_pred.npy'))
        panel_a_per_ch = np.load(os.path.join(d, 'panel_a_per_channel_pearson.npy'))
    else:
        panel_a_erp_true = erp_true
        panel_a_erp_pred = np.load(os.path.join(d, 'erp_pred_mean.npy'))
        panel_a_per_ch = np.load(os.path.join(d, 'per_channel_pearson_mean.npy'))
    return {
        'dir': d,
        'meta': meta,
        'erp_true': erp_true,
        'erp_pred': np.load(os.path.join(d, 'erp_pred_mean.npy')),
        'error_map': np.load(os.path.join(d, 'error_map_mean.npy')),
        'per_ch': np.load(os.path.join(d, 'per_channel_pearson_mean.npy')),
        'per_ch_std': np.load(os.path.join(d, 'per_channel_pearson_std.npy')),
        'panel_a_erp_true': panel_a_erp_true,
        'panel_a_erp_pred': panel_a_erp_pred,
        'panel_a_per_ch': panel_a_per_ch,
    }


def _get_topo_info():
    """Cached MNE Info for 63-channel standard_1020 montage."""
    import mne

    if not hasattr(_get_topo_info, '_info'):
        info = mne.create_info(CHANNELS, SFREQ, ch_types='eeg')
        info.set_montage('standard_1020', match_case=False, on_missing='ignore')
        _get_topo_info._info = info
    return _get_topo_info._info


def _plot_topomap_panel(
    fig,
    parent_spec,
    topo_true_list: List[np.ndarray],
    topo_pred_list: List[np.ndarray],
    vmin: float,
    vmax: float,
    times_ms: List[int] | None = None,
    panel_title: str = 'Scalp topography (grand-average ERP)',
    row_labels: tuple[str, str] = ('Ground truth', 'Generated'),
    label_col_ratio: float = 0.0,
    col_labels: List[str] | None = None,
    show_colorbar: bool = True,
    colorbar_label: str = 'Amplitude (a.u.)',
    colorbar_labelpad: float = 8,
    colorbar_width: float = 0.12,
    colorbar_gap: float = 0.28,
    row_hspace: float = 0.18,
    time_label_ratio: float = 0.09,
    title_ratio: float = 0.10,
    time_label_fontsize: float | None = None,
    row_label_fontsize: float | None = None,
    colorbar_label_fontsize: float | None = None,
    colorbar_tick_fontsize: float | None = None,
):
    """2×N topomaps with shared colorbar; compact spacing for Nature layout."""
    n_cols = len(topo_true_list)
    has_label_col = label_col_ratio > 0
    has_title = bool(panel_title)
    row_fs = FIG3_FONT['topo_row'] if row_label_fontsize is None else float(row_label_fontsize)
    cb_lab_fs = (
        FIG3_FONT['colorbar_label'] if colorbar_label_fontsize is None
        else float(colorbar_label_fontsize)
    )
    cb_tick_fs = (
        FIG3_FONT['colorbar_tick'] if colorbar_tick_fontsize is None
        else float(colorbar_tick_fontsize)
    )

    if has_label_col:
        width_ratios = [label_col_ratio] + [1.0] * n_cols
        col_off = 1
    else:
        width_ratios = [1.0] * n_cols
        col_off = 0
    if show_colorbar:
        width_ratios = width_ratios + [colorbar_gap, colorbar_width]

    # Skip empty title row; keep a thin time-label strip above the maps.
    if has_title:
        height_ratios = [title_ratio, time_label_ratio, 1.0, 1.0]
        row_time, row0, row1 = 1, 2, 3
        n_rows = 4
    else:
        height_ratios = [time_label_ratio, 1.0, 1.0]
        row_time, row0, row1 = 0, 1, 2
        n_rows = 3

    inner = parent_spec.subgridspec(
        n_rows, n_cols + col_off + (2 if show_colorbar else 0),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.02,
        hspace=row_hspace,
    )

    ax_title = None
    if has_title:
        ax_title = fig.add_subplot(inner[0, col_off:col_off + n_cols])
        ax_title.set_axis_off()
        ax_title.text(
            0.5, 0.45, panel_title,
            ha='center', va='center', fontsize=FIG3_FONT['panel_title'], fontweight='bold',
            transform=ax_title.transAxes, clip_on=False,
        )

    if time_label_fontsize is None:
        time_fs = FIG3_FONT['topo_time'] - (1 if col_labels else 0)
    else:
        time_fs = float(time_label_fontsize)

    for col in range(n_cols):
        ax_lbl = fig.add_subplot(inner[row_time, col + col_off])
        ax_lbl.set_axis_off()
        if col_labels is not None:
            lbl = col_labels[col]
        elif times_ms is not None:
            lbl = f'{times_ms[col]} ms'
        else:
            lbl = str(col)
        ax_lbl.text(
            0.5, 0.05, lbl, ha='center', va='bottom',
            fontsize=time_fs,
            transform=ax_lbl.transAxes,
        )

    axes = [[], []]
    row_label_axes = []
    mappable = None
    for row, row_idx in enumerate((row0, row1)):
        for col in range(n_cols):
            ax = fig.add_subplot(inner[row_idx, col + col_off])
            vals = topo_true_list[col] if row == 0 else topo_pred_list[col]
            im = _try_mne_topomap(vals, ax, vmin=vmin, vmax=vmax)
            if im is not None:
                mappable = im
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            axes[row].append(ax)

    for row, (row_idx, label) in enumerate(zip((row0, row1), row_labels)):
        if has_label_col:
            ax_lbl = fig.add_subplot(inner[row_idx, 0])
            ax_lbl.set_axis_off()
            ax_lbl.text(
                1.0, 0.5, label, ha='right', va='center',
                fontsize=row_fs, fontweight='bold',
                transform=ax_lbl.transAxes,
            )
            row_label_axes.append(ax_lbl)
        else:
            pos = axes[row][0].get_position()
            fig.text(
                pos.x0 - 0.008, pos.y0 + pos.height / 2,
                label, ha='right', va='center', fontsize=row_fs,
                fontweight='bold',
            )

    if mappable is not None and show_colorbar:
        # Invisible spacer axes so bbox_inches='tight' keeps the gap.
        ax_gap = fig.add_subplot(inner[row0:row1 + 1, col_off + n_cols])
        ax_gap.set_axis_off()
        ax_gap.set_xlim(0, 1)
        ax_gap.set_ylim(0, 1)
        ax_gap.plot([0], [0], alpha=0.0)
        cax = fig.add_subplot(inner[row0:row1 + 1, col_off + n_cols + 1])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(
            colorbar_label,
            fontsize=cb_lab_fs,
            labelpad=colorbar_labelpad,
        )
        cb.ax.tick_params(labelsize=cb_tick_fs, width=0.5, length=2, pad=2)
        cax.tick_params(pad=2)

    return axes, ax_title


def _try_mne_topomap(
    values: np.ndarray,
    ax,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot scalp map with MNE; fall back to horizontal mini-bar on failure."""
    try:
        import mne
        im, _ = mne.viz.plot_topomap(
            values,
            _get_topo_info(),
            axes=ax,
            show=False,
            cmap=TOPO_CMAP,
            vlim=(vmin, vmax),
            sphere=TOPO_SPHERE,
            contours=4,
            sensors=True,
            outlines='head',
            extrapolate='head',
        )
        return im
    except TypeError:
        # Older MNE without extrapolate / fewer kwargs.
        try:
            import mne
            im, _ = mne.viz.plot_topomap(
                values,
                _get_topo_info(),
                axes=ax,
                show=False,
                cmap=TOPO_CMAP,
                vlim=(vmin, vmax),
                sphere=TOPO_SPHERE,
                contours=4,
            )
            return im
        except Exception:
            pass
    except Exception:
        pass

    order = np.argsort(values)
    print('WARNING: MNE topomap failed; falling back to sorted bar plot')
    ax.barh(np.arange(len(CHANNELS)), values[order], color=PALETTE['true'], height=0.8)
    ax.set_yticks(np.arange(0, len(CHANNELS), 8))
    ax.set_yticklabels([CHANNELS[i] for i in order[::8]], fontsize=FIG3_FONT['tick'])
    ax.set_xlabel('μV (a.u.)', fontsize=FIG3_FONT['axis_label'])
    return None


def _panel_a_top_indices(meta: Dict, per_ch: np.ndarray) -> List[int]:
    if 'panel_a_top_indices' in meta:
        return [int(i) for i in meta['panel_a_top_indices'][:PANEL_A_N_CHANNELS]]
    if 'panel_a_top8_indices' in meta:
        return [int(i) for i in meta['panel_a_top8_indices'][:PANEL_A_N_CHANNELS]]
    return np.argsort(per_ch)[-PANEL_A_N_CHANNELS:][::-1].tolist()


def _plot_panel_a_subject_average(
    fig,
    parent_spec,
    erp_true: np.ndarray,
    erp_pred: np.ndarray,
    per_ch: np.ndarray,
    meta: Dict,
    t_ms: np.ndarray,
    *,
    stacked_header: bool = False,
    draw_header: bool = True,
    row_hspace: float | None = None,
    channel_fontsize: float | None = None,
    axis_fontsize: float | None = None,
    tick_fontsize: float | None = None,
):
    """Panel a/b: top-6 channel waveforms (Nature mini-grid).

    If draw_header=False, only the waveform grid is drawn into parent_spec
    (caller owns the shared title/legend strip).
    """
    top_idx = _panel_a_top_indices(meta, per_ch)
    n_rows = int(np.ceil(len(top_idx) / PANEL_A_N_COLS))
    hspace = 0.42 if row_hspace is None else float(row_hspace)
    ch_fs = FIG3_FONT['channel_tag'] if channel_fontsize is None else float(channel_fontsize)
    axis_fs = 6.5 if axis_fontsize is None else float(axis_fontsize)
    tick_fs = 5.5 if tick_fontsize is None else float(tick_fontsize)

    ax_hdr = None
    if draw_header:
        hdr_h = 0.26 if stacked_header else 0.14
        inner = parent_spec.subgridspec(
            n_rows + 1, PANEL_A_N_COLS,
            height_ratios=[hdr_h] + [1.0] * n_rows,
            hspace=hspace, wspace=0.28,
        )
        ax_hdr = fig.add_subplot(inner[0, :])
        ax_hdr.set_axis_off()
        ax_hdr.text(
            0.5, 0.95, 'Example waveforms (top channels)',
            ha='center', va='top', fontsize=9, fontweight='bold',
            transform=ax_hdr.transAxes,
        )
        ax_hdr.legend(
            handles=[
                Line2D([0], [0], color='#0F4D92', lw=1.2, label='Ground truth'),
                Line2D([0], [0], color='#DC2626', lw=1.15, ls=(0, (3.5, 1.4)), label='Generated'),
            ],
            loc='upper center', bbox_to_anchor=(0.5, 0.42),
            ncol=2, frameon=False, fontsize=5.5,
            handlelength=1.6, handletextpad=0.35, columnspacing=1.2,
        )
        grid_spec = inner
        grid_row0 = 1
    else:
        grid_spec = parent_spec.subgridspec(
            n_rows, PANEL_A_N_COLS,
            hspace=hspace, wspace=0.28,
        )
        grid_row0 = 0

    # Shared y-lim across mini panels
    sel_t = erp_true[top_idx]
    sel_p = erp_pred[top_idx]
    y_abs = float(np.max(np.abs(np.concatenate([sel_t, sel_p]))))
    ylim = (-y_abs * 1.08, y_abs * 1.08)

    axes = []
    for k, ci in enumerate(top_idx):
        row, col = divmod(k, PANEL_A_N_COLS)
        ax = fig.add_subplot(grid_spec[grid_row0 + row, col])
        axes.append(ax)
        ch_name = CHANNELS[ci]
        ax.plot(t_ms, erp_true[ci], color='#0F4D92', lw=1.05, zorder=3)
        ax.plot(
            t_ms, erp_pred[ci], color='#DC2626', lw=1.0,
            ls=(0, (3.2, 1.3)), zorder=3,
        )
        ax.axhline(0, color='#D1D5DB', lw=0.45, zorder=1)
        ax.set_ylim(*ylim)
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
        # Channel tag + r near top-right, just inside axes edge.
        r_ch = float(per_ch[ci])
        ax.text(
            0.97, 1.05, f'{ch_name} (r={r_ch:.2f})',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=ch_fs, fontweight='bold', color='#374151',
            zorder=5, clip_on=False,
        )
        ax.tick_params(length=2.0, width=0.55, labelsize=tick_fs)
        if row == n_rows - 1:
            ax.set_xlabel('Time (ms)', fontsize=axis_fs, labelpad=1.5)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel('Amplitude', fontsize=axis_fs, labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
        _despine(ax)

    return ax_hdr if ax_hdr is not None else axes[0]


def _plot_channel_correlation_bars(ax, per_ch: np.ndarray, per_ch_std: np.ndarray):
    """Horizontal bar chart: Pearson r (x) vs channel (y)."""
    order = np.argsort(per_ch)
    sorted_ch = [CHANNELS[i] for i in order]
    sorted_r = per_ch[order]
    sorted_std = per_ch_std[order]

    y = np.arange(len(CHANNELS))
    norm = mpl.colors.Normalize(vmin=-0.05, vmax=0.55)
    cmap = mpl.cm.RdBu_r
    colors = cmap(norm(sorted_r))

    ax.barh(
        y, sorted_r, xerr=sorted_std, height=0.82,
        color=colors, edgecolor='none', alpha=0.92,
        error_kw={'elinewidth': 0.45, 'capsize': 0, 'ecolor': PALETTE['neutral']},
    )
    mean_r = float(np.mean(per_ch))
    ax.axvline(mean_r, color=PALETTE['pred'], ls='--', lw=0.9, label=f'Mean r = {mean_r:.3f}')

    tick_idx = np.arange(0, len(CHANNELS), 4)
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([sorted_ch[i] for i in tick_idx], fontsize=FIG3_FONT['tick'])
    ax.set_xlabel('Pearson r', fontsize=FIG3_FONT['axis_label'])
    ax.set_ylabel('Channel (sorted by r)', fontsize=FIG3_FONT['axis_label'])
    ax.set_xlim(-0.05, max(0.55, sorted_r.max() + 0.08))
    ax.set_ylim(-0.6, len(CHANNELS) - 0.4)
    ax.legend(fontsize=FIG3_FONT['legend'], loc='lower right')
    ax.set_title('Per-channel Pearson r', fontweight='bold', fontsize=FIG3_FONT['panel_title'])
    ax.tick_params(axis='both', labelsize=FIG3_FONT['tick'])


def plot_fig3_visualizations():
    """Fig3 visualization suite (5 panels; a/b Nature restyle, c/d/e prior style)."""
    _setup_nature_rc()
    data = _load_aggregate()
    t_ms = _time_axis_ms(data['erp_true'].shape[1])
    meta = data['meta']

    fig = plt.figure(figsize=(7.4, 8.8))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.15, 1.0, 1.05],
        hspace=0.42,
        wspace=0.32,
    )

    # --- a: top-6 channel waveforms ---
    ax_a = _plot_panel_a_subject_average(
        fig, gs[0, :], data['panel_a_erp_true'], data['panel_a_erp_pred'],
        data['panel_a_per_ch'], meta, t_ms,
    )

    # --- b: occipital ERP ---
    ax_b = fig.add_subplot(gs[1, 0])
    _plot_panel_b_occipital_erp(ax_b, t_ms)

    # --- c: regional reconstruction (prior boxplot style) ---
    ax_c = fig.add_subplot(gs[1, 1])
    _plot_panel_c_region_boxplot(ax_c)

    # --- d: error heatmap (prior style) ---
    ax_d = fig.add_subplot(gs[2, 0])
    err = data['error_map']
    im = ax_d.imshow(
        err, aspect='auto', origin='lower', cmap='YlOrRd',
        extent=[t_ms[0], t_ms[-1], 0, len(CHANNELS)],
        interpolation='nearest',
    )
    ax_d.set_yticks(np.linspace(0.5, len(CHANNELS) - 0.5, 5))
    ax_d.set_yticklabels(['Fp', 'F', 'C', 'P', 'O'], fontsize=FIG3_FONT['tick'])
    ax_d.set_xlabel('Time (ms)', fontsize=FIG3_FONT['axis_label'])
    ax_d.set_ylabel('Channel (approx.)', fontsize=FIG3_FONT['axis_label'])
    ax_d.set_title(
        'Mean absolute error heatmap',
        fontweight='bold', fontsize=FIG3_FONT['panel_title'],
    )
    ax_d.tick_params(axis='x', labelsize=FIG3_FONT['tick'])
    cb = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.02)
    cb.set_label('|error| (a.u.)', fontsize=FIG3_FONT['colorbar_label'])
    cb.ax.tick_params(labelsize=FIG3_FONT['colorbar_tick'])

    # --- e: per-channel Pearson r (prior style) ---
    ax_e = fig.add_subplot(gs[2, 1])
    _plot_channel_correlation_bars(ax_e, data['per_ch'], data['per_ch_std'])

    label_axes([ax_a, ax_b, ax_c, ax_d, ax_e], start='a')

    stem = fig_path(FIG3)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def _erp_topo_in_bins(erp: np.ndarray, bins_ms: List[tuple[int, int]]) -> List[np.ndarray]:
    """Mean scalp map per channel within each [t0, t1) ms window."""
    n_time = erp.shape[1]
    out = []
    for t0, t1 in bins_ms:
        i0 = ms_to_idx(t0)
        i1 = min(ms_to_idx(t1), n_time)
        i0 = min(i0, n_time - 1)
        if i1 <= i0:
            out.append(erp[:, i0])
        else:
            out.append(erp[:, i0:i1].mean(axis=1))
    return out


def plot_fig3_supp_topo_100ms():
    """Supplementary: test-set grand-mean ERP topomaps per 100 ms window (GT vs Gen)."""
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)

    _setup_fig3_rc()
    data = _load_aggregate()
    bins_ms = TOPO_100MS_BINS
    col_labels = [f'{t0}–{t1}' for t0, t1 in bins_ms]
    topo_true = _erp_topo_in_bins(data['erp_true'], bins_ms)
    topo_pred = _erp_topo_in_bins(data['erp_pred'], bins_ms)
    all_topo = topo_true + topo_pred
    vmax = float(np.max(np.abs(np.stack(all_topo))))
    vmin = -vmax

    n_sub = data['meta'].get('n_subjects', 10)
    n_test = data['meta'].get('panel_a_n_test_images', 200)

    n_cols = len(bins_ms)
    cell = 0.98
    fig = plt.figure(figsize=(cell * n_cols + 1.90, cell * 2 + 0.86))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[0.11, 1.0], hspace=0.07,
        left=0.04, right=0.92, top=0.97, bottom=0.03,
    )
    ax_title = fig.add_subplot(outer[0, 0])
    ax_title.set_axis_off()
    ax_title.set_xlim(0, 1)
    ax_title.set_ylim(0, 1)
    ax_title.plot([0], [0], alpha=0.0)
    ax_title.text(
        0.01, 0.58, 'a', fontsize=FIG3_FONT['panel_title'], fontweight='bold',
        va='center', ha='left', transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.58,
        f'Test-set grand-average ERP topographies (n = {n_sub}, {n_test} images)',
        ha='center', va='center', fontsize=FIG3_FONT['panel_title'], fontweight='bold',
        transform=ax_title.transAxes,
    )
    _plot_topomap_panel(
        fig, outer[1, 0], topo_true, topo_pred, vmin, vmax,
        panel_title='',
        row_labels=('GT', 'Gen'),
        label_col_ratio=0.038,
        col_labels=col_labels,
    )
    stem = fig_path(FIG4)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    # Keep designed margins (tight bbox would collapse title/colorbar gaps).
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_fig3_erp_allchannels():
    """Fig5: GT vs Gen ERP heatmaps (Nature style, side-by-side)."""
    _setup_nature_rc()
    data = _load_aggregate()
    t_ms = _time_axis_ms(data['erp_true'].shape[1])
    erp_t = np.asarray(data['erp_true'], dtype=np.float64)
    erp_p = np.asarray(data['erp_pred'], dtype=np.float64)
    n_ch = erp_t.shape[0]
    n_sub = data['meta'].get('n_subjects', 10)
    n_test = data['meta'].get('panel_a_n_test_images', 200)

    vmax = float(np.max(np.abs(np.concatenate([erp_t, erp_p], axis=0))))
    if vmax <= 0:
        vmax = 1e-6

    region_ticks = [
        (0, 'Fp'),
        (14, 'F'),
        (30, 'C'),
        (54, 'P'),
        (n_ch - 1, 'O'),
    ]
    region_bounds = [15, 35, 55]

    fig = plt.figure(figsize=(7.4, 2.85))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[0.12, 1.0],
        width_ratios=[1.0, 1.0, 0.045],
        hspace=0.08,
        wspace=0.18,
        left=0.08, right=0.92, top=0.90, bottom=0.20,
    )

    ax_hdr = fig.add_subplot(gs[0, :2])
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.5, 0.20,
        f'Grand-average ERP (n = {n_sub}, {n_test} images)',
        ha='center', va='center', fontsize=9, fontweight='bold',
        transform=ax_hdr.transAxes,
    )

    panels = [
        (0, 'a', 'Ground truth', erp_t),
        (1, 'b', 'Generated', erp_p),
    ]
    last_im = None
    axes = []
    for col, letter, title, arr in panels:
        ax = fig.add_subplot(gs[1, col])
        axes.append(ax)
        last_im = ax.imshow(
            arr, aspect='auto', origin='lower', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[float(t_ms[0]), float(t_ms[-1]), -0.5, n_ch - 0.5],
            interpolation='nearest',
            rasterized=True,
        )
        for b in region_bounds:
            ax.axhline(b - 0.5, color='white', lw=0.55, alpha=0.65, zorder=2)
        for t0 in (80, 180):
            ax.axvline(
                t0, color='#9A3412', lw=0.55, ls=(0, (2.5, 1.8)),
                alpha=0.50, zorder=3,
            )

        ax.set_yticks([i for i, _ in region_ticks])
        ax.set_xlim(float(t_ms[0]), float(t_ms[-1]))
        ax.set_ylim(-0.5, n_ch - 0.5)
        ax.set_xlabel('Time (ms)', fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=3)
        ax.tick_params(length=2.2, width=0.55, labelsize=6.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
            spine.set_color('#6B7280')

        ax.text(
            -0.10, 1.12, letter, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left', clip_on=False,
        )

        if col == 0:
            ax.set_yticklabels([lab for _, lab in region_ticks], fontsize=6.5)
            ax.set_ylabel('Channel (ant. → post.)', fontsize=7)
            ax.text(
                130, 1.03, 'early visual',
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5.5, color='#9A3412',
                clip_on=False,
            )
        else:
            ax.set_yticklabels([])

    cax = fig.add_subplot(gs[1, 2])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label('Amplitude (a.u.)', fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.55)
    cb.outline.set_linewidth(0.55)

    stem = fig_path(FIG5)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_all_visualizations():
    plot_fig3_visualizations()
    plot_fig3_erp_allchannels()
    plot_fig3_supp_topo_100ms()
