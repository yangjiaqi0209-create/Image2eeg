"""Paper Results Figure 1: EEG response prediction (merged).

Panel order adcbef (old letters) → relettered a–f:
------------------------
  a  Spatiotemporal ERP topographies (100 ms)     ← old a
  b  Per-channel correlation (ant.→post.)         ← old d
  c  Regional prediction quality                  ← old c
  d  Example waveforms (top channels)             ← old b
  e  Occipital grand-average ERP                  ← old e
  f  Occipital ERP components (C1/P1/N1)          ← old f
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.eeg_gen_eval.config import CHANNELS, FIG_DIR
from analysis.eeg_gen_eval.figure_names import FINAL_FIG1, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _despine,
    _load_erp_component_fidelity_arrays,
    _load_summary,
    _plot_erp_fidelity_dotplot,
    _setup_nature_rc,
    add_panel_label,
)
from analysis.eeg_gen_eval.helpers.plot_visualizations import (
    BRAIN_REGION_COLORS,
    BRAIN_REGION_ORDER,
    TOPO_100MS_BINS,
    _channel_region,
    _erp_topo_in_bins,
    _load_aggregate,
    _plot_panel_a_subject_average,
    _plot_panel_b_occipital_erp,
    _plot_panel_c_region_boxplot,
    _plot_topomap_panel,
    _time_axis_ms,
)

# Final Fig1 type scale (Nature multi-panel, readable at print size).
FIG1_FS = {
    'panel_title': 10,
    'panel_letter': 12,
    'axis': 8,
    'tick': 7,
    'legend': 6.5,
    'annot': 6.5,
    'mean_annot': 8.0,
    'channel_tag': 7.5,
    'topo_time': 6.0,
    'region': 6.5,
}


def _setup_fig1_rc():
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FIG1_FS['tick'],
        'axes.labelsize': FIG1_FS['axis'],
        'axes.titlesize': FIG1_FS['panel_title'],
        'xtick.labelsize': FIG1_FS['tick'],
        'ytick.labelsize': FIG1_FS['tick'],
        'legend.fontsize': FIG1_FS['legend'],
    })


def _plot_per_channel_correlation_bars(ax, per_ch_mean: np.ndarray, per_ch_std: np.ndarray):
    """Per-channel Pearson r grouped by five mutually exclusive scalp regions."""
    n = len(CHANNELS)
    x = np.arange(n, dtype=float)
    mean_r = float(per_ch_mean.mean())

    grouped_indices = {
        region: [i for i, ch in enumerate(CHANNELS) if _channel_region(ch) == region]
        for region in BRAIN_REGION_ORDER
    }
    active = [
        (region, color)
        for region, color in zip(BRAIN_REGION_ORDER, BRAIN_REGION_COLORS)
        if grouped_indices[region]
    ]
    order = np.asarray([
        i for region, _ in active for i in grouped_indices[region]
    ], dtype=int)
    if len(order) != n or len(np.unique(order)) != n:
        raise ValueError('Each EEG channel must map to exactly one scalp region')
    ordered_mean = per_ch_mean[order]
    ordered_std = per_ch_std[order]
    ordered_channels = [CHANNELS[i] for i in order]

    regions = []
    start = 0
    for region, color in active:
        stop = start + len(grouped_indices[region])
        regions.append((start, stop, color, region))
        start = stop

    colors = np.empty(n, dtype=object)
    for gi, (x0, x1, color, _name) in enumerate(regions):
        colors[x0:x1] = color
        ax.axvspan(
            x0 - 0.5, x1 - 0.5,
            color='#F8FAFC' if gi % 2 == 0 else '#F1F5F9',
            lw=0, zorder=0,
        )

    ax.bar(
        x, ordered_mean, width=0.82, color=list(colors), edgecolor='none',
        alpha=0.90, zorder=2,
    )
    ax.errorbar(
        x, ordered_mean, yerr=ordered_std, fmt='none',
        ecolor='#94A3B8', elinewidth=0.45, capsize=0, zorder=3,
    )

    ax.axhline(mean_r, color='#B91C1C', ls=(0, (3.5, 2.2)), lw=0.9, zorder=4)
    ax.text(
        0.02, 0.93, f'mean r = {mean_r:.3f}',
        transform=ax.transAxes, ha='left', va='top',
        fontsize=FIG1_FS['mean_annot'], color='#B91C1C', zorder=5,
    )

    tick_idx = [int((x0 + x1 - 1) // 2) for x0, x1, _color, _name in regions]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([ordered_channels[i] for i in tick_idx], fontsize=FIG1_FS['tick'])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Pearson r', fontsize=FIG1_FS['axis'])
    ax.set_xlabel('Channel (grouped by region)', fontsize=FIG1_FS['axis'])
    _despine(ax)
    ax.tick_params(length=2.5, width=0.6, labelsize=FIG1_FS['tick'])
    return regions


def plot_final_fig1_eeg_response_prediction():
    """Merged Results Fig1; content order adcbef (relettered a–f)."""
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)
    _setup_fig1_rc()

    summary = _load_summary()
    per_ch_mean = np.array(summary['per_channel_pearson']['mean'])
    per_ch_std = np.array(summary['per_channel_pearson']['std'])
    comps, fid_arr = _load_erp_component_fidelity_arrays()

    vis = _load_aggregate()
    t_ms = _time_axis_ms(vis['erp_true'].shape[1])
    meta = vis['meta']

    bins_ms = TOPO_100MS_BINS
    col_labels = [f'{t0}–{t1} ms' for t0, t1 in bins_ms]
    topo_true = _erp_topo_in_bins(vis['erp_true'], bins_ms)
    topo_pred = _erp_topo_in_bins(vis['erp_pred'], bins_ms)
    vmax = float(np.max(np.abs(np.stack(topo_true + topo_pred))))
    vmin = -vmax

    fig = plt.figure(figsize=(7.2, 8.35))
    outer = fig.add_gridspec(
        4, 1,
        height_ratios=[0.86, 0.92, 1.02, 0.74],
        hspace=0.26,
        left=0.075, right=0.935, top=0.982, bottom=0.048,
    )

    # a — topo (old a)
    gs0 = outer[0].subgridspec(2, 1, height_ratios=[0.14, 1.0], hspace=0.06)
    ax_a_title = fig.add_subplot(gs0[0, 0])
    ax_a_title.set_axis_off()
    ax_a_title.text(
        0.5, 0.45,
        'Grand-average ERP topographies',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_a_title.transAxes,
    )
    add_panel_label(ax_a_title, 'a', x=-0.02, y=1.12, fontsize=FIG1_FS['panel_letter'])

    _plot_topomap_panel(
        fig, gs0[1, 0], topo_true, topo_pred, vmin, vmax,
        panel_title='',
        row_labels=('GT', 'Pred'),
        label_col_ratio=0.045,
        col_labels=col_labels,
        colorbar_label='a.u.',
        colorbar_labelpad=2,
        colorbar_width=0.10,
        colorbar_gap=0.22,
        row_hspace=0.0,
        time_label_ratio=0.12,
        time_label_fontsize=FIG1_FS['topo_time'],
        row_label_fontsize=8.5,
        colorbar_label_fontsize=FIG1_FS['axis'],
        colorbar_tick_fontsize=FIG1_FS['tick'],
    )

    # b — per-channel correlation (old d)
    gs1 = outer[1].subgridspec(
        2, 1, height_ratios=[0.15, 1.0], hspace=0.14,
    )
    ax_b_hdr = fig.add_subplot(gs1[0, 0])
    ax_b_hdr.set_axis_off()
    ax_b_hdr.text(
        0.5, 0.55, 'Per-channel correlation',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_b_hdr.transAxes,
    )
    ax_b = fig.add_subplot(gs1[1, 0])
    regions = _plot_per_channel_correlation_bars(ax_b, per_ch_mean, per_ch_std)
    for x0, x1, _color, name in regions:
        ax_b.text(
            0.5 * (x0 + x1) - 0.5, 1.01, name,
            transform=ax_b.get_xaxis_transform(),
            ha='center', va='bottom', fontsize=FIG1_FS['region'], color='#6B7280',
            clip_on=False,
        )
    add_panel_label(ax_b_hdr, 'b', x=-0.02, y=1.15, fontsize=FIG1_FS['panel_letter'])

    # c | d — regional (old c) | waveforms (old b)
    gs2 = outer[2].subgridspec(
        3, 2,
        height_ratios=[0.13, 0.10, 1.0],
        width_ratios=[1.0, 1.32],
        hspace=0.08, wspace=0.26,
    )
    ax_c_hdr = fig.add_subplot(gs2[0, 0])
    ax_c_hdr.set_axis_off()
    ax_c_hdr.text(
        0.5, 0.45, 'Regional prediction quality',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_c_hdr.transAxes,
    )
    ax_c_pad = fig.add_subplot(gs2[1, 0])
    ax_c_pad.set_axis_off()

    ax_d_hdr = fig.add_subplot(gs2[0, 1])
    ax_d_hdr.set_axis_off()
    ax_d_hdr.text(
        0.5, 0.45, 'Example waveforms',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_d_hdr.transAxes,
    )
    ax_d_leg = fig.add_subplot(gs2[1, 1])
    ax_d_leg.set_axis_off()
    ax_d_leg.legend(
        handles=[
            Line2D([0], [0], color='#0F4D92', lw=1.25, label='GT'),
            Line2D([0], [0], color='#DC2626', lw=1.15, ls=(0, (3.5, 1.4)), label='Pred'),
        ],
        loc='center', bbox_to_anchor=(0.5, 0.5),
        ncol=2, frameon=False, fontsize=FIG1_FS['legend'],
        handlelength=1.35, handletextpad=0.3, columnspacing=0.85,
    )

    ax_c = fig.add_subplot(gs2[2, 0])
    _plot_panel_c_region_boxplot(
        ax_c, compact=True, show_title=False,
        annot_fontsize=FIG1_FS['mean_annot'],
        axis_fontsize=FIG1_FS['axis'],
        tick_fontsize=FIG1_FS['tick'],
    )
    _plot_panel_a_subject_average(
        fig, gs2[2, 1],
        vis['panel_a_erp_true'], vis['panel_a_erp_pred'],
        vis['panel_a_per_ch'], meta, t_ms,
        draw_header=False,
        row_hspace=0.42,
        channel_fontsize=FIG1_FS['channel_tag'],
        axis_fontsize=FIG1_FS['axis'],
        tick_fontsize=FIG1_FS['tick'],
    )
    add_panel_label(ax_c_hdr, 'c', x=-0.03, y=1.25, fontsize=FIG1_FS['panel_letter'])
    add_panel_label(ax_d_hdr, 'd', x=-0.03, y=1.25, fontsize=FIG1_FS['panel_letter'])

    # e | f — occipital ERP | components
    gs3 = outer[3].subgridspec(
        2, 2,
        height_ratios=[0.14, 1.0],
        width_ratios=[1.55, 1.0],
        hspace=0.05, wspace=0.28,
    )
    ax_e_hdr = fig.add_subplot(gs3[0, 0])
    ax_e_hdr.set_axis_off()
    ax_e_hdr.text(
        0.5, 0.50, 'Occipital ERP comparison',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_e_hdr.transAxes,
    )
    ax_f_hdr = fig.add_subplot(gs3[0, 1])
    ax_f_hdr.set_axis_off()
    ax_f_hdr.text(
        0.5, 0.50, 'Occipital ERP components',
        ha='center', va='center', fontsize=FIG1_FS['panel_title'], fontweight='bold',
        transform=ax_f_hdr.transAxes,
    )
    ax_e = fig.add_subplot(gs3[1, 0])
    _plot_panel_b_occipital_erp(
        ax_e, t_ms, compact=True,
        annot_fontsize=FIG1_FS['annot'],
        legend_fontsize=FIG1_FS['legend'],
        axis_fontsize=FIG1_FS['axis'],
        tick_fontsize=FIG1_FS['tick'],
        show_early_visual=False,
    )
    ax_e.set_title('')
    ax_f = fig.add_subplot(gs3[1, 1])
    _plot_erp_fidelity_dotplot(
        ax_f, fid_arr, comps, compact=True,
        legend_fontsize=FIG1_FS['legend'],
        axis_fontsize=FIG1_FS['axis'],
        tick_fontsize=FIG1_FS['tick'],
        short_legend=True,
    )
    ax_f.set_title('')
    add_panel_label(ax_e_hdr, 'e', x=-0.04, y=1.20, fontsize=FIG1_FS['panel_letter'])
    add_panel_label(ax_f_hdr, 'f', x=-0.04, y=1.20, fontsize=FIG1_FS['panel_letter'])

    stem = fig_path(FINAL_FIG1)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


if __name__ == '__main__':
    plot_final_fig1_eeg_response_prediction()
