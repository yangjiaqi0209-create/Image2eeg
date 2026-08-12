"""Paper Results Figure 2: frequency / spectral fidelity.

Panel order cbda (relettered a–d), two rows:
--------------------------------------------
  a  Topomap band power (GT / Gen)              ← fig2e / old c
  b  True–Shuffle band fidelity ← fig2c / old b
  c  Band × channel waveform similarity         ← fig6  / old d
  d  Visual cortex PSD (occipital ROI)          ← fig2b / old a
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analysis.eeg_gen_eval.config import CHANNELS, CHANNELS_O, EEG_BANDS, FIG_DIR, RAW_DIR
from analysis.eeg_gen_eval.figure_names import FINAL_FIG2, fig_path
from analysis.eeg_gen_eval.compute.metrics import psd_mean_sem
from analysis.eeg_gen_eval.helpers.plot_band_heatmap import (
    _load_band_channel_data,
    plot_band_channel_heatmap_on_axes,
    region_channel_layout,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _mean_log_bandpower_topo,
    _plot_fig2_topomap,
    _plot_psd_curves,
    _setup_nature_rc,
    add_panel_label,
    calculate_frequency_fidelity_statistics,
    load_frequency_fidelity_data,
    plot_frequency_fidelity,
)

_FIDELITY_SHORT = {
    'All': 'All',
    'Delta': 'δ',
    'Theta': 'θ',
    'Alpha': 'α',
    'Beta': 'β',
    'Gamma': 'γ',
}

# Final Fig2 type scale (readable at print size).
FIG2_FS = {
    'panel_title': 10,
    'panel_letter': 12,
    'axis': 8,
    'tick': 7,
    'legend': 6.5,
    'grey': 7.0,
    'row_label': 8.0,
    'colorbar_label': 7.5,
    'colorbar_tick': 6.5,
    'star': 7.5,
    'star_ns': 6.5,
}


def _setup_fig2_rc():
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FIG2_FS['tick'],
        'axes.labelsize': FIG2_FS['axis'],
        'axes.titlesize': FIG2_FS['panel_title'],
        'xtick.labelsize': FIG2_FS['tick'],
        'ytick.labelsize': FIG2_FS['tick'],
        'legend.fontsize': FIG2_FS['legend'],
    })


def _load_fig2_spectral_payload():
    """Load occipital PSD + bandpower topo arrays used by fig2 b/e."""
    o_idx = [CHANNELS.index(ch) for ch in CHANNELS_O]
    occ_true, occ_pred = [], []
    occ_true_sem, occ_pred_sem = [], []
    occ_freqs = None
    topo_true_list, topo_pred_list = [], []

    for sub_dir in sorted(os.listdir(RAW_DIR)):
        if not sub_dir.startswith('sub-'):
            continue
        d = os.path.join(RAW_DIR, sub_dir)
        if not os.path.isfile(os.path.join(d, 'y_true.npy')):
            continue

        y_true = np.load(os.path.join(d, 'y_true.npy'))
        y_pred = np.load(os.path.join(d, 'y_pred.npy'))

        pt = psd_mean_sem(y_true[:, o_idx, :])
        pp = psd_mean_sem(y_pred[:, o_idx, :])
        occ_freqs = pt['freqs_hz']
        occ_true.append(pt['psd_mean'])
        occ_pred.append(pp['psd_mean'])
        occ_true_sem.append(pt['psd_sem'])
        occ_pred_sem.append(pp['psd_sem'])

        bp_t = np.load(os.path.join(d, 'bandpower_true.npy')).mean(axis=0)
        bp_p = np.load(os.path.join(d, 'bandpower_pred.npy')).mean(axis=0)
        all_t = _mean_log_bandpower_topo(y_true, 0.5, 45.0)
        all_p = _mean_log_bandpower_topo(y_pred, 0.5, 45.0)
        topo_true_list.append(np.column_stack([all_t, bp_t]))
        topo_pred_list.append(np.column_stack([all_p, bp_p]))

    return {
        'occ_freqs': occ_freqs,
        'occ_true': np.stack(occ_true).mean(axis=0),
        'occ_pred': np.stack(occ_pred).mean(axis=0),
        'occ_true_sem': np.stack(occ_true_sem).mean(axis=0),
        'occ_pred_sem': np.stack(occ_pred_sem).mean(axis=0),
        'topo_true': np.stack(topo_true_list).mean(axis=0),
        'topo_pred': np.stack(topo_pred_list).mean(axis=0),
    }


def _plot_bandpower_topomap_strip(fig, parent_spec, topo_true, topo_pred):
    """Compact Nature-style GT/Gen × band topomap strip."""
    topo_band_specs = [('All', 0.5, 45.0)] + [
        (name.capitalize(), fmin, fmax) for name, fmin, fmax in EEG_BANDS
    ]
    topo_labels = [lab for lab, _, _ in topo_band_specs]
    topo_short = {
        'All': 'All', 'Delta': 'δ', 'Theta': 'θ',
        'Alpha': 'α', 'Beta': 'β', 'Gamma': 'γ',
    }
    n_bands = len(topo_labels)

    vmin = float(min(topo_true.min(), topo_pred.min()))
    vmax = float(max(topo_true.max(), topo_pred.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Same title↔grey spacing style as panels c/d; grey sits flush above maps.
    outer_a = parent_spec.subgridspec(
        2, 1, height_ratios=[0.16, 1.0], hspace=0.0,
    )
    ax_title = fig.add_subplot(outer_a[0, 0])
    ax_title.set_axis_off()
    ax_title.text(
        0.5, 0.98, 'Topomap band power',
        ha='center', va='top', fontsize=FIG2_FS['panel_title'], fontweight='bold',
        transform=ax_title.transAxes,
    )
    # Grey band labels under title (aligned across map columns, skipping GT/Gen label col).
    label_w, map_w, gap_w, cbar_w = 0.07, 1.0, 0.10, 0.11
    total_w = label_w + map_w * n_bands + gap_w + cbar_w
    for bi, name in enumerate(topo_labels):
        x = (label_w + map_w * (bi + 0.5)) / total_w
        ax_title.text(
            x, 0.0, topo_short.get(name, name),
            ha='center', va='bottom', fontsize=FIG2_FS['grey'], color='#6B7280',
            transform=ax_title.transAxes,
        )

    width_ratios = [label_w] + [map_w] * n_bands + [gap_w, cbar_w]
    inner = outer_a[1].subgridspec(
        2, n_bands + 3,
        width_ratios=width_ratios,
        height_ratios=[1.0, 1.0],
        wspace=0.03,
        hspace=-0.06,  # pull GT/Gen map rows closer
    )

    for row_idx, label in ((0, 'GT'), (1, 'Pred')):
        ax_lbl = fig.add_subplot(inner[row_idx, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(
            1.0, 0.5, label, ha='right', va='center', fontsize=FIG2_FS['row_label'],
            fontweight='bold', color='#374151', transform=ax_lbl.transAxes,
        )

    last_im = None
    for bi, name in enumerate(topo_labels):
        ax_t = fig.add_subplot(inner[0, bi + 1])
        ax_p = fig.add_subplot(inner[1, bi + 1])
        _plot_fig2_topomap(topo_true[:, bi], ax_t, vmin=vmin, vmax=vmax)
        last_im = _plot_fig2_topomap(topo_pred[:, bi], ax_p, vmin=vmin, vmax=vmax)

    ax_gap = fig.add_subplot(inner[0:2, n_bands + 1])
    ax_gap.set_axis_off()

    cax = fig.add_subplot(inner[0:2, n_bands + 2])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label('log BP', fontsize=FIG2_FS['colorbar_label'], labelpad=2)
    cb.ax.tick_params(labelsize=FIG2_FS['colorbar_tick'], length=1.8, width=0.5, pad=1.2)
    ticks = np.linspace(vmin, vmax, 4)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f'{t:.1f}' for t in ticks])
    cb.outline.set_linewidth(0.5)

    return ax_title


def _polish_fidelity_panel(ax):
    """Readable tick labels + quieter legend for the narrow column."""
    labels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticklabels(
        [_FIDELITY_SHORT.get(lab, lab) for lab in labels],
        fontsize=FIG2_FS['tick'],
    )
    ax.set_xlabel('Band', fontsize=FIG2_FS['axis'])
    ax.set_ylabel('Bandpower r', fontsize=FIG2_FS['axis'])
    ax.tick_params(axis='both', labelsize=FIG2_FS['tick'])
    for txt in ax.texts:
        s = txt.get_text()
        if s in ('***', '**', '*', 'ns'):
            txt.set_fontsize(
                FIG2_FS['star_ns'] if s == 'ns' else FIG2_FS['star']
            )
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(
        fontsize=FIG2_FS['legend'], loc='upper right', frameon=False,
        handlelength=0.9, handletextpad=0.3, borderaxespad=0.1,
        labelspacing=0.2, columnspacing=0.6,
    )


def _polish_psd_panel(ax):
    """Readable axis labels + short GT/Gen legend (no in-axes band labels)."""
    ax.set_xlabel('Frequency (Hz)', fontsize=FIG2_FS['axis'])
    ax.set_ylabel('PSD (a.u.)', fontsize=FIG2_FS['axis'])
    ax.tick_params(labelsize=FIG2_FS['tick'])
    # Drop any grey band letters from the axes — they live in the shared header.
    band_labs = {'δ', 'θ', 'α', 'β', 'γ'}
    for txt in list(ax.texts):
        if txt.get_text() in band_labs:
            txt.remove()
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(
        handles=[
            Line2D([0], [0], color='#0F4D92', lw=1.35, label='GT'),
            Line2D([0], [0], color='#DC2626', lw=1.2, ls=(0, (3.5, 1.4)), label='Pred'),
        ],
        loc='lower left', fontsize=FIG2_FS['legend'], frameon=False,
        handlelength=1.3, handletextpad=0.3, borderaxespad=0.15,
        labelspacing=0.2,
    )


# Heatmap | colorbar column ratios (shared by panel-c header grey + body).
_HM_WR, _CB_WR, _HM_WS = 1.0, 0.022, 0.03


def _plot_heatmap_body(fig, parent_spec, data):
    """Heatmap + slim colorbar (no title strip)."""
    gs = parent_spec.subgridspec(
        1, 2, width_ratios=[_HM_WR, _CB_WR], wspace=_HM_WS,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    im, _, _ = plot_band_channel_heatmap_on_axes(
        ax, cax, data,
        show_title=False, show_meta=False, show_region_labels=False,
    )
    ax.set_xlabel('Channel (grouped by region)', fontsize=FIG2_FS['axis'])
    ax.set_ylabel('Band', fontsize=FIG2_FS['axis'])
    ax.tick_params(labelsize=FIG2_FS['tick'])
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontsize(FIG2_FS['tick'])
    if im is not None and getattr(im, 'colorbar', None) is not None:
        im.colorbar.set_label('Pearson r', fontsize=FIG2_FS['colorbar_label'], labelpad=2)
        im.colorbar.ax.tick_params(
            labelsize=FIG2_FS['colorbar_tick'], length=1.6, width=0.45, pad=1.0,
        )
    return ax


def plot_final_fig2_frequency_spectral():
    """Composite Results Fig2 in order cbda (two rows), polished layout."""
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)
    _setup_fig2_rc()

    payload = _load_fig2_spectral_payload()
    fidelity_all = calculate_frequency_fidelity_statistics(
        load_frequency_fidelity_data(occipital=False),
    )

    tm = np.asarray(fidelity_all['true_mean'], dtype=np.float64)
    sm = np.asarray(fidelity_all['shuffle_mean'], dtype=np.float64)
    te = np.asarray(fidelity_all.get('true_sem', np.zeros_like(tm)), dtype=np.float64)
    se = np.asarray(fidelity_all.get('shuffle_sem', np.zeros_like(sm)), dtype=np.float64)
    tops = np.maximum(tm + te, sm + se)
    ts = fidelity_all.get('true_by_subject')
    ss = fidelity_all.get('shuffle_by_subject')
    if ts is not None:
        tops = np.maximum(tops, np.nanmax(np.asarray(ts), axis=0))
    if ss is not None:
        tops = np.maximum(tops, np.nanmax(np.asarray(ss), axis=0))
    fid_ylim = (0.0, min(1.15, float(np.max(tops)) * 1.20))

    fig = plt.figure(figsize=(7.2, 4.55))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.05, 1.0],
        hspace=0.28,
        left=0.07, right=0.975, top=0.968, bottom=0.10,
    )

    # Row 1: a topo | b fidelity
    gs0 = outer[0].subgridspec(1, 2, width_ratios=[2.15, 0.82], wspace=0.26)
    ax_a_title = _plot_bandpower_topomap_strip(
        fig, gs0[0, 0], payload['topo_true'], payload['topo_pred'],
    )
    gs0_b = gs0[0, 1].subgridspec(
        2, 1, height_ratios=[0.16, 1.0], hspace=0.03,
    )
    ax_b_hdr = fig.add_subplot(gs0_b[0, 0])
    ax_b_hdr.set_axis_off()
    # Title only — panel letter placed later in figure coords at full-column corner.
    ax_b_hdr.text(
        0.5, 0.35, 'True–Shuffle band fidelity',
        ha='center', va='center',
        fontsize=FIG2_FS['panel_title'] - 1.0, fontweight='bold',
        transform=ax_b_hdr.transAxes, color='#111827',
    )
    ax_b = fig.add_subplot(gs0_b[1, 0])
    plot_frequency_fidelity(
        ax_b, fidelity_all,
        title='',
        ylim=fid_ylim, show_legend=True,
    )
    _polish_fidelity_panel(ax_b)

    # Row 2: shared header — tight to plot, slight air between title and grey
    gs1 = outer[1].subgridspec(
        2, 2,
        height_ratios=[0.18, 1.0],
        width_ratios=[2.15, 0.82],
        hspace=0.02, wspace=0.26,
    )

    data_hm = _load_band_channel_data()
    _, region_bands, _ = region_channel_layout()
    n_ch = int(np.asarray(data_hm['mean']).shape[1])

    # Title on full panel width; grey labels only over the heatmap (not colorbar).
    ax_c_hdr = fig.add_subplot(gs1[0, 0])
    ax_c_hdr.set_axis_off()
    ax_c_hdr.text(
        0.5, 0.98, 'Band–channel similarity',
        ha='center', va='top', fontsize=FIG2_FS['panel_title'], fontweight='bold',
        transform=ax_c_hdr.transAxes,
    )
    gs_c_grey = gs1[0, 0].subgridspec(
        1, 2, width_ratios=[_HM_WR, _CB_WR], wspace=_HM_WS,
    )
    ax_c_grey = fig.add_subplot(gs_c_grey[0, 0])
    ax_c_grey.set_axis_off()
    for x0, x1, name in region_bands:
        ax_c_grey.text(
            (x0 + x1) / (2.0 * n_ch), 0.0, name,
            ha='center', va='bottom', fontsize=FIG2_FS['grey'], color='#6B7280',
            transform=ax_c_grey.transAxes,
        )

    ax_d_hdr = fig.add_subplot(gs1[0, 1])
    ax_d_hdr.set_axis_off()
    ax_d_hdr.text(
        0.5, 0.98, 'Occipital PSD',
        ha='center', va='top', fontsize=FIG2_FS['panel_title'], fontweight='bold',
        transform=ax_d_hdr.transAxes,
    )
    for f0, f1, lab in (
        (0.5, 4, 'δ'), (4, 8, 'θ'), (8, 13, 'α'), (13, 30, 'β'), (30, 45, 'γ'),
    ):
        ax_d_hdr.text(
            (0.5 * (f0 + f1)) / 45.0, 0.0, lab,
            ha='center', va='bottom', fontsize=FIG2_FS['grey'], color='#6B7280',
            transform=ax_d_hdr.transAxes,
        )

    _plot_heatmap_body(fig, gs1[1, 0], data_hm)

    ax_d = fig.add_subplot(gs1[1, 1])
    _plot_psd_curves(
        ax_d,
        payload['occ_freqs'],
        payload['occ_true'], payload['occ_pred'],
        payload['occ_true_sem'], payload['occ_pred_sem'],
        title='',
        show_legend=False,
    )
    _polish_psd_panel(ax_d)

    add_panel_label(ax_a_title, 'a', x=-0.02, y=1.18, fontsize=FIG2_FS['panel_letter'])
    add_panel_label(ax_c_hdr, 'c', x=-0.03, y=1.15, fontsize=FIG2_FS['panel_letter'])
    add_panel_label(ax_d_hdr, 'd', x=-0.10, y=1.15, fontsize=FIG2_FS['panel_letter'])

    # Panel b: letter at top-left of the full column (header + bars), in the a|b gutter.
    fig.canvas.draw()
    b_x0 = min(ax_b_hdr.get_position().x0, ax_b.get_position().x0)
    b_y1 = max(ax_b_hdr.get_position().y1, ax_b.get_position().y1)
    fig.text(
        b_x0 - 0.022,
        b_y1 + 0.004,
        'b',
        transform=fig.transFigure,
        ha='right',
        va='bottom',
        fontsize=FIG2_FS['panel_letter'],
        fontweight='bold',
        color='#111827',
        clip_on=False,
        zorder=20,
    )

    stem = fig_path(FINAL_FIG2)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


if __name__ == '__main__':
    plot_final_fig2_frequency_spectral()
