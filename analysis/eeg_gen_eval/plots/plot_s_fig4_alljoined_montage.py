"""Supplementary Figure S4: Alljoined selected panels (native redraw).

Reading order a–g:
  a metrics | b grand-average ERP heatmaps
  c regional box | d occipital ERP
  e occipital PSD | f matched-pair | g RDM correspondence

Panels are drawn with the same axes helpers as the source figures
(no PNG cropping). Output is written to analysis/eeg_gen_eval/figures/.

Usage:
  PYTHONPATH=. python -m analysis.eeg_gen_eval.plots.plot_s_fig4_alljoined_montage
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# Switch dataset BEFORE importing helpers so their RAW_DIR / FIG_DIR bind correctly.
# Keep a module alias — `from config import RAW_DIR` would stay stale after switch.
from analysis.eeg_gen_eval import config as cfg

cfg.set_active_dataset('alljoined')

from analysis.eeg_gen_eval.compute.compute_rsa import load_rsa_data, rsa_correlation  # noqa: E402
from analysis.eeg_gen_eval.figure_names import S_FIG4  # noqa: E402
from analysis.eeg_gen_eval.helpers.plot_band_heatmap import region_channel_layout  # noqa: E402
from analysis.eeg_gen_eval.plots.plot_final_fig2 import (  # noqa: E402
    _load_fig2_spectral_payload,
    _polish_psd_panel,
)
from analysis.eeg_gen_eval.plots.plot_final_fig3 import (  # noqa: E402
    NEUTRAL_COLOR,
    TRUE_COLOR,
    _load_matched_pair_data,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (  # noqa: E402
    _despine,
    _ensure_rmse_stats,
    _load_summary,
    _plot_psd_curves,
    _setup_nature_rc,
    add_panel_label,
)
from analysis.eeg_gen_eval.plots.plot_s_fig1 import (  # noqa: E402
    _cbar_compact,
    _draw_channel_heatmap,
    _plot_metrics_compact,
)
from analysis.eeg_gen_eval.helpers.plot_visualizations import (  # noqa: E402
    _load_aggregate,
    _plot_panel_b_occipital_erp,
    _plot_panel_c_region_boxplot,
    _time_axis_ms,
)

# Unified type scale for a dense double-column supplementary montage.
FS = {
    'title': 8.5,
    'letter': 11.0,
    'axis': 7.5,
    'tick': 6.5,
    'legend': 6.0,
    'annot': 6.5,
    'grey': 6.0,
    'sublabel': 7.0,
    'cbar': 6.5,
}

TEXT = '#111827'

# Always publish S_fig4 under the main figures/ directory.
OUT_FIG_DIR = os.path.join(cfg.OUT_ROOT, 'figures')


def _setup_rc() -> None:
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FS['tick'],
        'axes.labelsize': FS['axis'],
        'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'axes.linewidth': 0.65,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 2.2,
        'ytick.major.size': 2.2,
        'axes.labelcolor': TEXT,
        'text.color': TEXT,
    })


def _header(fig, spec, letter: str, title: str, *, label_x: float = -0.02):
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(
        0.5, 0.42, title,
        ha='center', va='center',
        fontsize=FS['title'], fontweight='bold', color=TEXT,
        transform=ax.transAxes, clip_on=False,
    )
    add_panel_label(ax, letter, x=label_x, y=1.05, fontsize=FS['letter'])
    return ax


def _apply_region_yticks(ax) -> None:
    """Replace THINGS-hardcoded ERP heatmap y-ticks with Alljoined regions."""
    for line in list(ax.lines):
        line.remove()
    _, region_bands, _ = region_channel_layout()
    ticks = [0.5 * (a + b - 1) for a, b, _ in region_bands]
    labels = [name[0] for _, _, name in region_bands]  # F/C/P/O
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=FS['tick'])
    for a, _b, _name in region_bands[1:]:
        ax.axhline(a - 0.5, color='white', lw=0.55, alpha=0.8, zorder=2)


def _load_per_channel_all(summary: dict) -> np.ndarray | None:
    """Subject × channel Pearson r; ignore stale THINGS-shaped cache files."""
    n_ch = len(cfg.CHANNELS)
    path = os.path.join(cfg.RAW_DIR, 'per_channel_pearson_all.npy')
    if os.path.isfile(path):
        arr = np.load(path)
        if arr.ndim == 2 and arr.shape[1] == n_ch:
            return arr
    rows = []
    for meta in summary.get('per_subject', []):
        sub = str(meta['subject'])
        cand = [
            os.path.join(cfg.RAW_DIR, 'visualization', sub, 'per_channel_pearson.npy'),
            os.path.join(cfg.RAW_DIR, sub, 'per_channel_pearson.npy'),
        ]
        hit = next((p for p in cand if os.path.isfile(p)), None)
        if hit is None:
            return None
        row = np.asarray(np.load(hit), dtype=np.float64).reshape(-1)
        if row.size != n_ch:
            return None
        rows.append(row)
    return np.stack(rows, axis=0) if rows else None


def plot_s_fig4_alljoined_montage() -> str:
    """Native redraw of Alljoined S Fig4; save under figures/."""
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba_cache')
    os.makedirs(os.environ['NUMBA_CACHE_DIR'], exist_ok=True)
    _setup_rc()

    summary = _load_summary()
    _ensure_rmse_stats(summary)
    per_ch_all = _load_per_channel_all(summary)

    vis = _load_aggregate()
    t_ms = _time_axis_ms(vis['erp_true'].shape[1])
    erp_t = np.asarray(vis['erp_true'], dtype=np.float64)
    erp_p = np.asarray(vis['erp_pred'], dtype=np.float64)
    vmax = float(np.max(np.abs(np.concatenate([erp_t, erp_p], axis=0))))
    if vmax <= 0:
        vmax = 1e-6

    payload = _load_fig2_spectral_payload()
    pair_stats, matched_cos, random_cos = _load_matched_pair_data()

    rsa_meta, rdm_gt_raw, rdm_pred_raw = load_rsa_data()
    rsa_order = np.asarray(rsa_meta['sort_order'], dtype=int)
    rdm_gt = rdm_gt_raw[np.ix_(rsa_order, rsa_order)]
    rdm_pred = rdm_pred_raw[np.ix_(rsa_order, rsa_order)]
    rsa_stats = rsa_correlation(rdm_gt, rdm_pred)
    iu = np.triu_indices(rdm_gt.shape[0], k=1)

    # Nature double-column (~183 mm); three content rows after dropping e/g.
    fig = plt.figure(figsize=(7.2, 7.6), facecolor='white')
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[1.18, 0.95, 1.05],
        hspace=0.32,
        left=0.068, right=0.972, top=0.968, bottom=0.045,
    )

    # ── Row 1: a metrics | b ERP heatmaps ─────────────────────────
    gs0 = outer[0].subgridspec(
        2, 1, height_ratios=[0.14, 1.0], hspace=0.02,
    )
    gs0b = gs0[1, 0].subgridspec(
        1, 5,
        width_ratios=[1.08, 0.10, 1.12, 1.12, 0.038],
        wspace=0.08,
    )
    gs0h = gs0[0, 0].subgridspec(
        1, 5,
        width_ratios=[1.08, 0.10, 1.12, 1.12, 0.038],
        wspace=0.08,
    )
    _header(fig, gs0h[0, 0], 'a', 'Prediction metrics', label_x=-0.08)
    ax_b_hdr = fig.add_subplot(gs0h[0, 2:4])
    ax_b_hdr.set_axis_off()
    ax_b_hdr.text(
        0.5, 0.42, 'Grand-average ERP',
        ha='center', va='center',
        fontsize=FS['title'], fontweight='bold', color=TEXT,
        transform=ax_b_hdr.transAxes, clip_on=False,
    )
    add_panel_label(ax_b_hdr, 'b', x=-0.02, y=1.05, fontsize=FS['letter'])

    ax_a = fig.add_subplot(gs0b[0, 0])
    _plot_metrics_compact(ax_a, summary, per_ch_all)
    ax_a.set_ylabel('Value', fontsize=FS['axis'])
    ax_a.tick_params(labelsize=FS['tick'])
    _GROUP_TAGS = {'Error ↓', 'Waveform ↑', 'Spectral ↑', 'Repr. ↑'}
    for txt in ax_a.texts:
        if txt.get_text() in _GROUP_TAGS:
            txt.set_position((txt.get_position()[0], -0.155))
            txt.set_fontsize(5.0)

    ax_gt = fig.add_subplot(gs0b[0, 2])
    im_b = _draw_channel_heatmap(
        ax_gt, erp_t, t_ms, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        ylabel='Channel', yticklabels=True,
    )
    _apply_region_yticks(ax_gt)
    ax_gt.text(
        0.5, 1.02, 'GT', transform=ax_gt.transAxes,
        ha='center', va='bottom', fontsize=FS['sublabel'],
        fontweight='bold', color='#374151', clip_on=False,
    )
    ax_gt.set_xlabel('Time (ms)', fontsize=FS['axis'], labelpad=1)
    ax_gt.set_ylabel('Channel', fontsize=FS['axis'], labelpad=1)
    ax_gt.tick_params(labelsize=FS['tick'], pad=1.0)

    ax_pred = fig.add_subplot(gs0b[0, 3])
    _draw_channel_heatmap(
        ax_pred, erp_p, t_ms, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
        ylabel=None, yticklabels=False,
    )
    _apply_region_yticks(ax_pred)
    ax_pred.set_yticklabels([])
    ax_pred.text(
        0.5, 1.02, 'Pred', transform=ax_pred.transAxes,
        ha='center', va='bottom', fontsize=FS['sublabel'],
        fontweight='bold', color='#374151', clip_on=False,
    )
    ax_pred.set_xlabel('Time (ms)', fontsize=FS['axis'], labelpad=1)
    ax_pred.tick_params(labelsize=FS['tick'], pad=1.0)

    cax_b = fig.add_subplot(gs0b[0, 4])
    _cbar_compact(cax_b, im_b, 'Amp.')

    # ── Row 2: c regional | d occipital ERP ───────────────────────
    gs1 = outer[1].subgridspec(
        2, 2,
        height_ratios=[0.15, 1.0],
        width_ratios=[1.0, 1.35],
        hspace=0.03, wspace=0.26,
    )
    _header(fig, gs1[0, 0], 'c', 'Regional waveform correlation', label_x=-0.06)
    _header(fig, gs1[0, 1], 'd', 'Occipital ERP comparison', label_x=-0.04)

    ax_c = fig.add_subplot(gs1[1, 0])
    _plot_panel_c_region_boxplot(
        ax_c, compact=True, show_title=False,
        annot_fontsize=FS['annot'] + 0.5,
        axis_fontsize=FS['axis'],
        tick_fontsize=FS['tick'],
    )
    ax_c.tick_params(labelsize=FS['tick'])

    ax_d = fig.add_subplot(gs1[1, 1])
    _plot_panel_b_occipital_erp(
        ax_d, t_ms, compact=True,
        annot_fontsize=FS['annot'],
        legend_fontsize=FS['legend'],
        axis_fontsize=FS['axis'],
        tick_fontsize=FS['tick'],
        show_early_visual=False,
    )
    ax_d.set_title('')
    ax_d.tick_params(labelsize=FS['tick'])

    # ── Row 3: e occipital PSD | f matched-pair | g RDM ────────────
    gs2 = outer[2].subgridspec(
        2, 3,
        height_ratios=[0.20, 1.0],
        width_ratios=[1.05, 1.0, 1.0],
        hspace=0.04, wspace=0.30,
    )

    # Shared title baseline for e|f|g (same y as _header).
    ax_e_hdr = _header(fig, gs2[0, 0], 'e', 'Occipital PSD', label_x=-0.08)
    for f0, f1, lab in (
        (0.5, 4, 'δ'), (4, 8, 'θ'), (8, 13, 'α'), (13, 30, 'β'), (30, 45, 'γ'),
    ):
        ax_e_hdr.text(
            (0.5 * (f0 + f1)) / 45.0, 0.0, lab,
            ha='center', va='bottom',
            fontsize=FS['grey'], color='#6B7280',
            transform=ax_e_hdr.transAxes, clip_on=False,
        )
    _header(fig, gs2[0, 1], 'f', 'Matched-pair specificity', label_x=-0.08)
    _header(fig, gs2[0, 2], 'g', 'RDM correspondence', label_x=-0.08)

    ax_e = fig.add_subplot(gs2[1, 0])
    _plot_psd_curves(
        ax_e,
        payload['occ_freqs'],
        payload['occ_true'], payload['occ_pred'],
        payload['occ_true_sem'], payload['occ_pred_sem'],
        title='',
        show_legend=False,
    )
    _polish_psd_panel(ax_e)
    ax_e.set_xlabel('Frequency (Hz)', fontsize=FS['axis'], labelpad=2)
    ax_e.set_ylabel('PSD (a.u.)', fontsize=FS['axis'], labelpad=1)
    ax_e.tick_params(labelsize=FS['tick'])
    leg = ax_e.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(FS['legend'])

    ax_f = fig.add_subplot(gs2[1, 1])
    bins = np.linspace(-0.05, 0.85, 35)
    _, _, random_patches = ax_f.hist(
        random_cos, bins=bins, density=True, alpha=0.38,
        color=NEUTRAL_COLOR, label='Random pairing',
        edgecolor='white', linewidth=0.35,
    )
    _, _, matched_patches = ax_f.hist(
        matched_cos, bins=bins, density=True, alpha=0.68,
        color=TRUE_COLOR, label='Matched image',
        edgecolor='white', linewidth=0.35,
    )
    ax_f.axvline(
        pair_stats['matched_cos_mean'], color=TRUE_COLOR,
        linestyle='--', linewidth=1.0,
    )
    ax_f.axvline(
        pair_stats['random_cos_mean'], color=NEUTRAL_COLOR,
        linestyle='--', linewidth=1.0,
    )
    ax_f.set_xlabel('Encoder cosine similarity', fontsize=FS['axis'])
    ax_f.set_ylabel('Density', fontsize=FS['axis'])
    ax_f.set_ylim(0, ax_f.get_ylim()[1] * 1.15)
    leg_f = ax_f.legend(
        handles=[matched_patches[0], random_patches[0]],
        labels=['Matched image', 'Random pairing'],
        frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
        loc='upper right', fontsize=FS['legend'],
        handlelength=1.0, handletextpad=0.35, labelspacing=0.2, borderaxespad=0.15,
    )
    _despine(ax_f)
    ax_f.tick_params(labelsize=FS['tick'])

    ax_g = fig.add_subplot(gs2[1, 2])
    x_rdm, y_rdm = rdm_gt[iu], rdm_pred[iu]
    limit = max(float(x_rdm.max()), float(y_rdm.max())) * 1.02
    ax_g.hexbin(
        x_rdm, y_rdm, gridsize=40, mincnt=1, bins='log',
        cmap='Blues', linewidths=0, alpha=0.92, rasterized=True,
    )
    ax_g.plot(
        [0, limit], [0, limit], linestyle='--',
        color='#8A8A8A', linewidth=0.7,
    )
    ax_g.set_xlim(0, limit)
    ax_g.set_ylim(0, limit)
    ax_g.set_aspect('equal', adjustable='box')
    ax_g.set_xticks(np.linspace(0, np.floor(limit * 2) / 2, 3))
    ax_g.set_yticks(np.linspace(0, np.floor(limit * 2) / 2, 3))
    ax_g.set_xlabel('GT RDM dissimilarity', fontsize=FS['axis'], labelpad=2)
    ax_g.set_ylabel('Pred RDM dissimilarity', fontsize=FS['axis'], labelpad=2)
    ax_g.text(
        0.97, 0.04,
        rf"Spearman $\rho$ = {rsa_stats['rsa_spearman']:.3f}",
        transform=ax_g.transAxes, ha='right', va='bottom',
        fontsize=FS['annot'], color='#303030',
    )
    _despine(ax_g)
    ax_g.tick_params(labelsize=FS['tick'])

    fig.canvas.draw()
    leg_bbox = leg_f.get_window_extent(fig.canvas.get_renderer())
    leg_axes = leg_bbox.transformed(ax_f.transAxes.inverted())
    ax_f.text(
        leg_axes.x1, leg_axes.y0 - 0.02,
        rf"$\Delta\mu$ = {pair_stats['cos_separation']:.3f}",
        transform=ax_f.transAxes, ha='right', va='top',
        fontsize=FS['annot'], color='#303030',
    )

    stem = os.path.join(OUT_FIG_DIR, S_FIG4)
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{png,svg}}')
    return stem


if __name__ == '__main__':
    plot_s_fig4_alljoined_montage()
