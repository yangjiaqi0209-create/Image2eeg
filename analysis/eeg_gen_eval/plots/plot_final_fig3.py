"""Paper Results Figure 3: core EEG representational-alignment evidence.

Contents:
  a  EEG–image similarity                            ← fig7
  b  Matched-pair specificity                        ← fig11a
  c  GT / Pred RDM geometry                          ← fig14
  d  RDM correspondence (RSA)                        ← fig14
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FormatStrFormatter

from analysis.eeg_gen_eval.compute.compute_rsa import load_rsa_data, rsa_correlation
from analysis.eeg_gen_eval.config import FIG_DIR
from analysis.eeg_gen_eval.figure_names import FINAL_FIG3, fig_path
from analysis.eeg_gen_eval.helpers.plot_eeg_embedding import (
    EMBED_DIR,
    compute_matched_pair_stats,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _despine,
    _setup_nature_rc,
    add_panel_label,
    save_pub,
)
from analysis.eeg_gen_eval.helpers.plot_rsa import CMAP_RDM
from analysis.eeg_gen_eval.helpers.plot_similarity_matrix import (
    CATEGORY_COLORS,
    _category_boundaries,
    _colorbar_ticks,
    _draw_category_axis_bars,
    _load_similarity_data,
    _plot_sim_panel,
    _reorder as reorder_similarity,
    _shared_vlim,
)

FS = {
    'title': 10.0,
    'subtitle': 7.0,
    'axis': 8.0,
    'tick': 7.0,
    'legend': 6.5,
    'annot': 7.0,
    'letter': 12.0,
    'cbar': 6.5,
}

TRUE_COLOR = '#3B73A8'
NEUTRAL_COLOR = '#A7A7A7'
TEXT_COLOR = '#202124'
JOURNAL_CATEGORY_COLORS = {
    'animal': '#fbb4ae',
    'food': '#ffd9a8',
    'vehicle': '#b3cde4',
    'tool': '#cceac4',
    'others': '#decae5',
}


def _setup_rc() -> None:
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FS['tick'],
        'axes.labelsize': FS['axis'],
        'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'axes.titleweight': 'semibold',
        'axes.labelcolor': TEXT_COLOR,
        'axes.edgecolor': '#3F3F3F',
        'xtick.color': '#3F3F3F',
        'ytick.color': '#3F3F3F',
        'text.color': TEXT_COLOR,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
    })


def _compact_rdm_panel(
    ax,
    matrix: np.ndarray,
    *,
    title: str,
    edges: list[int],
    categories: list[str],
    cmap,
    vmin: float,
    vmax: float,
    ylabel: str = '',
    xlabel: str = 'Concept j',
):
    image = ax.imshow(
        matrix, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='nearest', origin='upper',
        rasterized=True,
    )
    _draw_category_axis_bars(ax, edges, categories, matrix.shape[0])
    ax.set_xlim(-0.5, matrix.shape[0] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1.5)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, fontweight='semibold', pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color('#B5B5B5')
    return image


def _match_cbar_height_to_image(ax, cax) -> None:
    """Align colorbar axes to the imshow square (exclude title / axis labels)."""
    fig = ax.figure
    fig.canvas.draw()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    fig_pts = fig.transFigure.inverted().transform(ax.transData.transform(corners))
    y_lo = float(fig_pts[:, 1].min())
    y_hi = float(fig_pts[:, 1].max())
    cpos = cax.get_position()
    cax.set_position([cpos.x0, y_lo, cpos.width, y_hi - y_lo])


def _refine_rdm_geometry(ax, edges: list[int], n: int) -> None:
    """Add crisp vector guides without smoothing the pairwise RDM values."""
    frame_effect = [
        path_effects.Stroke(
            linewidth=1.15, foreground='white', alpha=0.58,
        ),
        path_effects.Normal(),
    ]
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        side = stop - start
        frame = Rectangle(
            (start - 0.5, start - 0.5), side, side,
            fill=False, edgecolor='#8E96A1', linewidth=0.52,
            zorder=5,
        )
        frame.set_path_effects(frame_effect)
        ax.add_patch(frame)
    ax.plot(
        [-0.5, n - 0.5], [-0.5, n - 0.5],
        color='white', linewidth=0.42, alpha=0.92,
        solid_capstyle='butt', zorder=6,
    )


def _image_fig_bbox(ax) -> tuple[float, float, float, float]:
    """Return (x0, y0, width, height) of the equal-aspect image in figure coords."""
    fig = ax.figure
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    fp = fig.transFigure.inverted().transform(ax.transData.transform(corners))
    x_lo, x_hi = float(fp[:, 0].min()), float(fp[:, 0].max())
    y_lo, y_hi = float(fp[:, 1].min()), float(fp[:, 1].max())
    return x_lo, y_lo, x_hi - x_lo, y_hi - y_lo


def _pack_heatmap_pair(ax_left, ax_right, cax, *, gap: float = 0.006) -> None:
    """Pull two letterboxed equal-aspect heatmaps together; keep cbar on the right."""
    fig = ax_left.figure
    fig.canvas.draw()
    lx, _, lw, _ = _image_fig_bbox(ax_left)
    rx, _, _, _ = _image_fig_bbox(ax_right)
    shift = (lx + lw + gap) - rx
    # Only close an existing gap (negative shift moves the right panel left).
    if shift >= -1e-4:
        return
    for ax in (ax_right, cax):
        pos = ax.get_position()
        ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])


def _load_matched_pair_data() -> tuple[dict, np.ndarray, np.ndarray]:
    stats_path = os.path.join(EMBED_DIR, 'matched_pair_stats.json')
    arrays_path = os.path.join(EMBED_DIR, 'matched_pair_arrays.npz')
    if not os.path.isfile(stats_path) or not os.path.isfile(arrays_path):
        compute_matched_pair_stats()
    with open(stats_path) as f:
        stats = json.load(f)
    arrays = np.load(arrays_path)
    return stats, arrays['matched_cos'], arrays['random_cos']


def plot_final_fig3_representational_alignment() -> str:
    _setup_rc()
    CATEGORY_COLORS.update(JOURNAL_CATEGORY_COLORS)

    # Cross-modal similarity.
    sim_meta = _load_similarity_data()
    sim_order = sim_meta['sort_order']
    sim_gt = reorder_similarity(sim_meta['sim_gt'], sim_order)
    sim_pred = reorder_similarity(sim_meta['sim_gen'], sim_order)
    category_order = sim_meta['category_order']
    edges = _category_boundaries(sim_meta['block_sizes'], category_order)
    sim_vmin, sim_vmax = _shared_vlim(sim_gt, sim_pred)

    # Matched-pair specificity.
    pair_stats, matched_cos, random_cos = _load_matched_pair_data()

    # RDM / RSA.
    rsa_meta, rdm_gt_raw, rdm_pred_raw = load_rsa_data()
    rsa_order = np.asarray(rsa_meta['sort_order'], dtype=int)
    rdm_gt = rdm_gt_raw[np.ix_(rsa_order, rsa_order)]
    rdm_pred = rdm_pred_raw[np.ix_(rsa_order, rsa_order)]
    rsa_edges = _category_boundaries(rsa_meta['block_sizes'], category_order)
    rsa_stats = rsa_correlation(rdm_gt, rdm_pred)
    iu = np.triu_indices(rdm_gt.shape[0], k=1)
    rdm_vmax = float(np.percentile(np.maximum(rdm_gt, rdm_pred)[iu], 99.8))

    # Reading order: a → b → c → d.
    fig = plt.figure(figsize=(7.2, 4.85))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.08],
        hspace=0.23,
        left=0.062, right=0.972, top=0.978, bottom=0.118,
    )
    gs_top = outer[0].subgridspec(
        1, 2, width_ratios=[1.88, 1.00], wspace=0.28,
    )
    gs_bottom = outer[1].subgridspec(
        1, 2, width_ratios=[2.15, 1.00], wspace=0.34,
    )

    # a — EEG–image similarity.
    group_a = gs_top[0, 0].subgridspec(
        2, 1, height_ratios=[0.18, 1.0], hspace=0.14,
    )
    ax_a_hdr = fig.add_subplot(group_a[0, 0])
    ax_a_hdr.set_axis_off()
    ax_a_hdr.text(
        0.5, 0.55, 'EEG–image similarity',
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        transform=ax_a_hdr.transAxes,
    )
    add_panel_label(
        ax_a_hdr, 'a', x=-0.02, y=1.15, fontsize=FS['letter'],
    )
    gs1 = group_a[1, 0].subgridspec(
        1, 3, width_ratios=[1, 1, 0.030], wspace=0.00,
    )
    ax_sim_gt = fig.add_subplot(gs1[0, 0])
    ax_sim_pred = fig.add_subplot(gs1[0, 1])
    cax_sim = fig.add_subplot(gs1[0, 2])
    _plot_sim_panel(
        ax_sim_gt, sim_gt, title='', letter='',
        edges=edges, category_order=category_order,
        vmin=sim_vmin, vmax=sim_vmax, show_ylabel=True,
    )
    im_sim = _plot_sim_panel(
        ax_sim_pred, sim_pred, title='', letter='',
        edges=edges, category_order=category_order,
        vmin=sim_vmin, vmax=sim_vmax, show_ylabel=False,
    )
    for ax in (ax_sim_gt, ax_sim_pred):
        for edge in edges[1:-1]:
            boundary = edge - 0.5
            ax.axvline(
                boundary, color='white', linewidth=0.45,
                alpha=0.90, zorder=4,
            )
            ax.axhline(
                boundary, color='white', linewidth=0.45,
                alpha=0.90, zorder=4,
            )
        ax.set_xlabel('Image CLIP feature', labelpad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
            spine.set_color('#6B7280')
    ax_sim_gt.set_ylabel('GT EEG feature', labelpad=1.5)
    ax_sim_pred.set_ylabel('Pred EEG feature', labelpad=1.5)
    for ax in (ax_sim_gt, ax_sim_pred):
        ax.xaxis.label.set_size(FS['axis'])
        ax.yaxis.label.set_size(FS['axis'])
    cb_sim = fig.colorbar(im_sim, cax=cax_sim, extend='max')
    cb_sim.set_ticks(_colorbar_ticks(sim_vmin, sim_vmax, n_ticks=3))
    cb_sim.set_label(
        'Cosine similarity', fontsize=FS['cbar'],
        rotation=90, labelpad=2,
    )
    cb_sim.ax.tick_params(labelsize=FS['cbar'], length=2, width=0.5)
    cb_sim.outline.set_linewidth(0.5)

    # b — matched-pair specificity.
    group_b = gs_top[0, 1].subgridspec(
        2, 1, height_ratios=[0.18, 1.0], hspace=0.14,
    )
    ax_b_hdr = fig.add_subplot(group_b[0, 0])
    ax_b_hdr.set_axis_off()
    ax_b_hdr.text(
        0.5, 0.55, 'Matched-pair specificity',
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        transform=ax_b_hdr.transAxes,
    )
    add_panel_label(
        ax_b_hdr, 'b', x=-0.06, y=1.15, fontsize=FS['letter'],
    )
    ax_match = fig.add_subplot(group_b[1, 0])
    bins = np.linspace(-0.05, 0.85, 35)
    _, _, random_patches = ax_match.hist(
        random_cos, bins=bins, density=True, alpha=0.38,
        color=NEUTRAL_COLOR, label='Random pairing',
        edgecolor='white', linewidth=0.35,
    )
    _, _, matched_patches = ax_match.hist(
        matched_cos, bins=bins, density=True, alpha=0.68,
        color=TRUE_COLOR, label='Matched image',
        edgecolor='white', linewidth=0.35,
    )
    ax_match.axvline(
        pair_stats['matched_cos_mean'], color=TRUE_COLOR,
        linestyle='--', linewidth=1.0,
    )
    ax_match.axvline(
        pair_stats['random_cos_mean'], color=NEUTRAL_COLOR,
        linestyle='--', linewidth=1.0,
    )
    ax_match.set_xlabel('Encoder cosine similarity')
    ax_match.set_ylabel('Density')
    ax_match.set_ylim(0, ax_match.get_ylim()[1] * 1.17)
    ax_match.legend(
        handles=[matched_patches[0], random_patches[0]],
        labels=['Matched image', 'Random pairing'],
        frameon=True, facecolor='white', edgecolor='none', framealpha=0.92,
        loc='upper left', handlelength=1.1,
        handletextpad=0.4, labelspacing=0.25, borderaxespad=0.15,
    )
    ax_match.text(
        0.97, 0.95, rf"$\Delta\mu$ = {pair_stats['cos_separation']:.3f}",
        transform=ax_match.transAxes, ha='right', va='top',
        fontsize=FS['annot'], color='#303030',
    )
    _despine(ax_match)

    # c — GT / Pred RDM geometry.
    group_c = gs_bottom[0, 0].subgridspec(
        2, 1, height_ratios=[0.18, 1.0], hspace=0.10,
    )
    ax_c_hdr = fig.add_subplot(group_c[0, 0])
    ax_c_hdr.set_axis_off()
    ax_c_hdr.text(
        0.5, 0.55, 'RDM geometry',
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        transform=ax_c_hdr.transAxes,
    )
    add_panel_label(
        ax_c_hdr, 'c', x=-0.02, y=1.15, fontsize=FS['letter'],
    )
    _rdm_cbar_w = 0.070
    gs_rdm_pair = group_c[1, 0].subgridspec(
        1, 3, width_ratios=[1.0, 1.0, _rdm_cbar_w], wspace=0.04,
    )
    ax_rdm_gt = fig.add_subplot(gs_rdm_pair[0, 0])
    ax_rdm_pred = fig.add_subplot(gs_rdm_pair[0, 1])
    cax_rdm = fig.add_subplot(gs_rdm_pair[0, 2])
    _compact_rdm_panel(
        ax_rdm_gt, rdm_gt, title='GT RDM', edges=rsa_edges,
        categories=category_order, cmap=CMAP_RDM, vmin=0, vmax=rdm_vmax,
        ylabel='Concept i',
    )
    im_rdm = _compact_rdm_panel(
        ax_rdm_pred, rdm_pred, title='Pred RDM', edges=rsa_edges,
        categories=category_order, cmap=CMAP_RDM, vmin=0, vmax=rdm_vmax,
        ylabel='',
    )
    for ax in (ax_rdm_gt, ax_rdm_pred):
        _refine_rdm_geometry(ax, rsa_edges, rdm_gt.shape[0])
    cb_rdm = fig.colorbar(im_rdm, cax=cax_rdm)
    cb_rdm.set_ticks(np.linspace(0, rdm_vmax, 3))
    cb_rdm.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb_rdm.set_label('1−cos', fontsize=FS['cbar'], labelpad=2)
    cb_rdm.ax.tick_params(labelsize=FS['cbar'], length=1.5, width=0.4, pad=0.6)
    cb_rdm.outline.set_linewidth(0.4)

    # d — quantitative RDM correspondence.
    group_d = gs_bottom[0, 1].subgridspec(
        2, 1, height_ratios=[0.18, 1.0], hspace=0.12,
    )
    ax_d_hdr = fig.add_subplot(group_d[0, 0])
    ax_d_hdr.set_axis_off()
    ax_d_hdr.text(
        0.5, 0.55, 'RDM correspondence',
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        transform=ax_d_hdr.transAxes,
    )
    add_panel_label(
        ax_d_hdr, 'd', x=-0.06, y=1.15, fontsize=FS['letter'],
    )
    ax_rsa = fig.add_subplot(group_d[1, 0])
    x_rdm, y_rdm = rdm_gt[iu], rdm_pred[iu]
    limit = max(float(x_rdm.max()), float(y_rdm.max())) * 1.02
    ax_rsa.hexbin(
        x_rdm, y_rdm, gridsize=43, mincnt=1, bins='log',
        cmap='Blues', linewidths=0, alpha=0.92, rasterized=True,
    )
    ax_rsa.plot(
        [0, limit], [0, limit], linestyle='--',
        color='#8A8A8A', linewidth=0.7,
    )
    ax_rsa.set_xlim(0, limit)
    ax_rsa.set_ylim(0, limit)
    ax_rsa.set_aspect('equal')
    ax_rsa.set_xticks(np.linspace(0, np.floor(limit * 2) / 2, 3))
    ax_rsa.set_yticks(np.linspace(0, np.floor(limit * 2) / 2, 3))
    ax_rsa.set_xlabel('GT RDM dissimilarity', labelpad=2)
    ax_rsa.set_ylabel('Pred RDM dissimilarity', labelpad=2)
    ax_rsa.text(
        0.97, 0.04,
        rf"Spearman $\rho$ = {rsa_stats['rsa_spearman']:.3f}",
        transform=ax_rsa.transAxes, ha='right', va='bottom',
        fontsize=FS['annot'], color='#303030',
    )
    _despine(ax_rsa)

    for ax in (ax_match, ax_rsa):
        ax.tick_params(labelsize=FS['tick'], width=0.55, length=2.2)
    for ax in (ax_match, ax_rsa):
        ax.xaxis.label.set_size(FS['axis'])
        ax.yaxis.label.set_size(FS['axis'])
    for ax in (ax_rdm_gt, ax_rdm_pred):
        ax.title.set_size(FS['subtitle'])
        ax.title.set_fontweight('normal')
        ax.title.set_color('#5F6368')

    # One shared category key for matrix panels.
    category_handles = [
        Patch(
            facecolor=CATEGORY_COLORS[c], edgecolor='white',
            linewidth=0.35, label=c.capitalize(),
        )
        for c in category_order
    ]
    fig.legend(
        handles=category_handles,
        loc='lower center', bbox_to_anchor=(0.39, 0.012),
        ncol=len(category_handles), frameon=False,
        fontsize=FS['legend'], handlelength=1.15, handleheight=0.85,
        handletextpad=0.35, columnspacing=1.15,
    )

    _pack_heatmap_pair(ax_sim_gt, ax_sim_pred, cax_sim, gap=0.030)
    _match_cbar_height_to_image(ax_sim_pred, cax_sim)
    _match_cbar_height_to_image(ax_rdm_pred, cax_rdm)

    stem = fig_path(FINAL_FIG3)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    save_pub(fig, stem, svg_dpi=300)
    plt.close(fig)
    print(f'Wrote {stem}.{{png,svg}}')
    return stem


def main() -> None:
    plot_final_fig3_representational_alignment()


if __name__ == '__main__':
    main()
