"""Supplementary Figure S3: representational-alignment diagnostics.

Layout:
  Left:  a Complete five-category image-retrieval gallery
  Right: b Joint UMAP | c RDM residual
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

from analysis.eeg_gen_eval.compute.compute_retrieval_gallery import compute_retrieval_examples
from analysis.eeg_gen_eval.compute.compute_rsa import load_rsa_data
from analysis.eeg_gen_eval.concept_categories import CATEGORY_ORDER
from analysis.eeg_gen_eval.config import FIG_DIR
from analysis.eeg_gen_eval.figure_names import S_FIG3, fig_path
from analysis.eeg_gen_eval.helpers.plot_eeg_embedding import (
    _scatter_joint,
    compute_joint_embeddings,
    load_encoder_embeddings,
)
from analysis.eeg_gen_eval.plots.plot_final_fig3 import (
    JOURNAL_CATEGORY_COLORS,
    _compact_rdm_panel,
    _match_cbar_height_to_image,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _setup_nature_rc,
    add_panel_label,
    save_pub,
)
from analysis.eeg_gen_eval.helpers.plot_retrieval_gallery import draw_retrieval_gallery
from analysis.eeg_gen_eval.helpers.plot_similarity_matrix import (
    CATEGORY_COLORS,
    _category_boundaries,
)

S_FS = {
    'title': 9.5,
    'subtitle': 7.2,
    'axis': 8.0,
    'tick': 7.0,
    'legend': 6.5,
    'annot': 7.0,
    'letter': 11.5,
    'cbar': 7.0,
}

GT_COLOR = '#3B7FC4'
PRED_COLOR = '#E85A5A'
TEXT_COLOR = '#202124'


def _setup_rc() -> None:
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': S_FS['tick'],
        'axes.labelsize': S_FS['axis'],
        'axes.titlesize': S_FS['title'],
        'xtick.labelsize': S_FS['tick'],
        'ytick.labelsize': S_FS['tick'],
        'legend.fontsize': S_FS['legend'],
        'axes.labelcolor': TEXT_COLOR,
        'text.color': TEXT_COLOR,
    })


def _panel_header(fig, spec, label: str, title: str, *, label_x: float = -0.05):
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(
        0.5, 0.88, title,
        ha='center', va='center',
        fontsize=S_FS['title'], fontweight='bold',
        transform=ax.transAxes,
    )
    add_panel_label(
        ax, label, x=label_x, y=1.10, fontsize=S_FS['letter'],
    )
    return ax


def plot_s_fig3_representational_alignment_supp(
    sub: int = 1,
    seed: int = 42,
) -> str:
    """Assemble retrieval, UMAP and residual-RDM diagnostics."""
    _setup_rc()
    CATEGORY_COLORS.update(JOURNAL_CATEGORY_COLORS)

    # Joint UMAP payload.
    emb_true, emb_pred, _, _ = load_encoder_embeddings()
    _, _, umap_true, umap_pred = compute_joint_embeddings(
        emb_true, emb_pred,
    )

    # Residual-RDM payload.
    rsa_meta, rdm_gt_raw, rdm_pred_raw = load_rsa_data()
    rsa_order = np.asarray(rsa_meta['sort_order'], dtype=int)
    rdm_gt = rdm_gt_raw[np.ix_(rsa_order, rsa_order)]
    rdm_pred = rdm_pred_raw[np.ix_(rsa_order, rsa_order)]
    rdm_diff = rdm_pred - rdm_gt
    category_order = rsa_meta.get('category_order', CATEGORY_ORDER)
    rsa_edges = _category_boundaries(rsa_meta['block_sizes'], category_order)
    iu = np.triu_indices(rdm_diff.shape[0], k=1)
    diff_lim = float(np.percentile(np.abs(rdm_diff[iu]), 99.5))

    # Retrieval payload is loaded from cache when available.
    retrieval_meta = compute_retrieval_examples(sub=sub, seed=seed)

    # Final-size supplementary figure: ~183 mm double-column width.
    fig = plt.figure(figsize=(7.2, 7.8))
    fig.patch.set_facecolor('white')
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1.85, 1.0],
        wspace=0.12,
        left=0.035, right=0.985, top=0.985, bottom=0.030,
    )

    # a — complete retrieval gallery: retain width, reduce panel height.
    gs_left = outer[0, 0].subgridspec(
        2, 1, height_ratios=[0.76, 0.24], hspace=0.0,
    )
    draw_retrieval_gallery(
        fig, gs_left[0, 0], retrieval_meta,
        panel_label='a',
        title='Image retrieval examples',
        font_scale=0.62,
        category_colors=JOURNAL_CATEGORY_COLORS,
        wrap_footer=True,
        pred_label='Pred',
        show_title_context=False,
        show_footer=False,
    )

    # Match the combined b/c vertical span exactly to panel a.
    gs_right_outer = outer[0, 1].subgridspec(
        2, 1, height_ratios=[0.76, 0.24], hspace=0.0,
    )
    gs_right = gs_right_outer[0, 0].subgridspec(2, 1, hspace=0.04)

    # b — joint UMAP.
    group_b = gs_right[0, 0].subgridspec(
        2, 1, height_ratios=[0.17, 1.0], hspace=0.12,
    )
    _panel_header(fig, group_b[0, 0], 'b', 'Joint UMAP', label_x=-0.10)
    ax_umap = fig.add_subplot(group_b[1, 0])
    umap_overlap = _scatter_joint(
        ax_umap, umap_true, umap_pred,
        title='Joint UMAP', show_legend=True,
        true_color=GT_COLOR, pred_color=PRED_COLOR,
    )
    for text in list(ax_umap.texts):
        if text.get_text().startswith('KDE overlap'):
            text.remove()
    for collection in ax_umap.collections[:2]:
        collection.set_alpha(0.22)
        collection.set_sizes([8.0])
    ax_umap.set_title('')
    ax_umap.set_box_aspect(1)
    ax_umap.set_anchor('N')
    ax_umap.set_xlabel('UMAP 1')
    ax_umap.set_ylabel('UMAP 2')
    ax_umap.xaxis.label.set_size(9.0)
    ax_umap.yaxis.label.set_size(9.0)
    ax_umap.yaxis.labelpad = -2.0
    ax_umap.tick_params(labelsize=8.0, width=0.6, length=2.8)
    for spine in ax_umap.spines.values():
        spine.set_linewidth(0.6)
    ax_umap.text(
        0.96, 0.04, f'KDE overlap = {umap_overlap:.2f}',
        transform=ax_umap.transAxes,
        ha='right', va='bottom',
        fontsize=S_FS['subtitle'], color=TEXT_COLOR,
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 1.2},
    )
    legend = ax_umap.get_legend()
    if legend is not None:
        for text, label in zip(legend.get_texts(), ('GT EEG', 'Pred EEG')):
            text.set_text(label)
            text.set_fontsize(7.5)
        legend.set_frame_on(False)

    # c — residual RDM.
    group_c = gs_right[1, 0].subgridspec(
        2, 1, height_ratios=[0.10, 1.0], hspace=0.02,
    )
    _panel_header(fig, group_c[0, 0], '', '', label_x=-0.10)
    gs_res = group_c[1, 0].subgridspec(
        1, 2, width_ratios=[1.0, 0.065], wspace=0.08,
    )
    ax_res = fig.add_subplot(gs_res[0, 0])
    cax_res = fig.add_subplot(gs_res[0, 1])
    im_res = _compact_rdm_panel(
        ax_res, rdm_diff, title='', edges=rsa_edges,
        categories=category_order, cmap='RdBu_r',
        vmin=-diff_lim, vmax=diff_lim,
        ylabel='Concept i',
    )
    ax_res.set_anchor('S')
    ax_res.set_title(
        'RDM residual', fontsize=S_FS['title'],
        fontweight='bold', pad=4,
    )
    ax_res.xaxis.label.set_size(8.5)
    ax_res.yaxis.label.set_size(8.5)
    for edge in rsa_edges[1:-1]:
        boundary = edge - 0.5
        ax_res.axvline(
            boundary, color='white', linewidth=0.35,
            alpha=0.72, zorder=3,
        )
        ax_res.axhline(
            boundary, color='white', linewidth=0.35,
            alpha=0.72, zorder=3,
        )
    add_panel_label(
        ax_res, 'c', x=-0.10, y=1.09, fontsize=S_FS['letter'],
    )
    cb_res = fig.colorbar(im_res, cax=cax_res)
    cb_res.set_ticks([-diff_lim, 0, diff_lim])
    cb_res.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb_res.set_label('Δ(1−cos)', fontsize=S_FS['cbar'], labelpad=2)
    cb_res.ax.tick_params(
        labelsize=S_FS['cbar'], length=1.5, width=0.4, pad=0.6,
    )
    cb_res.outline.set_linewidth(0.4)

    _match_cbar_height_to_image(ax_res, cax_res)

    stem = fig_path(S_FIG3)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Wrote {stem}.{{png,svg}}')
    return stem


# Backward-compatible alias
plot_s_fig2_representational_alignment_supp = plot_s_fig3_representational_alignment_supp


if __name__ == '__main__':
    plot_s_fig3_representational_alignment_supp()
