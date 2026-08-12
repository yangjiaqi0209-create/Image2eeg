"""Gallery figure: per-category query image + GT / Gen top-5 retrieval."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from analysis.eeg_gen_eval.compute.compute_retrieval_gallery import compute_retrieval_examples
from analysis.eeg_gen_eval.concept_categories import CATEGORY_ORDER
from analysis.eeg_gen_eval.config import BASELINE_LABEL, FIG_DIR, IMAGE_ROOT
from analysis.eeg_gen_eval.figure_names import FIG8, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import _setup_nature_rc, add_panel_label, save_pub

CATEGORY_COLORS = {
    'animal': '#4C78A8',
    'food': '#F58518',
    'vehicle': '#54A24B',
    'tool': '#E45756',
    'others': '#B279A2',
}

GALLERY_FONT = {
    'title': 15,
    'header': 12,
    'category': 12,
    'concept': 11,
    'row_label': 11,
    'footer': 10,
}

QUERY_EDGE = '#333333'

TOP_LABELS = ['Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5']
LABEL_TO_STRIP = [0.06, 1.0]   # GT/Gen label | five-image strip
LABEL_STRIP_WSPACE = 0.03      # gap between GT/Gen text and Top 1
QUERY_TO_RETR_WSPACE = 0.03     # gap between query image and GT/Gen column

QUERY_SIZE = 88
RETR_SIZE = 88


def _load_rgb(relpath: str, size: int) -> np.ndarray:
    path = os.path.join(IMAGE_ROOT, relpath)
    img = Image.open(path).convert('RGB')
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img)
    h, w = arr.shape[:2]
    pad = np.ones((size, size, 3), dtype=np.uint8) * 255
    y0 = (size - h) // 2
    x0 = (size - w) // 2
    pad[y0:y0 + h, x0:x0 + w] = arr
    return pad


def _show_image(
    ax,
    relpath: str,
    *,
    size: int,
    edge: Optional[str] = None,
    lw: float = 2.6,
):
    ax.imshow(_load_rgb(relpath, size), aspect='equal', interpolation='nearest')
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if edge is not None:
        ax.add_patch(Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes,
            fill=False, edgecolor=edge, linewidth=lw, clip_on=False,
        ))


def _build_strip(paths: Sequence[str], size: int) -> np.ndarray:
    return np.concatenate([_load_rgb(p, size) for p in paths], axis=1)


def _show_retrieval_strip(
    ax,
    strip: np.ndarray,
    *,
    size: int,
    hit_cols: Sequence[int],
    cat_color: str,
    lw: float = 2.8,
):
    h, w = strip.shape[:2]
    ax.imshow(strip, aspect='auto', interpolation='nearest')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for k in hit_cols:
        ax.add_patch(Rectangle(
            (k * size, 0), size, size,
            fill=False, edgecolor=cat_color, linewidth=lw, clip_on=False,
        ))


def _concept_label(relpath: str) -> str:
    folder = relpath.split('/')[1]
    return folder.split('_', 1)[1].replace('_', ' ')


def _draw_retrieval_row(
    fig,
    gs,
    row: int,
    ex: Dict,
    *,
    mode: str,
    cat_color: str,
    row_label: str,
    ret_col: int = 2,
    font_scale: float = 1.0,
):
    q_idx = ex['query_idx']
    if mode == 'gt':
        indices, paths = ex['gt_top5_idx'], ex['gt_top5_paths']
    else:
        indices, paths = ex['gen_top5_idx'], ex['gen_top5_paths']

    sg = gs[row, ret_col].subgridspec(
        1, 2, wspace=LABEL_STRIP_WSPACE, width_ratios=LABEL_TO_STRIP,
    )

    ax_lbl = fig.add_subplot(sg[0, 0])
    ax_lbl.set_axis_off()
    ax_lbl.text(
        0.5, 0.5, row_label, ha='center', va='center',
        fontsize=GALLERY_FONT['row_label'] * font_scale, fontweight='bold',
        rotation=90, color='#333333', transform=ax_lbl.transAxes,
    )

    hit_cols = [k for k, idx in enumerate(indices) if idx == q_idx]
    ax_strip = fig.add_subplot(sg[0, 1])
    _show_retrieval_strip(
        ax_strip, _build_strip(paths, RETR_SIZE),
        size=RETR_SIZE, hit_cols=hit_cols, cat_color=cat_color,
        lw=2.8 * font_scale,
    )


def draw_retrieval_gallery(
    fig,
    parent_spec,
    meta: Dict,
    *,
    panel_label: str = '',
    title: str = 'Image retrieval examples',
    font_scale: float = 1.0,
    category_colors: Optional[Dict[str, str]] = None,
    wrap_footer: bool = False,
    pred_label: str = 'Gen',
    show_title_context: bool = True,
    show_footer: bool = True,
):
    """Draw the complete five-category retrieval gallery into a subplot spec."""
    examples: List[Dict] = meta['examples']
    sub_tag = meta['subject']
    rows_per_cat = 2
    n_cat = len(CATEGORY_ORDER)
    n_body_rows = n_cat * rows_per_cat
    ret_col = 2
    colors = category_colors or CATEGORY_COLORS

    # col0 category | col1 query | col2 GT/Gen label + Top 1–5 strip
    width_ratios = [0.20, 0.72, 2.58]
    footer_ratio = (0.34 if wrap_footer else 0.22) if show_footer else 0.02
    height_ratios = [0.25, 0.20] + [1.0] * n_body_rows + [footer_ratio]
    gs = parent_spec.subgridspec(
        2 + n_body_rows + 1, len(width_ratios),
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=QUERY_TO_RETR_WSPACE, hspace=0.08,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_axis_off()
    display_title = (
        f'{title} ({sub_tag}, 200-way test set)'
        if show_title_context else title
    )
    ax_title.text(
        0.5, 0.55, display_title,
        ha='center', va='center',
        fontsize=GALLERY_FONT['title'] * font_scale,
        fontweight='bold', transform=ax_title.transAxes,
    )
    if panel_label:
        add_panel_label(
            ax_title, panel_label, x=-0.01, y=1.05,
            fontsize=GALLERY_FONT['header'] * font_scale,
        )

    ax_h_gt = fig.add_subplot(gs[1, 1])
    ax_h_gt.set_axis_off()
    ax_h_gt.text(0.5, 0.5, 'Ground Truth', ha='center', va='center',
                 fontsize=GALLERY_FONT['header'] * font_scale, fontweight='bold',
                 transform=ax_h_gt.transAxes)

    sg_h = gs[1, ret_col].subgridspec(
        1, 2, wspace=LABEL_STRIP_WSPACE, width_ratios=LABEL_TO_STRIP,
    )
    fig.add_subplot(sg_h[0, 0]).set_axis_off()
    ax_h_top = fig.add_subplot(sg_h[0, 1])
    ax_h_top.set_axis_off()
    for k, label in enumerate(TOP_LABELS):
        ax_h_top.text(
            (k + 0.5) / len(TOP_LABELS), 0.5, label,
            ha='center', va='center',
            fontsize=GALLERY_FONT['header'] * font_scale, fontweight='bold',
            transform=ax_h_top.transAxes,
        )

    for i, ex in enumerate(examples):
        r_gt = 2 + i * rows_per_cat
        r_gen = r_gt + 1
        cat = ex['category']
        color = colors.get(cat, '#888888')

        ax_cat = fig.add_subplot(gs[r_gt:r_gen + 1, 0])
        ax_cat.set_axis_off()
        ax_cat.add_patch(Rectangle(
            (0.08, 0.04), 0.22, 0.92, transform=ax_cat.transAxes,
            color=color, clip_on=False,
        ))
        ax_cat.text(
            0.62, 0.5, cat.capitalize(), ha='left', va='center',
            fontsize=GALLERY_FONT['category'] * font_scale, fontweight='bold',
            rotation=90, transform=ax_cat.transAxes,
        )

        ax_q = fig.add_subplot(gs[r_gt:r_gen + 1, 1])
        _show_image(
            ax_q, ex['query_path'], size=QUERY_SIZE, edge=QUERY_EDGE,
            lw=2.6 * font_scale,
        )
        ax_q.text(
            0.5, -0.06, _concept_label(ex['query_path']),
            transform=ax_q.transAxes, ha='center', va='top',
            fontsize=GALLERY_FONT['concept'] * font_scale, color='#222222',
        )

        _draw_retrieval_row(
            fig, gs, r_gt, ex, mode='gt', cat_color=color, row_label='GT',
            font_scale=font_scale,
        )
        _draw_retrieval_row(
            fig, gs, r_gen, ex, mode='gen', cat_color=color, row_label=pred_label,
            font_scale=font_scale,
        )

    ax_footer = fig.add_subplot(gs[-1, :])
    ax_footer.set_axis_off()
    footer = (
        'Category-colored border = correct match · random query per category with '
        f'GT & {pred_label} both in top-5 (seed = {meta["seed"]}) · '
        f'{BASELINE_LABEL} EEG encoder × fovea CLIP'
    )
    if wrap_footer:
        footer = (
            'Category-colored border = correct match · random query per category · '
            f'GT & {pred_label} both in top-5 (seed = {meta["seed"]})\n'
            f'{BASELINE_LABEL} EEG encoder × fovea CLIP'
        )
    if show_footer:
        ax_footer.text(
            0.5, 0.5,
            footer,
            ha='center', va='center',
            fontsize=GALLERY_FONT['footer'] * font_scale,
            color='#555555', transform=ax_footer.transAxes,
        )


def plot_fig7_retrieval_gallery(sub: int = 1, seed: int = 42):
    """Five categories; query | GT/Gen + Top 1–5 strip."""
    _setup_nature_rc()
    meta = compute_retrieval_examples(sub=sub, seed=seed)
    fig = plt.figure(figsize=(8.4, 11.0))
    fig.patch.set_facecolor('white')
    parent = fig.add_gridspec(
        1, 1, left=0.07, right=0.995, top=0.985, bottom=0.025,
    )[0, 0]
    draw_retrieval_gallery(
        fig, parent, meta,
        panel_label='a', title='Image retrieval by category',
    )

    stem = fig_path(FIG8)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_all_retrieval_gallery(sub: int = 1, seed: int = 42, force: bool = False):
    compute_retrieval_examples(sub=sub, seed=seed, force=force)
    plot_fig7_retrieval_gallery(sub=sub, seed=seed)
