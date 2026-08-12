"""Figure: 200×200 EEG–image feature similarity matrices (category-sorted)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle

from analysis.eeg_gen_eval.config import FIG_DIR, RAW_DIR
from analysis.eeg_gen_eval.figure_names import FIG7, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import _setup_nature_rc, add_panel_label, save_pub

# White → blue (shared scale across GT / Gen)
CMAP_SIM = LinearSegmentedColormap.from_list(
    'sim_pub',
    ['#ffffff', '#F1F5F9', '#DBEAFE', '#93C5FD', '#3B82F6', '#1D4ED8', '#0F4D92'],
    N=256,
)

DIAG_BOX_COLOR = '#9CA3AF'

CATEGORY_COLORS = {
    'animal': '#0F4D92',
    'food': '#B45309',
    'vehicle': '#3D8B8C',
    'tool': '#DC2626',
    'others': '#6B7280',
}


def _load_similarity_data() -> Dict:
    d = os.path.join(RAW_DIR, 'similarity_matrix')
    meta_path = os.path.join(d, 'meta.json')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f'Missing similarity cache {meta_path}. '
            'Re-run the similarity compute step that produced raw/similarity_matrix/.'
        )
    with open(meta_path) as f:
        meta = json.load(f)
    meta['sim_gt'] = np.load(os.path.join(d, 'sim_gt_mean.npy'))
    meta['sim_gen'] = np.load(os.path.join(d, 'sim_gen_mean.npy'))
    meta['sort_order'] = np.load(os.path.join(d, 'sort_order.npy')).tolist()
    return meta


def _reorder(sim: np.ndarray, order: List[int]) -> np.ndarray:
    idx = np.array(order, dtype=int)
    return sim[np.ix_(idx, idx)]


def _category_boundaries(block_sizes: Dict[str, int], category_order: List[str]) -> List[int]:
    edges = [0]
    for cat in category_order:
        edges.append(edges[-1] + int(block_sizes.get(cat, 0)))
    return edges


def _auto_vlim(sim: np.ndarray) -> Tuple[float, float]:
    vmin = 0.0
    vmax = float(np.percentile(sim, 99.5))
    diag = float(np.diag(sim).max())
    vmax = max(vmax, diag * 1.02)
    if vmax <= vmin + 1e-8:
        vmax = float(np.max(sim))
    return vmin, vmax


def _shared_vlim(*sims: np.ndarray) -> Tuple[float, float]:
    """Single color scale for all panels so GT vs gen are directly comparable."""
    vmax = max(_auto_vlim(sim)[1] for sim in sims)
    return 0.0, vmax


def _add_category_legend(fig, category_order: List[str]):
    handles = [
        Patch(
            facecolor=CATEGORY_COLORS.get(cat, '#888888'),
            edgecolor='none',
            label=cat.capitalize(),
        )
        for cat in category_order
    ]
    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.48, -0.02),
        fontsize=5.5,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=1.1,
    )


def _draw_category_axis_bars(ax, edges: List[int], category_order: List[str], n: int):
    """Colored segments flush with heatmap left / bottom edges."""
    bar_lw = 2.8
    y_bottom = n - 0.5
    x_left = -0.5
    for cat, a, b in zip(category_order, edges[:-1], edges[1:]):
        if b <= a:
            continue
        color = CATEGORY_COLORS.get(cat, '#888888')
        x0, x1 = a - 0.5, b - 0.5
        y0, y1 = a - 0.5, b - 0.5
        ax.plot(
            [x0, x1], [y_bottom, y_bottom],
            color=color, linewidth=bar_lw, solid_capstyle='butt',
            clip_on=False, zorder=6,
        )
        ax.plot(
            [x_left, x_left], [y0, y1],
            color=color, linewidth=bar_lw, solid_capstyle='butt',
            clip_on=False, zorder=6,
        )


def _outline_diagonal_blocks(ax, edges: List[int]):
    """Box each on-diagonal category block only."""
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        side = b - a
        ax.add_patch(
            Rectangle(
                (a - 0.5, a - 0.5), side, side,
                fill=False, edgecolor=DIAG_BOX_COLOR, linewidth=0.7, zorder=5,
            )
        )


def _colorbar_ticks(vmin: float, vmax: float, n_ticks: int = 5) -> np.ndarray:
    tick_vals = np.linspace(vmin, vmax, n_ticks)
    tick_vals[0] = 0.0
    return np.unique(np.round(tick_vals, 3))


def _plot_sim_panel(
    ax,
    sim: np.ndarray,
    *,
    title: str,
    letter: str,
    edges: List[int],
    category_order: List[str],
    vmin: float,
    vmax: float,
    show_ylabel: bool = True,
):
    n = sim.shape[0]
    im = ax.imshow(
        sim, cmap=CMAP_SIM, vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='nearest', origin='upper',
        rasterized=True,
    )
    _outline_diagonal_blocks(ax, edges)
    _draw_category_axis_bars(ax, edges, category_order, n)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Image CLIP feature', fontsize=7, labelpad=4)
    if show_ylabel:
        ax.set_ylabel('EEG feature', fontsize=7, labelpad=4)
    ax.set_title(title, fontweight='bold', fontsize=8, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('#9CA3AF')
    add_panel_label(ax, letter, x=-0.08, y=1.08)
    return im


def plot_fig6_eeg_image_similarity():
    """Two-panel 200×200 similarity matrices, shared color scale (Nature style)."""
    _setup_nature_rc()
    meta = _load_similarity_data()
    order = meta['sort_order']
    sim_gt = _reorder(meta['sim_gt'], order)
    sim_gen = _reorder(meta['sim_gen'], order)
    edges = _category_boundaries(meta['block_sizes'], meta['category_order'])
    n_sub = meta['n_subjects']
    vmin, vmax = _shared_vlim(sim_gt, sim_gen)
    cbar_ticks = _colorbar_ticks(vmin, vmax)

    fig = plt.figure(figsize=(7.2, 3.55))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 0.045],
        wspace=0.22,
        left=0.08, right=0.92, top=0.82, bottom=0.18,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    _plot_sim_panel(
        ax_a, sim_gt,
        title='Ground truth',
        letter='a',
        edges=edges,
        category_order=meta['category_order'],
        vmin=vmin,
        vmax=vmax,
        show_ylabel=True,
    )
    im = _plot_sim_panel(
        ax_b, sim_gen,
        title='Generated',
        letter='b',
        edges=edges,
        category_order=meta['category_order'],
        vmin=vmin,
        vmax=vmax,
        show_ylabel=False,
    )

    cb = fig.colorbar(im, cax=cax, extend='max')
    cb.set_label('Cosine similarity', fontsize=7, labelpad=3)
    cb.set_ticks(cbar_ticks)
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.55)
    cb.outline.set_linewidth(0.55)

    fig.suptitle(
        f'EEG–image feature similarity  ·  200 concepts  ·  n = {n_sub}',
        fontsize=9, fontweight='bold', y=0.96,
    )
    _add_category_legend(fig, meta['category_order'])

    stem = fig_path(FIG7)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white', bbox_inches='tight')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_all_similarity(subs: List[int] | None = None, force: bool = False):
    del subs, force  # regenerator removed; use existing raw/similarity_matrix cache
    plot_fig6_eeg_image_similarity()
