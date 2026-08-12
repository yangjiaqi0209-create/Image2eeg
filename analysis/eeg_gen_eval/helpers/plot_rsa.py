"""Visualize GT vs Gen EEG RDM and RSA."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.eeg_gen_eval.concept_categories import CATEGORY_ORDER
from analysis.eeg_gen_eval.compute.compute_rsa import (
    compute_rsa,
    compute_rsa_permutation_test,
    load_rsa_data,
    rsa_correlation,
)
from analysis.eeg_gen_eval.config import BASELINE_LABEL, FIG_DIR, RAW_DIR
from analysis.eeg_gen_eval.figure_names import FIG14, FIG15, FIG16, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import _setup_nature_rc, add_panel_label, label_axes, save_pub
from analysis.eeg_gen_eval.helpers.plot_similarity_matrix import (
    CATEGORY_COLORS,
    _add_category_legend,
    _category_boundaries,
    _draw_category_axis_bars,
    _outline_diagonal_blocks,
)

CMAP_RDM = LinearSegmentedColormap.from_list(
    'rdm',
    ['#ffffff', '#f7f0f5', '#e8d4e6', '#d4a8cf', '#b87db5', '#9a5299', '#7a2e7d'],
    N=256,
)


def _reorder(mat: np.ndarray, order: List[int]) -> np.ndarray:
    idx = np.array(order, dtype=int)
    return mat[np.ix_(idx, idx)]


def _rdm_vlim(rdm: np.ndarray, pct: float = 99.5) -> Tuple[float, float]:
    iu = np.triu_indices(rdm.shape[0], k=1)
    vals = rdm[iu]
    return 0.0, float(np.percentile(vals, pct))


def _plot_rdm_panel(
    ax,
    rdm: np.ndarray,
    *,
    title: str,
    edges: List[int],
    category_order: List[str],
    vmin: float,
    vmax: float,
    ylabel: str,
):
    n = rdm.shape[0]
    im = ax.imshow(
        rdm, cmap=CMAP_RDM, vmin=vmin, vmax=vmax,
        aspect='equal', interpolation='nearest', origin='upper',
        rasterized=True,
    )
    _outline_diagonal_blocks(ax, edges)
    _draw_category_axis_bars(ax, edges, category_order, n)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Test concept', fontsize=11, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color('#cccccc')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4.5%', pad=0.12)
    cb = ax.figure.colorbar(im, cax=cax, extend='max')
    cb.set_label('Dissimilarity (1 − cos)', fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9, width=0.4, length=2)
    cb.outline.set_linewidth(0.4)
    return im


def plot_fig_rsa_rdm(force: bool = False):
    if force:
        import shutil
        rsa_dir = os.path.join(RAW_DIR, 'rsa')
        if os.path.isdir(rsa_dir):
            shutil.rmtree(rsa_dir)
        compute_rsa(force=True)

    meta, rdm_gt, rdm_gen = load_rsa_data()
    order = meta['sort_order']
    rdm_gt = _reorder(rdm_gt, order)
    rdm_gen = _reorder(rdm_gen, order)
    rdm_diff = rdm_gen - rdm_gt
    edges = _category_boundaries(meta['block_sizes'], meta.get('category_order', CATEGORY_ORDER))
    category_order = meta.get('category_order', CATEGORY_ORDER)
    rsa = rsa_correlation(rdm_gt, rdm_gen)
    n_sub = meta['n_subjects']

    vmin, vmax = _rdm_vlim(np.maximum(rdm_gt, rdm_gen))

    _setup_nature_rc()
    fig = plt.figure(figsize=(13.2, 4.8))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.45)

    ax_gt = fig.add_subplot(gs[0, 0])
    ax_gen = fig.add_subplot(gs[0, 1])
    ax_diff = fig.add_subplot(gs[0, 2])
    ax_sc = fig.add_subplot(gs[0, 3])

    _plot_rdm_panel(
        ax_gt, rdm_gt,
        title='GT EEG RDM',
        edges=edges,
        category_order=category_order,
        vmin=vmin, vmax=vmax,
        ylabel='GT EEG (brain emb.)',
    )
    _plot_rdm_panel(
        ax_gen, rdm_gen,
        title='Generated EEG RDM',
        edges=edges,
        category_order=category_order,
        vmin=vmin, vmax=vmax,
        ylabel='Gen EEG (brain emb.)',
    )

    diff_lim = float(np.percentile(np.abs(rdm_diff[np.triu_indices(rdm_diff.shape[0], k=1)]), 99))
    im_d = ax_diff.imshow(
        rdm_diff, cmap='RdBu_r', vmin=-diff_lim, vmax=diff_lim,
        aspect='equal', interpolation='nearest', origin='upper',
        rasterized=True,
    )
    _outline_diagonal_blocks(ax_diff, edges)
    _draw_category_axis_bars(ax_diff, edges, category_order, rdm_diff.shape[0])
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    ax_diff.set_xlabel('Test concept', fontsize=11, labelpad=6)
    ax_diff.set_ylabel('Δ dissimilarity', fontsize=11, labelpad=8)
    ax_diff.set_title('Gen − GT', fontweight='bold', fontsize=12, pad=10)
    divider = make_axes_locatable(ax_diff)
    cax_d = divider.append_axes('right', size='4.5%', pad=0.12)
    cb_d = fig.colorbar(im_d, cax=cax_d)
    cb_d.set_label('Δ (1 − cos)', fontsize=10)
    cb_d.ax.tick_params(labelsize=9)

    iu = np.triu_indices(rdm_gt.shape[0], k=1)
    x = rdm_gt[iu]
    y = rdm_gen[iu]
    ax_sc.scatter(x, y, s=4, alpha=0.35, c='#4C78A8', edgecolors='none', rasterized=True)
    lim = max(x.max(), y.max()) * 1.02
    ax_sc.plot([0, lim], [0, lim], '--', color='#888888', lw=0.8, zorder=0)
    ax_sc.set_xlim(0, lim)
    ax_sc.set_ylim(0, lim)
    ax_sc.set_aspect('equal')
    ax_sc.set_xlabel('GT RDM', fontsize=11)
    ax_sc.set_ylabel('Gen RDM', fontsize=11)
    ax_sc.set_title(
        f"RSA ρ = {rsa['rsa_spearman']:.3f}\n"
        f"(per-subj: {meta['rsa_per_subject_mean']:.3f} "
        f"± {meta['rsa_per_subject_std']:.3f})",
        fontweight='bold', fontsize=11, pad=8,
    )

    _add_category_legend(fig, category_order)
    fig.suptitle(
        f'EEG representational geometry — {BASELINE_LABEL}\n'
        f'200 test concepts, mean over {n_sub} subjects (brain-encoder RDM)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.82, bottom=0.16)
    label_axes([ax_gt, ax_gen, ax_diff, ax_sc], start='a')

    stem = fig_path(FIG14)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return meta


def plot_fig_rsa_per_subject():
    """Bar chart of per-subject RSA Spearman."""
    meta_path = os.path.join(RAW_DIR, 'rsa', 'meta.json')
    if not os.path.isfile(meta_path):
        compute_rsa()
    with open(meta_path) as f:
        meta = json.load(f)

    subs = [s['subject'] for s in meta['per_subject']]
    vals = [s['rsa_spearman'] for s in meta['per_subject']]
    mean_v = meta['rsa_per_subject_mean']

    _setup_nature_rc()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    xs = np.arange(len(subs))
    ax.bar(xs, vals, color='#4C78A8', edgecolor='white', linewidth=0.5, width=0.72)
    ax.axhline(mean_v, color='#E45756', ls='--', lw=1.0, label=f'mean = {mean_v:.3f}')
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace('sub-', '') for s in subs], fontsize=9)
    ax.set_xlabel('Subject')
    ax.set_ylabel('RSA (Spearman ρ)')
    ax.set_title(f'GT vs Gen RDM correlation — {BASELINE_LABEL}', fontweight='bold', fontsize=11)
    ax.set_ylim(0, min(1.0, max(vals) * 1.15))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    add_panel_label(ax, 'a')
    stem = fig_path(FIG15)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_fig_rsa_permutation_null(force: bool = False):
    """Nature-style RSA figure: subject strip + null KDE (no histogram)."""
    from scipy.stats import gaussian_kde

    from analysis.eeg_gen_eval.helpers.plot_quality import PALETTE

    perm_path = os.path.join(RAW_DIR, 'rsa', 'permutation_meta.json')
    if force or not os.path.isfile(perm_path):
        compute_rsa_permutation_test(force=force)
    with open(perm_path) as f:
        perm = json.load(f)

    null_iter = np.load(os.path.join(RAW_DIR, 'rsa', 'null_rsa_group_iter.npy'))
    g = perm['group']
    subjects = [s['subject'].replace('sub-', '') for s in perm['per_subject']]
    obs_vals = np.array([s['rsa_observed'] for s in perm['per_subject']], dtype=float)
    obs_mean = float(g['rsa_observed_mean'])
    obs_std = float(g['rsa_observed_std'])
    obs_sem = float(obs_std / np.sqrt(len(obs_vals)))
    p = float(g['p_one_sided_iter_mean'])
    p_str = 'P < 0.001' if p <= 0.001 else f'P = {p:.3f}'
    null_lo, null_hi = np.quantile(null_iter, [0.025, 0.975])

    _setup_nature_rc()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.75),
        gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.45},
    )
    fig.patch.set_facecolor('white')
    ax, ax_n = axes

    # a | Per-subject RSA as strip + mean ± SEM at right
    xs = np.arange(len(obs_vals))
    ax.scatter(
        xs, obs_vals, s=22, color=PALETTE['true'],
        edgecolors='white', linewidths=0.4, zorder=3,
        label='subject',
    )
    ax.errorbar(
        len(xs) + 0.35, obs_mean, yerr=obs_sem,
        fmt='D', color=PALETTE['pred'], ecolor=PALETTE['pred'],
        elinewidth=0.9, capsize=2.5, markersize=4.5,
        markeredgecolor='white', markeredgewidth=0.4, zorder=4,
        label=r'mean $\pm$ s.e.m.',
    )
    ax.axhline(obs_mean, color=PALETTE['pred'], ls='--', lw=0.7, alpha=0.75, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(subjects)
    ax.set_xlabel('Subject')
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_ylim(0.35, 0.58)
    ax.set_xlim(-0.7, len(xs) + 1.0)
    ax.legend(loc='lower right', fontsize=6, handlelength=1.2)
    ax.text(
        0.03, 0.97,
        f'n = 10\n{p_str}',
        transform=ax.transAxes, ha='left', va='top', fontsize=6, color='#444444',
    )
    add_panel_label(ax, 'a')

    # b | Null as filled KDE
    kde = gaussian_kde(null_iter, bw_method='scott')
    x_grid = np.linspace(null_iter.min() - 0.002, null_iter.max() + 0.002, 400)
    dens = kde(x_grid)
    ax_n.fill_between(x_grid, dens, color='#C5C5C5', alpha=0.85, linewidth=0, zorder=1)
    ax_n.plot(x_grid, dens, color='#555555', lw=1.0, zorder=3)
    ci_mask = (x_grid >= null_lo) & (x_grid <= null_hi)
    ax_n.fill_between(
        x_grid[ci_mask], dens[ci_mask],
        color=PALETTE['neutral'], alpha=0.35, linewidth=0, zorder=2,
        label='95% CI',
    )
    ax_n.axvline(0.0, color='#222222', ls='--', lw=0.8, zorder=4)
    rng = np.random.default_rng(0)
    rug = null_iter[rng.choice(len(null_iter), size=min(120, len(null_iter)), replace=False)]
    y0 = -0.03 * dens.max()
    ax_n.plot(
        rug, np.full_like(rug, y0), '|', color='#888888',
        markersize=4, alpha=0.45, zorder=2,
    )
    ax_n.set_xlabel(r'Spearman $\rho$')
    ax_n.set_ylabel('Density')
    ax_n.set_ylim(y0 * 1.8, dens.max() * 1.18)
    ax_n.set_xlim(x_grid.min(), x_grid.max())
    ax_n.legend(loc='upper right', fontsize=6, handlelength=1.0)
    ax_n.text(
        0.03, 0.97,
        'null (1,000 shuffles)',
        transform=ax_n.transAxes, ha='left', va='top', fontsize=6, color='#444444',
    )
    add_panel_label(ax_n, 'b')

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18)
    stem = fig_path(FIG16)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    print(
        f'Caption: (a) Per-subject GT-Gen RSA '
        f'(mean $\\rho$ = {obs_mean:.3f} $\\pm$ {obs_sem:.3f} s.e.m.). '
        f'(b) Null density from 1,000 image-EEG shuffles '
        f'(95% CI [{null_lo:.3f}, {null_hi:.3f}]). {p_str}.'
    )
    return perm


def plot_all_rsa(force: bool = False):
    meta = plot_fig_rsa_rdm(force=force)
    plot_fig_rsa_per_subject()
    plot_fig_rsa_permutation_null(force=force)
    return meta


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    plot_all_rsa(force=args.force)
