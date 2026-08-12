"""Fig19/20: Loss-group ablation — bars (fig17-style) + Δ views (fig18-style)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.eeg_gen_eval.config import FIG_DIR, REPO_ROOT
from analysis.eeg_gen_eval.figure_names import FIG19, FIG20, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _setup_nature_rc,
    add_panel_label,
    label_axes,
    save_pub,
)

EXT_RAW = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'extended_ablation')
LOSS_RAW = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'loss_ablation')

PANELS = [
    ('test_corr', 'Pearson r'),
    ('bandpower_corr', 'Bandpower corr'),
    ('test_semantic_cosine', 'Semantic cos'),
    ('rsa', r'RSA ($\rho$)'),
]
METRIC_SHORT = ['Pearson r', 'Bandpower', 'Semantic cos', r'RSA ($\rho$)']

FIG_FONT = {
    'tick': 7,
    'axis_label': 8.5,
    'annot': 7.2,
    'cbar': 7.5,
}

COLOR_TWO_STAGE = '#1B4F72'
COLOR_SINGLE = '#3D6F93'
COLOR_ABLATION = '#8FA6B8'

CMAP = mcolors.LinearSegmentedColormap.from_list(
    'ablation_delta',
    [
        '#67001F', '#B2182B', '#D6604D', '#F4A582', '#FDDBC7',
        '#F7F7F7',
        '#D1E5F0', '#92C5DE', '#4393C3', '#2166AC', '#053061',
    ],
    N=512,
)


def _load_rows() -> List[Dict]:
    """Ordered rows for the five loss-ablation conditions."""
    with open(os.path.join(EXT_RAW, 'summary.json')) as f:
        e0 = json.load(f)['summary']['E0_full']
    with open(os.path.join(EXT_RAW, 'rsa_summary.json')) as f:
        e0_rsa = json.load(f)['E0_full']
    with open(os.path.join(LOSS_RAW, 'summary.json')) as f:
        loss = json.load(f)['summary']

    def pack(label: str, short: str, mean_src: dict, rsa_mean: float, rsa_std: float, n: int = 10):
        row = {
            'label': label,
            'short': short,
            'n': n,
            'test_corr_mean': float(mean_src['test_corr_mean']),
            'test_corr_std': float(mean_src['test_corr_std']),
            'bandpower_corr_mean': float(mean_src['bandpower_corr_mean']),
            'bandpower_corr_std': float(mean_src['bandpower_corr_std']),
            'test_semantic_cosine_mean': float(mean_src['test_semantic_cosine_mean']),
            'test_semantic_cosine_std': float(mean_src['test_semantic_cosine_std']),
            'rsa_mean': float(rsa_mean),
            'rsa_std': float(rsa_std),
        }
        for base in ('test_corr', 'bandpower_corr', 'test_semantic_cosine', 'rsa'):
            row[f'{base}_sem'] = row[f'{base}_std'] / np.sqrt(n) if n > 1 else 0.0
        return row

    return [
        pack('Full (two-stage)', 'Full\n(two-stage)', e0, e0_rsa['rsa_mean'], e0_rsa['rsa_std']),
        pack('Full (single-stage)', 'Full\n(single-stage)', loss['full'],
             loss['full']['rsa_mean'], loss['full']['rsa_std']),
        pack('w/o time loss', 'w/o time', loss['no_time'],
             loss['no_time']['rsa_mean'], loss['no_time']['rsa_std']),
        pack('w/o frequency loss', 'w/o freq', loss['no_freq'],
             loss['no_freq']['rsa_mean'], loss['no_freq']['rsa_std']),
        pack('w/o semantic loss', 'w/o semantic', loss['no_semantic'],
             loss['no_semantic']['rsa_mean'], loss['no_semantic']['rsa_std']),
    ]


def plot_loss_ablation_bars(out_dir: str | None = None) -> str:
    """Fig19: four-metric bar chart (fig17 style)."""
    _setup_nature_rc()
    rows = _load_rows()
    n_cond = len(rows)
    x = np.arange(n_cond)
    x_labels = [r['short'] for r in rows]
    colors = [COLOR_TWO_STAGE, COLOR_SINGLE] + [COLOR_ABLATION] * (n_cond - 2)

    # Reference line = two-stage Full
    ref = {
        'test_corr': rows[0]['test_corr_mean'],
        'bandpower_corr': rows[0]['bandpower_corr_mean'],
        'test_semantic_cosine': rows[0]['test_semantic_cosine_mean'],
        'rsa': rows[0]['rsa_mean'],
    }

    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.85), sharey=False)
    for ax, (key, ylabel) in zip(axes, PANELS):
        ys = np.array([r[f'{key}_mean'] for r in rows], dtype=float)
        es = np.array([r[f'{key}_sem'] for r in rows], dtype=float)
        bars = ax.bar(
            x, ys, yerr=es, width=0.72, capsize=1.8,
            color=colors, edgecolor='none',
            error_kw=dict(elinewidth=0.65, capthick=0.65, ecolor='#2A2A2A'),
            zorder=3,
        )
        bars[0].set_edgecolor('#0D2F4A')
        bars[0].set_linewidth(0.8)
        bars[1].set_edgecolor('#2A5570')
        bars[1].set_linewidth(0.7)

        ax.axhline(
            ref[key], color=COLOR_TWO_STAGE, linestyle=(0, (3, 2.5)),
            linewidth=0.75, alpha=0.65, zorder=2,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            x_labels, rotation=35, ha='right', rotation_mode='anchor',
            fontsize=FIG_FONT['tick'], linespacing=0.95,
        )
        ax.set_ylabel(ylabel, fontsize=FIG_FONT['axis_label'], labelpad=1.5)
        ax.tick_params(axis='y', labelsize=FIG_FONT['tick'], length=2.2, width=0.5)
        ax.tick_params(axis='x', length=0, pad=1.5)

        y_lo = float(np.min(ys - es))
        y_hi = float(np.max(ys + es))
        span = max(y_hi - y_lo, 1e-3)
        ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.16 * span)
        ax.set_xlim(-0.55, n_cond - 0.45)
        ax.yaxis.grid(True, linestyle=':', linewidth=0.4, color='#C8C8C8', zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.7)
            ax.spines[spine].set_color('#333333')

    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.30, top=0.90, wspace=0.40)
    label_axes(axes, start='a')

    out = out_dir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    base = fig_path(FIG19) if out == FIG_DIR else os.path.join(out, FIG19)
    save_pub(fig, base)
    plt.close(fig)
    return f'{base}.svg'


def plot_loss_ablation_delta(out_dir: str | None = None) -> str:
    """Fig20: Δ heatmap + diverging bars vs Full (single-stage)."""
    _setup_nature_rc()
    rows = _load_rows()
    # baseline = Full (single-stage); compare two-stage + three ablations
    base_idx = 1
    keys = [k for k, _ in PANELS]
    base_means = np.array([rows[base_idx][f'{k}_mean'] for k in keys], dtype=float)
    compare = [rows[0]] + rows[2:]  # two-stage, no_time, no_freq, no_semantic
    labels = [r['short'].replace('\n', ' ') for r in compare]
    # shorter labels for y-axis
    labels = [
        'Full (two-stage)',
        'w/o time',
        'w/o freq',
        'w/o semantic',
    ]
    means = np.array([[r[f'{k}_mean'] for k in keys] for r in compare], dtype=float)
    delta = means - base_means[None, :]
    pct = 100.0 * delta / np.maximum(np.abs(base_means[None, :]), 1e-8)
    metric_names = [n for _, n in PANELS]
    n_abl, n_met = pct.shape
    y = np.arange(n_abl)

    fig = plt.figure(figsize=(7.8, 5.4))
    gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[0.95, 1.25],
        hspace=0.40,
        left=0.14, right=0.96, top=0.90, bottom=0.08,
    )

    # ── a) heatmap ───────────────────────────────────────────────────
    gs_a = gs[0].subgridspec(1, 3, width_ratios=[0.45, 2.5, 0.45], wspace=0.08)
    ax_h = fig.add_subplot(gs_a[0, 1])
    vmax = float(np.nanmax(np.abs(pct)))
    vmax = max(np.ceil(vmax / 10.0) * 10.0, 10.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    xx = np.arange(n_met + 1, dtype=float)
    yy = np.arange(n_abl + 1, dtype=float)
    mesh = ax_h.pcolormesh(
        xx, yy, pct, cmap=CMAP, norm=norm,
        edgecolors='#F7F7F7', linewidths=0.45,
        antialiased=True, shading='flat',
    )
    ax_h.set_xlim(0, n_met)
    ax_h.set_ylim(n_abl, 0)
    ax_h.set_xticks(np.arange(n_met) + 0.5)
    ax_h.set_xticklabels(METRIC_SHORT, fontsize=FIG_FONT['tick'])
    ax_h.set_yticks(np.arange(n_abl) + 0.5)
    ax_h.set_yticklabels(labels, fontsize=FIG_FONT['tick'])
    ax_h.tick_params(length=0, pad=3)
    ax_h.xaxis.tick_top()

    for i in range(n_abl):
        for j in range(n_met):
            val = pct[i, j]
            rgba = CMAP(norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = '#FFFFFF' if lum < 0.52 else '#111111'
            ax_h.text(
                j + 0.5, i + 0.5, f'{val:+.1f}%',
                ha='center', va='center',
                fontsize=FIG_FONT['annot'], color=txt_color,
                fontweight='medium', clip_on=True,
            )

    for spine in ax_h.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color('#4A4A4A')

    divider = make_axes_locatable(ax_h)
    cax = divider.append_axes('right', size='4.5%', pad=0.08)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=FIG_FONT['cbar'], length=2.4, width=0.5)
    cbar.set_label(r'$\Delta$ vs Full (single-stage) (%)', fontsize=FIG_FONT['axis_label'], labelpad=4)
    cbar.outline.set_linewidth(0.5)
    add_panel_label(ax_h, 'a', x=-0.28, y=1.16)

    # ── b) diverging bars ────────────────────────────────────────────
    gs_b = gs[1].subgridspec(1, 4, wspace=0.28)
    div_axes = [fig.add_subplot(gs_b[0, j]) for j in range(4)]
    for j, ax in enumerate(div_axes):
        d = delta[:, j]
        colors = ['#2166AC' if v >= 0 else '#B2182B' for v in d]
        ax.barh(y, d, height=0.62, color=colors, edgecolor='none', zorder=3)
        ax.axvline(0, color='#333333', linewidth=0.7, zorder=2)
        ax.set_yticks(y)
        if j == 0:
            ax.set_yticklabels(labels, fontsize=FIG_FONT['tick'])
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(rf'$\Delta$ {metric_names[j]}', fontsize=FIG_FONT['axis_label'], labelpad=2)
        ax.tick_params(axis='x', labelsize=FIG_FONT['tick'] - 0.5, length=2.2, width=0.5)
        ax.tick_params(axis='y', length=0, pad=2)
        ax.invert_yaxis()
        ax.xaxis.grid(True, linestyle=':', linewidth=0.4, color='#C8C8C8', zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.6)
            ax.spines[spine].set_color('#333333')
        lim = float(np.max(np.abs(d))) * 1.15 + 1e-4
        ax.set_xlim(-lim, lim)

    add_panel_label(div_axes[0], 'b', x=-0.42, y=1.08)

    out = out_dir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    base = fig_path(FIG20) if out == FIG_DIR else os.path.join(out, FIG20)
    save_pub(fig, base)
    plt.close(fig)
    return f'{base}.svg'


def main():
    p = argparse.ArgumentParser(description='Plot loss-group ablation fig19/fig20')
    p.add_argument('--out_dir', default=FIG_DIR)
    p.add_argument('--which', choices=('bars', 'delta', 'all'), default='all')
    args = p.parse_args()
    if args.which in ('bars', 'all'):
        print(f'Wrote {plot_loss_ablation_bars(args.out_dir)}')
    if args.which in ('delta', 'all'):
        print(f'Wrote {plot_loss_ablation_delta(args.out_dir)}')


if __name__ == '__main__':
    main()
