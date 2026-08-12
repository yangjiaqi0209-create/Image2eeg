"""Fig17: Selected generator ablation (structural + extended), SCI journal style."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.eeg_gen_eval.config import FIG_DIR, REPO_ROOT
from analysis.eeg_gen_eval.figure_names import FIG17, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import _setup_nature_rc, label_axes, save_pub

STRUCT_RAW = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'structural_ablation')
EXT_RAW = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'extended_ablation')

# (display label, short x-tick, source, condition_key)
SELECTED: List[Tuple[str, str, str, str]] = [
    ('Full (Ours)', 'Full', 'ext', 'E0_full'),
    ('w/o Fovea CLIP', 'w/o FoveaBlur', 'ext', 'E1_sharp'),
    ('w/o Dilated TCN', 'w/o Dilated', 'struct', 'S1_no_dconv'),
    ('w/o Transformer', 'w/o Transformer', 'struct', 'S3_tcn_only'),
    ('w/o self-attention', 'w/o self-attn', 'ext', 'E5_no_self_attn'),
    ('hidden=128', 'H=128', 'ext', 'E8_h128'),
    ('hidden=512', 'H=512', 'ext', 'E9_h512'),
]

# time / frequency / semantic / representation
PANELS = [
    ('test_corr', 'Pearson r', True),
    ('bandpower_corr', 'Bandpower corr', True),
    ('test_semantic_cosine', 'Semantic cos', True),
    ('rsa', r'RSA ($\rho$)', True),
]

FIG_FONT = {
    'tick': 7,
    'axis_label': 8.5,
    'panel_title': 9,
}

# Full = navy; ablations = single muted slate (SCI-style, not rainbow)
COLOR_FULL = '#1B4F72'
COLOR_ABLATION = '#8FA6B8'


def _load_bundle() -> Dict[str, Dict]:
    with open(os.path.join(STRUCT_RAW, 'summary.json')) as f:
        s_sum = json.load(f)['summary']
    with open(os.path.join(EXT_RAW, 'summary.json')) as f:
        e_sum = json.load(f)['summary']
    with open(os.path.join(STRUCT_RAW, 'rsa_summary.json')) as f:
        s_rsa = json.load(f)
    with open(os.path.join(EXT_RAW, 'rsa_summary.json')) as f:
        e_rsa = json.load(f)

    rows = []
    for label, short, src, key in SELECTED:
        meta = e_sum[key] if src == 'ext' else s_sum[key]
        rsa = e_rsa[key] if src == 'ext' else s_rsa[key]
        n = int(meta.get('n_subjects', rsa.get('n_subjects', 10)))
        row = {
            'label': label,
            'short': short,
            'n': n,
            'test_corr_mean': float(meta['test_corr_mean']),
            'test_corr_std': float(meta['test_corr_std']),
            'bandpower_corr_mean': float(meta['bandpower_corr_mean']),
            'bandpower_corr_std': float(meta['bandpower_corr_std']),
            'test_semantic_cosine_mean': float(meta['test_semantic_cosine_mean']),
            'test_semantic_cosine_std': float(meta['test_semantic_cosine_std']),
            'rsa_mean': float(rsa['rsa_mean']),
            'rsa_std': float(rsa['rsa_std']),
        }
        # SEM for journal-style uncertainty of the mean
        for base in ('test_corr', 'bandpower_corr', 'test_semantic_cosine', 'rsa'):
            std = row[f'{base}_std']
            row[f'{base}_sem'] = std / np.sqrt(n) if n > 1 else 0.0
        rows.append(row)
    return {'rows': rows}


def plot_selected_ablation(out_dir: str | None = None) -> str:
    _setup_nature_rc()
    data = _load_bundle()
    rows = data['rows']
    n_cond = len(rows)
    x = np.arange(n_cond)
    x_labels = [r['short'] for r in rows]
    full_means = {
        'test_corr': rows[0]['test_corr_mean'],
        'bandpower_corr': rows[0]['bandpower_corr_mean'],
        'test_semantic_cosine': rows[0]['test_semantic_cosine_mean'],
        'rsa': rows[0]['rsa_mean'],
    }

    fig_w = 7.4  # ~188 mm, double-column
    fig_h = 2.75
    fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), sharey=False)
    bar_colors = [COLOR_FULL] + [COLOR_ABLATION] * (n_cond - 1)

    for ax, (key, ylabel, _higher) in zip(axes, PANELS):
        ys = np.array([r[f'{key}_mean'] for r in rows], dtype=float)
        es = np.array([r[f'{key}_sem'] for r in rows], dtype=float)

        bars = ax.bar(
            x, ys, yerr=es, width=0.78, capsize=1.8,
            color=bar_colors,
            edgecolor='none',
            error_kw=dict(
                elinewidth=0.65, capthick=0.65, ecolor='#2A2A2A',
            ),
            zorder=3,
        )
        bars[0].set_edgecolor('#0D2F4A')
        bars[0].set_linewidth(0.8)

        # Full baseline
        ax.axhline(
            full_means[key], color=COLOR_FULL, linestyle=(0, (3, 2.5)),
            linewidth=0.75, alpha=0.65, zorder=2,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            x_labels, rotation=40, ha='right', rotation_mode='anchor',
            fontsize=FIG_FONT['tick'],
        )
        ax.set_ylabel(ylabel, fontsize=FIG_FONT['axis_label'], labelpad=1.5)
        ax.tick_params(axis='y', labelsize=FIG_FONT['tick'], length=2.2, width=0.5)
        ax.tick_params(axis='x', length=0, pad=1.5)

        y_lo = float(np.min(ys - es))
        y_hi = float(np.max(ys + es))
        span = max(y_hi - y_lo, 1e-3)
        ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.16 * span)
        ax.set_xlim(-0.6, n_cond - 0.4)

        ax.yaxis.grid(True, linestyle=':', linewidth=0.4, color='#C8C8C8', zorder=0)
        ax.set_axisbelow(True)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        for spine in ('left', 'bottom'):
            ax.spines[spine].set_linewidth(0.7)
            ax.spines[spine].set_color('#333333')

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.28, top=0.90, wspace=0.42)
    label_axes(axes, start='a')

    out = out_dir or FIG_DIR
    os.makedirs(out, exist_ok=True)
    # Keep figure_names stem; also write the canonical FIG17 path
    base = fig_path(FIG17) if out == FIG_DIR else os.path.join(out, FIG17)
    save_pub(fig, base)
    plt.close(fig)
    return f'{base}.svg'


def main():
    p = argparse.ArgumentParser(description='Plot selected generator ablation (fig17)')
    p.add_argument('--out_dir', default=FIG_DIR)
    args = p.parse_args()
    path = plot_selected_ablation(args.out_dir)
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
