"""Paper Results Figure 4: predictor & loss ablations.

Contents:
  a  Predictor architecture ablation bars          ← fig17
  b  Loss-group ablation bars                      ← fig19
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from analysis.eeg_gen_eval.config import FIG_DIR
from analysis.eeg_gen_eval.figure_names import FINAL_FIG4, fig_path
from analysis.eeg_gen_eval.helpers.plot_loss_ablation import (
    _load_rows as _load_loss_rows,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _despine,
    _setup_nature_rc,
    add_panel_label,
    save_pub,
)
from analysis.eeg_gen_eval.helpers.plot_selected_ablation import _load_bundle

# Journal palette: deep navy Ours, cool secondary, soft slate ablations.
BAR_OURS = '#1B3F66'
BAR_SECOND = '#6B9BB8'
BAR_ABLATION = '#B0BEC8'
ERR = '#3F4A56'
BASELINE = '#1B3F66'

METRIC_PANELS = [
    ('test_corr', 'Pearson $r$'),
    ('bandpower_corr', 'Bandpower corr.'),
    ('test_semantic_cosine', 'Semantic cos.'),
]

FS = {
    'title': 9.5,
    'subtitle': 7.2,
    'axis': 7.2,
    'tick': 6.4,
    'letter': 11.5,
    'legend': 6.2,
}

GRID = '#EEF0F2'
SPINE = '#6B7280'
TEXT = '#111827'
MUTED = '#6B7280'


def _setup_rc() -> None:
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FS['tick'],
        'axes.labelsize': FS['axis'],
        'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'axes.labelcolor': TEXT,
        'text.color': TEXT,
        'xtick.color': SPINE,
        'ytick.color': SPINE,
        'axes.edgecolor': SPINE,
        'axes.linewidth': 0.65,
    })


def _header(fig, parent, title: str, letter: str, *, legend_handles=None):
    """Centered title + panel letter; optional legend on the right."""
    gs = parent.subgridspec(2, 1, height_ratios=[0.26, 1.0], hspace=0.10)
    ax_hdr = fig.add_subplot(gs[0, 0])
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.5, 0.42, title,
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        color=TEXT, transform=ax_hdr.transAxes,
    )
    add_panel_label(ax_hdr, letter, x=-0.01, y=1.08, fontsize=FS['letter'])
    if legend_handles:
        ax_hdr.legend(
            handles=legend_handles,
            loc='center right', bbox_to_anchor=(1.0, 0.42),
            frameon=False, fontsize=FS['legend'],
            handlelength=1.8, handletextpad=0.4,
            borderaxespad=0.0,
        )
    return gs[1, 0]


def _bar_xlabels(rows: list[dict], *, ours_as: str) -> list[str]:
    labels = []
    for i, r in enumerate(rows):
        raw = r['short'].replace('\n', ' ')
        if i == 0:
            labels.append(ours_as)
        elif 'single-stage' in raw.lower():
            labels.append('Single-stage')
        else:
            if raw in ('w/o Fovea', 'w/o Fovea CLIP'):
                raw = 'w/o FoveaBlur'
            labels.append(raw)
    return labels


def _draw_metric_bars(
    fig,
    parent,
    rows: list[dict],
    *,
    colors: list[str],
    zero_origin: tuple[bool, ...] | None = None,
    ref_index: int = 0,
    x_labels: list[str] | None = None,
) -> None:
    """Polished metric bars with Ours baseline."""
    n_met = len(METRIC_PANELS)
    gs = parent.subgridspec(1, n_met, wspace=0.26)
    n_cond = len(rows)
    x = np.arange(n_cond, dtype=float)
    if x_labels is None:
        x_labels = _bar_xlabels(rows, ours_as='Ours')
    if zero_origin is None:
        zero_origin = tuple(False for _ in METRIC_PANELS)

    for i, (key, ylabel) in enumerate(METRIC_PANELS):
        ax = fig.add_subplot(gs[0, i])
        ys = np.array([r[f'{key}_mean'] for r in rows], dtype=float)
        es = np.array([r[f'{key}_sem'] for r in rows], dtype=float)
        ref_y = float(ys[ref_index])

        ax.yaxis.grid(True, linestyle='-', linewidth=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)

        ax.axhline(
            ref_y, color=BASELINE, linestyle=(0, (2.8, 1.8)),
            linewidth=0.9, alpha=0.80, zorder=1.5,
        )

        bars = ax.bar(
            x, ys, width=0.68, color=colors, edgecolor='none',
            zorder=2, alpha=1.0,
        )
        bars[ref_index].set_edgecolor('#0D2438')
        bars[ref_index].set_linewidth(0.7)

        ax.errorbar(
            x, ys, yerr=es, fmt='none',
            ecolor=ERR, elinewidth=0.7, capsize=1.5,
            capthick=0.65, zorder=3,
        )

        y_hi = float(np.max(ys + es))
        y_lo = float(np.min(ys - es))

        ax.set_xticks(x)
        ax.set_xticklabels(
            x_labels, rotation=32, ha='right', rotation_mode='anchor',
            fontsize=FS['tick'],
        )
        ax.set_title(
            ylabel, fontsize=FS['subtitle'], fontweight='normal',
            color=MUTED, pad=2.5,
        )
        if i == 0:
            ax.set_ylabel('Score', fontsize=FS['axis'], labelpad=2.5)
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='y', length=2.2, width=0.55, labelsize=FS['tick'], pad=1.2)
        ax.tick_params(axis='x', length=0, pad=1.5)

        if zero_origin[i]:
            ax.set_ylim(0.0, y_hi * 1.10)
        else:
            span = max(y_hi - y_lo, 1e-3)
            ax.set_ylim(y_lo - 0.14 * span, y_hi + 0.16 * span)
        ax.set_xlim(-0.55, n_cond - 0.45)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))

        _despine(ax)
        ax.spines['left'].set_linewidth(0.65)
        ax.spines['bottom'].set_linewidth(0.65)
        ax.spines['left'].set_color(SPINE)
        ax.spines['bottom'].set_color(SPINE)


def plot_final_fig4_ablations() -> str:
    _setup_rc()

    struct_rows = _load_bundle()['rows']
    loss_rows = _load_loss_rows()

    fig = plt.figure(figsize=(7.2, 4.70))
    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.0, 1.0],
        hspace=0.46,
        left=0.085, right=0.975, top=0.955, bottom=0.10,
    )

    ours_line = Line2D(
        [0], [0], color=BASELINE, linestyle=(0, (2.8, 1.8)),
        linewidth=1.0, label='Ours',
    )

    # a — architecture ablation bars
    body_a = _header(
        fig, outer[0], 'Architecture ablation', 'a',
        legend_handles=[ours_line],
    )
    struct_colors = [BAR_OURS] + [BAR_ABLATION] * (len(struct_rows) - 1)
    _draw_metric_bars(
        fig, body_a, struct_rows, colors=struct_colors,
        ref_index=0,
        x_labels=_bar_xlabels(struct_rows, ours_as='Ours'),
    )

    # b — loss ablation bars
    body_b = _header(
        fig, outer[1], 'Loss ablation', 'b',
        legend_handles=[ours_line],
    )
    loss_colors = (
        [BAR_OURS, BAR_SECOND] + [BAR_ABLATION] * (len(loss_rows) - 2)
    )
    loss_xlabels = []
    for i, r in enumerate(loss_rows):
        raw = r['short'].replace('\n', ' ')
        if i == 0:
            loss_xlabels.append('Ours')
        elif 'single-stage' in raw.lower():
            loss_xlabels.append('Single-stage')
        else:
            loss_xlabels.append(raw.replace('Full (two-stage)', 'Ours'))
    _draw_metric_bars(
        fig, body_b, loss_rows, colors=loss_colors,
        zero_origin=(True, False, False),
        ref_index=0,
        x_labels=loss_xlabels,
    )

    stem = fig_path(FINAL_FIG4)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Wrote {stem}.{{png,svg}}')
    return stem


def main() -> None:
    plot_final_fig4_ablations()


if __name__ == '__main__':
    main()
