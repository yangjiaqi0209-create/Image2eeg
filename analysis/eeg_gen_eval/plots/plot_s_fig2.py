"""Supplementary Figure S2: multi-example single-image GT vs Pred.

Each row:
  left   — stimulus image (no frame)
  middle — best-K time-domain waveforms (Final Fig1d style)
  right  — best log-PSD-r channel (Final Fig2d style)
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image
from scipy.signal import welch

from analysis.eeg_gen_eval.config import (
    CHANNELS,
    DATA_DIR,
    FIG_DIR,
    IMAGE_ROOT,
    RAW_DIR,
    SFREQ,
)
from analysis.eeg_gen_eval.figure_names import S_FIG2, fig_path
from analysis.eeg_gen_eval.helpers.plot_quality import (
    _despine,
    _plot_psd_curves,
    _setup_nature_rc,
)

SUB = 8
N_EXAMPLES = 5
N_TOP_CH = 3
IMG_SIZE = 280
FMAX_HZ = 45.0
DEFAULT_IMAGE_INDICES = [136, 78, 17, 93, 11]

# Match Final Fig1d / Fig2d
GT = '#0F4D92'
PRED = '#DC2626'
MUTED = '#6B7280'
TEXT = '#374151'
ZERO = '#D1D5DB'

FS = {
    'letter': 12.0,
    'title': 10.0,
    'legend': 6.5,
    'concept': 7.0,
    'meta': 6.0,
    'channel': 7.5,
    'axis': 8.0,
    'tick': 7.0,
    'band': 6.5,
}


def _per_channel_r(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    rs = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    for c in range(y_true.shape[0]):
        a = y_true[c] - y_true[c].mean()
        b = y_pred[c] - y_pred[c].mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den > 1e-12:
            rs[c] = float(np.dot(a, b) / den)
    return rs


def _score_all_images(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return np.array([
        float(np.nanmean(_per_channel_r(y_pred[i], y_true[i])))
        for i in range(y_true.shape[0])
    ])


def _test_image_relpaths(sub: int) -> list[str]:
    test_path = os.path.join(DATA_DIR, f'sub-{sub:02d}', 'test.pt')
    data = torch.load(test_path, map_location='cpu', weights_only=False)
    img = np.asarray(data['img'])
    if img.ndim == 1:
        return [str(p) for p in img]
    return [str(p) for p in img[:, 0]]


def _load_rgb(relpath: str, size: int = IMG_SIZE) -> np.ndarray:
    """Load RGB thumbnail without letterbox padding (avoids fake white frame)."""
    path = os.path.join(IMAGE_ROOT, relpath)
    img = Image.open(path).convert('RGB')
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img)


def _concept_name(relpath: str) -> str:
    parts = relpath.replace('\\', '/').split('/')
    for p in parts:
        if '_' in p and len(p) >= 6 and p[:5].isdigit():
            return p.split('_', 1)[1].replace('_', ' ')
    return os.path.splitext(os.path.basename(relpath))[0].replace('_', ' ')


def _channel_psd(
    eeg_ct: np.ndarray,
    channel_i: int,
    *,
    sfreq: float = SFREQ,
    fmax: float = FMAX_HZ,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD for a single channel. eeg_ct: [C, T]."""
    y = np.asarray(eeg_ct[int(channel_i)], dtype=np.float64)
    y = y - y.mean()
    nperseg = min(128, y.shape[-1])
    freqs, psd = welch(
        y, fs=sfreq, window='hann',
        nperseg=nperseg, noverlap=nperseg // 2,
        detrend='constant',
    )
    mask = freqs <= fmax
    return freqs[mask], np.maximum(psd[mask], 1e-16)


def _psd_log_corr(freqs: np.ndarray, psd_t: np.ndarray, psd_p: np.ndarray) -> float:
    a = np.log10(psd_t)
    b = np.log10(psd_p)
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return float('nan')
    return float(np.dot(a, b) / den)


def _best_psd_channel(
    yt: np.ndarray,
    yp: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, float]:
    """Channel with highest GT–Pred log-PSD Pearson r (single-channel)."""
    n_ch = yt.shape[0]
    best_i, best_r = 0, -np.inf
    best = None
    for ci in range(n_ch):
        freqs, psd_t = _channel_psd(yt, ci)
        _, psd_p = _channel_psd(yp, ci)
        r = _psd_log_corr(freqs, psd_t, psd_p)
        if np.isfinite(r) and r > best_r:
            best_r = float(r)
            best_i = ci
            best = (freqs, psd_t, psd_p)
    assert best is not None
    return best_i, best[0], best[1], best[2], float(best_r)


def _plot_wave_fig1d(
    ax,
    t_ms: np.ndarray,
    yt: np.ndarray,
    yp: np.ndarray,
    *,
    ch_name: str,
    r_ch: float,
    ylim: tuple[float, float],
    show_xlabel: bool,
    show_yticklabels: bool,
):
    """One mini waveform panel matching Final Fig1d."""
    ax.plot(t_ms, yt, color=GT, lw=1.05, zorder=3)
    ax.plot(t_ms, yp, color=PRED, lw=1.0, ls=(0, (3.2, 1.3)), zorder=3)
    ax.axhline(0, color=ZERO, lw=0.45, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.0, 1000.0)
    ax.set_xticks([0, 500, 1000])
    # Electrode + r inside axes, top-right.
    ax.text(
        0.97, 0.96, f'{ch_name} (r={r_ch:.2f})',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=FS['channel'], fontweight='bold', color=TEXT,
        zorder=5, clip_on=True,
    )
    ax.tick_params(length=2.0, width=0.55, labelsize=FS['tick'])
    if show_xlabel:
        ax.set_xlabel('Time (ms)', fontsize=FS['axis'], labelpad=1.5)
    else:
        ax.tick_params(labelbottom=False)
    if not show_yticklabels:
        ax.tick_params(labelleft=False)
        ax.set_ylabel('')
    else:
        # Numbers outside; Amplitude snug against tick labels.
        ax.tick_params(axis='y', direction='out', pad=3.0, which='major')
        ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
        ax.set_ylabel(
            'Amplitude (a.u.)', fontsize=FS['axis'], labelpad=1.0,
            color='#111827',
        )
    ax.tick_params(axis='x', direction='out', pad=0.5)
    ax.set_facecolor('none')
    ax.patch.set_visible(False)
    _despine(ax)


def _plot_psd_fig2d(
    ax,
    freqs: np.ndarray,
    psd_t: np.ndarray,
    psd_p: np.ndarray,
    *,
    ch_name: str,
    psd_r: float,
    show_xlabel: bool,
    show_band_labels: bool,
    show_legend: bool,
):
    """Single-channel PSD panel matching Final Fig2d via shared helper."""
    zeros = np.zeros_like(psd_t)
    _plot_psd_curves(
        ax, freqs, psd_t, psd_p, zeros, zeros,
        title='',
        show_legend=False,
    )
    ax.set_title('')
    # Always strip δ–γ strip from helper (too crowded in multi-row layout).
    for txt in list(ax.texts):
        t = txt.get_text().strip()
        if t in {'δ', 'θ', 'α', 'β', 'γ', 'delta', 'theta', 'alpha', 'beta', 'gamma'}:
            txt.remove()
    if show_band_labels:
        for f0, f1, lab in (
            (0.5, 4, 'δ'), (4, 8, 'θ'), (8, 13, 'α'),
            (13, 30, 'β'), (30, 45, 'γ'),
        ):
            ax.text(
                (0.5 * (f0 + f1)) / 45.0, 1.02, lab,
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=FS['band'], color=MUTED, clip_on=False, zorder=4,
            )
    # Electrode + r inside axes, top-right (same as time panels).
    ax.text(
        0.97, 0.96,
        f'{ch_name} (r={psd_r:.2f})',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=FS['channel'], fontweight='bold', color=TEXT,
        zorder=5, clip_on=True,
    )
    ax.set_xlabel(
        'Frequency (Hz)' if show_xlabel else '',
        fontsize=FS['axis'],
    )
    if not show_xlabel:
        ax.tick_params(labelbottom=False)
        ax.set_xlabel('')
    # Numbers outside; unit snug against scientific tick labels.
    ax.set_ylabel(
        'PSD (a.u.)', fontsize=FS['axis'], labelpad=0.2,
        color='#111827',
    )
    ax.tick_params(
        labelsize=FS['tick'], length=2.5, width=0.6,
        axis='y', direction='out', pad=3.5, which='major',
    )
    ax.tick_params(axis='y', which='minor', left=False, labelleft=False)
    ax.tick_params(axis='x', direction='out', pad=0.8)
    ax.set_facecolor('none')
    ax.patch.set_visible(False)
    # No in-axes legend — shared GT/Pred legend lives in the figure header.
    _ = show_legend
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def plot_s_fig2_single_image_waveforms(
    *,
    sub: int = SUB,
    n_examples: int = N_EXAMPLES,
    n_top_ch: int = N_TOP_CH,
    image_indices: list[int] | None = None,
) -> str:
    _setup_nature_rc()
    plt.rcParams.update({
        'font.size': FS['tick'],
        'axes.labelsize': FS['axis'],
        'axes.titlesize': FS['title'],
        'xtick.labelsize': FS['tick'],
        'ytick.labelsize': FS['tick'],
        'legend.fontsize': FS['legend'],
        'axes.linewidth': 0.7,
    })

    sub_tag = f'sub-{sub:02d}'
    d = os.path.join(RAW_DIR, sub_tag)
    y_pred = np.load(os.path.join(d, 'y_pred.npy'))
    y_true = np.load(os.path.join(d, 'y_true.npy'))
    img_paths = _test_image_relpaths(sub)
    if len(img_paths) != y_true.shape[0]:
        raise ValueError(
            f'{sub_tag}: n_images={len(img_paths)} != n_eeg={y_true.shape[0]}'
        )

    scores = _score_all_images(y_pred, y_true)
    if image_indices is None:
        if sub == SUB and n_examples == len(DEFAULT_IMAGE_INDICES):
            image_indices = list(DEFAULT_IMAGE_INDICES)
        else:
            order = np.argsort(-scores, kind='stable')
            image_indices = [int(i) for i in order[:n_examples]]
    else:
        image_indices = [int(i) for i in image_indices]
        n_examples = len(image_indices)

    rows = []
    for img_i in image_indices:
        yt = y_true[img_i]
        yp = y_pred[img_i]
        per_ch = _per_channel_r(yp, yt)
        top_idx = np.argsort(np.nan_to_num(per_ch, nan=-1.0))[-n_top_ch:][::-1]
        psd_ch, freqs, psd_t, psd_p, psd_r = _best_psd_channel(yt, yp)
        rows.append({
            'image_index': int(img_i),
            'yt': yt,
            'yp': yp,
            'per_ch': per_ch,
            'top_idx': top_idx,
            'psd_ch': psd_ch,
            'mean_r': float(np.nanmean(per_ch)),
            'relpath': img_paths[img_i],
            'concept': _concept_name(img_paths[img_i]),
            'freqs': freqs,
            'psd_t': psd_t,
            'psd_p': psd_p,
            'psd_r': psd_r,
        })

    t_ms = np.arange(y_true.shape[2]) * (1000.0 / SFREQ)
    n_wave = n_top_ch

    # Header: letter + title; GT/Pred legend lives in bottom-right PSD.
    fig = plt.figure(figsize=(8.8, 5.9))
    outer = fig.add_gridspec(
        n_examples, 1,
        hspace=0.13,
        left=0.030, right=0.985, top=0.955, bottom=0.055,
    )
    ax_hdr = fig.add_axes([0.030, 0.960, 0.955, 0.032])
    ax_hdr.set_axis_off()
    ax_hdr.text(
        0.0, 0.50, 'a',
        ha='left', va='center', fontsize=FS['letter'], fontweight='bold',
        color='#111827', transform=ax_hdr.transAxes,
    )
    ax_hdr.text(
        0.5, 0.50,
        f'Single-image prediction  ({sub_tag})',
        ha='center', va='center', fontsize=FS['title'], fontweight='bold',
        color='#111827', transform=ax_hdr.transAxes,
    )

    examples_meta = []
    last_psd_ax = None
    for row_i, row in enumerate(rows):
        # (stim | y-margin | waves) | (y-margin | PSD≈1.5× one wave)
        # Empty margins reserve space for outside tick numbers + ylabel.
        body = outer[row_i].subgridspec(
            1, 2,
            width_ratios=[3.70, 1.66],
            wspace=0.025,
        )
        left = body[0, 0].subgridspec(
            1, 3,
            width_ratios=[0.52, 0.28, 3.00],
            wspace=0.012,
        )
        right = body[0, 1].subgridspec(
            1, 2,
            width_ratios=[0.26, 1.05],
            wspace=0.008,
        )
        last = row_i == n_examples - 1

        # ---- stimulus: no frame; caption centered below ----
        stim = left[0, 0].subgridspec(
            2, 1, height_ratios=[1.0, 0.34], hspace=0.10,
        )
        ax_img = fig.add_subplot(stim[0, 0])
        rgb = _load_rgb(row['relpath'])
        ax_img.imshow(rgb, aspect='equal', interpolation='bilinear')
        ax_img.set_anchor('N')
        ax_img.axis('off')
        ax_img.set_facecolor('none')
        ax_img.patch.set_visible(False)

        ax_cap = fig.add_subplot(stim[1, 0])
        ax_cap.set_axis_off()
        ax_cap.set_facecolor('none')
        ax_cap.patch.set_visible(False)
        ax_cap.text(
            0.5, 0.88, row['concept'],
            ha='center', va='top', fontsize=FS['concept'],
            fontweight='bold', color='#111827', transform=ax_cap.transAxes,
        )
        ax_cap.text(
            0.5, 0.18, rf'mean $r$ = {row["mean_r"]:.2f}',
            ha='center', va='center', fontsize=FS['meta'],
            color=MUTED, transform=ax_cap.transAxes,
        )

        # ---- empty left margin for Amplitude ylabel + outside ticks ----
        ax_amp_pad = fig.add_subplot(left[0, 1])
        ax_amp_pad.set_axis_off()
        ax_amp_pad.set_facecolor('none')
        ax_amp_pad.patch.set_visible(False)

        # ---- time-domain: Final Fig1d ----
        wave_gs = left[0, 2].subgridspec(1, n_wave, wspace=0.14)
        sel = np.concatenate(
            [row['yt'][row['top_idx']], row['yp'][row['top_idx']]], axis=0,
        )
        y_abs = float(np.max(np.abs(sel)))
        # Row 2 (freezer): fixed symmetric range for readable ticks.
        if row_i == 1:
            ylim = (-1.0, 1.0)
        else:
            ylim = (-y_abs * 1.08, y_abs * 1.08)

        for k, ci in enumerate(row['top_idx']):
            ax = fig.add_subplot(wave_gs[0, k])
            _plot_wave_fig1d(
                ax, t_ms, row['yt'][ci], row['yp'][ci],
                ch_name=CHANNELS[ci],
                r_ch=float(row['per_ch'][ci]),
                ylim=ylim,
                show_xlabel=last,
                show_yticklabels=(k == 0),
            )
            if row_i == 1:
                ax.set_yticks([-1.0, 0.0, 1.0])
            ax.set_zorder(1)
            ax.set_facecolor('none')
            ax.patch.set_visible(False)

        # ---- empty left margin for PSD ylabel + outside scientific ticks ----
        ax_psd_pad = fig.add_subplot(right[0, 0])
        ax_psd_pad.set_axis_off()
        ax_psd_pad.set_facecolor('none')
        ax_psd_pad.patch.set_visible(False)

        # ---- frequency-domain (~1.5× one time panel) ----
        ax_f = fig.add_subplot(right[0, 1])
        _plot_psd_fig2d(
            ax_f,
            row['freqs'], row['psd_t'], row['psd_p'],
            ch_name=CHANNELS[row['psd_ch']],
            psd_r=row['psd_r'],
            show_xlabel=last,
            show_band_labels=False,
            show_legend=False,
        )
        ax_f.set_zorder(2)
        ax_f.set_facecolor('none')
        ax_f.patch.set_visible(False)
        if last:
            last_psd_ax = ax_f

        examples_meta.append({
            'image_index': row['image_index'],
            'relpath': row['relpath'],
            'concept': row['concept'],
            'mean_channel_r': row['mean_r'],
            'top_channels': [CHANNELS[int(i)] for i in row['top_idx']],
            'top_channel_r': [float(row['per_ch'][i]) for i in row['top_idx']],
            'psd_log_r': row['psd_r'],
            'psd_channel': CHANNELS[row['psd_ch']],
        })

    # GT/Pred legend: lower-left of the bottom-right PSD panel.
    if last_psd_ax is not None:
        last_psd_ax.legend(
            handles=[
                Line2D([0], [0], color=GT, lw=1.25, label='GT'),
                Line2D([0], [0], color=PRED, lw=1.15, ls=(0, (3.5, 1.4)), label='Pred'),
            ],
            loc='lower left', frameon=False, fontsize=FS['legend'],
            handlelength=1.25, handletextpad=0.3, columnspacing=0.7,
            borderaxespad=0.25, labelspacing=0.2,
        )

    meta = {
        'subject': sub_tag,
        'n_examples': n_examples,
        'n_top_channels': n_top_ch,
        'selection': 'fixed shuffled examples (see DEFAULT_IMAGE_INDICES)',
        'time_style': 'Final Fig1d (_plot_panel_a_subject_average)',
        'psd_style': 'Final Fig2d (_plot_psd_curves), best log-PSD-r channel',
        'examples': examples_meta,
        'raw_dir': d,
    }
    stem = fig_path(S_FIG2)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    with open(f'{stem}_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(
        f'Wrote {stem}.{{png,svg}}  '
        f'({sub_tag}, {n_examples} examples × {n_top_ch} time ch + PSD)'
    )
    return stem


def parse_args():
    p = argparse.ArgumentParser(
        description='S Fig2: stimulus + time waveforms + frequency PSD',
    )
    p.add_argument('--sub', type=int, default=SUB)
    p.add_argument('--n_examples', type=int, default=N_EXAMPLES)
    p.add_argument('--n_top_ch', type=int, default=N_TOP_CH)
    p.add_argument('--image_indices', type=int, nargs='*', default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    plot_s_fig2_single_image_waveforms(
        sub=args.sub,
        n_examples=args.n_examples,
        n_top_ch=args.n_top_ch,
        image_indices=args.image_indices,
    )


# Backward-compatible alias
plot_fig13_single_image = plot_s_fig2_single_image_waveforms


if __name__ == '__main__':
    main()
