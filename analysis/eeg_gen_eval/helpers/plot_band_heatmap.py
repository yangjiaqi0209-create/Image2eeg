"""Heatmap: band × channel similarity (Gen vs GT)."""

from __future__ import annotations

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from analysis.eeg_gen_eval.config import CHANNELS, EEG_BANDS, FIG_DIR, RAW_DIR
from analysis.eeg_gen_eval.figure_names import FIG6, fig_path
from analysis.eeg_gen_eval.compute.metrics import band_channel_waveform_correlation
from analysis.eeg_gen_eval.helpers.plot_quality import _setup_nature_rc, add_panel_label, save_pub
from analysis.eeg_gen_eval.helpers.plot_visualizations import BRAIN_REGION_ORDER, _channel_region

_BAND_SHORT = {
    'delta': 'δ',
    'theta': 'θ',
    'alpha': 'α',
    'beta': 'β',
    'gamma': 'γ',
}


def region_channel_layout():
    """Group channels into five mutually exclusive scalp regions (same as fig1b).

    Returns
    -------
    order : np.ndarray of int
        Permutation of original channel indices.
    region_bands : list[(start, stop, name)]
        Half-open index ranges in the reordered channel axis.
    ordered_channels : list[str]
        Channel names after regrouping.
    """
    grouped_indices = {
        region: [i for i, ch in enumerate(CHANNELS) if _channel_region(ch) == region]
        for region in BRAIN_REGION_ORDER
    }
    # Skip empty regions (e.g. Alljoined 32-ch has no Temporal electrodes).
    active_regions = [r for r in BRAIN_REGION_ORDER if grouped_indices[r]]
    order = np.asarray([
        i for region in active_regions for i in grouped_indices[region]
    ], dtype=int)
    n = len(CHANNELS)
    if len(order) != n or len(np.unique(order)) != n:
        raise ValueError('Each EEG channel must map to exactly one scalp region')
    region_bands = []
    start = 0
    for region in active_regions:
        stop = start + len(grouped_indices[region])
        region_bands.append((start, stop, region))
        start = stop
    ordered_channels = [CHANNELS[i] for i in order]
    return order, region_bands, ordered_channels


# Module-level bands for callers that only need label ranges (same layout as heatmap).
_, _REGION_BANDS, _ = region_channel_layout()


def compute_band_channel_similarity(subs: List[int] | None = None) -> Dict:
    """Per-subject then grand-mean (band, channel) Pearson r matrices."""
    subs = subs or list(range(1, 11))
    band_names = [b[0] for b in EEG_BANDS]
    mats = []
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        d = os.path.join(RAW_DIR, sub_tag)
        y_pred = np.load(os.path.join(d, 'y_pred.npy'))
        y_true = np.load(os.path.join(d, 'y_true.npy'))
        mat, names = band_channel_waveform_correlation(y_pred, y_true)
        assert names == band_names
        mats.append(mat)
        np.save(os.path.join(d, 'band_channel_correlation.npy'), mat)

    stack = np.stack(mats, axis=0)
    mean_mat = stack.mean(axis=0)
    std_mat = stack.std(axis=0, ddof=0)

    out_dir = RAW_DIR
    np.save(os.path.join(out_dir, 'band_channel_correlation_mean.npy'), mean_mat)
    np.save(os.path.join(out_dir, 'band_channel_correlation_std.npy'), std_mat)

    meta = {
        'n_subjects': len(subs),
        'band_names': band_names,
        'n_channels': mean_mat.shape[1],
        'mean_over_subjects': True,
        'value': (
            'Pearson r of band-limited waveforms (Gen vs GT, '
            'pooled over test images per subject)'
        ),
    }
    with open(os.path.join(out_dir, 'band_channel_correlation_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return {
        'mean': mean_mat,
        'std': std_mat,
        'band_names': band_names,
        'meta': meta,
    }


def _load_band_channel_data(data: Dict | None = None, subs: List[int] | None = None) -> Dict:
    if data is not None:
        return data
    mean_path = os.path.join(RAW_DIR, 'band_channel_correlation_mean.npy')
    if not os.path.isfile(mean_path):
        return compute_band_channel_similarity(subs)
    with open(os.path.join(RAW_DIR, 'band_channel_correlation_meta.json')) as f:
        meta = json.load(f)
    return {
        'mean': np.load(mean_path),
        'band_names': meta['band_names'],
        'meta': meta,
    }


def plot_band_channel_heatmap_on_axes(
    ax,
    cax,
    data: Dict | None = None,
    *,
    show_title: bool = True,
    show_meta: bool = True,
    show_region_labels: bool = True,
    title: str = 'Band–channel waveform similarity',
):
    """Draw band×channel Pearson-r heatmap into existing axes (+ colorbar axes)."""
    data = _load_band_channel_data(data)
    mat = np.asarray(data['mean'], dtype=np.float64)
    band_names = list(data['band_names'])
    meta = data.get('meta', {})
    order, region_bands, ordered_channels = region_channel_layout()
    mat = mat[:, order]
    n_bands, n_ch = mat.shape
    n_sub = int(meta.get('n_subjects', 10))
    mean_r = float(np.nanmean(mat))

    vmax = float(np.nanpercentile(mat, 98))
    vmax = max(0.55, min(vmax, 0.95))
    vmin = 0.0

    im = ax.imshow(
        mat, aspect='auto', origin='lower', cmap='RdBu_r',
        vmin=vmin, vmax=vmax,
        extent=[-0.5, n_ch - 0.5, -0.5, n_bands - 0.5],
        interpolation='nearest',
        rasterized=True,
    )

    for x0, x1, name in region_bands:
        if x0 > 0:
            ax.axvline(x0 - 0.5, color='#F8FAFC', lw=0.8, alpha=0.9, zorder=3)
        if show_region_labels:
            ax.text(
                0.5 * (x0 + x1) - 0.5, 1.03, name,
                transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=5.5, color='#6B7280',
                clip_on=False,
            )

    # One landmark tick near the middle of each anatomical region (fig1b style).
    tick_idx = [int((x0 + x1 - 1) // 2) for x0, x1, _ in region_bands]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(
        [ordered_channels[i] for i in tick_idx],
        fontsize=6.5,
    )
    ax.set_xlabel('Channel (grouped by region)', fontsize=7)

    ax.set_yticks(range(n_bands))
    ax.set_yticklabels(
        [_BAND_SHORT.get(b, b) for b in band_names],
        fontsize=8,
    )
    ax.set_ylabel('Frequency band', fontsize=7)
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.set_ylim(-0.5, n_bands - 0.5)
    ax.tick_params(length=2.2, width=0.55, labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color('#6B7280')

    cb = ax.figure.colorbar(im, cax=cax)
    cb.set_label('Pearson r', fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.55)
    cb.outline.set_linewidth(0.55)

    if show_title:
        ax.set_title(title, fontweight='bold', fontsize=9, pad=14, loc='center')
    if show_meta:
        ax.text(
            0.5, 1.14,
            f'n = {n_sub}  ·  mean r = {mean_r:.3f}',
            transform=ax.transAxes,
            ha='center', va='bottom', fontsize=5.5, color='#6B7280',
            clip_on=False,
        )
    return im, mean_r, n_sub


def plot_fig5_band_channel_similarity(data: Dict | None = None, subs: List[int] | None = None):
    """Nature-style heatmap: frequency band × channel Pearson r (Gen vs GT)."""
    _setup_nature_rc()
    data = _load_band_channel_data(data, subs)

    fig = plt.figure(figsize=(7.2, 2.65))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.035], wspace=0.06,
        left=0.10, right=0.92, top=0.72, bottom=0.26,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    plot_band_channel_heatmap_on_axes(ax, cax, data)

    add_panel_label(ax, 'a', x=-0.08, y=1.22)

    stem = fig_path(FIG6)
    os.makedirs(os.path.dirname(stem) or FIG_DIR, exist_ok=True)
    fig.savefig(f'{stem}.svg', facecolor='white')
    fig.savefig(f'{stem}.png', dpi=600, facecolor='white')
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')


def plot_all_band_heatmaps(subs: List[int] | None = None):
    data = compute_band_channel_similarity(subs)
    plot_fig5_band_channel_similarity(data)
