"""t-SNE / UMAP of test-set EEG brain-encoder embeddings (gt vs G2_fovea gen)."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from analysis.eeg_gen_eval.compute.array_utils import img_per_sample, trial_format
from analysis.eeg_gen_eval.concept_categories import (
    CATEGORY_ORDER,
    category_for_concept,
    concept_from_relpath,
)
from analysis.eeg_gen_eval.config import (
    BASELINE_LABEL,
    BRAIN_ENCODER_EXP,
    DATA_DIR,
    FIG_DIR,
    eval_brain_ckpt_path,
)
from analysis.eeg_gen_eval.helpers.plot_quality import (
    PALETTE,
    _despine,
    _setup_nature_rc,
    label_axes,
    save_pub,
)
from analysis.eeg_gen_eval.figure_names import FIG10, FIG11, FIG12, fig_path
from analysis.eeg_gen_eval.helpers.plot_similarity_matrix import CATEGORY_COLORS

FEATURE_TYPE = 'brain_encoder'
DEFAULT_SINGLE_SUB = 1
RANDOM_STATE = 42
PCA_DIM = 50
TSNE_PERPLEXITY = 30
UMAP_NEIGHBORS = 30


def _embed_dir() -> str:
    """Always follow the active config.RAW_DIR (THINGS vs alljoined)."""
    from analysis.eeg_gen_eval import config as cfg
    return os.path.join(cfg.RAW_DIR, 'eeg_embedding')


def __getattr__(name: str):
    if name == 'EMBED_DIR':
        return _embed_dir()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def _brain_config(n_ch: int, seq_len: int, z_dim: int = 1024) -> dict:
    return {
        'target': 'encoder.models.EEGProjectLayer',
        'params': {'c_num': n_ch, 'z_dim': z_dim, 'timesteps': [0, seq_len]},
    }


def _test_concepts(sub: int = 1) -> List[str]:
    from analysis.eeg_gen_eval import config as cfg
    test_path = os.path.join(cfg.DATA_DIR, f'sub-{sub:02d}', 'test.pt')
    data = torch.load(test_path, map_location='cpu', weights_only=False)
    paths = img_per_sample(np.asarray(data['img']), trial_format(data))
    return [concept_from_relpath(str(p)) for p in paths]


def load_pooled_eeg(
    subs: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-subject (200, C, T) arrays; labels repeat per subject."""
    from analysis.eeg_gen_eval import config as cfg
    subs = subs or list(range(1, 11))
    concepts = _test_concepts(subs[0])
    categories = np.array([category_for_concept(c) for c in concepts])

    true_blocks, gen_blocks, label_blocks = [], [], []
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        true_path = os.path.join(cfg.RAW_DIR, sub_tag, 'y_true.npy')
        pred_path = os.path.join(cfg.RAW_DIR, sub_tag, 'y_pred.npy')
        if not os.path.isfile(true_path) or not os.path.isfile(pred_path):
            raise FileNotFoundError(
                f'Missing {true_path} or {pred_path}. '
                'Run evaluate with default baseline (exp_a) first.'
            )
        y_true = np.load(true_path)
        y_pred = np.load(pred_path)
        if y_true.shape != y_pred.shape:
            raise ValueError(f'{sub_tag}: y_true {y_true.shape} != y_pred {y_pred.shape}')
        true_blocks.append(y_true)
        gen_blocks.append(y_pred)
        label_blocks.append(categories)

    return (
        np.concatenate(true_blocks, axis=0),
        np.concatenate(gen_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
    )


@torch.no_grad()
def encode_pooled_eeg(
    subs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-subject intra-subject EEG encoder -> normalized 1024-d embeddings."""
    from analysis.eeg_gen_eval import config as cfg
    from predictor.data import eeg_to_ubp_embedding, load_frozen_ubp_brain

    subs = subs or list(range(1, 11))
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    concepts = _test_concepts(subs[0])
    categories = np.array([category_for_concept(c) for c in concepts])
    true_blocks, gen_blocks, label_blocks = [], [], []

    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        true_path = os.path.join(cfg.RAW_DIR, sub_tag, 'y_true.npy')
        pred_path = os.path.join(cfg.RAW_DIR, sub_tag, 'y_pred.npy')
        y_true = np.load(true_path)
        y_pred = np.load(pred_path)

        brain_ckpt = eval_brain_ckpt_path(sub, seed=0)
        brain = load_frozen_ubp_brain(
            _brain_config(y_true.shape[1], y_true.shape[2]),
            brain_ckpt,
            device,
        )
        if brain is None:
            raise FileNotFoundError(f'Brain encoder checkpoint not found: {brain_ckpt}')

        y_true_t = torch.from_numpy(y_true).float().to(device)
        y_pred_t = torch.from_numpy(y_pred).float().to(device)
        emb_true = eeg_to_ubp_embedding(y_true_t, brain).cpu().numpy()
        emb_gen = eeg_to_ubp_embedding(y_pred_t, brain).cpu().numpy()
        true_blocks.append(emb_true)
        gen_blocks.append(emb_gen)
        label_blocks.append(categories)
        print(f'  {sub_tag} encoder embeddings done')

    return (
        np.concatenate(true_blocks, axis=0),
        np.concatenate(gen_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
    )


def _pca_reduce(X: np.ndarray, n_components: int = PCA_DIM) -> np.ndarray:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(Xs)


def compute_tsne(X: np.ndarray, perplexity: float = TSNE_PERPLEXITY) -> np.ndarray:
    Xp = _pca_reduce(X)
    perp = min(perplexity, max(5.0, (Xp.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=perp,
        init='pca',
        learning_rate='auto',
        random_state=RANDOM_STATE,
        max_iter=1000,
    ).fit_transform(Xp)


def compute_umap(X: np.ndarray, n_neighbors: int = UMAP_NEIGHBORS) -> np.ndarray:
    import umap

    Xp = _pca_reduce(X)
    n_neighbors = min(n_neighbors, Xp.shape[0] - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='euclidean',
        random_state=RANDOM_STATE,
    )
    return reducer.fit_transform(Xp)


def _encoder_paths() -> Dict[str, str]:
    return {
        'emb_true': os.path.join(_embed_dir(), 'emb_true.npy'),
        'emb_gen': os.path.join(_embed_dir(), 'emb_gen.npy'),
        'categories': os.path.join(_embed_dir(), 'categories.npy'),
    }


def load_encoder_embeddings(
    subs: Optional[List[int]] = None,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Cached 1024-d brain-encoder embeddings for all requested subjects."""
    os.makedirs(_embed_dir(), exist_ok=True)
    paths = _encoder_paths()
    meta_path = os.path.join(_embed_dir(), 'meta.json')

    if (
        not force
        and all(os.path.isfile(paths[k]) for k in paths)
        and os.path.isfile(meta_path)
    ):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('feature_type') == FEATURE_TYPE:
            return (
                np.load(paths['emb_true']),
                np.load(paths['emb_gen']),
                np.load(paths['categories']),
                meta,
            )

    X_true, X_gen, categories = encode_pooled_eeg(subs, device=device)
    np.save(paths['emb_true'], X_true)
    np.save(paths['emb_gen'], X_gen)
    np.save(paths['categories'], categories)

    meta = {
        'n_samples': int(X_true.shape[0]),
        'n_subjects': len(subs or list(range(1, 11))),
        'feature_type': FEATURE_TYPE,
        'feature_dim': int(X_true.shape[1]),
        'brain_encoder_exp': BRAIN_ENCODER_EXP,
        'baseline': BASELINE_LABEL,
        'pca_dim': PCA_DIM,
        'tsne_perplexity': TSNE_PERPLEXITY,
        'umap_neighbors': UMAP_NEIGHBORS,
        'random_state': RANDOM_STATE,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return X_true, X_gen, categories, meta


def _subject_slice(
    emb_true: np.ndarray,
    emb_gen: np.ndarray,
    categories: np.ndarray,
    sub: int,
    n_per_sub: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = (sub - 1) * n_per_sub
    sl = slice(idx, idx + n_per_sub)
    return emb_true[sl], emb_gen[sl], categories[sl]


def compute_all_embeddings(
    subs: Optional[List[int]] = None,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    os.makedirs(_embed_dir(), exist_ok=True)
    meta_path = os.path.join(_embed_dir(), 'meta.json')
    keys = ('tsne_true', 'tsne_gen', 'umap_true', 'umap_gen', 'categories')
    paths = {k: os.path.join(_embed_dir(), f'{k}.npy') for k in keys}

    if not force and all(os.path.isfile(paths[k]) for k in keys) and os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get('feature_type') == FEATURE_TYPE:
            out = {k: np.load(paths[k]) for k in keys}
            out['meta'] = meta
            return out

    X_true, X_gen, categories, meta = load_encoder_embeddings(
        subs, force=force, device=device,
    )

    tsne_true = compute_tsne(X_true)
    tsne_gen = compute_tsne(X_gen)
    umap_true = compute_umap(X_true)
    umap_gen = compute_umap(X_gen)

    for k, arr in (
        ('tsne_true', tsne_true),
        ('tsne_gen', tsne_gen),
        ('umap_true', umap_true),
        ('umap_gen', umap_gen),
        ('categories', categories),
    ):
        np.save(paths[k], arr)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return {
        'tsne_true': tsne_true,
        'tsne_gen': tsne_gen,
        'umap_true': umap_true,
        'umap_gen': umap_gen,
        'categories': categories,
        'meta': meta,
    }


def compute_joint_embeddings(
    emb_true: np.ndarray,
    emb_gen: np.ndarray,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit t-SNE / UMAP once on stacked GT+Gen encoder embeddings."""
    os.makedirs(_embed_dir(), exist_ok=True)
    paths = {
        'tsne_true': os.path.join(_embed_dir(), 'joint_tsne_true.npy'),
        'tsne_gen': os.path.join(_embed_dir(), 'joint_tsne_gen.npy'),
        'umap_true': os.path.join(_embed_dir(), 'joint_umap_true.npy'),
        'umap_gen': os.path.join(_embed_dir(), 'joint_umap_gen.npy'),
    }
    if not force and all(os.path.isfile(p) for p in paths.values()):
        return (
            np.load(paths['tsne_true']),
            np.load(paths['tsne_gen']),
            np.load(paths['umap_true']),
            np.load(paths['umap_gen']),
        )

    stacked = np.vstack([emb_true, emb_gen])
    tsne_all = compute_tsne(stacked)
    umap_all = compute_umap(stacked)
    n = emb_true.shape[0]
    out = (
        tsne_all[:n],
        tsne_all[n:],
        umap_all[:n],
        umap_all[n:],
    )
    for key, arr in zip(paths.keys(), out):
        np.save(paths[key], arr)
    return out


def _scatter_by_category(ax, xy: np.ndarray, categories: np.ndarray, *, title: str):
    for cat in CATEGORY_ORDER:
        mask = categories == cat
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=8,
            c=CATEGORY_COLORS[cat],
            alpha=0.75,
            linewidths=0,
            rasterized=True,
        )
    ax.set_title(title, fontsize=9, pad=6)
    ax.set_xlabel('Dim 1', fontsize=8)
    ax.set_ylabel('Dim 2', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal', adjustable='datalim')


def _density_overlap_contours(
    ax,
    xy_true: np.ndarray,
    xy_gen: np.ndarray,
    *,
    true_color: str = '#0F4D92',
    pred_color: str = '#DC2626',
) -> float:
    """Draw one smooth 80% KDE boundary and return density overlap."""
    combined = np.vstack([xy_true, xy_gen])
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    pad = np.maximum((hi - lo) * 0.06, 1e-6)
    gx = np.linspace(lo[0] - pad[0], hi[0] + pad[0], 140)
    gy = np.linspace(lo[1] - pad[1], hi[1] + pad[1], 140)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.vstack([xx.ravel(), yy.ravel()])

    # Slightly broader than Scott's rule to avoid fragmented, noisy contours.
    dt = gaussian_kde(xy_true.T, bw_method=0.45)(grid).reshape(xx.shape)
    dg = gaussian_kde(xy_gen.T, bw_method=0.45)(grid).reshape(xx.shape)
    pt = dt / dt.sum()
    pg = dg / dg.sum()
    overlap = float(np.minimum(pt, pg).sum())

    def mass_threshold(density: np.ndarray, mass: float = 0.80) -> float:
        vals = np.sort(density.ravel())[::-1]
        cumulative = np.cumsum(vals) / vals.sum()
        return float(vals[np.searchsorted(cumulative, mass)])

    ax.contour(
        xx, yy, dt,
        levels=[mass_threshold(dt)],
        colors=[true_color],
        linewidths=1.15,
        alpha=0.95,
        zorder=4,
    )
    ax.contour(
        xx, yy, dg,
        levels=[mass_threshold(dg)],
        colors=[pred_color],
        linewidths=1.15,
        linestyles='--',
        alpha=0.95,
        zorder=4,
    )
    return overlap


def _scatter_joint(
    ax,
    xy_true: np.ndarray,
    xy_gen: np.ndarray,
    *,
    title: str,
    show_legend: bool = True,
    true_color: str = '#0F4D92',
    pred_color: str = '#DC2626',
) -> float:
    """Nature-style joint embedding scatter + 80% KDE contours."""
    ax.scatter(
        xy_true[:, 0], xy_true[:, 1],
        s=10, c=true_color, alpha=0.28, linewidths=0,
        label='Ground truth', rasterized=True, zorder=2,
    )
    ax.scatter(
        xy_gen[:, 0], xy_gen[:, 1],
        s=10, c=pred_color, alpha=0.28, linewidths=0,
        label='Generated', rasterized=True, zorder=2,
    )
    overlap = _density_overlap_contours(
        ax, xy_true, xy_gen, true_color=true_color, pred_color=pred_color,
    )

    ax.set_title(title, fontweight='bold', fontsize=8, pad=4)
    ax.set_xlabel('Dim 1', fontsize=7)
    ax.set_ylabel('Dim 2', fontsize=7)
    ax.tick_params(length=2.2, width=0.55, labelsize=6.5)
    ax.set_aspect('equal', adjustable='datalim')
    _despine(ax)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)

    ax.text(
        0.03, 0.04,
        f'KDE overlap = {overlap:.2f}',
        transform=ax.transAxes,
        ha='left', va='bottom',
        fontsize=6, color='#374151',
        zorder=6,
    )

    if show_legend:
        legend_handles = [
            Line2D(
                [0], [0], marker='o', linestyle='-',
                color=true_color, markerfacecolor=true_color,
                markersize=3.5, linewidth=1.0, label='Ground truth',
            ),
            Line2D(
                [0], [0], marker='o', linestyle=(0, (3.5, 1.6)),
                color=pred_color, markerfacecolor=pred_color,
                markersize=3.5, linewidth=1.0, label='Generated',
            ),
        ]
        ax.legend(
            handles=legend_handles,
            loc='upper right',
            frameon=False,
            fontsize=5.5,
            handlelength=1.5,
            handletextpad=0.35,
            borderaxespad=0.2,
            labelspacing=0.25,
        )
    return overlap


def plot_eeg_embedding(
    subs: Optional[List[int]] = None,
    force: bool = False,
) -> str:
    data = compute_all_embeddings(subs, force=force)
    categories = data['categories']

    _setup_nature_rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    panels = [
        (axes[0, 0], data['tsne_true'], 'Ground-truth EEG (encoder)\nt-SNE'),
        (axes[0, 1], data['umap_true'], 'Ground-truth EEG (encoder)\nUMAP'),
        (axes[1, 0], data['tsne_gen'], f'Generated EEG (encoder, {BASELINE_LABEL})\nt-SNE'),
        (axes[1, 1], data['umap_gen'], f'Generated EEG (encoder, {BASELINE_LABEL})\nUMAP'),
    ]
    for ax, xy, title in panels:
        _scatter_by_category(ax, xy, categories, title=title)

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=CATEGORY_COLORS[c],
            markersize=5,
            label=c.capitalize(),
        )
        for c in CATEGORY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(CATEGORY_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )

    n = data['meta']['n_samples']
    n_sub = data['meta']['n_subjects']
    fig.suptitle(
        f'EEG encoder embedding manifold ({n} samples = {n_sub} subjects × 200 images)',
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    stem = os.path.join(FIG_DIR, 'fig15_eeg_tsne_umap')
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


def plot_eeg_embedding_joint(
    subs: Optional[List[int]] = None,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    emb_true, emb_gen, _, meta = load_encoder_embeddings(
        subs, force=force, device=device,
    )
    tsne_true, tsne_gen, umap_true, umap_gen = compute_joint_embeddings(
        emb_true, emb_gen, force=force,
    )

    _setup_nature_rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    overlap_tsne = _scatter_joint(
        axes[0], tsne_true, tsne_gen, title='Joint t-SNE', show_legend=True,
    )
    overlap_umap = _scatter_joint(
        axes[1], umap_true, umap_gen, title='Joint UMAP', show_legend=False,
    )

    label_axes(axes, start='a')
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.90, wspace=0.28)

    stem = fig_path(FIG10)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    print(f'Density overlap: t-SNE={overlap_tsne:.3f}, UMAP={overlap_umap:.3f}')
    return stem


def plot_eeg_embedding_single(
    sub: int = DEFAULT_SINGLE_SUB,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    emb_true, emb_gen, categories, meta = load_encoder_embeddings(
        force=force, device=device,
    )
    true_sub, gen_sub, cats_sub = _subject_slice(emb_true, emb_gen, categories, sub)

    tsne_true = compute_tsne(true_sub)
    tsne_gen = compute_tsne(gen_sub)
    umap_true = compute_umap(true_sub)
    umap_gen = compute_umap(gen_sub)

    _setup_nature_rc()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    panels = [
        (axes[0, 0], tsne_true, 'Ground-truth EEG (encoder)\nt-SNE'),
        (axes[0, 1], umap_true, 'Ground-truth EEG (encoder)\nUMAP'),
        (axes[1, 0], tsne_gen, f'Generated EEG (encoder, {BASELINE_LABEL})\nt-SNE'),
        (axes[1, 1], umap_gen, f'Generated EEG (encoder, {BASELINE_LABEL})\nUMAP'),
    ]
    for ax, xy, title in panels:
        _scatter_by_category(ax, xy, cats_sub, title=title)

    legend_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=CATEGORY_COLORS[c],
            markersize=5,
            label=c.capitalize(),
        )
        for c in CATEGORY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=len(CATEGORY_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.suptitle(
        f'EEG encoder embedding — sub-{sub:02d} (200 test images, 80-trial mean)',
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()

    stem = os.path.join(FIG_DIR, f'fig17_eeg_tsne_umap_sub{sub:02d}')
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


def compute_matched_pair_stats(
    subs: Optional[List[int]] = None,
    force: bool = False,
    device: Optional[torch.device] = None,
    n_random: int = 5000,
    seed: int = RANDOM_STATE,
) -> Dict:
    """Matched GT–Gen pair distances in encoder & joint-DR space."""
    subs = subs or list(range(1, 11))
    emb_true, emb_gen, _, meta = load_encoder_embeddings(
        subs, force=force, device=device,
    )
    tsne_true, tsne_gen, umap_true, umap_gen = compute_joint_embeddings(
        emb_true, emb_gen, force=force,
    )

    matched_cos = (emb_true * emb_gen).sum(axis=1)
    matched_enc_dist = 1.0 - matched_cos
    matched_tsne = np.linalg.norm(tsne_true - tsne_gen, axis=1)
    matched_umap = np.linalg.norm(umap_true - umap_gen, axis=1)

    rng = np.random.default_rng(seed)
    n_per_sub = 200
    random_cos = []
    for sub_idx in range(len(subs)):
        sl = slice(sub_idx * n_per_sub, (sub_idx + 1) * n_per_sub)
        gt_blk = emb_true[sl]
        gen_blk = emb_gen[sl]
        n_blk = gt_blk.shape[0]
        n_draw = max(n_random // len(subs), n_blk)
        ii = rng.integers(0, n_blk, size=n_draw)
        jj = rng.integers(0, n_blk, size=n_draw)
        cross = (gt_blk[ii] * gen_blk[jj]).sum(axis=1)
        off = ii != jj
        random_cos.extend(cross[off].tolist())
    random_cos = np.array(random_cos, dtype=np.float64)
    random_enc_dist = 1.0 - random_cos

    per_sub = []
    for sub_idx, sub in enumerate(subs):
        sl = slice(sub_idx * n_per_sub, (sub_idx + 1) * n_per_sub)
        cos_sub = matched_cos[sl]
        per_sub.append({
            'subject': f'sub-{sub:02d}',
            'n': int(n_per_sub),
            'matched_cos_mean': float(cos_sub.mean()),
            'matched_cos_std': float(cos_sub.std()),
            'matched_enc_dist_mean': float((1.0 - cos_sub).mean()),
            'matched_umap_dist_mean': float(matched_umap[sl].mean()),
            'matched_tsne_dist_mean': float(matched_tsne[sl].mean()),
        })

    stats = {
        'baseline': BASELINE_LABEL,
        'brain_encoder_exp': BRAIN_ENCODER_EXP,
        'n_matched': int(matched_cos.shape[0]),
        'n_subjects': len(subs),
        'matched_cos_mean': float(matched_cos.mean()),
        'matched_cos_median': float(np.median(matched_cos)),
        'matched_cos_std': float(matched_cos.std()),
        'random_cos_mean': float(random_cos.mean()),
        'random_cos_median': float(np.median(random_cos)),
        'matched_enc_dist_mean': float(matched_enc_dist.mean()),
        'random_enc_dist_mean': float(random_enc_dist.mean()),
        'matched_umap_dist_mean': float(matched_umap.mean()),
        'matched_tsne_dist_mean': float(matched_tsne.mean()),
        'cos_separation': float(matched_cos.mean() - random_cos.mean()),
        'per_subject': per_sub,
    }

    os.makedirs(_embed_dir(), exist_ok=True)
    out_path = os.path.join(_embed_dir(), 'matched_pair_stats.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    np.savez(
        os.path.join(_embed_dir(), 'matched_pair_arrays.npz'),
        matched_cos=matched_cos,
        random_cos=random_cos,
        matched_umap=matched_umap,
        matched_tsne=matched_tsne,
    )
    return stats


def plot_matched_pair_distance(
    subs: Optional[List[int]] = None,
    force: bool = False,
    single_sub: int = DEFAULT_SINGLE_SUB,
    device: Optional[torch.device] = None,
) -> str:
    """Fig: matched-pair distance GT↔Gen vs random cross-pairs."""
    subs = subs or list(range(1, 11))
    emb_true, emb_gen, _, meta = load_encoder_embeddings(
        subs, force=force, device=device,
    )
    tsne_true, tsne_gen, umap_true, umap_gen = compute_joint_embeddings(
        emb_true, emb_gen, force=force,
    )

    arrays_path = os.path.join(_embed_dir(), 'matched_pair_arrays.npz')
    stats_path = os.path.join(_embed_dir(), 'matched_pair_stats.json')
    if force or not os.path.isfile(arrays_path):
        stats = compute_matched_pair_stats(
            subs, force=force, device=device,
        )
    else:
        with open(stats_path) as f:
            stats = json.load(f)
    data = np.load(arrays_path)
    matched_cos = data['matched_cos']
    random_cos = data['random_cos']
    matched_umap = data['matched_umap']

    true_sub, gen_sub, _ = _subject_slice(
        emb_true, emb_gen, np.zeros(200, dtype=object), single_sub,
    )
    stacked_sub = np.vstack([true_sub, gen_sub])
    umap_sub_all = compute_umap(stacked_sub)
    umap_t_sub = umap_sub_all[:200]
    umap_g_sub = umap_sub_all[200:]

    _setup_nature_rc()
    fig = plt.figure(figsize=(7.4, 5.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.38, wspace=0.32)

    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.linspace(-0.05, 0.85, 35)
    ax_a.hist(
        matched_cos, bins=bins, density=True, alpha=0.65,
        color=PALETTE['true'], label='Matched (same image)', edgecolor='white', linewidth=0.3,
    )
    ax_a.hist(
        random_cos, bins=bins, density=True, alpha=0.45,
        color=PALETTE['neutral'], label='Random cross-pair', edgecolor='white', linewidth=0.3,
    )
    ax_a.axvline(stats['matched_cos_mean'], color=PALETTE['true'], ls='--', lw=1.0, alpha=0.9)
    ax_a.axvline(stats['random_cos_mean'], color=PALETTE['neutral'], ls='--', lw=1.0, alpha=0.9)
    ax_a.set_xlabel('Encoder cosine similarity', fontsize=8)
    ax_a.set_ylabel('Density', fontsize=8)
    ax_a.set_title('Encoder space', fontsize=9, pad=5)
    ax_a.legend(fontsize=6.5, loc='upper left', frameon=False)
    ax_a.tick_params(labelsize=7)
    sep = stats['cos_separation']
    ax_a.text(
        0.97, 0.95, f'Δmean = {sep:.3f}',
        transform=ax_a.transAxes, ha='right', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.9),
    )

    ax_b = fig.add_subplot(gs[0, 1])
    per_sub = stats['per_subject']
    xs = np.arange(len(per_sub))
    means = [p['matched_cos_mean'] for p in per_sub]
    stds = [p['matched_cos_std'] for p in per_sub]
    ax_b.bar(xs, means, yerr=stds, capsize=2, color=PALETTE['accent'], alpha=0.85,
             edgecolor='white', linewidth=0.4, error_kw={'linewidth': 0.8, 'capthick': 0.8})
    ax_b.axhline(stats['matched_cos_mean'], color=PALETTE['true'], ls='--', lw=1.0, alpha=0.8)
    ax_b.axhline(stats['random_cos_mean'], color=PALETTE['neutral'], ls=':', lw=1.0, alpha=0.8)
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels([p['subject'].replace('sub-', '') for p in per_sub], fontsize=6.5)
    ax_b.set_xlabel('Subject', fontsize=8)
    ax_b.set_ylabel('Matched cosine (mean ± s.d.)', fontsize=8)
    ax_b.set_title('Per subject', fontsize=9, pad=5)
    ax_b.tick_params(labelsize=7)
    ax_b.set_ylim(0, max(means) + max(stds) + 0.08)

    ax_c = fig.add_subplot(gs[1, 0])
    bins_u = np.linspace(0, float(np.percentile(matched_umap, 99.5)), 35)
    ax_c.hist(
        matched_umap, bins=bins_u, density=True, alpha=0.65,
        color=PALETTE['pred'], label='Matched UMAP dist.', edgecolor='white', linewidth=0.3,
    )
    ax_c.axvline(float(matched_umap.mean()), color=PALETTE['pred'], ls='--', lw=1.0)
    ax_c.set_xlabel('Joint UMAP Euclidean distance', fontsize=8)
    ax_c.set_ylabel('Density', fontsize=8)
    ax_c.set_title('Joint UMAP (pooled)', fontsize=9, pad=5)
    ax_c.legend(fontsize=6.5, loc='upper right', frameon=False)
    ax_c.tick_params(labelsize=7)
    ax_c.text(
        0.03, 0.95,
        f'mean = {matched_umap.mean():.2f}\nmedian = {np.median(matched_umap):.2f}',
        transform=ax_c.transAxes, ha='left', va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#cccccc', alpha=0.9),
    )

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.scatter(
        umap_t_sub[:, 0], umap_t_sub[:, 1],
        s=12, c=PALETTE['true'], alpha=0.55, linewidths=0, label='GT', rasterized=True,
    )
    ax_d.scatter(
        umap_g_sub[:, 0], umap_g_sub[:, 1],
        s=12, c=PALETTE['pred'], alpha=0.55, linewidths=0, label='Gen', rasterized=True,
    )
    for i in range(umap_t_sub.shape[0]):
        ax_d.plot(
            [umap_t_sub[i, 0], umap_g_sub[i, 0]],
            [umap_t_sub[i, 1], umap_g_sub[i, 1]],
            color='#bbbbbb', alpha=0.25, linewidth=0.4, zorder=0,
        )
    ax_d.set_title(f'Matched pairs — sub-{single_sub:02d} UMAP', fontsize=9, pad=5)
    ax_d.set_xlabel('Dim 1', fontsize=8)
    ax_d.set_ylabel('Dim 2', fontsize=8)
    ax_d.legend(fontsize=6.5, loc='best', frameon=False, markerscale=0.8)
    ax_d.tick_params(labelsize=7)
    ax_d.set_aspect('equal', adjustable='datalim')

    fig.suptitle(
        f'Matched-pair distance (GT↔Gen, {meta["n_subjects"]} subjects × 200 images)',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    label_axes([ax_a, ax_b, ax_c, ax_d], start='a')

    stem = fig_path(FIG11)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    print(
        f'  encoder: matched cos={stats["matched_cos_mean"]:.3f}  '
        f'random={stats["random_cos_mean"]:.3f}  Δ={stats["cos_separation"]:.3f}'
    )
    return stem


def _encoder_pc1(
    emb: np.ndarray,
    pca_fit: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, PCA]:
    """First PC coordinate; fit on ``pca_fit`` (default: ``emb``)."""
    fit = emb if pca_fit is None else pca_fit
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    pca.fit(fit)
    pc1 = pca.transform(emb)[:, 0]
    return pc1, pca


def plot_encoder_pc1_histogram(
    sub: int = DEFAULT_SINGLE_SUB,
    force: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    """Gen encoder PC1 histogram (with GT reference & category breakdown)."""
    emb_true, emb_gen, _, _ = load_encoder_embeddings(force=force, device=device)
    true_sub, gen_sub, _ = _subject_slice(
        emb_true, emb_gen, np.zeros(200, dtype=object), sub,
    )
    concepts = _test_concepts(sub)
    categories = np.array([category_for_concept(c) for c in concepts])

    pc1_gen, pca = _encoder_pc1(gen_sub, pca_fit=np.vstack([true_sub, gen_sub]))
    pc1_gt = pca.transform(true_sub)[:, 0]
    var1 = float(pca.explained_variance_ratio_[0])

    _setup_nature_rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    bins = np.linspace(
        min(pc1_gen.min(), pc1_gt.min()) - 0.05,
        max(pc1_gen.max(), pc1_gt.max()) + 0.05,
        30,
    )
    ax = axes[0]
    ax.hist(
        pc1_gt, bins=bins, density=True, alpha=0.55,
        color=PALETTE['true'], label='GT', edgecolor='white', linewidth=0.3,
    )
    ax.hist(
        pc1_gen, bins=bins, density=True, alpha=0.55,
        color=PALETTE['pred'], label='Gen', edgecolor='white', linewidth=0.3,
    )
    ax.axvline(np.median(pc1_gen), color=PALETTE['pred'], ls='--', lw=1.0, alpha=0.9)
    ax.axvline(np.median(pc1_gt), color=PALETTE['true'], ls='--', lw=1.0, alpha=0.9)
    ax.set_xlabel('Encoder PC1', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title(
        f'PC1 distribution (joint fit, var={var1:.1%})',
        fontsize=9, pad=5,
    )
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    ax.tick_params(labelsize=7)
    ax.text(
        0.03, 0.97,
        f'Gen skew={float(((pc1_gen - pc1_gen.mean()) ** 3).mean() / pc1_gen.std() ** 3):.2f}',
        transform=ax.transAxes, ha='left', va='top', fontsize=7,
    )

    ax2 = axes[1]
    for cat in CATEGORY_ORDER:
        mask = categories == cat
        if not np.any(mask):
            continue
        ax2.hist(
            pc1_gen[mask], bins=bins, density=True, alpha=0.50,
            color=CATEGORY_COLORS[cat], label=cat.capitalize(),
            edgecolor='white', linewidth=0.3,
        )
    ax2.axvline(np.median(pc1_gen), color='#333333', ls='--', lw=1.0, alpha=0.7)
    ax2.set_xlabel('Gen encoder PC1', fontsize=8)
    ax2.set_ylabel('Density', fontsize=8)
    ax2.set_title('Gen PC1 by category', fontsize=9, pad=5)
    ax2.legend(fontsize=6.5, frameon=False, loc='upper right')
    ax2.tick_params(labelsize=7)

    fig.suptitle(
        f'Encoder PC1 — sub-{sub:02d} (200 test images, 80-trial mean)',
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    label_axes(axes, start='a')

    stem = fig_path(FIG12)
    save_pub(fig, stem)
    plt.close(fig)
    print(f'Saved {stem}.{{svg,png}}')
    return stem


def plot_all_eeg_embedding(
    subs: Optional[List[int]] = None,
    force: bool = False,
    single_sub: int = DEFAULT_SINGLE_SUB,
    device: Optional[torch.device] = None,
):
    plot_eeg_embedding_joint(subs, force=force, device=device)
    plot_matched_pair_distance(
        subs, force=force, single_sub=single_sub, device=device,
    )
    plot_encoder_pc1_histogram(sub=single_sub, force=force, device=device)


if __name__ == '__main__':
    plot_all_eeg_embedding(force=False)
