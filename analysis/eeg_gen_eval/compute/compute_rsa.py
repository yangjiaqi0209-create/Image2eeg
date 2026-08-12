"""Compute EEG–EEG RDM and GT vs Gen RSA on the 200-way test set."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.eeg_gen_eval.concept_categories import (
    CATEGORY_ORDER,
    concept_from_relpath,
    sort_indices_by_category,
)
from analysis.eeg_gen_eval.config import (
    BASELINE_LABEL,
    DATA_DIR,
    N_TEST_CLASSES,
    RAW_DIR,
    eval_brain_ckpt_path,
)


def _brain_config(n_ch: int, seq_len: int, z_dim: int = 1024) -> dict:
    return {
        'target': 'encoder.models.EEGProjectLayer',
        'params': {'c_num': n_ch, 'z_dim': z_dim, 'timesteps': [0, seq_len]},
    }


def rdm_from_embeddings(emb: torch.Tensor) -> np.ndarray:
    """Pairwise dissimilarity: 1 - cosine similarity (embeddings assumed L2-normalized)."""
    sim = (emb @ emb.T).cpu().numpy()
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm.astype(np.float64)


def rsa_correlation(rdm_a: np.ndarray, rdm_b: np.ndarray) -> Dict[str, float]:
    """RSA: correlate upper-triangle entries of two RDMs."""
    n = rdm_a.shape[0]
    iu = np.triu_indices(n, k=1)
    va = rdm_a[iu]
    vb = rdm_b[iu]
    rho, _ = spearmanr(va, vb)
    r, _ = pearsonr(va, vb)
    return {
        'rsa_spearman': float(rho),
        'rsa_pearson': float(r),
        'n_pairs': int(len(va)),
    }


@torch.no_grad()
def compute_rsa(
    subs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    force: bool = False,
) -> Dict:
    """Per-subject and mean RDM/RSA for GT vs generated EEG (brain-encoder space)."""
    from analysis.eeg_gen_eval.compute.evaluate import _load_test_averaged
    from predictor.data import eeg_to_ubp_embedding, load_frozen_ubp_brain

    subs = subs or list(range(1, 11))
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(RAW_DIR, 'rsa')
    meta_path = os.path.join(out_dir, 'meta.json')
    if not force and os.path.isfile(meta_path):
        with open(meta_path) as f:
            return json.load(f)

    rdm_gt_sum = np.zeros((N_TEST_CLASSES, N_TEST_CLASSES), dtype=np.float64)
    rdm_gen_sum = np.zeros((N_TEST_CLASSES, N_TEST_CLASSES), dtype=np.float64)
    per_sub: List[Dict] = []
    concepts: List[str] | None = None

    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        pred_path = os.path.join(RAW_DIR, sub_tag, 'y_pred.npy')
        if not os.path.isfile(pred_path):
            raise FileNotFoundError(f'Missing {pred_path}. Run evaluate first.')

        y_true, _ = _load_test_averaged(sub, device)
        y_true = torch.from_numpy(y_true).float()
        y_pred = torch.from_numpy(np.load(pred_path)).float()

        if concepts is None:
            test_path = os.path.join(DATA_DIR, sub_tag, 'test.pt')
            data = torch.load(test_path, map_location='cpu', weights_only=False)
            img_paths = np.asarray(data['img'])[:, 0]
            concepts = [concept_from_relpath(str(p)) for p in img_paths]

        brain_ckpt = eval_brain_ckpt_path(sub, seed=0)
        brain = load_frozen_ubp_brain(
            _brain_config(y_true.shape[1], y_true.shape[2]),
            brain_ckpt,
            device,
        )
        if brain is None:
            raise FileNotFoundError(f'Brain encoder checkpoint not found: {brain_ckpt}')

        emb_gt = eeg_to_ubp_embedding(y_true.to(device), brain)
        emb_gen = eeg_to_ubp_embedding(y_pred.to(device), brain)
        rdm_gt = rdm_from_embeddings(emb_gt)
        rdm_gen = rdm_from_embeddings(emb_gen)
        rsa = rsa_correlation(rdm_gt, rdm_gen)

        rdm_gt_sum += rdm_gt
        rdm_gen_sum += rdm_gen
        per_sub.append({'subject': sub_tag, **rsa})
        print(f'  {sub_tag} RSA Spearman={rsa["rsa_spearman"]:.4f}')

    n_ok = len(per_sub)
    rdm_gt_mean = rdm_gt_sum / n_ok
    rdm_gen_mean = rdm_gen_sum / n_ok
    rsa_mean = rsa_correlation(rdm_gt_mean, rdm_gen_mean)
    spearman_vals = [s['rsa_spearman'] for s in per_sub]

    assert concepts is not None
    order, sorted_cats, block_sizes = sort_indices_by_category(concepts)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'rdm_gt_mean.npy'), rdm_gt_mean.astype(np.float32))
    np.save(os.path.join(out_dir, 'rdm_gen_mean.npy'), rdm_gen_mean.astype(np.float32))
    np.save(os.path.join(out_dir, 'sort_order.npy'), np.array(order, dtype=np.int32))

    meta = {
        'n_subjects': n_ok,
        'subjects': [s['subject'] for s in per_sub],
        'n_test': N_TEST_CLASSES,
        'concepts': concepts,
        'sort_order': order,
        'sorted_categories': sorted_cats,
        'block_sizes': block_sizes,
        'category_order': CATEGORY_ORDER,
        'baseline': BASELINE_LABEL,
        'embedding': 'UBP-Brain (EEGProjectLayer, fovea RN50)',
        'rdm_metric': '1 - cosine similarity between brain embeddings',
        'per_subject': per_sub,
        'rsa_mean_rdm': rsa_mean,
        'rsa_per_subject_mean': float(np.mean(spearman_vals)),
        'rsa_per_subject_std': float(np.std(spearman_vals, ddof=1)) if n_ok > 1 else 0.0,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def _upper_triangle(rdm: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(rdm.shape[0], k=1)
    return rdm[iu]


def _spearman_upper(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    rho, _ = spearmanr(_upper_triangle(rdm_a), _upper_triangle(rdm_b))
    return float(rho)


def _permute_gen_correspondence(rdm_gt: np.ndarray, emb_gen: torch.Tensor, perm: np.ndarray) -> float:
    """Null RSA: shuffle image→gen-EEG pairing, then correlate GT vs shuffled Gen RDM."""
    emb_shuf = emb_gen[perm]
    rdm_shuf = rdm_from_embeddings(emb_shuf)
    return _spearman_upper(rdm_gt, rdm_shuf)


@torch.no_grad()
def compute_rsa_permutation_test(
    subs: Optional[List[int]] = None,
    device: Optional[torch.device] = None,
    n_perm: int = 1000,
    seed: int = 2023,
    force: bool = False,
) -> Dict:
    """
    Permutation test for RSA significance.

    Null: randomly shuffle image→generated-EEG correspondence (Image_i → EEG_j),
    recompute Gen RDM, correlate with fixed GT RDM. One-sided p-value tests whether
    observed RSA exceeds the null (structure preservation beyond chance pairing).
    """
    from analysis.eeg_gen_eval.compute.evaluate import _load_test_averaged
    from predictor.data import eeg_to_ubp_embedding, load_frozen_ubp_brain

    subs = subs or list(range(1, 11))
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.join(RAW_DIR, 'rsa')
    perm_meta_path = os.path.join(out_dir, 'permutation_meta.json')
    null_path = os.path.join(out_dir, 'null_rsa_group.npy')
    if not force and os.path.isfile(perm_meta_path):
        with open(perm_meta_path) as f:
            return json.load(f)

    meta_path = os.path.join(out_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        compute_rsa(subs=subs, device=device, force=False)

    rng = np.random.default_rng(seed)
    per_sub: List[Dict] = []
    null_group: List[float] = []
    cached: List[Dict] = []

    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        pred_path = os.path.join(RAW_DIR, sub_tag, 'y_pred.npy')
        y_true, _ = _load_test_averaged(sub, device)
        y_true = torch.from_numpy(y_true).float()
        y_pred = torch.from_numpy(np.load(pred_path)).float()

        brain_ckpt = eval_brain_ckpt_path(sub, seed=0)
        brain = load_frozen_ubp_brain(
            _brain_config(y_true.shape[1], y_true.shape[2]),
            brain_ckpt,
            device,
        )
        if brain is None:
            raise FileNotFoundError(f'Brain encoder checkpoint not found: {brain_ckpt}')

        emb_gt = eeg_to_ubp_embedding(y_true.to(device), brain)
        emb_gen = eeg_to_ubp_embedding(y_pred.to(device), brain)
        rdm_gt = rdm_from_embeddings(emb_gt)
        obs = _spearman_upper(rdm_gt, rdm_from_embeddings(emb_gen))

        null_vals = np.empty(n_perm, dtype=np.float64)
        n = emb_gen.shape[0]
        for k in range(n_perm):
            perm = rng.permutation(n)
            null_vals[k] = _permute_gen_correspondence(rdm_gt, emb_gen, perm)
        p_one_sided = float((np.sum(null_vals >= obs) + 1) / (n_perm + 1))
        per_sub.append({
            'subject': sub_tag,
            'rsa_observed': obs,
            'null_mean': float(null_vals.mean()),
            'null_std': float(null_vals.std(ddof=1)),
            'null_q95': float(np.quantile(null_vals, 0.95)),
            'p_one_sided': p_one_sided,
        })
        null_group.extend(null_vals.tolist())
        cached.append({'rdm_gt': rdm_gt, 'emb_gen': emb_gen})
        print(
            f'  {sub_tag} obs={obs:.4f} null={null_vals.mean():.4f} '
            f'p={p_one_sided:.4g}'
        )

    obs_vals = [s['rsa_observed'] for s in per_sub]
    obs_mean = float(np.mean(obs_vals))
    null_group_arr = np.asarray(null_group, dtype=np.float64)

    null_group_iter = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        iter_vals = []
        for item in cached:
            n = item['emb_gen'].shape[0]
            perm = rng.permutation(n)
            iter_vals.append(
                _permute_gen_correspondence(item['rdm_gt'], item['emb_gen'], perm)
            )
        null_group_iter[k] = float(np.mean(iter_vals))

    p_group = float((np.sum(null_group_iter >= obs_mean) + 1) / (n_perm + 1))

    os.makedirs(out_dir, exist_ok=True)
    np.save(null_path, null_group_arr.astype(np.float32))
    np.save(os.path.join(out_dir, 'null_rsa_group_iter.npy'), null_group_iter.astype(np.float32))

    perm_meta = {
        'n_perm': n_perm,
        'seed': seed,
        'null_hypothesis': (
            'Shuffle image→generated-EEG correspondence; '
            'GT RDM fixed, Gen RDM recomputed from permuted embeddings.'
        ),
        'test': 'one-sided (observed RSA > null)',
        'p_value_correction': '(count(null >= obs) + 1) / (n_perm + 1)',
        'per_subject': per_sub,
        'group': {
            'rsa_observed_mean': obs_mean,
            'rsa_observed_std': float(np.std(obs_vals, ddof=1)) if len(obs_vals) > 1 else 0.0,
            'null_mean': float(null_group_arr.mean()),
            'null_std': float(null_group_arr.std(ddof=1)),
            'null_iter_mean': float(null_group_iter.mean()),
            'null_iter_std': float(null_group_iter.std(ddof=1)),
            'p_one_sided_iter_mean': p_group,
            'n_subjects_significant_p001': int(sum(s['p_one_sided'] <= 0.001 for s in per_sub)),
            'n_subjects_significant_p01': int(sum(s['p_one_sided'] <= 0.01 for s in per_sub)),
        },
    }
    with open(perm_meta_path, 'w') as f:
        json.dump(perm_meta, f, indent=2)

    # Merge summary into main meta if present.
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta['permutation_test'] = perm_meta
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    return perm_meta


def load_rsa_data() -> Tuple[Dict, np.ndarray, np.ndarray]:
    from analysis.eeg_gen_eval import config as cfg

    out_dir = os.path.join(cfg.RAW_DIR, 'rsa')
    meta_path = os.path.join(out_dir, 'meta.json')
    if not os.path.isfile(meta_path):
        compute_rsa()
    with open(meta_path) as f:
        meta = json.load(f)
    rdm_gt = np.load(os.path.join(out_dir, 'rdm_gt_mean.npy'))
    rdm_gen = np.load(os.path.join(out_dir, 'rdm_gen_mean.npy'))
    return meta, rdm_gt, rdm_gen


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(description='Compute GT vs Gen EEG RDM and RSA')
    p.add_argument('--subs', type=str, default='1-10')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--force', action='store_true')
    p.add_argument('--perm', action='store_true', help='Run permutation test only')
    p.add_argument('--n_perm', type=int, default=1000)
    p.add_argument('--seed', type=int, default=2023)
    args = p.parse_args()

    if '-' in args.subs:
        a, b = args.subs.split('-')
        subs = list(range(int(a), int(b) + 1))
    else:
        subs = [int(x) for x in args.subs.split(',')]

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.perm:
        perm_meta = compute_rsa_permutation_test(
            subs=subs, device=device, n_perm=args.n_perm, seed=args.seed, force=args.force,
        )
        g = perm_meta['group']
        print(
            f"Group RSA={g['rsa_observed_mean']:.4f} | "
            f"null={g['null_iter_mean']:.4f} | p={g['p_one_sided_iter_mean']:.4g}"
        )
    else:
        meta = compute_rsa(subs=subs, device=device, force=args.force)
        print(
            f"Mean RDM RSA (Spearman): {meta['rsa_mean_rdm']['rsa_spearman']:.4f} | "
            f"Per-subject mean±std: {meta['rsa_per_subject_mean']:.4f} "
            f"± {meta['rsa_per_subject_std']:.4f}"
        )
