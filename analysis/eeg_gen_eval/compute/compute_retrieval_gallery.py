"""Compute per-category retrieval examples (GT / Gen top-5).

Cache hits (raw/retrieval_gallery/*.json) need no generator / encoder imports.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.eeg_gen_eval.concept_categories import (
    CATEGORY_ORDER,
    category_for_concept,
    concept_from_relpath,
)
from analysis.eeg_gen_eval.config import DATA_DIR, RAW_DIR, eval_brain_ckpt_path


def _both_in_top5_candidates(
    by_cat: Dict[str, List[int]],
    sim_gt: np.ndarray,
    sim_gen: np.ndarray,
    cat: str,
) -> List[int]:
    """Indices where the query appears in both GT and Gen top-5 retrieval."""
    out = []
    for q in by_cat[cat]:
        gt_top5 = np.argsort(-sim_gt[q])[:5]
        gen_top5 = np.argsort(-sim_gen[q])[:5]
        if q in gt_top5 and q in gen_top5:
            out.append(q)
    return out


def _rank_in_top5(top5: np.ndarray, q: int) -> int:
    return int(np.where(top5 == q)[0][0]) + 1


def _brain_config(n_ch: int, seq_len: int, z_dim: int = 1024) -> dict:
    return {
        'target': 'encoder.models.EEGProjectLayer',
        'params': {'c_num': n_ch, 'z_dim': z_dim, 'timesteps': [0, seq_len]},
    }


def compute_retrieval_examples(
    sub: int = 1,
    seed: int = 42,
    device: Optional['torch.device'] = None,
    force: bool = False,
) -> Dict:
    """One random query per category; GT & Gen both retrieve query within top-5."""
    out_dir = os.path.join(RAW_DIR, 'retrieval_gallery')
    meta_path = os.path.join(out_dir, f'sub-{sub:02d}_examples.json')
    if os.path.isfile(meta_path) and not force:
        with open(meta_path) as f:
            return json.load(f)

    import torch
    import torch.nn.functional as F
    from analysis.eeg_gen_eval.compute.evaluate import _load_test_averaged
    from predictor.data import eeg_to_ubp_embedding, load_frozen_ubp_brain

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sub_tag = f'sub-{sub:02d}'
    pred_path = os.path.join(RAW_DIR, sub_tag, 'y_pred.npy')
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f'Missing {pred_path}. Run evaluate first.')

    with torch.no_grad():
        y_true, clip = _load_test_averaged(sub, device)
        y_true = torch.from_numpy(y_true).float()
        y_pred = torch.from_numpy(np.load(pred_path)).float()

        test_path = os.path.join(DATA_DIR, sub_tag, 'test.pt')
        data = torch.load(test_path, map_location='cpu', weights_only=False)
        img_paths = [str(p) for p in np.asarray(data['img'])[:, 0]]
        concepts = [concept_from_relpath(p) for p in img_paths]

        brain_ckpt = eval_brain_ckpt_path(sub, seed=0)
        brain = load_frozen_ubp_brain(
            _brain_config(y_true.shape[1], y_true.shape[2]), brain_ckpt, device,
        )
        if brain is None:
            raise FileNotFoundError(f'Brain encoder checkpoint not found: {brain_ckpt}')

        clip_n = F.normalize(clip.to(device), dim=1)
        sim_gt = (eeg_to_ubp_embedding(y_true.to(device), brain) @ clip_n.T).cpu().numpy()
        sim_gen = (eeg_to_ubp_embedding(y_pred.to(device), brain) @ clip_n.T).cpu().numpy()

        by_cat: Dict[str, List[int]] = {c: [] for c in CATEGORY_ORDER}
        for i, name in enumerate(concepts):
            by_cat[category_for_concept(name)].append(i)

        rng = np.random.default_rng(seed)
        examples = []
        for cat in CATEGORY_ORDER:
            candidates = _both_in_top5_candidates(by_cat, sim_gt, sim_gen, cat)
            if not candidates:
                raise RuntimeError(
                    f'No query in category {cat} with GT and Gen both in top-5 (sub={sub}).',
                )
            q = int(rng.choice(candidates))
            gt_top5 = np.argsort(-sim_gt[q])[:5].tolist()
            gen_top5 = np.argsort(-sim_gen[q])[:5].tolist()
            examples.append({
                'category': cat,
                'query_idx': q,
                'query_path': img_paths[q],
                'concept': concepts[q],
                'gt_top5_idx': gt_top5,
                'gt_top5_paths': [img_paths[i] for i in gt_top5],
                'gt_top5_scores': [float(sim_gt[q, i]) for i in gt_top5],
                'gen_top5_idx': gen_top5,
                'gen_top5_paths': [img_paths[i] for i in gen_top5],
                'gen_top5_scores': [float(sim_gen[q, i]) for i in gen_top5],
                'gt_rank': _rank_in_top5(np.asarray(gt_top5), q),
                'gen_rank': _rank_in_top5(np.asarray(gen_top5), q),
                'gt_top1_correct': gt_top5[0] == q,
                'gen_top1_correct': gen_top5[0] == q,
            })

    meta = {
        'subject': sub_tag,
        'seed': seed,
        'examples': examples,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return meta
