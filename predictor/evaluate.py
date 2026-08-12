"""Evaluate image-to-EEG generator on held-out test split."""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from predictor.losses import correlation_loss, time_loss
from predictor.model import load_any_generator_from_checkpoint
from predictor.data import eeg_to_ubp_embedding, load_frozen_ubp_brain


@torch.no_grad()
def compute_gt_gen_encoder_metrics(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    brain: nn.Module,
    *,
    compute_random: bool = True,
) -> Dict[str, float]:
    """Matched GT↔Gen distances in frozen brain-encoder space."""
    emb_g = eeg_to_ubp_embedding(y_hat, brain)
    emb_t = eeg_to_ubp_embedding(y, brain)
    matched_cos = (emb_g * emb_t).sum(dim=1)
    metrics = {
        'matched_encoder_cosine': matched_cos.mean().item(),
        'matched_encoder_cosine_median': matched_cos.median().item(),
        'matched_encoder_distance': (1.0 - matched_cos).mean().item(),
    }
    if compute_random and y_hat.shape[0] >= 2:
        sim = emb_g @ emb_t.T
        mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
        random_cos = sim[mask]
        metrics['random_encoder_cosine'] = random_cos.mean().item()
        metrics['encoder_cosine_separation'] = (
            metrics['matched_encoder_cosine'] - metrics['random_encoder_cosine']
        )
    return metrics


@torch.no_grad()
def evaluate_subject(
    model_path: str,
    clip_test: torch.Tensor,
    eeg_test: torch.Tensor,
    device: torch.device,
    result_dir: str,
    *,
    brain_config: Optional[dict] = None,
    ubp_ckpt: Optional[str] = None,
    batch_size: int = 200,
) -> Dict[str, float]:
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    seq_len = int(ckpt.get('seq_len', eeg_test.shape[-1]))
    n_ch = int(ckpt.get('n_channels', eeg_test.shape[1]))
    model = load_any_generator_from_checkpoint(ckpt, n_ch, seq_len).to(device)
    model.eval()

    brain = None
    if brain_config is not None and ubp_ckpt and os.path.isfile(ubp_ckpt):
        brain = load_frozen_ubp_brain(brain_config, ubp_ckpt, device)

    clip_features = clip_test.detach().clone()
    preds, targets = [], []
    for i in range(0, len(eeg_test), batch_size):
        cf = clip_features[i:i + batch_size].to(device).clone()
        y = eeg_test[i:i + batch_size].to(device)
        y_hat = model(cf)
        preds.append(y_hat.cpu())
        targets.append(y.cpu())

    y_hat = torch.cat(preds, dim=0)
    y = torch.cat(targets, dim=0)

    mse = time_loss(y_hat, y).item()
    corr = (1.0 - correlation_loss(y_hat, y)).item()

    metrics = {
        'test_mse': mse,
        'test_corr': corr,
        'n_test': int(y.shape[0]),
    }

    if brain is not None:
        y_hat_d = y_hat.to(device)
        y_d = y.to(device)
        clip_emb = F.normalize(clip_features.to(device), dim=1)
        eeg_emb = eeg_to_ubp_embedding(y_hat_d, brain)
        metrics['test_semantic_cosine'] = (eeg_emb * clip_emb).sum(dim=1).mean().item()

        sim = eeg_emb @ clip_emb.T
        top1 = (sim.argmax(dim=1) == torch.arange(sim.shape[0], device=device)).float().mean()
        top5 = (
            sim.topk(5, dim=1).indices
            == torch.arange(sim.shape[0], device=device).unsqueeze(1)
        ).any(dim=1).float().mean()
        metrics['test_retrieval_top1'] = top1.item()
        metrics['test_retrieval_top5'] = top5.item()

        enc = compute_gt_gen_encoder_metrics(y_hat_d, y_d, brain)
        metrics.update({f'test_{k}': v for k, v in enc.items()})

    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Test metrics:', metrics)
    return metrics
