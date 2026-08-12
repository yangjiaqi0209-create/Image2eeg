"""Run generator inference on test set (80-trial average) and save raw metrics."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from analysis.eeg_gen_eval.compute.array_utils import (
    eeg_per_sample as _eeg_per_sample,
    img_per_sample as _img_per_sample,
    trial_format as _trial_format,
)
from analysis.eeg_gen_eval.config import (
    CHANNELS,
    CKPT_DIR,
    DATA_DIR,
    IMAGE_ROOT,
    N_TEST_TRIALS,
    RAW_DIR,
    DATASET_PROFILES,
    active_dataset,
    eval_brain_ckpt_path,
    generator_eval_ckpt_path,
)
from analysis.eeg_gen_eval.compute.metrics import compute_all_metrics


def _brain_config(n_ch: int, seq_len: int, z_dim: int = 1024) -> dict:
    return {
        'target': 'encoder.models.EEGProjectLayer',
        'params': {'c_num': n_ch, 'z_dim': z_dim, 'timesteps': [0, seq_len]},
    }


def _clip_load_kwargs(device: torch.device) -> dict:
    profile = DATASET_PROFILES.get(active_dataset(), {})
    kwargs = {
        'data_dir': DATA_DIR,
        'device': device,
        'image_root': profile.get('image_root'),
        'feature_dir': profile.get('feature_dir'),
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def _generator_deps():
    """Lazy import — cached figure redraw does not need the generator stack."""
    from predictor.data import (
        clip_input_from_checkpoint,
        load_clip_features,
        load_frozen_ubp_brain,
        vision_backbone_from_checkpoint,
    )
    from predictor.evaluate import compute_gt_gen_encoder_metrics
    from predictor.model import load_any_generator_from_checkpoint
    return {
        'load_any_generator_from_checkpoint': load_any_generator_from_checkpoint,
        'clip_input_from_checkpoint': clip_input_from_checkpoint,
        'load_clip_features': load_clip_features,
        'vision_backbone_from_checkpoint': vision_backbone_from_checkpoint,
        'compute_gt_gen_encoder_metrics': compute_gt_gen_encoder_metrics,
        'load_frozen_ubp_brain': load_frozen_ubp_brain,
    }


def _load_train_averaged(
    sub: int,
    device: torch.device,
    model_type: str = 'RN50',
) -> tuple[np.ndarray, torch.Tensor]:
    """Load train EEG averaged over trials per image, and medium-blur CLIP features."""
    train_path = os.path.join(DATA_DIR, f'sub-{sub:02d}', 'train.pt')
    data = torch.load(train_path, map_location='cpu', weights_only=False)
    eeg = np.asarray(data['eeg'], dtype=np.float32)
    img_paths = np.asarray(data['img'])
    fmt = _trial_format(data)

    if eeg.ndim != 4 and eeg.ndim != 3:
        raise ValueError(f'Expected train EEG (N, trials, C, T), got {eeg.shape}')

    ch_idx = [CHANNELS.index(ch) for ch in CHANNELS]
    if eeg.ndim == 4:
        eeg = eeg[:, :, ch_idx, :]
        y_true = _eeg_per_sample(eeg, fmt)
    else:
        y_true = eeg[:, ch_idx, :]

    img_per_concept = _img_per_sample(img_paths, fmt)
    load_clip_features = _generator_deps()['load_clip_features']
    clip = load_clip_features(
        img_per_concept,
        model_type=model_type,
        clip_input='fovea',
        blur_kernel_size=51,
        mode='train',
        tag='medium',
        **_clip_load_kwargs(device),
    )
    return y_true, clip


def _load_test_averaged(
    sub: int,
    device: torch.device,
    clip_input: str = 'fovea',
    blur_kernel_size: int = 51,
    model_type: str = 'RN50',
) -> tuple[np.ndarray, torch.Tensor]:
    """Load test EEG; THINGS averages reps, NOD keeps one row per unique image."""
    test_path = os.path.join(DATA_DIR, f'sub-{sub:02d}', 'test.pt')
    data = torch.load(test_path, map_location='cpu', weights_only=False)
    eeg = np.asarray(data['eeg'], dtype=np.float32)
    img_paths = np.asarray(data['img'])
    fmt = _trial_format(data)

    if eeg.ndim != 4 and eeg.ndim != 3:
        raise ValueError(f'Expected test EEG (N, trials, C, T), got {eeg.shape}')
    if eeg.ndim == 4:
        n_trials = eeg.shape[1]
        if fmt != 'per_image' and n_trials != N_TEST_TRIALS:
            print(f'  warning: sub-{sub:02d} has {n_trials} trials (expected {N_TEST_TRIALS})')

    ch_idx = [CHANNELS.index(ch) for ch in CHANNELS]
    if eeg.ndim == 4:
        eeg = eeg[:, :, ch_idx, :]
        y_true = _eeg_per_sample(eeg, fmt)
    else:
        y_true = eeg[:, ch_idx, :]

    img_per_concept = _img_per_sample(img_paths, fmt)
    load_clip_features = _generator_deps()['load_clip_features']
    clip = load_clip_features(
        img_per_concept,
        model_type=model_type,
        clip_input=clip_input,
        blur_kernel_size=blur_kernel_size,
        mode='test',
        tag='medium',
        **_clip_load_kwargs(device),
    )
    return y_true, clip


def _backbone_from_ckpt(ckpt_path: Optional[str]) -> str:
    if ckpt_path and os.path.isfile(ckpt_path):
        return _generator_deps()['vision_backbone_from_checkpoint'](ckpt_path)
    return 'RN50'


@torch.no_grad()
def predict_subject(sub: int, device: torch.device, ckpt_path: Optional[str] = None) -> np.ndarray:
    deps = _generator_deps()
    if ckpt_path is None:
        ckpt_path = generator_eval_ckpt_path(CKPT_DIR, sub)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    clip_input = deps['clip_input_from_checkpoint'](ckpt_path, default='ubp')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt.get('args') or {}
    blur_kernel_size = int(args.get('blur_kernel_size', 51))
    model_type = deps['vision_backbone_from_checkpoint'](ckpt_path)

    y_true, clip_test = _load_test_averaged(
        sub, device,
        clip_input=clip_input,
        blur_kernel_size=blur_kernel_size,
        model_type=model_type,
    )
    n_ch = int(ckpt.get('n_channels', y_true.shape[1]))
    seq_len = int(ckpt.get('seq_len', y_true.shape[2]))
    model = deps['load_any_generator_from_checkpoint'](ckpt, n_ch, seq_len).to(device)
    model.eval()

    preds = []
    bs = 64
    for i in range(0, len(clip_test), bs):
        cf = clip_test[i:i + bs].to(device)
        preds.append(model(cf).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).astype(np.float32)
    assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
    return y_pred, y_true


@torch.no_grad()
def compute_train_erp(
    sub: int,
    device: torch.device,
    ckpt_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Grand-mean ERP over all train images: (n_ch, T) for true and predicted."""
    if ckpt_path is None:
        ckpt_path = generator_eval_ckpt_path(CKPT_DIR, sub)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    deps = _generator_deps()
    model_type = _backbone_from_ckpt(ckpt_path)
    y_true, clip_train = _load_train_averaged(sub, device, model_type=model_type)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    n_ch = int(ckpt.get('n_channels', y_true.shape[1]))
    seq_len = int(ckpt.get('seq_len', y_true.shape[2]))
    model = deps['load_any_generator_from_checkpoint'](ckpt, n_ch, seq_len).to(device)
    model.eval()

    preds = []
    bs = 64
    for i in range(0, len(clip_train), bs):
        cf = clip_train[i:i + bs].to(device)
        preds.append(model(cf).cpu().numpy())
    y_pred = np.concatenate(preds, axis=0).astype(np.float32)
    assert y_pred.shape == y_true.shape, (y_pred.shape, y_true.shape)
    return y_true.mean(axis=0), y_pred.mean(axis=0)


def evaluate_subject(sub: int, device: torch.device, ckpt_path: Optional[str] = None) -> Dict:
    sub_tag = f'sub-{sub:02d}'
    out_dir = os.path.join(RAW_DIR, sub_tag)
    os.makedirs(out_dir, exist_ok=True)

    y_pred, y_true = predict_subject(sub, device, ckpt_path)
    result = compute_all_metrics(y_pred, y_true)
    summary = result['summary']
    arrays = result['arrays']

    np.save(os.path.join(out_dir, 'y_pred.npy'), y_pred)
    np.save(os.path.join(out_dir, 'y_true.npy'), y_true)
    for key, arr in arrays.items():
        np.save(os.path.join(out_dir, f'{key}.npy'), arr)

    meta = {
        'subject': sub_tag,
        'checkpoint': ckpt_path or generator_eval_ckpt_path(CKPT_DIR, sub),
        **summary,
    }

    deps = _generator_deps()
    brain_ckpt = eval_brain_ckpt_path(sub, seed=0)
    brain = deps['load_frozen_ubp_brain'](
        _brain_config(y_true.shape[1], y_true.shape[2]),
        brain_ckpt,
        device,
    )
    if brain is not None:
        y_hat_t = torch.from_numpy(y_pred).float().to(device)
        y_true_t = torch.from_numpy(y_true).float().to(device)
        enc = deps['compute_gt_gen_encoder_metrics'](y_hat_t, y_true_t, brain)
        meta.update(enc)

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    enc_msg = ''
    if 'matched_encoder_distance' in meta:
        enc_msg = (
            f' enc_dist={meta["matched_encoder_distance"]:.4f}'
            f' enc_cos={meta["matched_encoder_cosine"]:.4f}'
        )
    print(f'{sub_tag}: MSE={summary["mse"]:.4f} MAE={summary["mae"]:.4f} '
          f'r={summary["pearson_r"]:.4f} R²_wave={summary["r2_per_sample_waveform_mean"]:.4f}'
          f'{enc_msg}')
    return meta


def recompute_metrics_from_raw(subs: List[int]) -> Dict:
    """Recompute metrics from saved y_pred/y_true without re-running inference."""
    all_meta = []
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        out_dir = os.path.join(RAW_DIR, sub_tag)
        y_pred = np.load(os.path.join(out_dir, 'y_pred.npy'))
        y_true = np.load(os.path.join(out_dir, 'y_true.npy'))
        result = compute_all_metrics(y_pred, y_true)
        summary = result['summary']
        arrays = result['arrays']
        for key, arr in arrays.items():
            np.save(os.path.join(out_dir, f'{key}.npy'), arr)
        meta_path = os.path.join(out_dir, 'metrics.json')
        prev = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                prev = json.load(f)
        meta = {
            'subject': sub_tag,
            'checkpoint': prev.get('checkpoint'),
            **summary,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        all_meta.append(meta)
        print(f'{sub_tag}: R² vs mean={summary["r2_per_sample_vs_mean_mean"]:.4f}, '
              f'R² waveform={summary["r2_per_sample_waveform_mean"]:.4f} '
              f'(median={summary["r2_per_sample_waveform_median"]:.4f})')

    summary = _aggregate_summaries(all_meta)
    summary_path = os.path.join(RAW_DIR, 'summary_all_subjects.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved aggregate summary -> {summary_path}')
    return summary


def evaluate_all_subjects(
    subs: List[int],
    device: torch.device,
    ckpt_path: Optional[str] = None,
    *,
    ckpt_dir: Optional[str] = None,
) -> Dict:
    os.makedirs(RAW_DIR, exist_ok=True)
    all_meta = []
    for sub in subs:
        sub_ckpt = ckpt_path
        if sub_ckpt is None and ckpt_dir is not None:
            sub_ckpt = generator_eval_ckpt_path(ckpt_dir, sub)
        all_meta.append(evaluate_subject(sub, device, sub_ckpt))

    summary = _aggregate_summaries(all_meta)
    summary_path = os.path.join(RAW_DIR, 'summary_all_subjects.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Saved aggregate summary -> {summary_path}')
    return summary


def _aggregate_summaries(metas: List[Dict]) -> Dict:
    scalar_keys = [
        'mse', 'rmse', 'nmse', 'mae', 'pearson_r',
        'r2_per_sample_vs_mean_mean', 'r2_per_sample_vs_mean_median',
        'r2_per_sample_waveform_mean', 'r2_per_sample_waveform_median',
        'r2_pooled_vs_mean',
        'fft_l1_magnitude_error', 'fft_mse_magnitude_error',
    ]
    agg = {'n_subjects': len(metas), 'subjects': [m['subject'] for m in metas], 'per_subject': metas}

    for key in scalar_keys:
        vals = [m[key] for m in metas]
        agg[key] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=0)), 'values': vals}

    for band in metas[0]['bandpower_abs_error']:
        errs = [m['bandpower_abs_error'][band] for m in metas]
        corrs = [m['bandpower_correlation'][band] for m in metas]
        agg.setdefault('bandpower_abs_error', {})[band] = {
            'mean': float(np.mean(errs)), 'std': float(np.std(errs, ddof=0)),
        }
        agg.setdefault('bandpower_correlation', {})[band] = {
            'mean': float(np.mean(corrs)), 'std': float(np.std(corrs, ddof=0)),
        }

    for win in metas[0]['time_window_pearson']:
        vals = [m['time_window_pearson'][win] for m in metas]
        agg.setdefault('time_window_pearson', {})[win] = {
            'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=0)),
        }

    per_ch = np.stack([
        np.load(os.path.join(RAW_DIR, m['subject'], 'per_channel_pearson.npy'))
        for m in metas
    ], axis=0)
    per_t = np.stack([
        np.load(os.path.join(RAW_DIR, m['subject'], 'per_timepoint_pearson.npy'))
        for m in metas
    ], axis=0)
    np.save(os.path.join(RAW_DIR, 'per_channel_pearson_all.npy'), per_ch)
    np.save(os.path.join(RAW_DIR, 'per_timepoint_pearson_all.npy'), per_t)
    agg['per_channel_pearson'] = {
        'mean': per_ch.mean(axis=0).tolist(),
        'std': per_ch.std(axis=0, ddof=0).tolist(),
    }
    agg['per_timepoint_pearson'] = {
        'mean': per_t.mean(axis=0).tolist(),
        'std': per_t.std(axis=0, ddof=0).tolist(),
    }
    return agg
