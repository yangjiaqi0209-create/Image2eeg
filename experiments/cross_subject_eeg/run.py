"""Audit THINGS-EEG2, rerun final models, and evaluate valid common-space data.

Run from the repository root with ``python -m experiments.cross_subject_eeg.run``.
Without a verified common-space bundle, only within-subject reproduction runs.
No off-diagonal correlations or specificity tests are computed in MVNN space.
"""

from __future__ import annotations
from project_paths import load_json, load_json_file, resolve_path, verify_source

import argparse
import csv
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from figures.eeg_gen_eval.config import CHANNELS, CHANNELS_O
from figures.eeg_gen_eval.compute.metrics import (
    global_pearson, mean_per_channel_pearson, per_channel_pearson,
)
from predictor.model import load_any_generator_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
SUBJECTS = [f'sub-{i:02d}' for i in range(1, 11)]
METRICS = ('whole_scalp', 'mean_channel', 'occipital')


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy-2 array pickles in the existing NumPy-1 EEG environment."""

    def find_class(self, module, name):
        if np.lib.NumpyVersion(np.__version__) < '2.0.0' and module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def load_payload(path):
    # Only use on the user's trusted local .pt files, just like the project loader.
    compat = types.ModuleType('numpy_compat_pickle')
    compat.Unpickler = NumpyCompatUnpickler
    compat.load, compat.loads = pickle.load, pickle.loads
    return torch.load(path, map_location='cpu', pickle_module=compat)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_payload(data, canonical_ids=None):
    eeg = np.asarray(data['eeg'])
    if eeg.shape != (200, 80, 63, 250):
        raise ValueError(f'Expected all 200 images × 80 trials × 63 channels × 250 times: {eeg.shape}')
    if not np.isfinite(eeg).all():
        raise ValueError('Nonfinite EEG; no silent trial or image exclusion is allowed')
    imgs = np.asarray(data['img']).astype(str)
    if imgs.shape != (200, 80) or not np.all(imgs == imgs[:, :1]):
        raise ValueError('Image identity is not constant across the 80 repetitions')
    ids = imgs[:, 0].tolist()
    if len(set(ids)) != 200:
        raise ValueError('Test images must have 200 unique IDs')
    channels = list(data.get('ch_names') or [])
    if len(channels) != len(set(channels)) or set(channels) != set(CHANNELS):
        raise ValueError('Missing, duplicated or unexpected EEG channel names')
    if canonical_ids is None:
        canonical_ids = ids
    if set(ids) != set(canonical_ids):
        raise ValueError('Test image ID sets differ between subjects')
    idx = [ids.index(image) for image in canonical_ids]
    ch_idx = [channels.index(c) for c in CHANNELS]
    # Model heads are in the training channel order, so retain both orders.
    y_model_order = eeg.mean(axis=1, dtype=np.float32)[idx]
    y_canonical = y_model_order[:, ch_idx]
    session = np.asarray(data.get('session'))
    if session.shape != (200, 80) or set(np.unique(session)) != {0, 1, 2, 3}:
        raise ValueError('Expected four session labels for every test image')
    counts = [[int(np.sum(row == s)) for s in range(4)] for row in session]
    if not np.all(np.array(counts) == 20):
        raise ValueError('Expected 20 repetitions per image per session')
    times = np.asarray(data.get('times'), dtype=np.float64)
    if times.ndim != 1 or len(times) < eeg.shape[-1]:
        raise ValueError('Missing usable time metadata')
    # The preprocessing crops epochs to their last 250 samples but stores full times.
    used_times = times[-eeg.shape[-1]:]
    if not np.allclose(used_times, np.arange(250) / 250, atol=1e-6):
        raise ValueError('Expected 0–996 ms at 250 Hz; never relabel an unknown time axis')
    info = {
        'shape': list(eeg.shape), 'image_ids_identical_order': ids == canonical_ids,
        'image_ids_same_set': True, 'channels': channels,
        'channel_order_matches_canonical': channels == list(CHANNELS),
        'session_repetitions_per_image': [20, 20, 20, 20],
        'stored_times_length': len(times), 'used_time_range_s': used_times[[0, -1]].tolist(),
        'payload_keys': sorted(data),
        'possible_transform_keys': [k for k in data if any(
            token in k.lower() for token in ('sigma', 'cov', 'whiten', 'zscore', 'z_score', 'std', 'scale'))],
    }
    return y_canonical, y_model_order, canonical_ids, ch_idx, info


def waveform_metrics(pred, target):
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError('Waveforms must have matched [image, channel, time] shapes')
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('Nonfinite waveforms')
    roi = [CHANNELS.index(ch) for ch in CHANNELS_O]
    values = {
        # Exact published functions: pool all image × channel × time values;
        # or pool image × time per channel, then average channel correlations.
        'whole_scalp': global_pearson(pred, target),
        'mean_channel': mean_per_channel_pearson(pred, target),
        'occipital': mean_per_channel_pearson(pred[:, roi], target[:, roi]),
    }
    # The existing helper omits undefined channels. Here undefined r is blocking.
    pc = per_channel_pearson(pred, target)
    if not np.isfinite(pc).all() or not np.isfinite(list(values.values())).all():
        raise ValueError('Undefined Pearson correlation; no silent channel omission')
    return values


def training_waveform_r(pred, target):
    a = pred.astype(np.float64) - pred.mean(axis=-1, keepdims=True, dtype=np.float64)
    b = target.astype(np.float64) - target.mean(axis=-1, keepdims=True, dtype=np.float64)
    return float(np.mean(np.sum(a * b, axis=-1) /
                         (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)))


def require_common_space(metadata):
    if metadata.get('signal_space') != 'unwhitened_electrode':
        raise ValueError('Cross-subject evaluation requires unwhitened electrode-space waveforms')
    if metadata.get('subject_specific_whitening_remaining') is not False:
        raise ValueError('Subject-specific whitening must have been removed')
    if metadata.get('session_mixture_resolved') is not True:
        raise ValueError('Session-specific whitening of the averaged model output is unresolved')
    for key in ('units', 'reference', 'prediction_reconstruction', 'target_reconstruction',
                'normalization_history', 'provenance'):
        if not metadata.get(key):
            raise ValueError(f'Common-space metadata must document {key}')


def evaluate_common_space(bundle_path, out_dir, image_ids):
    """Require arrays already valid in a shared space; do not guess an inverse.

    Schema: predictions/targets [10,200,63,250], subjects [10], image_ids [200],
    channels [63], times_s [250], metadata_json scalar UTF-8 string.
    The metadata declarations are a necessary gate, not independent proof of
    reconstruction validity. Review the recorded upstream reconstruction first.
    """
    from experiments.cross_subject_eeg.statistics import summarize_matrices

    with np.load(bundle_path, allow_pickle=False) as bundle:
        meta = load_json(str(bundle['metadata_json'].item()))
        require_common_space(meta)
        if bundle['subjects'].tolist() != SUBJECTS or bundle['image_ids'].tolist() != image_ids:
            raise ValueError('Common-space subjects / image IDs are not in canonical order')
        if bundle['channels'].tolist() != list(CHANNELS):
            raise ValueError('Common-space channels are not in canonical electrode order')
        if not np.allclose(bundle['times_s'], np.arange(250) / 250, atol=1e-6):
            raise ValueError('Common-space time grid differs')
        predictions, targets = bundle['predictions'], bundle['targets']
    if predictions.shape != (10, 200, 63, 250) or targets.shape != predictions.shape:
        raise ValueError('Common-space bundle must contain all 10 subjects and 200 test images')
    matrices = {k: np.empty((10, 10)) for k in METRICS}
    for i, pred in enumerate(predictions):
        for j, target in enumerate(targets):
            for name, value in waveform_metrics(pred, target).items():
                matrices[name][i, j] = value
    for name, matrix in matrices.items():
        write_csv(out_dir / f'matrix_{name}.csv', [
            {'model_subject': sub, **dict(zip(SUBJECTS, row))}
            for sub, row in zip(SUBJECTS, matrix)])
    summary, pairs = summarize_matrices(matrices, SUBJECTS)
    write_json(out_dir / 'statistics.json', summary)
    write_csv(out_dir / 'within_cross_pairs.csv', pairs)
    write_json(out_dir / 'common_space_provenance.json', {
        'bundle_sha256': digest(bundle_path), 'metadata': meta,
        'upstream_reconstruction_requires_scientific_review': True,
    })
    return summary


def run(args):
    out = args.output.resolve()
    # Fresh output directories prevent accidental reuse/overwriting of original results.
    out.mkdir(parents=True, exist_ok=False)
    (out / 'predictions').mkdir()
    torch.set_num_threads(args.threads)
    torch.manual_seed(2023)
    device = torch.device(args.device)
    audit = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'command': sys.argv, 'python': platform.python_version(),
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
        'threads': args.threads,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        'signal_space': 'subject_and_session_specific_mvnn',
        'cross_subject_status': 'BLOCKED', 'subjects': {},
        'metric_definitions': {
            'whole_scalp': 'Pearson after flattening image × channel × time; published global_pearson',
            'mean_channel': 'Mean of 63 channel Pearson r, each pooled across image × time',
            'occipital': 'Mean of 8 occipital channel Pearson r, each pooled across image × time',
            'training_waveform_r': 'Mean image × channel Pearson along time; differs from pooled metrics',
        },
        'occipital_channels': list(CHANNELS_O),
        'normalization_audit': {
            'mvnn': 'Training covariance per subject and session; no matrices saved by save_split',
            'zscore': 'No explicit EEG z-score found in inspected current or 0f837c3 THINGS preprocessing/loaders; historical execution not fully proven',
            'session_average': 'Each prediction targets the mean of 80 separately session-whitened repetitions',
        },
        'reproduction_tolerances': {
            'target_max_abs': 2e-6,
            'metric_abs': args.metric_atol,
            'prediction_relative_l2': args.prediction_rtol,
            'interpretation': 'Numerical agreement, not bitwise equality; tolerances declared in this run',
        },
        'source_sha256': {str(p.relative_to(REPO)): digest(p) for p in [
            Path(__file__), REPO / 'src/predictor/model.py', REPO / 'src/predictor/data.py',
            REPO / 'src/preprocess/process_eeg_whiten.py',
            REPO / 'figures/eeg_gen_eval/compute/metrics.py',
        ]},
    }
    rows, image_ids = [], None
    common_features, first_feature_spec = None, None
    for subject in SUBJECTS:
        print(f'{subject}: auditing EEG and final checkpoint', flush=True)
        test_path = args.data_dir / subject / 'test.pt'
        data = load_payload(test_path)
        target, target_order, image_ids, ch_idx, info = validate_payload(data, image_ids)
        del data
        # Verify train/test channel order; output-head order follows training data.
        train_path = args.data_dir / subject / 'train.pt'
        train_data = load_payload(train_path)
        if list(train_data.get('ch_names') or []) != info['channels']:
            raise ValueError(f'{subject}: train/test channel order mismatch')
        info['train_payload_keys'] = sorted(train_data)
        info['train_channel_order_matches_test'] = True
        del train_data
        checkpoint = args.ckpt_dir / subject / 'last.pt'
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        ca = ckpt.get('args', {})
        if ca.get('sub') != int(subject[-2:]):
            raise ValueError('Checkpoint subject does not match model row')
        spec = (ca.get('clip_input', 'fovea'), int(ca.get('blur_kernel_size', 51)),
                ca.get('vision_backbone', 'RN50'))
        if spec != ('fovea', 51, 'RN50'):
            raise ValueError(f'Unexpected feature specification: {spec}')
        if common_features is None:
            cached = load_payload(args.feature_cache)['img_features']
            if set(cached) != set(image_ids):
                raise ValueError('Feature cache does not exactly cover the 200 test image IDs')
            common_features = torch.stack([cached[k] for k in image_ids]).float()
            first_feature_spec = spec
        if spec != first_feature_spec:
            raise ValueError('All models must receive the identical image features')
        if (ckpt.get('n_channels'), ckpt.get('seq_len')) != (63, 250):
            raise ValueError('Unexpected final model dimensions')
        model = load_any_generator_from_checkpoint(ckpt, 63, 250).to(device).eval()
        with torch.inference_mode():
            pred_order = np.concatenate([
                model(common_features[i:i + args.batch_size].to(device)).cpu().numpy()
                for i in range(0, 200, args.batch_size)], axis=0)
        pred = pred_order[:, ch_idx]
        np.save(out / 'predictions' / f'{subject}.npy', pred)
        values = waveform_metrics(pred, target)
        paper_dir = args.paper_raw / subject
        paper_meta = load_json((paper_dir / 'metrics.json').read_text())
        paper_pred = np.load(paper_dir / 'y_pred.npy')
        paper_target = np.load(paper_dir / 'y_true.npy')
        # Historical raw arrays have no image IDs: establish their order by full
        # equality of the recalculated 80-trial target averages, then compare predictions.
        max_target_error = float(np.max(np.abs(target - paper_target)))
        max_pred_error = float(np.max(np.abs(pred - paper_pred)))
        pred_relative_l2 = float(np.linalg.norm(pred.astype(np.float64) - paper_pred) /
                                 np.linalg.norm(paper_pred.astype(np.float64)))
        expected = waveform_metrics(paper_pred, paper_target)
        delta_r = float(values['whole_scalp'] - paper_meta['pearson_r'])
        passed = (max_target_error <= 2e-6 and pred_relative_l2 <= args.prediction_rtol
                  and abs(delta_r) <= args.metric_atol
                  and all(abs(values[k] - expected[k]) <= args.metric_atol for k in METRICS))
        row = {'subject': subject, **values,
               'training_waveform_r': training_waveform_r(pred_order, target_order),
               'paper_whole_scalp': paper_meta['pearson_r'],
               'paper_mean_channel': expected['mean_channel'],
               'paper_occipital': expected['occipital'],
               'whole_scalp_difference_from_paper': delta_r,
               'max_abs_prediction_difference': max_pred_error,
               'prediction_relative_l2_difference': pred_relative_l2,
               'max_abs_target_difference': max_target_error,
               'bitwise_prediction_reproduction': bool(np.array_equal(pred, paper_pred)),
               'reproduction_passed': passed}
        rows.append(row)
        info.update({'test_sha256': digest(test_path), 'checkpoint_sha256': digest(checkpoint),
                     'checkpoint_args': ca, 'reproduction': row,
                     'paper_metrics_sha256': digest(paper_dir / 'metrics.json')})
        audit['subjects'][subject] = info
        write_csv(out / 'within_subject_reproduction.csv', rows)
        write_json(out / 'audit.json', audit)
        print(f"{subject}: pooled r={values['whole_scalp']:.8f}; channel r={values['mean_channel']:.8f}; reproduction={passed}", flush=True)
        del ckpt, model, pred, pred_order, target, target_order, paper_pred, paper_target
    write_csv(out / 'image_alignment.csv', [
        {'row_index': i, 'image_id': img, **{sub: True for sub in SUBJECTS}}
        for i, img in enumerate(image_ids)])
    write_json(out / 'image_ids.json', image_ids)
    audit['feature_cache_sha256'] = digest(args.feature_cache)
    audit['within_subject_all_passed'] = all(r['reproduction_passed'] for r in rows)
    from experiments.cross_subject_eeg.statistics import mean_sem
    write_json(out / 'within_subject_summary.json', {
        'n_subjects': 10, 'n_images': 200, 'n_trials_averaged_per_image': 80,
        'signal_space': 'subject_and_session_specific_mvnn',
        'interpretation': 'Within-subject reproduction only; no subject-specificity inference',
        'metrics': {k: mean_sem([r[k] for r in rows]) for k in METRICS},
        'max_whole_scalp_difference_from_paper': max(abs(r['whole_scalp_difference_from_paper']) for r in rows),
        'max_relative_l2_prediction_difference': max(r['prediction_relative_l2_difference'] for r in rows),
        'all_targets_identical_to_paper_cache': all(r['max_abs_target_difference'] == 0 for r in rows),
    })
    if not audit['within_subject_all_passed']:
        audit['blocker'] = 'Within-subject reproduction failed; inspect differences before cross-subject evaluation'
    elif args.common_space_bundle:
        evaluate_common_space(args.common_space_bundle, out, image_ids)
        audit['cross_subject_status'] = 'COMPUTED_FROM_DECLARED_COMMON_SPACE_BUNDLE'
    else:
        audit['blocker'] = (
            'Missing per-subject/session MVNN transforms and validated reconstruction of session-averaged model outputs. '
            'No formal off-diagonal correlations, specificity p-values or Figures A/B were computed.'
        )
    write_json(out / 'audit.json', audit)
    write_json(out / 'experiment_status.json', {
        'within_subject_reproduced': audit['within_subject_all_passed'],
        'cross_subject_status': audit['cross_subject_status'],
        'subject_specificity_conclusion': None,
        'reason': audit.get('blocker'),
        'formal_figures_A_B_available': audit['cross_subject_status'].startswith('COMPUTED'),
    })
    print(audit.get('blocker', 'Common-space matrices and statistics saved.'), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=REPO / 'data/things-eeg/Preprocessed_data_250Hz_whiten')
    p.add_argument('--ckpt-dir', type=Path, default=REPO / 'archive/legacy_paper/checkpoints/predictor/THINGSEEG2/Ours/full')
    p.add_argument('--feature-cache', type=Path, default=REPO / 'data/things-eeg/Image_feature/FoveaBlur/generator_fovea_medium_k51_RN50_test.pt')
    p.add_argument('--paper-raw', type=Path, default=REPO / 'archive/legacy_paper/analysis/raw')
    p.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--metric-atol', type=float, default=1e-4,
                   help='Numerical reproduction tolerance for r (not bitwise equality)')
    p.add_argument('--prediction-rtol', type=float, default=1e-3,
                   help='Maximum prediction relative L2 error against the paper cache')
    p.add_argument('--common-space-bundle', type=Path)
    args = p.parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.metric_atol < 0 or args.prediction_rtol < 0:
        p.error('threads/batch-size must be positive and tolerances nonnegative')
    run(args)


if __name__ == '__main__':
    main()
