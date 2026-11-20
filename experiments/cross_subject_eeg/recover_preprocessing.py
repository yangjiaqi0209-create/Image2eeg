"""Recover training-derived session MVNN transforms from the original archives.

Reuses the exact project epoch selection. This does not invent an inverse of a
session-averaged model output, and does not compute cross-subject correlations.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import scipy.linalg
from sklearn.discriminant_analysis import _cov
from threadpoolctl import threadpool_limits

from experiments.cross_subject_eeg.run import REPO, digest, load_payload, write_json
from preprocess import process_eeg_whiten as preprocessing


def batched_auto_cov(trials):
    """Vectorized sklearn _cov(X, shrinkage='auto'), with time observations.

    Input [trial, channel, time]. Implements StandardScaler, Ledoit-Wolf, and
    rescaling; compares against the actual sklearn function before any recovery.
    """
    x = np.asarray(trials, dtype=np.float64).transpose(0, 2, 1)
    x = x - x.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(x * x, axis=1))
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError('Constant or invalid training EEG channel')
    z = x / scale[:, None, :]
    z = z - z.mean(axis=1, keepdims=True)
    n, c = z.shape[1:]
    empirical = z.transpose(0, 2, 1) @ z / n
    trace = np.diagonal(empirical, axis1=1, axis2=2).sum(axis=1)
    mu = trace / c
    delta_sum = np.square(empirical).sum(axis=(1, 2))
    beta_sum = np.square(np.square(z).sum(axis=2)).sum(axis=1)
    beta = (beta_sum / n - delta_sum) / (c * n)
    delta = (delta_sum - 2 * mu * trace + c * mu**2) / c
    beta = np.minimum(beta, delta)
    shrinkage = np.divide(beta, delta, out=np.zeros_like(beta), where=beta != 0)
    covariance = empirical * (1 - shrinkage[:, None, None])
    covariance[:, np.arange(c), np.arange(c)] += (shrinkage * mu)[:, None]
    return covariance * scale[:, :, None] * scale[:, None, :]


def training_covariance(epochs, batch_size=128):
    trials = epochs.reshape(-1, *epochs.shape[-2:])
    first = trials[:32]
    reference = np.stack([_cov(trial.T, shrinkage='auto') for trial in first])
    fast = batched_auto_cov(first)
    relative_error = float(np.linalg.norm(fast - reference) / np.linalg.norm(reference))
    if relative_error > 1e-10:
        raise ValueError(f'Vectorized covariance differs from sklearn: {relative_error}')
    total = np.zeros_like(reference[0])
    for i in range(0, len(trials), batch_size):
        total += batched_auto_cov(trials[i:i + batch_size]).sum(axis=0)
    return total / len(trials), relative_error


def unpack_member(archive, original_name, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(original_name) as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def recover_session(archive_path, subject, session, work_dir, processed, destination):
    started = time.perf_counter()
    # The project epoch function loops from ses-01. A private staging directory
    # maps the selected original session to that name without changing sources.
    prefix = f'{subject}/ses-{session:02d}'
    stage = work_dir / subject / 'ses-01'
    test_path = stage / 'raw_eeg_test.npy'
    train_path = stage / 'raw_eeg_training.npy'
    with zipfile.ZipFile(archive_path) as archive:
        unpack_member(archive, f'{prefix}/raw_eeg_test.npy', test_path)
        unpack_member(archive, f'{prefix}/raw_eeg_training.npy', train_path)
    previous = preprocessing.N_SESSIONS
    preprocessing.N_SESSIONS = 1
    try:
        train, conditions, channels, train_times = preprocessing.epoch_data(
            'train', int(subject[-2:]), str(work_dir))
        covariance, covariance_error = training_covariance(train[0])
        n_training_conditions = len(conditions[0])
        del train, conditions
        gc.collect()
        test, test_conditions, test_channels, times = preprocessing.epoch_data(
            'test', int(subject[-2:]), str(work_dir))
    finally:
        preprocessing.N_SESSIONS = previous
    if channels != list(processed['ch_names']) or channels != test_channels:
        raise ValueError('Recovered training / test channels differ from existing payload')
    if not np.array_equal(test_conditions[0], np.arange(1, 201)):
        raise ValueError('Raw test events are not the expected ordered 200 image IDs')
    if not np.allclose(times[-250:], np.arange(250) / 250, atol=1e-7):
        raise ValueError('Recovered time grid differs')
    whitening = scipy.linalg.fractional_matrix_power(covariance, -.5)
    inverse = scipy.linalg.fractional_matrix_power(covariance, .5)
    if np.iscomplexobj(whitening) or np.iscomplexobj(inverse):
        raise ValueError('Training covariance produced complex transforms')
    raw_test = test[0]
    whitened = preprocessing._whiten_partition(raw_test, whitening)
    stored = np.asarray(processed['eeg'])[:, np.asarray(processed['session'])[0] == session - 1]
    if stored.shape != whitened.shape:
        raise ValueError('Recovered test trial structure differs')
    quantized = whitened.astype(np.float16)
    # Validate actual historical whitening and trial choice, before using W.
    quantization_match = float(np.mean(quantized == stored))
    error_norm = np.linalg.norm(whitened - stored.astype(np.float64))
    relative_error = float(error_norm / np.linalg.norm(stored.astype(np.float64)))
    passed = bool(quantization_match >= .99 and relative_error <= 5e-4)
    meta = {
        'subject': subject, 'session': session,
        'archive': str(archive_path), 'archive_member_prefix': prefix,
        'n_training_conditions': n_training_conditions,
        'n_training_trials': n_training_conditions * 2,
        'covariance_reference_relative_error': covariance_error,
        'float16_exact_match_fraction': quantization_match,
        'whitened_vs_stored_relative_l2': relative_error,
        'historical_preprocessing_match_passed': passed,
        'elapsed_s': time.perf_counter() - started,
        'raw_test_units': 'same values as MNE epoch_data output; no further rescaling',
        'normalization': 'Project baseline correction and training-derived MVNN only; no added z-score',
    }
    if not passed:
        write_json(destination.with_suffix('.failed.json'), meta)
        raise ValueError(f'{subject} session {session}: historical preprocessing mismatch {meta}')
    np.savez_compressed(
        destination, training_covariance=covariance, whitening=whitening, inverse=inverse,
        raw_test_mean=raw_test.mean(axis=1), whitened_test_mean=whitened.mean(axis=1),
        channels=np.array(channels), times_s=times[-250:],
        image_ids=np.asarray(processed['img'])[:, 0].astype(str),
        metadata_json=np.array(json.dumps(meta)),
    )
    write_json(destination.with_suffix('.json'), meta)
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--sessions', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_hash = digest(Path(__file__))
    for sub in args.subjects:
        subject = f'sub-{sub:02d}'
        archive_path = args.archive_dir / f'{subject}.zip'
        processed = load_payload(REPO / 'data/things-eeg/Preprocessed_data_250Hz_whiten' / subject / 'test.pt')
        subject_dir = args.output / subject
        subject_dir.mkdir(exist_ok=True)
        for session in args.sessions:
            dest = subject_dir / f'session-{session:02d}.npz'
            if dest.exists():
                print(f'RECOVERY: {subject} session {session} already saved; preserving', flush=True)
                continue
            print(f'RECOVERY: {subject} session {session} starting', flush=True)
            with tempfile.TemporaryDirectory(prefix='eeg_recovery_', dir=args.output) as tmp:
                with threadpool_limits(limits=args.threads):
                    meta = recover_session(archive_path, subject, session, Path(tmp), processed, dest)
            meta['recovery_code_sha256'] = source_hash
            write_json(dest.with_suffix('.json'), meta)
            print(f'RECOVERY: {subject} session {session} PASSED; match={meta["float16_exact_match_fraction"]:.6f}; seconds={meta["elapsed_s"]:.1f}', flush=True)
        del processed
        gc.collect()


if __name__ == '__main__':
    main()
