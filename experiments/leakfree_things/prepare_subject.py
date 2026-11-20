"""Freeze the shared image split and refit per-session MVNN on training images only.

The 740 validation images are the seed-2023 prefix used by the original
generator. They are removed from the encoder training set and from every
session covariance. Test images never enter the covariance. Historical
whitening (covariance on all 16,540 training images) is checked against the
stored test.pt before the restricted transform is kept.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import time
import types
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
from threadpoolctl import threadpool_limits

from experiments.cross_subject_eeg.recover_preprocessing import training_covariance
from preprocess import process_eeg_whiten as preprocessing

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('EEG_RUN_ROOT', str(REPO / 'results' / 'things_eeg2_leakfree')))
if not OUT.is_absolute():
    OUT = REPO / OUT
OFFICIAL = REPO / 'data' / 'things-eeg' / 'Preprocessed_data_250Hz_whiten'
def raw_eeg_root() -> Path:
    value = os.environ.get('THINGS_EEG_RAW_ROOT')
    if not value:
        raise RuntimeError('Set THINGS_EEG_RAW_ROOT to the THINGS-EEG2 raw_eeg directory')
    return Path(value)
IMG_ROOT = REPO / 'data' / 'things-eeg' / 'Image_set_Resize'
OLD_FEAT = REPO / 'data' / 'things-eeg' / 'Image_feature' / 'FoveaBlur'
SPLIT_DIR = OUT / 'split'
N_TOTAL = 16540
N_VAL = 740
N_TRAIN = 15800
N_TEST = 200
SEED = 2023


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


def _pickle_module():
    module = types.ModuleType('pickle_compat')
    module.Unpickler = _CompatUnpickler
    for attr in (
        'dump', 'dumps', 'load', 'loads', 'HIGHEST_PROTOCOL', 'Pickler',
        'PicklingError', 'UnpicklingError', 'DEFAULT_PROTOCOL',
    ):
        if hasattr(pickle, attr):
            setattr(module, attr, getattr(pickle, attr))
    return module


PICKLE = _pickle_module()


def load_pt(path):
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=PICKLE)


def image_column(payload):
    return np.asarray(payload['img'])[:, 0].astype(str)


def official_train_images():
    payload = load_pt(OFFICIAL / 'sub-01' / 'train.pt')
    images = image_column(payload)
    if len(images) != N_TOTAL:
        raise ValueError(f'Official train images are {len(images)}, expected {N_TOTAL}')
    meta, _, _ = preprocessing.collect_image_metadata(
        str(IMG_ROOT / 'train_images'), str(IMG_ROOT),
    )
    if list(meta) != list(images):
        raise ValueError('Image-folder order does not match official train.pt')
    return images


def ensure_split(images):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / 'image_split.npz'
    perm = np.random.default_rng(SEED).permutation(N_TOTAL)
    val_index = np.asarray(perm[:N_VAL], dtype=np.int64)
    train_index = np.asarray(perm[N_VAL:], dtype=np.int64)
    if path.is_file():
        saved = np.load(path, allow_pickle=False)
        if not np.array_equal(saved['train_index'], train_index) or not np.array_equal(saved['val_index'], val_index):
            raise ValueError(f'Existing split differs from seed {SEED}: {path}')
    else:
        test_images = image_column(load_pt(OFFICIAL / 'sub-01' / 'test.pt'))
        if len(test_images) != N_TEST:
            raise ValueError(f'Official test images are {len(test_images)}, expected {N_TEST}')
        train_images = images[train_index]
        val_images = images[val_index]
        if set(train_images) & set(val_images) or set(train_images) & set(test_images) or set(val_images) & set(test_images):
            raise ValueError('Train, validation, and test image paths overlap')
        np.savez(
            path,
            seed=np.int64(SEED),
            train_index=train_index,
            val_index=val_index,
            train_images=train_images,
            val_images=val_images,
            test_images=test_images,
        )
        meta = {
            'seed': SEED,
            'rule': 'np.random.default_rng(2023).permutation(16540); val=prefix 740; train=the other 15800',
            'n_train': N_TRAIN,
            'n_val': N_VAL,
            'n_test': N_TEST,
            'pairwise_disjoint': True,
            'shared_by': 'encoder and generator',
            'test_source': 'official THINGS-EEG2 test.pt order, not carved from train',
        }
        (SPLIT_DIR / 'image_split.json').write_text(json.dumps(meta, indent=2) + '\n')
        print(f'Wrote {path}')
    saved = np.load(path, allow_pickle=False)
    return saved['train_index'], saved['val_index'], saved['test_images'].astype(str)


def epoch_one(mode, subject, session, stage_root):
    stage = stage_root / f'sub-{subject:02d}' / 'ses-01'
    if stage.parent.exists():
        shutil.rmtree(stage.parent)
    stage.mkdir(parents=True)
    source = raw_eeg_root() / f'sub-{subject:02d}' / f'ses-{session:02d}'
    for name in source.iterdir():
        os.symlink(name, stage / name.name)
    previous = preprocessing.N_SESSIONS
    preprocessing.N_SESSIONS = 1
    try:
        epoched, conditions, channels, times = preprocessing.epoch_data(
            mode, subject, str(stage_root),
        )
    finally:
        preprocessing.N_SESSIONS = previous
        shutil.rmtree(stage.parent)
    return epoched[0], conditions[0], list(channels), times


def _real_matrix_power(matrix, exponent):
    out = scipy.linalg.fractional_matrix_power(matrix, exponent)
    if np.iscomplexobj(out):
        if np.max(np.abs(out.imag)) > 1e-8:
            raise ValueError(f'Complex whitening, imag max {np.max(np.abs(out.imag))}')
        out = out.real
    return np.asarray(out, dtype=np.float64)


def historical_match(whitened_test, stored_test, session):
    stored = np.asarray(stored_test['eeg'])[:, np.asarray(stored_test['session'])[0] == session - 1]
    if stored.shape != whitened_test.shape:
        raise ValueError(f'session {session}: shape {whitened_test.shape} != stored {stored.shape}')
    quantized = whitened_test.astype(np.float16)
    match = float(np.mean(quantized == stored))
    denom = np.linalg.norm(stored.astype(np.float64))
    relative = float(np.linalg.norm(whitened_test - stored.astype(np.float64)) / denom)
    passed = bool(match >= 0.99 and relative <= 5e-4)
    return {
        'session': session,
        'float16_exact_match_fraction': match,
        'whitened_vs_stored_relative_l2': relative,
        'passed': passed,
    }


def save_payload(path, eeg, sessions, images, labels, texts, channels, times):
    repeats = eeg.shape[1]
    images = np.asarray(images)
    payload = {
        'eeg': np.asarray(eeg, dtype=np.float16),
        'label': np.tile(np.asarray(labels)[:, None], (1, repeats)),
        'img': np.tile(images[:, None], (1, repeats)),
        'text': np.tile(np.asarray(texts)[:, None], (1, repeats)),
        'session': sessions,
        'ch_names': channels,
        'times': times,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path, pickle_protocol=5)
    print(f'Wrote {path} eeg={payload["eeg"].shape} dtype={payload["eeg"].dtype}', flush=True)


def link_features(train_images, val_images, test_images):
    enc_dir = OUT / 'encoder_features' / 'FoveaBlur'
    gen_dir = OUT / 'generator_features'
    enc_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)
    old_train = load_pt(OLD_FEAT / 'eeg_intra-subject_fixed_fovea_EEGProjectLayer_RN50_train.pt')
    old_test = load_pt(OLD_FEAT / 'eeg_intra-subject_fixed_fovea_EEGProjectLayer_RN50_test.pt')
    train_keys = set(old_train['img_features'])
    missing = [p for p in list(train_images) + list(val_images) if p not in train_keys]
    if missing:
        raise ValueError(f'{len(missing)} train/val images missing from the frozen CLIP cache, e.g. {missing[:3]}')
    test_keys = set(old_test['img_features'])
    missing_test = [p for p in test_images if p not in test_keys]
    if missing_test:
        raise ValueError(f'{len(missing_test)} test images missing from the frozen CLIP cache')
    # Re-save instead of symlinking: the historical caches were pickled with
    # NumPy 2 and the training loaders do not use a compatibility unpickler.
    for mode, payload in (('train', old_train), ('val', old_train), ('test', old_test)):
        dest = enc_dir / f'THINGSEEG2_leakfree_{mode}.pt'
        if not dest.exists():
            torch.save(payload, dest)
    gen_train = load_pt(OLD_FEAT / 'generator_fovea_medium_k51_RN50_train.pt')
    gen_test = load_pt(OLD_FEAT / 'generator_fovea_medium_k51_RN50_test.pt')
    gen_train_keys = set(gen_train['img_features'])
    missing_gen = [p for p in list(train_images) + list(val_images) if p not in gen_train_keys]
    if missing_gen:
        raise ValueError(f'{len(missing_gen)} images missing from the generator CLIP cache')
    for name, payload in (
        ('generator_fovea_medium_k51_RN50_train.pt', gen_train),
        ('generator_fovea_medium_k51_RN50_test.pt', gen_test),
    ):
        dest = gen_dir / name
        if not dest.exists():
            torch.save(payload, dest)
    val_path = gen_dir / 'generator_fovea_medium_k51_RN50_val.pt'
    if not val_path.exists():
        val_feats = {p: gen_train['img_features'][p] for p in val_images}
        torch.save({'img_features': val_feats}, val_path)
    print(f'CLIP caches linked under {OUT}', flush=True)


def prepare(subject):
    if not str(OUT).endswith('things_eeg2_leakfree'):
        raise RuntimeError(OUT)
    images = official_train_images()
    official_subject = image_column(load_pt(OFFICIAL / f'sub-{subject:02d}' / 'train.pt'))
    if not np.array_equal(official_subject, images):
        raise ValueError(f'sub-{subject:02d} train image order differs from sub-01')
    train_index, val_index, test_images = ensure_split(images)
    train_set = set(train_index.tolist())
    val_set = set(val_index.tolist())
    subject_dir = OUT / 'preprocessed' / f'sub-{subject:02d}'
    manifest_path = OUT / 'mvnn' / f'sub-{subject:02d}' / 'manifest.json'
    if (subject_dir / 'train.pt').is_file() and (subject_dir / 'val.pt').is_file() and (subject_dir / 'test.pt').is_file() and manifest_path.is_file():
        print(f'sub-{subject:02d} preprocessed files already exist; leaving them in place', flush=True)
        saved = np.load(SPLIT_DIR / 'image_split.npz', allow_pickle=False)
        link_features(saved['train_images'].astype(str), saved['val_images'].astype(str), test_images)
        return
    stored_test = load_pt(OFFICIAL / f'sub-{subject:02d}' / 'test.pt')
    meta_images, meta_labels, meta_texts = preprocessing.collect_image_metadata(
        str(IMG_ROOT / 'train_images'), str(IMG_ROOT),
    )
    test_meta_images, test_labels, test_texts = preprocessing.collect_image_metadata(
        str(IMG_ROOT / 'test_images'), str(IMG_ROOT),
    )
    if list(test_meta_images) != list(test_images):
        raise ValueError('Test image-folder order does not match official test.pt')
    all_conditions = []
    whitened_train, whitened_test, conditions = [], [], []
    channels = times = None
    session_meta = []
    mvnn_dir = OUT / 'mvnn' / f'sub-{subject:02d}'
    mvnn_dir.mkdir(parents=True, exist_ok=True)
    stage_root = OUT / '_epoch_stage' / f'sub-{subject:02d}'
    stage_root.mkdir(parents=True, exist_ok=True)
    for session in range(1, 5):
        started = time.perf_counter()
        print(f'sub-{subject:02d} session {session}: epoching', flush=True)
        train_epochs, train_conditions, channels, times = epoch_one('train', subject, session, stage_root)
        image_rows = train_conditions.astype(int) - 1
        if set(image_rows.tolist()) - (train_set | val_set):
            raise ValueError(f'session {session} contains an image outside the 16540 training images')
        fit_mask = np.isin(image_rows, train_index)
        val_mask = np.isin(image_rows, val_index)
        if int(fit_mask.sum() + val_mask.sum()) != len(image_rows):
            raise ValueError(f'session {session} fit/val partition is incomplete')
        with threadpool_limits(limits=8):
            cov_all, err_all = training_covariance(train_epochs)
            cov_fit, err_fit = training_covariance(train_epochs[fit_mask])
        hist_w = _real_matrix_power(cov_all, -0.5)
        fit_w = _real_matrix_power(cov_fit, -0.5)
        test_epochs, test_conditions, test_channels, _ = epoch_one('test', subject, session, stage_root)
        if test_channels != channels or not np.array_equal(test_conditions, np.arange(1, N_TEST + 1)):
            raise ValueError(f'session {session} test events are not 1..200')
        hist = historical_match(
            preprocessing._whiten_partition(test_epochs, hist_w), stored_test, session,
        )
        hist.update({
            'covariance_reference_relative_error': err_all,
            'n_training_conditions_all': int(len(image_rows)),
            'n_fit_conditions': int(fit_mask.sum()),
            'n_excluded_val_conditions': int(val_mask.sum()),
        })
        print(f'sub-{subject:02d} session {session} historical match {hist}', flush=True)
        if not hist['passed']:
            (mvnn_dir / f'ses-{session:02d}.failed.json').write_text(json.dumps(hist, indent=2) + '\n')
            raise RuntimeError(f'sub-{subject:02d} session {session} does not reproduce stored test.pt')
        whitened_train.append(preprocessing._whiten_partition(train_epochs, fit_w))
        whitened_test.append(preprocessing._whiten_partition(test_epochs, fit_w))
        conditions.append(train_conditions)
        all_conditions.append(image_rows)
        np.savez_compressed(
            mvnn_dir / f'ses-{session:02d}.npz',
            training_covariance=cov_fit,
            historical_covariance=cov_all,
            whitening=fit_w,
            fit_image_index=image_rows[fit_mask],
            excluded_val_image_index=image_rows[val_mask],
        )
        hist['restricted_covariance_reference_relative_error'] = err_fit
        hist['elapsed_s'] = time.perf_counter() - started
        session_meta.append(hist)
        del train_epochs, test_epochs, cov_all, cov_fit
        gc.collect()
    covered = set(np.concatenate(all_conditions).tolist())
    if covered != set(range(N_TOTAL)):
        raise ValueError(f'training image rows cover {len(covered)} ids, expected {N_TOTAL}')
    merged_train, train_sessions = preprocessing.merge_train_sessions(whitened_train, conditions)
    if merged_train.shape[0] != N_TOTAL:
        raise ValueError(f'merged train images {merged_train.shape[0]} != {N_TOTAL}')
    merged_test, test_sessions = preprocessing.merge_test_sessions(whitened_test)
    save_payload(
        subject_dir / 'train.pt', merged_train[train_index], train_sessions[train_index],
        [meta_images[i] for i in train_index], [meta_labels[i] for i in train_index],
        [meta_texts[i] for i in train_index], channels, times,
    )
    save_payload(
        subject_dir / 'val.pt', merged_train[val_index], train_sessions[val_index],
        [meta_images[i] for i in val_index], [meta_labels[i] for i in val_index],
        [meta_texts[i] for i in val_index], channels, times,
    )
    save_payload(
        subject_dir / 'test.pt', merged_test, test_sessions,
        test_meta_images, test_labels, test_texts, channels, times,
    )
    del merged_train, merged_test
    manifest = {
        'subject': f'sub-{subject:02d}',
        'mvnn': 'original per-session fractional-power MVNN; covariance fit on the 15800 training images only',
        'excluded_from_fit': '740 validation images and 200 test images',
        'historical_all_16540_covariance_reproduces_stored_test': True,
        'sessions': session_meta,
        'shapes': {'train': [N_TRAIN, 4, 63, 250], 'val': [N_VAL, 4, 63, 250], 'test': [N_TEST, 80, 63, 250]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    saved = np.load(SPLIT_DIR / 'image_split.npz', allow_pickle=False)
    link_features(saved['train_images'].astype(str), saved['val_images'].astype(str), test_images)
    print(f'sub-{subject:02d} preprocessing done', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, required=True)
    args = parser.parse_args()
    prepare(args.subject)


if __name__ == '__main__':
    main()
