"""
Convert Alljoined-1.6M official preprocessed flat EEG + stim_order.parquet
to UBP-compatible train.pt / test.pt.

Input (per subject, from HuggingFace preprocessed_eeg/sub-XX/):
  - preprocessed_eeg_training_flat.npy  (pickled dict, key preprocessed_eeg_data)
  - preprocessed_eeg_test_flat.npy
  - stim_order.parquet

Output (mirrors THINGS-EEG layout):
  - train.pt: eeg (16540, 4, 32, 250), trial_format repeated
  - test.pt:  eeg (200, 80, 32, 250)

Images are mapped to in-repo THINGS paths under Image_set_Resize via
experiment_metadata_categories.parquet + folder scan.

Usage:
  python preprocess/convert_alljoined.py --subject 1
  python preprocess/convert_alljoined.py --subject 1 --verify
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
import torch

from encoder.data import ALLJOINED_CHANNELS
from encoder.paths import eeg_data_root, repo_root

RE_SFREQ = 250.0
POST_STIM_SAMPLES = 250
N_TRAIN_IMAGES = 16540
N_TEST_IMAGES = 200
TRAIN_REPS = 4
TEST_REPS = 80
DEFAULT_EEG_ROOT = eeg_data_root()
DEFAULT_REPO_ROOT = str(repo_root())


def get_args():
    p = argparse.ArgumentParser('Convert Alljoined-1.6M preprocessed EEG to UBP .pt')
    p.add_argument('--subject', type=int, required=True, help='Subject number (1-20)')
    p.add_argument(
        '--hf_root',
        default=None,
        help='HF download root containing preprocessed_eeg/ (default: $UBP_EEG_DATA_ROOT/alljoined-1.6M/raw_hf)',
    )
    p.add_argument(
        '--output_dir',
        default=None,
        help='UBP .pt output base (default: $UBP_EEG_DATA_ROOT/alljoined-1.6M/ubp_preprocessed)',
    )
    p.add_argument(
        '--image_root',
        default=None,
        help='THINGS Image_set_Resize root (default: repo data/things-eeg/Image_set_Resize)',
    )
    p.add_argument(
        '--categories_parquet',
        default=None,
        help='experiment_metadata_categories.parquet (default: under hf_root/preprocessed_eeg/)',
    )
    p.add_argument('--verify', action='store_true', help='Check image paths exist after conversion')
    p.add_argument('--force', action='store_true', help='Overwrite existing .pt files')
    return p.parse_args()


def _load_flat_eeg(path: str) -> Tuple[np.ndarray, Optional[List[str]], Optional[np.ndarray]]:
    """Load official Alljoined flat npy (pickled dict inside .npy)."""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict):
        eeg = np.asarray(raw['preprocessed_eeg_data'])
        ch_names = raw.get('ch_names')
        times = raw.get('times')
    else:
        eeg = np.asarray(raw)
        ch_names, times = None, None

    if eeg.ndim != 3:
        raise ValueError(f'Expected (trials, channels, time), got {eeg.shape} from {path}')

    # Official preprocessing already crops to 250 post-stim samples; guard anyway.
    if eeg.shape[-1] > POST_STIM_SAMPLES:
        eeg = eeg[..., -POST_STIM_SAMPLES:]
    elif eeg.shape[-1] < POST_STIM_SAMPLES:
        raise ValueError(f'Expected >={POST_STIM_SAMPLES} time samples, got {eeg.shape[-1]}')

    return eeg, ch_names, times


def _build_image_lookup(image_root: Path) -> Dict[Tuple[str, int, int], str]:
    """Map (partition, category_num, category_img_num) -> THINGS-relative path."""
    lookup: Dict[Tuple[str, int, int], str] = {}
    for split, part in (('train', 'stim_train'), ('test', 'stim_test')):
        base = image_root / f'{split}_images'
        if not base.is_dir():
            raise FileNotFoundError(f'Missing THINGS image directory: {base}')
        for folder in sorted(os.listdir(base)):
            m = re.match(r'(\d+)_(.+)', folder)
            if not m:
                continue
            cat_num = int(m.group(1)) - 1
            img_dir = base / folder
            files = sorted(
                f for f in os.listdir(img_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            for img_idx, fn in enumerate(files):
                lookup[(part, cat_num, img_idx)] = f'{split}_images/{folder}/{fn}'
    return lookup


def _resolve_image_path(
    row: pd.Series,
    lookup: Dict[Tuple[str, int, int], str],
) -> str:
    key = (str(row['partition']), int(row['category_num']), int(row['category_img_num']))
    if key not in lookup:
        raise KeyError(
            f'No THINGS path for partition={key[0]} category_num={key[1]} '
            f'category_img_num={key[2]} (image_path={row.get("image_path")})'
        )
    return lookup[key]


def _text_from_path(rel_path: str) -> str:
    folder = rel_path.split('/')[1]
    return folder.split('_', 1)[1] if '_' in folder else folder


def _align_trials(
    eeg: np.ndarray,
    meta: pd.DataFrame,
    partition: str,
) -> pd.DataFrame:
    """Align stim_order rows with flat EEG trial count (drops excess metadata rows)."""
    df = meta[meta['partition'] == partition].copy()
    df = df.reset_index(drop=True)
    n_eeg, n_meta = eeg.shape[0], len(df)
    if n_meta < n_eeg:
        raise ValueError(
            f'partition={partition}: stim_order has {n_meta} rows but EEG has {n_eeg} trials'
        )
    if n_meta > n_eeg:
        df = df.iloc[:n_eeg].reset_index(drop=True)
    return df


def _group_repeated(
    eeg: np.ndarray,
    meta: pd.DataFrame,
    group_cols: List[str],
    max_reps: int,
    image_lookup: Dict[Tuple[str, int, int], str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group flat trials into (n_groups, max_reps, C, T) like THINGS .pt."""
    groups = meta.groupby(group_cols, sort=True)
    n_groups = groups.ngroups
    c_num, seq_len = eeg.shape[1], eeg.shape[2]
    out = np.zeros((n_groups, max_reps, c_num, seq_len), dtype=np.float32)
    img_list: List[List[str]] = []
    label_list: List[int] = []
    text_list: List[str] = []
    session_list = np.zeros((n_groups, max_reps), dtype=np.float32)

    sorted_keys = sorted(groups.groups.keys())
    for gi, key in enumerate(sorted_keys):
        idx_arr = np.sort(np.asarray(groups.groups[key]))
        if len(idx_arr) > max_reps:
            idx_arr = idx_arr[:max_reps]
        n = len(idx_arr)
        out[gi, :n] = eeg[idx_arr]
        row0 = meta.iloc[idx_arr[0]]
        rel = _resolve_image_path(row0, image_lookup)
        img_list.append([rel] * max_reps)
        label_list.append(int(row0['category_num']))
        text_list.append(_text_from_path(rel))
        for ri, trial_i in enumerate(idx_arr):
            session_list[gi, ri] = float(meta.iloc[trial_i]['session'])

    img_arr = np.array(img_list, dtype=object)
    label_arr = np.tile(np.array(label_list, dtype=np.int64)[:, None], (1, max_reps))
    text_arr = np.array([[t] * max_reps for t in text_list], dtype=object)

    return out, label_arr, img_arr, text_arr, session_list


def _save_pt(
    out_path: Path,
    eeg: np.ndarray,
    label: np.ndarray,
    img: np.ndarray,
    text: np.ndarray,
    session: np.ndarray,
    ch_names: List[str],
    times: np.ndarray,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'eeg': eeg.astype(np.float16),
        'label': label,
        'img': img,
        'text': text,
        'session': session,
        'ch_names': ch_names,
        'times': times,
        'trial_format': 'repeated',
    }
    torch.save(payload, out_path, pickle_protocol=5)
    print(f'Wrote {out_path}  eeg={payload["eeg"].shape}  dtype={payload["eeg"].dtype}')


def _verify_paths(img: np.ndarray, image_root: Path, n_check: int = 20):
    flat = img.reshape(-1)
    missing = []
    for p in flat[:n_check]:
        if not (image_root / str(p)).is_file():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError(f'Missing image files under {image_root}: {missing[:5]}')
    print(f'Verified {min(n_check, len(flat))} image paths under {image_root}')


def convert_subject(
    subject: int,
    hf_root: Path,
    output_dir: Path,
    image_root: Path,
    categories_parquet: Path,
    verify: bool = False,
    force: bool = False,
):
    sub_tag = f'sub-{subject:02d}'
    sub_hf = hf_root / 'preprocessed_eeg' / sub_tag
    train_npy = sub_hf / 'preprocessed_eeg_training_flat.npy'
    test_npy = sub_hf / 'preprocessed_eeg_test_flat.npy'
    stim_parquet = sub_hf / 'stim_order.parquet'

    for path in (train_npy, test_npy, stim_parquet, categories_parquet):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')

    out_sub = output_dir / sub_tag
    train_pt = out_sub / 'train.pt'
    test_pt = out_sub / 'test.pt'
    if not force and train_pt.is_file() and test_pt.is_file():
        print(f'SKIP {sub_tag}: {train_pt} and {test_pt} exist (use --force)')
        return

    image_lookup = _build_image_lookup(image_root)
    stim = pd.read_parquet(stim_parquet)

    eeg_train, ch_names, times = _load_flat_eeg(str(train_npy))
    eeg_test, _, _ = _load_flat_eeg(str(test_npy))

    ch_names = list(ch_names) if ch_names is not None else list(ALLJOINED_CHANNELS)
    if times is None:
        times = np.arange(POST_STIM_SAMPLES, dtype=np.float64) / RE_SFREQ

    meta_train = _align_trials(eeg_train, stim, 'stim_train')
    meta_test = _align_trials(eeg_test, stim, 'stim_test')

    group_cols = ['category_num', 'category_img_num']
    train_eeg, train_label, train_img, train_text, train_sess = _group_repeated(
        eeg_train, meta_train, group_cols, TRAIN_REPS, image_lookup,
    )
    test_eeg, test_label, test_img, test_text, test_sess = _group_repeated(
        eeg_test, meta_test, group_cols, TEST_REPS, image_lookup,
    )

    if train_eeg.shape[0] != N_TRAIN_IMAGES:
        print(
            f'WARNING {sub_tag}: train groups={train_eeg.shape[0]} '
            f'(expected {N_TRAIN_IMAGES})'
        )
    if test_eeg.shape[0] != N_TEST_IMAGES:
        print(
            f'WARNING {sub_tag}: test groups={test_eeg.shape[0]} '
            f'(expected {N_TEST_IMAGES})'
        )

    _save_pt(train_pt, train_eeg, train_label, train_img, train_text, train_sess, ch_names, times)
    _save_pt(test_pt, test_eeg, test_label, test_img, test_text, test_sess, ch_names, times)

    if verify:
        _verify_paths(train_img, image_root)
        _verify_paths(test_img, image_root)


def main():
    args = get_args()
    eeg_root = Path(DEFAULT_EEG_ROOT)
    hf_root = Path(args.hf_root or eeg_root / 'alljoined-1.6M' / 'raw_hf')
    output_dir = Path(args.output_dir or eeg_root / 'alljoined-1.6M' / 'ubp_preprocessed')
    image_root = Path(args.image_root or Path(DEFAULT_REPO_ROOT) / 'data/things-eeg/Image_set_Resize')
    categories_parquet = Path(
        args.categories_parquet or hf_root / 'preprocessed_eeg' / 'experiment_metadata_categories.parquet'
    )

    convert_subject(
        subject=args.subject,
        hf_root=hf_root,
        output_dir=output_dir,
        image_root=image_root,
        categories_parquet=categories_parquet,
        verify=args.verify,
        force=args.force,
    )


if __name__ == '__main__':
    main()
