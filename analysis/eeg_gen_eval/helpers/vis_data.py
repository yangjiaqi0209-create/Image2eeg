"""Compute arrays for section-III visualization experiments."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from analysis.eeg_gen_eval.config import CHANNELS, DATA_DIR, RAW_DIR, SFREQ
from analysis.eeg_gen_eval.compute.metrics import per_channel_pearson

VIS_DIR = os.path.join(RAW_DIR, 'visualization')
KEY_TIMEPOINTS_MS = [50, 100, 150, 200, 250]


def ms_to_idx(ms: float) -> int:
    return int(round(ms * SFREQ / 1000.0))


def _best_example_index(y_pred: np.ndarray, y_true: np.ndarray) -> int:
    """Image index with highest mean channel-wise time Pearson r."""
    n = y_true.shape[0]
    scores = []
    for i in range(n):
        rs = []
        for c in range(y_true.shape[1]):
            yt = y_true[i, c] - y_true[i, c].mean()
            yp = y_pred[i, c] - y_pred[i, c].mean()
            den = np.linalg.norm(yt) * np.linalg.norm(yp)
            if den > 1e-12:
                rs.append(float(np.dot(yt, yp) / den))
        scores.append(float(np.mean(rs)) if rs else -1.0)
    return int(np.argmax(scores))


def _test_image_relpath(sub: int, image_index: int) -> str:
    test_path = os.path.join(DATA_DIR, f'sub-{sub:02d}', 'test.pt')
    data = __import__('torch').load(test_path, map_location='cpu', weights_only=False)
    img = np.asarray(data['img'])
    return str(img[image_index, 0] if img.ndim > 1 else img[image_index])


def compute_subject_visualization(sub: int) -> Dict:
    sub_tag = f'sub-{sub:02d}'
    src = os.path.join(RAW_DIR, sub_tag)
    y_pred = np.load(os.path.join(src, 'y_pred.npy'))
    y_true = np.load(os.path.join(src, 'y_true.npy'))

    erp_true = y_true.mean(axis=0)
    erp_pred = y_pred.mean(axis=0)
    error_map = np.abs(y_true - y_pred).mean(axis=0)
    per_ch = per_channel_pearson(y_pred, y_true)

    ex_idx = _best_example_index(y_pred, y_true)
    time_idx = {f'{ms} ms': ms_to_idx(ms) for ms in KEY_TIMEPOINTS_MS}
    topo_true = {k: erp_true[:, idx] for k, idx in time_idx.items()}
    topo_pred = {k: erp_pred[:, idx] for k, idx in time_idx.items()}

    out_dir = os.path.join(VIS_DIR, sub_tag)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'erp_true_mean.npy'), erp_true)
    np.save(os.path.join(out_dir, 'erp_pred_mean.npy'), erp_pred)
    np.save(os.path.join(out_dir, 'error_map_mean.npy'), error_map)
    np.save(os.path.join(out_dir, 'per_channel_pearson.npy'), per_ch)
    np.save(os.path.join(out_dir, 'example_y_true.npy'), y_true[ex_idx])
    np.save(os.path.join(out_dir, 'example_y_pred.npy'), y_pred[ex_idx])

    meta = {
        'subject': sub_tag,
        'n_test_images': int(y_true.shape[0]),
        'example_image_index': ex_idx,
        'example_image_relpath': _test_image_relpath(sub, ex_idx),
        'example_mean_channel_r': float(per_ch.mean()),
        'key_timepoints_ms': KEY_TIMEPOINTS_MS,
        'key_timepoints_idx': time_idx,
        'channel_indices_display': {
            ch: CHANNELS.index(ch) for ch in ['Fp1', 'Cz', 'Pz', 'Oz']
        },
    }
    for k, v in topo_true.items():
        np.save(os.path.join(out_dir, f'topo_true_{k.replace(" ", "")}.npy'), v)
        np.save(os.path.join(out_dir, f'topo_pred_{k.replace(" ", "")}.npy'), topo_pred[k])

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def compute_aggregate_visualization(subs: List[int]) -> Dict:
    erp_t_list, erp_p_list, err_list, per_ch = [], [], [], []
    metas = []
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        d = os.path.join(VIS_DIR, sub_tag)
        if not os.path.isfile(os.path.join(d, 'erp_true_mean.npy')):
            compute_subject_visualization(sub)
        erp_t_list.append(np.load(os.path.join(d, 'erp_true_mean.npy')))
        erp_p_list.append(np.load(os.path.join(d, 'erp_pred_mean.npy')))
        err_list.append(np.load(os.path.join(d, 'error_map_mean.npy')))
        per_ch.append(np.load(os.path.join(d, 'per_channel_pearson.npy')))
        with open(os.path.join(d, 'meta.json')) as f:
            metas.append(json.load(f))

    erp_t = np.stack(erp_t_list).mean(axis=0)
    erp_p = np.stack(erp_p_list).mean(axis=0)
    err = np.stack(err_list).mean(axis=0)
    per_ch_mean = np.stack(per_ch).mean(axis=0)
    per_ch_std = np.stack(per_ch).std(axis=0, ddof=0)

    # Panel a: per-subject test-set average ERP; pick subject with best mean channel r
    sub_mean_r = [float(pc.mean()) for pc in per_ch]
    best_i = int(np.argmax(sub_mean_r))
    best_sub = subs[best_i]
    panel_a_erp_true = erp_t_list[best_i]
    panel_a_erp_pred = erp_p_list[best_i]
    panel_a_per_ch = per_ch[best_i]
    top_idx = np.argsort(panel_a_per_ch)[-6:][::-1]

    out_dir = os.path.join(VIS_DIR, 'aggregate')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'erp_true_mean.npy'), erp_t)
    np.save(os.path.join(out_dir, 'erp_pred_mean.npy'), erp_p)
    np.save(os.path.join(out_dir, 'error_map_mean.npy'), err)
    np.save(os.path.join(out_dir, 'per_channel_pearson_mean.npy'), per_ch_mean)
    np.save(os.path.join(out_dir, 'per_channel_pearson_std.npy'), per_ch_std)
    np.save(os.path.join(out_dir, 'panel_a_erp_true.npy'), panel_a_erp_true)
    np.save(os.path.join(out_dir, 'panel_a_erp_pred.npy'), panel_a_erp_pred)
    np.save(os.path.join(out_dir, 'panel_a_per_channel_pearson.npy'), panel_a_per_ch)

    time_idx = {f'{ms} ms': ms_to_idx(ms) for ms in KEY_TIMEPOINTS_MS}
    for k, idx in time_idx.items():
        np.save(os.path.join(out_dir, f'topo_true_{k.replace(" ", "")}.npy'), erp_t[:, idx])
        np.save(os.path.join(out_dir, f'topo_pred_{k.replace(" ", "")}.npy'), erp_p[:, idx])

    # Representative single-trial: best example across subjects
    best_sub, best_idx, best_score = None, None, -1.0
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        d = os.path.join(VIS_DIR, sub_tag)
        with open(os.path.join(d, 'meta.json')) as f:
            m = json.load(f)
        y_pred = np.load(os.path.join(RAW_DIR, sub_tag, 'y_pred.npy'))
        y_true = np.load(os.path.join(RAW_DIR, sub_tag, 'y_true.npy'))
        i = m['example_image_index']
        rs = []
        for c in range(y_true.shape[1]):
            yt = y_true[i, c] - y_true[i, c].mean()
            yp = y_pred[i, c] - y_pred[i, c].mean()
            den = np.linalg.norm(yt) * np.linalg.norm(yp)
            if den > 1e-12:
                rs.append(float(np.dot(yt, yp) / den))
        score = float(np.mean(rs)) if rs else -1.0
        if score > best_score:
            best_score, best_sub, best_idx = score, sub, i

    ex_true = np.load(os.path.join(RAW_DIR, f'sub-{best_sub:02d}', 'y_true.npy'))[best_idx]
    ex_pred = np.load(os.path.join(RAW_DIR, f'sub-{best_sub:02d}', 'y_pred.npy'))[best_idx]
    np.save(os.path.join(out_dir, 'example_y_true.npy'), ex_true)
    np.save(os.path.join(out_dir, 'example_y_pred.npy'), ex_pred)

    img_relpath = _test_image_relpath(best_sub, best_idx)

    meta = {
        'n_subjects': len(subs),
        'subjects': [f'sub-{s:02d}' for s in subs],
        'key_timepoints_ms': KEY_TIMEPOINTS_MS,
        'example_subject': f'sub-{best_sub:02d}',
        'example_image_index': int(best_idx),
        'example_image_relpath': img_relpath,
        'example_score': best_score,
        'panel_a_subject': f'sub-{best_sub:02d}',
        'panel_a_mean_r': float(sub_mean_r[best_i]),
        'panel_a_n_test_images': int(metas[best_i]['n_test_images']),
        'panel_a_top_indices': [int(i) for i in top_idx],
        'panel_a_top_channels': [CHANNELS[int(i)] for i in top_idx],
        'panel_a_top_r': [float(panel_a_per_ch[i]) for i in top_idx],
        'panel_a_per_subject_mean_r': {
            f'sub-{s:02d}': float(r) for s, r in zip(subs, sub_mean_r)
        },
        'per_subject_meta': metas,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def compute_aggregate_train_visualization(
    subs: List[int],
    device: Optional[torch.device] = None,
) -> Dict:
    """Grand-mean train ERP across subjects (requires generator inference once per subject)."""
    from analysis.eeg_gen_eval.compute.evaluate import compute_train_erp

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    erp_t_list, erp_p_list, n_train = [], [], []
    for sub in subs:
        sub_tag = f'sub-{sub:02d}'
        out_dir = os.path.join(VIS_DIR, sub_tag)
        erp_t_path = os.path.join(out_dir, 'erp_true_mean_train.npy')
        erp_p_path = os.path.join(out_dir, 'erp_pred_mean_train.npy')
        if not (os.path.isfile(erp_t_path) and os.path.isfile(erp_p_path)):
            os.makedirs(out_dir, exist_ok=True)
            print(f'Computing train ERP for {sub_tag} …')
            erp_t, erp_p = compute_train_erp(sub, device)
            np.save(erp_t_path, erp_t)
            np.save(erp_p_path, erp_p)
            train_path = os.path.join(DATA_DIR, sub_tag, 'train.pt')
            data = __import__('torch').load(train_path, map_location='cpu', weights_only=False)
            n_img = int(np.asarray(data['eeg']).shape[0])
        else:
            erp_t = np.load(erp_t_path)
            erp_p = np.load(erp_p_path)
            meta_path = os.path.join(out_dir, 'meta_train.json')
            if os.path.isfile(meta_path):
                with open(meta_path) as f:
                    n_img = int(json.load(f)['n_train_images'])
            else:
                train_path = os.path.join(DATA_DIR, sub_tag, 'train.pt')
                data = __import__('torch').load(train_path, map_location='cpu', weights_only=False)
                n_img = int(np.asarray(data['eeg']).shape[0])
        erp_t_list.append(erp_t)
        erp_p_list.append(erp_p)
        n_train.append(n_img)
        meta_sub = {
            'subject': sub_tag,
            'n_train_images': n_img,
            'split': 'train.pt (trial-averaged per image)',
        }
        with open(os.path.join(out_dir, 'meta_train.json'), 'w') as f:
            json.dump(meta_sub, f, indent=2)

    erp_t = np.stack(erp_t_list).mean(axis=0)
    erp_p = np.stack(erp_p_list).mean(axis=0)
    out_dir = os.path.join(VIS_DIR, 'aggregate')
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, 'erp_true_mean_train.npy'), erp_t)
    np.save(os.path.join(out_dir, 'erp_pred_mean_train.npy'), erp_p)
    meta = {
        'n_subjects': len(subs),
        'subjects': [f'sub-{s:02d}' for s in subs],
        'n_train_images_per_subject': int(np.mean(n_train)),
        'n_train_images_values': n_train,
        'split': 'train.pt (trial-averaged per image)',
    }
    with open(os.path.join(out_dir, 'meta_train.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return meta


def compute_all_visualizations(subs: List[int] | None = None) -> Dict:
    subs = subs or list(range(1, 11))
    for sub in subs:
        compute_subject_visualization(sub)
    return compute_aggregate_visualization(subs)
