"""Record leak-free checkpoint selection and run the one-shot held-out test.

Training must already have finished with the test split unread. This script
does not train and does not choose another checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from encoder.data import EEGDataset
from experiments.leakfree_things.prepare_subject import OUT, load_pt
from predictor.data import eeg_to_ubp_embedding, load_clip_features, load_frozen_ubp_brain
from predictor.model import load_any_generator_from_checkpoint

REPO = Path(__file__).resolve().parents[2]
BRAIN_CFG = {
    'target': 'encoder.models.EEGProjectLayer',
    'params': {'c_num': 63, 'z_dim': 1024, 'timesteps': [0, 250]},
}
HISTORICAL = {
    'source': 'results/audit_data_leakage/ablation_recheck.json Ours/full sub-01, historical whitening and historical encoder',
    'pooled_pearson': 0.539814293384552,
    'mean_channel_pearson': 0.3110654652118683,
    'gt_pred_encoder_cosine': 0.35293421149253845,
    'pred_clip_cosine': 0.3133077323436737,
    'encoder_test_top1': 0.390,
    'encoder_test_top1_source': 'historical TensorBoard test_top1_acc equals the last val_top1_acc, sub-01',
}


def _jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def encoder_dir(subject):
    return OUT / 'checkpoints' / 'encoder' / 'THINGSEEG2_leakfree' / f'sub-{subject:02d}_seed0'


def best_encoder(subject):
    files = sorted((encoder_dir(subject) / 'checkpoints').glob('epoch=*.ckpt'))
    if len(files) != 1:
        raise FileNotFoundError(f'expected one encoder epoch checkpoint, found {files}')
    return files[0].resolve()


def record_encoder(subject):
    ckpt_path = best_encoder(subject)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    record = {
        'rule': (
            'Train for the full 50-epoch budget. Keep the checkpoint with the '
            'highest validation top-1 (ModelCheckpoint mode=max, save_top_k=1). '
            'EarlyStopping patience is 50, so it does not stop training before '
            'the budget. The test split is not opened and is not a selection metric.'
        ),
        'checkpoint': str(ckpt_path.relative_to(REPO)),
        'epoch': int(ckpt.get('epoch', -1)),
        'global_step': int(ckpt.get('global_step', -1)),
        'callbacks': _jsonable(ckpt.get('callbacks', {})),
    }
    dest = OUT / 'selection' / f'sub-{subject:02d}_encoder.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote {dest}')


def pearson(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-12)


def retrieval(eeg, img, brain, device):
    eeg_z = brain(eeg.to(device))
    eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
    img_z = img.to(device)
    img_z = img_z / img_z.norm(dim=-1, keepdim=True)
    similarity = eeg_z @ img_z.T
    top = similarity.topk(5, dim=-1).indices.cpu().numpy()
    labels = np.arange(similarity.shape[0])
    top1 = float(np.mean(top[:, 0] == labels))
    top5 = float(np.mean((top == labels[:, None]).any(axis=1)))
    return top1, top5


def encoder_test(subject, device):
    config = OmegaConf.load(REPO / 'configs' / 'eeg' / 'fixed_fovea_leakfree.yaml')
    config.brain_backbone = 'EEGProjectLayer'
    config.vision_backbone = 'RN50'
    config.epoch = 50
    config.lr = 1e-4
    config.z_dim = 1024
    config.data.subjects = [f'sub-{subject:02d}']
    config.data.defer_test = False
    OmegaConf.resolve(config)
    dataset = EEGDataset(config, mode='test')
    if len(dataset) != 200:
        raise ValueError(f'encoder test set has {len(dataset)} images')
    batch = dataset[0]
    eeg = torch.stack([dataset[i]['eeg'] for i in range(len(dataset))])
    img = torch.stack([dataset[i]['img_features'] for i in range(len(dataset))])
    del batch
    brain = load_frozen_ubp_brain(BRAIN_CFG, str(best_encoder(subject)), device)
    top1, top5 = retrieval(eeg, img, brain, device)
    return {'test_top1_acc': top1, 'test_top5_acc': top5, 'n': 200, 'protocol': 'one 200-way batch, same as historical test_batch_size'}


def generator_metrics(subject, ckpt_path, eeg, images, data_dir, encoder_ckpt, device, mode):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = load_any_generator_from_checkpoint(ckpt, 63, 250).to(device).eval()
    clip = load_clip_features(
        images, data_dir=data_dir, mode=mode, clip_input='fovea', device=device,
        image_root=str(REPO / 'data' / 'things-eeg' / 'Image_set_Resize'),
        feature_dir=str(OUT / 'generator_features'),
    )
    with torch.no_grad():
        preds = [model(clip[i:i + 100].to(device)).cpu() for i in range(0, len(eeg), 100)]
    yh, y = torch.cat(preds), eeg
    pooled = float(pearson(yh.reshape(1, -1), y.reshape(1, -1)))
    per_channel = pearson(yh.permute(1, 0, 2).reshape(63, -1), y.permute(1, 0, 2).reshape(63, -1))
    brain = load_frozen_ubp_brain(BRAIN_CFG, encoder_ckpt, device)
    emb_g = F.normalize(eeg_to_ubp_embedding(yh.to(device), brain), dim=1)
    emb_t = F.normalize(eeg_to_ubp_embedding(y.to(device), brain), dim=1)
    clip_emb = F.normalize(clip.to(device), dim=1)
    return {
        'pooled_pearson': pooled,
        'mean_channel_pearson': float(per_channel.mean()),
        'gt_pred_encoder_cosine': float((emb_g * emb_t).sum(1).mean()),
        'pred_clip_cosine': float((emb_g * clip_emb).sum(1).mean()),
        'gt_clip_cosine': float((emb_t * clip_emb).sum(1).mean()),
        'encoder_ckpt': encoder_ckpt,
    }


def final_test(subject, device):
    tag = f'sub-{subject:02d}'
    stage2 = OUT / 'checkpoints' / 'generator' / 'stage2' / tag / 'best.pt'
    stage2_ckpt = torch.load(stage2, map_location='cpu', weights_only=False)
    args = stage2_ckpt.get('args') or {}
    new_encoder = str(best_encoder(subject))
    if Path(args.get('ubp_ckpt', '')).resolve() != Path(new_encoder).resolve():
        raise ValueError(f"stage2 ubp_ckpt is {args.get('ubp_ckpt')}, expected {new_encoder}")
    if args.get('early_stop_metric') != 'semantic_guard':
        raise ValueError(f"stage2 metric is {args.get('early_stop_metric')}")
    if 'THINGSEEG2_leakfree' not in str(args.get('ubp_ckpt')):
        raise ValueError('stage2 semantic_guard encoder is not the leak-free encoder')
    test = load_pt(OUT / 'preprocessed' / tag / 'test.pt')
    eeg = torch.as_tensor(np.asarray(test['eeg'])).float().mean(1)
    images = np.asarray(test['img'])[:, 0].astype(str)
    primary = generator_metrics(
        subject, stage2, eeg, images, str(OUT / 'preprocessed'), new_encoder, device, 'test',
    )
    old_encoder = str(REPO / 'checkpoints' / 'encoder' / 'THINGSEEG2' / f'{tag}_seed0' / 'checkpoints' / 'last.ckpt')
    secondary = generator_metrics(
        subject, stage2, eeg, images, str(OUT / 'preprocessed'), old_encoder, device, 'test',
    )
    official = load_pt(REPO / 'data' / 'things-eeg' / 'Preprocessed_data_250Hz_whiten' / tag / 'test.pt')
    old_eeg = torch.as_tensor(np.asarray(official['eeg'])).float().mean(1)
    old_images = np.asarray(official['img'])[:, 0].astype(str)
    diagnostic = generator_metrics(
        subject, stage2, old_eeg, old_images,
        str(REPO / 'data' / 'things-eeg' / 'Preprocessed_data_250Hz_whiten'),
        new_encoder, device, 'test',
    )
    enc = encoder_test(subject, device)
    anomalies = []
    if not all(np.isfinite(v) for v in (
        primary['pooled_pearson'], primary['mean_channel_pearson'], enc['test_top1_acc'],
    )):
        anomalies.append('non-finite metric')
    if primary['pooled_pearson'] < 0.15 or primary['mean_channel_pearson'] < 0.10:
        anomalies.append('waveform correlation collapsed relative to the historical sub-01 result')
    if enc['test_top1_acc'] < 0.05:
        anomalies.append('encoder test top-1 collapsed')
    report = {
        'subject': tag,
        'selection_not_changed_by_this_script': True,
        'encoder': enc,
        'generator_stage2_best': {
            'epoch': int(stage2_ckpt.get('epoch', -1)),
            'best_epoch': int(stage2_ckpt.get('best_epoch', -1)),
            'early_stop_metric': args.get('early_stop_metric'),
            'ubp_ckpt': args.get('ubp_ckpt'),
            'init_ckpt': args.get('init_ckpt'),
        },
        'primary_new_encoder_new_whitening': primary,
        'secondary_old_encoder_not_used_for_selection': secondary,
        'diagnostic_new_generator_on_historical_whitening_not_used_for_selection': diagnostic,
        'historical_reference': HISTORICAL,
        'anomalies': anomalies,
    }
    dest = OUT / 'evaluation' / f'{tag}_final.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if anomalies:
        raise SystemExit(f'anomalies: {anomalies}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('record-encoder', 'final'))
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'record-encoder':
        record_encoder(args.subject)
        return
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    final_test(args.subject, device)


if __name__ == '__main__':
    main()
