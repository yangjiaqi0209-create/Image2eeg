"""Dataset: CLIP features (sharp / fovea) + averaged EEG."""

from __future__ import annotations

import gc
import glob
import os
from typing import List, Optional, Tuple

from torch import Tensor

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm import tqdm

from encoder.data import THINGS_CHANNELS, resolve_clip_pretrained
from encoder.models import CLIP_BACKBONES
from encoder.utils import get_device, instantiate_from_config

CHANNELS = THINGS_CHANNELS

PRETRAIN_MAP = {k: {'pretrained': v['pretrained']} for k, v in CLIP_BACKBONES.items()}

CLIP_INPUT_CHOICES = ('sharp', 'fovea')


def _load_eeg_tensor(
    pt_path: str,
    timesteps: Tuple[int, int] = (0, 250),
    avg: bool = True,
    selected_ch: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    eeg = torch.from_numpy(np.asarray(data['eeg'])).float()
    img_paths = np.asarray(data['img'])
    file_channels = list(data.get('ch_names') or CHANNELS)

    if selected_ch is None:
        selected_ch = file_channels
    if selected_ch:
        idx = [file_channels.index(ch) for ch in selected_ch]
        eeg = eeg[:, :, idx]

    if avg:
        eeg = eeg.mean(dim=1)
    else:
        eeg = eeg.reshape(-1, *eeg.shape[2:])

    t0, t1 = timesteps
    eeg = eeg[..., t0:t1]

    if avg:
        img_paths = img_paths[:, 0]
    else:
        img_paths = img_paths.reshape(-1)

    return eeg, img_paths


def _transform_config(clip_input: str, blur_kernel_size: int) -> dict:
    if clip_input == 'sharp':
        return {'target': 'encoder.models.DirectT', 'params': {}}
    return {
        'target': 'encoder.models.FoveaBlur',
        'params': {
            'h': 224,
            'w': 224,
            'blur_kernel_size': blur_kernel_size,
            'curve_type': 'exp',
            'system_g': 3,
        },
    }


def _feature_subdir(clip_input: str) -> str:
    return {
        'sharp': 'DirectT',
        'fovea': 'FoveaBlur',
    }[clip_input]


@torch.no_grad()
def _encode_images_clip(
    img_paths: List[str],
    image_root: str,
    model_type: str,
    clip_input: str,
    blur_kernel_size: int,
    device: torch.device,
    batch_size: int = 128,
    desc: str = 'CLIP encode',
) -> dict:
    img_transform = instantiate_from_config(_transform_config(clip_input, blur_kernel_size))
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])

    pretrained = resolve_clip_pretrained(model_type, PRETRAIN_MAP[model_type]['pretrained'])
    vlmodel, _, _ = open_clip.create_model_and_transforms(
        model_type, device=str(device), pretrained=pretrained,
    )
    vlmodel.eval()
    for p in vlmodel.parameters():
        p.requires_grad = False

    unique_paths = sorted(set(img_paths))
    features: dict = {}
    for i in tqdm(range(0, len(unique_paths), batch_size), desc=desc):
        batch_paths = unique_paths[i:i + batch_size]
        imgs = []
        for rel in batch_paths:
            path = os.path.join(image_root, rel)
            img = img_transform(Image.open(path).convert('RGB'))
            imgs.append(process(img))
        batch = torch.stack(imgs).to(device)
        feat = vlmodel.encode_image(batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        for j, rel in enumerate(batch_paths):
            features[rel] = feat[j].float().cpu()

    del vlmodel
    torch.cuda.empty_cache()
    gc.collect()
    return features


def _feature_cache_path(
    feature_dir: str,
    model_type: str,
    mode: str,
    clip_input: str,
    blur_kernel_size: int,
    tag: str,
) -> str:
    os.makedirs(feature_dir, exist_ok=True)
    return os.path.join(
        feature_dir,
        f'generator_{clip_input}_{tag}_k{blur_kernel_size}_{model_type}_{mode}.pt',
    )


def load_clip_features(
    img_paths: np.ndarray,
    *,
    data_dir: str,
    model_type: str = 'RN50',
    clip_input: str = 'fovea',
    blur_kernel_size: int = 51,
    mode: str = 'train',
    tag: str = 'medium',
    device: Optional[torch.device] = None,
    image_root: Optional[str] = None,
    feature_dir: Optional[str] = None,
) -> torch.Tensor:
    """Load or compute CLIP features for the given blur/sharp input type."""
    if clip_input not in CLIP_INPUT_CHOICES:
        raise ValueError(f'clip_input must be one of {CLIP_INPUT_CHOICES}, got {clip_input!r}')

    image_root = image_root or os.path.join(data_dir, '../Image_set_Resize')
    if feature_dir is None:
        feature_dir = os.path.join(data_dir, '../Image_feature', _feature_subdir(clip_input))
    cache_path = _feature_cache_path(
        feature_dir, model_type, mode, clip_input, blur_kernel_size, tag,
    )

    if os.path.isfile(cache_path):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        feat_dict = cached['img_features']
    else:
        if device is None:
            device = torch.device(f'cuda:{get_device("auto")}' if torch.cuda.is_available() else 'cpu')
        feat_dict = _encode_images_clip(
            list(set(img_paths.tolist())),
            image_root,
            model_type,
            clip_input,
            blur_kernel_size,
            device,
            desc=f'CLIP {clip_input} {tag} k={blur_kernel_size}',
        )
        torch.save({'img_features': feat_dict}, cache_path)

    return torch.stack([feat_dict[p] for p in img_paths.tolist()]).float()


def load_subject_splits(
    sub: int,
    *,
    data_dir: str,
    model_type: str = 'RN50',
    clip_input: str = 'fovea',
    blur_kernel_size: int = 51,
    timesteps: Tuple[int, int] = (0, 250),
    selected_ch: Optional[List[str]] = None,
    seed: int = 2023,
    n_val: int = 740,
    device: Optional[torch.device] = None,
    image_root: Optional[str] = None,
    feature_dir: Optional[str] = None,
    avg: bool = True,
) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    Optional[torch.Tensor], Optional[torch.Tensor],
]:
    """
    Returns:
        clip_train, eeg_train, clip_val, eeg_val, clip_test, eeg_test
    """
    sub_dir = os.path.join(data_dir, f'sub-{sub:02d}')
    train_path = os.path.join(sub_dir, 'train.pt')
    test_path = os.path.join(sub_dir, 'test.pt')
    if not os.path.isfile(train_path):
        raise FileNotFoundError(train_path)

    eeg_train, img_train = _load_eeg_tensor(
        train_path, timesteps=timesteps, avg=avg, selected_ch=selected_ch,
    )
    clip_train = load_clip_features(
        img_train, data_dir=data_dir, model_type=model_type,
        clip_input=clip_input, blur_kernel_size=blur_kernel_size,
        mode='train', tag='medium', device=device,
        image_root=image_root, feature_dir=feature_dir,
    )

    val_path = os.path.join(sub_dir, 'val.pt')
    if os.path.isfile(val_path):
        eeg_va, img_va = _load_eeg_tensor(
            val_path, timesteps=timesteps, avg=avg, selected_ch=selected_ch,
        )
        clip_va = load_clip_features(
            img_va, data_dir=data_dir, model_type=model_type,
            clip_input=clip_input, blur_kernel_size=blur_kernel_size,
            mode='val', tag='medium', device=device,
            image_root=image_root, feature_dir=feature_dir,
        )
        clip_tr = clip_train
        eeg_tr = eeg_train
    else:
        rng = np.random.default_rng(seed)
        n = eeg_train.shape[0]
        perm = rng.permutation(n)
        n_val = min(n_val, n - 1)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        clip_tr = clip_train[train_idx]
        eeg_tr = eeg_train[train_idx]
        clip_va = clip_train[val_idx]
        eeg_va = eeg_train[val_idx]

    if os.path.isfile(test_path):
        eeg_test, img_test = _load_eeg_tensor(
            test_path, timesteps=timesteps, avg=avg, selected_ch=selected_ch,
        )
        clip_test = load_clip_features(
            img_test, data_dir=data_dir, model_type=model_type,
            clip_input=clip_input, blur_kernel_size=blur_kernel_size,
            mode='test', tag='medium', device=device,
            image_root=image_root, feature_dir=feature_dir,
        )
    else:
        clip_test, eeg_test = None, None

    return clip_tr, eeg_tr, clip_va, eeg_va, clip_test, eeg_test


def make_loaders(
    clip_train: torch.Tensor,
    eeg_train: torch.Tensor,
    clip_val: torch.Tensor,
    eeg_val: torch.Tensor,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 2023,
) -> Tuple[DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(seed)
    train_ds = TensorDataset(clip_train, eeg_train)
    val_ds = TensorDataset(clip_val, eeg_val)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(), generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _checkpoint_args(ckpt_path: str) -> dict:
    if not os.path.isfile(ckpt_path):
        return {}
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ckpt.get('args') or {}


def clip_input_from_checkpoint(ckpt_path: str, default: str = 'fovea') -> str:
    return _checkpoint_args(ckpt_path).get('clip_input', default)


def vision_backbone_from_checkpoint(ckpt_path: str, default: str = 'RN50') -> str:
    return _checkpoint_args(ckpt_path).get('vision_backbone', default)


# --- Frozen UBP brain encoder ---

def _encoder_dataset_dir(exp_name: Optional[str]) -> str:
    """Map dataset tags / legacy exp names → checkpoints/encoder/<dir>."""
    name = (exp_name or '').lower()
    if name in {'alljoined', 'thingseeg2'}:
        return 'Alljoined' if name == 'alljoined' else 'THINGSEEG2'
    if 'alljoined' in name:
        return 'Alljoined'
    return 'THINGSEEG2'


def resolve_encoder_ckpt_path(
    repo_root: str,
    sub: int,
    seed: int = 0,
    *,
    exp_name: str = 'THINGSEEG2',
    prefer_best: bool = True,
) -> str:
    """Return val-best checkpoint when present (epoch=*.ckpt), else last.ckpt."""
    ckpt_dir = os.path.join(
        repo_root,
        'checkpoints',
        'encoder',
        _encoder_dataset_dir(exp_name),
        f'sub-{sub:02d}_seed{seed}',
        'checkpoints',
    )
    if prefer_best:
        epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'epoch=*.ckpt')))
        if epoch_ckpts:
            return epoch_ckpts[0]
    return os.path.join(ckpt_dir, 'last.ckpt')


def default_ubp_ckpt_path(
    repo_root: str,
    sub: int,
    seed: int = 0,
    *,
    exp_name: Optional[str] = None,
    **_kwargs,
) -> str:
    return resolve_encoder_ckpt_path(
        repo_root, sub, seed, exp_name=exp_name or 'THINGSEEG2', prefer_best=True,
    )


def load_frozen_ubp_brain(
    brain_config: dict,
    ckpt_path: Optional[str],
    device: torch.device,
) -> Optional[nn.Module]:
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        return None

    brain = instantiate_from_config(brain_config)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    brain_state = {
        k.removeprefix('brain.'): v
        for k, v in state.items()
        if k.startswith('brain.')
    }
    if not brain_state:
        raise RuntimeError(f'No brain.* weights found in {ckpt_path}')

    brain.load_state_dict(brain_state, strict=True)
    brain.to(device)
    brain.eval()
    for p in brain.parameters():
        p.requires_grad = False
    return brain


def eeg_to_ubp_embedding(eeg: Tensor, brain: nn.Module) -> Tensor:
    """Map [B, C, T] EEG to normalized CLIP-space embedding."""
    z = brain(eeg.contiguous())
    return F.normalize(z, dim=-1)
