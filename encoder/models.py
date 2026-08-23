"""Brain encoder, CLIP registry, and fovea/sharp image transforms."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# --- CLIP ---

CLIP_BACKBONES: Dict[str, Dict] = {
    'RN50': {'pretrained': 'openai', 'z_dim': 1024},
}


def clip_z_dim(vision_backbone: str) -> int:
    if vision_backbone not in CLIP_BACKBONES:
        raise KeyError(
            f'Unknown vision_backbone {vision_backbone!r}; '
            f'choose from {list(CLIP_BACKBONES)}'
        )
    return CLIP_BACKBONES[vision_backbone]['z_dim']


# --- Brain encoder ---

class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return x + self.f(x)


class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super().__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1] - self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x


# --- Blur / image transforms ---

class DirectT:
    """Identity resize (sharp CLIP input)."""

    def __init__(self, h: int = 224, w: int = 224):
        self.size = (w, h)

    def __call__(self, x, U=None):
        if isinstance(x, Image.Image) and self.size:
            return x.resize(self.size, Image.BICUBIC)
        return x


class FoveaBlur:
    """Center-weighted Gaussian blur (exp falloff; matches manuscript UBP prior)."""

    def __init__(self, h, w, blur_kernel_size, curve_type='exp', system_g=3, *args, **kwargs):
        if curve_type != 'exp':
            raise ValueError(f'Only curve_type=exp is supported, got {curve_type!r}')
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0.0

        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1.0, distance / max_distance)
                y0 = np.exp(-system_g * x0)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.convertScaleAbs(img1 * (1 - alpha) + img2 * alpha)

    def __call__(self, img, blur_kernel_size=None):
        if blur_kernel_size is None:
            blur_kernel_size = self.blur_kernel_size
        h, w = self.mask.shape
        if isinstance(img, Image.Image):
            img = img.resize((w, h), Image.BILINEAR)
        img = np.array(img)
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h))
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blured = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        blended = self.alphaBlend(img, blured, 1 - self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
