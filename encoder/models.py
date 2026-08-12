"""Brain encoder, CLIP registry, and UBP blur transforms."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.optimize import fsolve
from torchvision.transforms import functional as TF


# --- CLIP ---

CLIP_BACKBONES: Dict[str, Dict] = {
    'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
}


def clip_z_dim(vision_backbone: str) -> int:
    if vision_backbone not in CLIP_BACKBONES:
        raise KeyError(
            f'Unknown vision_backbone {vision_backbone!r}; '
            f'choose from {list(CLIP_BACKBONES)}'
        )
    return CLIP_BACKBONES[vision_backbone]['z_dim']


def clip_pretrained_tag(vision_backbone: str) -> str:
    if vision_backbone not in CLIP_BACKBONES:
        raise KeyError(
            f'Unknown vision_backbone {vision_backbone!r}; '
            f'choose from {list(CLIP_BACKBONES)}'
        )
    return CLIP_BACKBONES[vision_backbone]['pretrained']


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
    def __init__(self, h: int = 224, w: int = 224):
        self.size = (w, h)

    def __call__(self, x, U=None):
        if isinstance(x, Image.Image) and self.size:
            return x.resize(self.size, Image.BICUBIC)
        return x


class UniformBlur:
    def __init__(self, blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)


class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0

        def equations(vars):
            t, r = vars
            return [r * (t - np.sin(t)) - 1, -r * (1 - np.cos(t)) + 1.0]

        _, r_solution = fsolve(equations, [1.0, 1.0])
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1, distance / max_distance)
                y0 = fun_degrade(x0, **kwargs)
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

    def linear(self, x, **kwargs):
        return 1 - x

    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

    def quadratic(self, x, **kwargs):
        return 1 - x ** 2

    def log(self, x, **kwargs):
        b = 1 / (np.e - 1)
        a = np.log(b) + 1
        return a - np.log(x + b)

    def brachistochrone(self, x, **kwargs):
        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        return -self.r * (1 - np.cos(t0)) + 1.0
