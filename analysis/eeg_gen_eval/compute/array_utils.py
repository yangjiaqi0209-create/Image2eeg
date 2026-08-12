"""Lightweight array helpers shared by cached redraw (no generator imports)."""

from __future__ import annotations

import numpy as np


def trial_format(data: dict) -> str | None:
    return data.get('trial_format')


def eeg_per_sample(eeg: np.ndarray, trial_fmt: str | None) -> np.ndarray:
    """(N, C, T): THINGS averages same-image reps; per_image keeps each unique image."""
    if eeg.ndim == 3:
        return eeg
    if eeg.ndim != 4:
        raise ValueError(f'Expected EEG (N, trials, C, T) or (N, C, T), got {eeg.shape}')
    if trial_fmt == 'per_image' or eeg.shape[1] == 1:
        return eeg[:, 0]
    return eeg.mean(axis=1)


def img_per_sample(img_paths: np.ndarray, trial_fmt: str | None) -> np.ndarray:
    if img_paths.ndim == 1:
        return img_paths
    if trial_fmt == 'per_image' or img_paths.shape[1] == 1:
        return img_paths[:, 0]
    return img_paths[:, 0]
