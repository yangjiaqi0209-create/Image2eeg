"""One invertible training-derived spatial transform shared by a subject's sessions.

This module does not change the existing session-MVNN preprocessing entrypoint.
Raw electrode space here means after the existing baseline/resampling/crop.
"""

from __future__ import annotations

import numpy as np


def fit_from_covariances(covariances, n_training_trials):
    """Pool trial-level shrinkage covariances, then take ONE matrix square root.

    Each input covariance is the mean of the original per-trial auto-shrinkage
    estimator, restricted to the eligible optimization-training images.
    Weight by the number of eligible trials, never average whitening matrices.
    """
    covariances = np.asarray(covariances, dtype=np.float64)
    counts = np.asarray(n_training_trials, dtype=np.float64)
    if (covariances.ndim != 3 or covariances.shape[1] != covariances.shape[2]
            or counts.shape != (len(covariances),) or np.any(counts <= 0)
            or not np.isfinite(covariances).all() or not np.isfinite(counts).all()):
        raise ValueError('Finite square covariances and positive training trial counts are required')
    covariance = np.average(covariances, axis=0, weights=counts)
    covariance = (covariance + covariance.T) / 2
    eigenvalues, vectors = np.linalg.eigh(covariance)
    if np.any(eigenvalues <= 0):
        raise ValueError('Covariance must be strictly positive definite; no silent eigenvalue clipping')
    whitening = (vectors * eigenvalues**-.5) @ vectors.T
    inverse = (vectors * eigenvalues**.5) @ vectors.T
    return covariance, whitening, inverse


def apply_spatial(x, matrix):
    """[... , channel, time] -> same shape, without changing averaging semantics."""
    x = np.asarray(x)
    matrix = np.asarray(matrix)
    if x.ndim < 2 or matrix.shape != (x.shape[-2], x.shape[-2]):
        raise ValueError('Channel shape does not match the spatial transform')
    return np.matmul(matrix, x)


def forward(x, whitening, mean=None, scale=None):
    """Optional fixed affine normalization AFTER whitening; identity by default.

    No new z-score is estimated or enabled by this module. If a future protocol
    explicitly uses fixed training-fitted mean/scale, they must be persisted.
    Trial-wise/time-varying nonlinear normalization is not substituted here.
    """
    z = apply_spatial(x, whitening)
    if mean is not None:
        z = z - mean
    if scale is not None:
        if not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
            raise ValueError('Normalization scale must be finite and positive')
        z = z / scale
    return z


def inverse(z, inverse_whitening, mean=None, scale=None):
    z = np.asarray(z)
    if scale is not None:
        if not np.isfinite(scale).all() or np.any(np.asarray(scale) <= 0):
            raise ValueError('Normalization scale must be finite and positive')
        z = z * scale
    if mean is not None:
        z = z + mean
    return apply_spatial(z, inverse_whitening)
