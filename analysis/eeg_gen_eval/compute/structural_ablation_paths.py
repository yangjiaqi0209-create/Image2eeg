"""Paths and labels for generator structural (component) ablation."""

from __future__ import annotations

import os
from typing import Dict, List

from analysis.eeg_gen_eval.config import REPO_ROOT

STRUCTURAL_RAW_DIR = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'structural_ablation')

FULL_CKPT_DIR = os.path.join(
    REPO_ROOT, 'checkpoints', 'predictor', 'THINGSEEG2', 'Ours', 'full',
)
FULL_RESULT_DIR = os.path.join(REPO_ROOT, 'results', 'generator_final', 'full')

ARCH_ROOT_CKPT = os.path.join(
    REPO_ROOT, 'checkpoints', 'predictor', 'THINGSEEG2', 'architecture_ablation',
)
STRUCTURAL_ABLATION_ROOT_RESULT = os.path.join(
    REPO_ROOT, 'results', 'generator_structural_ablation',
)
STRUCTURAL_ABLATION_ROOT_CKPT = ARCH_ROOT_CKPT

# Manuscript Fig4 structural set (keys kept for existing raw/ caches).
STRUCTURAL_CONDITIONS: Dict[str, Dict[str, object]] = {
    'S0_full': {
        'label': 'Full (Ours)',
        'generator_arch': 'full',
        'per_channel_heads': True,
        'ckpt_dir': FULL_CKPT_DIR,
        'result_dir': FULL_RESULT_DIR,
    },
    'S1_no_dconv': {
        'label': 'w/o Dilated TCN',
        'generator_arch': 'no_dconv',
        'per_channel_heads': True,
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'no_Dilated', 'full'),
        'result_dir': os.path.join(STRUCTURAL_ABLATION_ROOT_RESULT, 'no_dconv', 'full'),
    },
    'S3_tcn_only': {
        'label': 'w/o Transformer',
        'generator_arch': 'tcn_only',
        'per_channel_heads': True,
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'no_Transformer', 'full'),
        'result_dir': os.path.join(STRUCTURAL_ABLATION_ROOT_RESULT, 'tcn_only', 'full'),
    },
}

STRUCTURAL_ORDER: List[str] = [
    'S0_full',
    'S1_no_dconv',
    'S3_tcn_only',
]

WAVEFORM_METRIC_KEYS = [
    'test_mse',
    'test_corr',
]
SEMANTIC_METRIC_KEYS = [
    'test_semantic_cosine',
]
FREQUENCY_METRIC_KEYS = [
    'bandpower_corr',
    'fft_l1',
]
METRIC_KEYS = WAVEFORM_METRIC_KEYS + FREQUENCY_METRIC_KEYS + SEMANTIC_METRIC_KEYS

METRIC_COLUMNS: List[tuple] = [
    ('test_mse', 'MSE ↓', 4, True),
    ('test_corr', 'Pearson r ↑ (per-ch mean)', 3, False),
    ('bandpower_corr', 'Bandpower corr ↑', 3, False),
    ('fft_l1', 'FFT L1 ↓', 4, True),
    ('test_semantic_cosine', 'Semantic cos ↑', 3, False),
]


def structural_ckpt_path(condition: str, sub: int) -> str:
    meta = STRUCTURAL_CONDITIONS[condition]
    return os.path.join(str(meta['ckpt_dir']), f'sub-{sub:02d}', 'last.pt')


def structural_metrics_path(condition: str, sub: int) -> str:
    meta = STRUCTURAL_CONDITIONS[condition]
    return os.path.join(str(meta['result_dir']), f'sub-{sub:02d}', 'metrics.json')


def structural_eval_raw_dir(condition: str, sub: int) -> str:
    return os.path.join(STRUCTURAL_RAW_DIR, condition, f'sub-{sub:02d}')
