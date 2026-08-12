"""Paths and labels for extended Final generator ablation."""

from __future__ import annotations

import os
from typing import Dict, List

from analysis.eeg_gen_eval.config import REPO_ROOT

EXTENDED_RAW_DIR = os.path.join(REPO_ROOT, 'analysis', 'eeg_gen_eval', 'raw', 'extended_ablation')

FULL_CKPT_DIR = os.path.join(
    REPO_ROOT, 'checkpoints', 'predictor', 'THINGSEEG2', 'Ours', 'full',
)
FULL_RESULT_DIR = os.path.join(REPO_ROOT, 'results', 'generator_final', 'full')

ARCH_ROOT_CKPT = os.path.join(
    REPO_ROOT, 'checkpoints', 'predictor', 'THINGSEEG2', 'architecture_ablation',
)
EXTENDED_ROOT_RESULT = os.path.join(REPO_ROOT, 'results', 'generator_extended_ablation')
EXTENDED_ROOT_CKPT = ARCH_ROOT_CKPT

# Manuscript Fig4 extended set (keys kept for existing raw/ caches).
EXTENDED_CONDITIONS: Dict[str, Dict[str, object]] = {
    'E0_full': {
        'label': 'Full (Ours)',
        'variant': 'full',
        'generator_arch': 'full',
        'n_layers': 4,
        'hidden': 256,
        'clip_input': 'fovea',
        'ckpt_dir': FULL_CKPT_DIR,
        'result_dir': FULL_RESULT_DIR,
    },
    'E1_sharp': {
        'label': 'w/o Fovea CLIP (sharp)',
        'variant': 'no_FoveaBlur',
        'generator_arch': 'full',
        'n_layers': 4,
        'hidden': 256,
        'clip_input': 'sharp',
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'no_FoveaBlur', 'full'),
        'result_dir': os.path.join(EXTENDED_ROOT_RESULT, 'sharp', 'full'),
    },
    'E5_no_self_attn': {
        'label': 'w/o self-attention',
        'variant': 'no_self_attn',
        'generator_arch': 'no_self_attn',
        'n_layers': 4,
        'hidden': 256,
        'clip_input': 'fovea',
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'no_self_attn', 'full'),
        'result_dir': os.path.join(EXTENDED_ROOT_RESULT, 'no_self_attn', 'full'),
    },
    'E8_h128': {
        'label': 'hidden=128',
        'variant': 'h128',
        'generator_arch': 'full',
        'n_layers': 4,
        'hidden': 128,
        'clip_input': 'fovea',
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'h128', 'full'),
        'result_dir': os.path.join(EXTENDED_ROOT_RESULT, 'h128', 'full'),
    },
    'E9_h512': {
        'label': 'hidden=512',
        'variant': 'h512',
        'generator_arch': 'full',
        'n_layers': 4,
        'hidden': 512,
        'clip_input': 'fovea',
        'ckpt_dir': os.path.join(ARCH_ROOT_CKPT, 'h512', 'full'),
        'result_dir': os.path.join(EXTENDED_ROOT_RESULT, 'h512', 'full'),
    },
}

EXTENDED_ORDER: List[str] = [
    'E0_full',
    'E1_sharp',
    'E5_no_self_attn',
    'E8_h128',
    'E9_h512',
]

WAVEFORM_METRIC_KEYS = ['test_mse', 'test_corr']
SEMANTIC_METRIC_KEYS = ['test_semantic_cosine']
FREQUENCY_METRIC_KEYS = ['bandpower_corr', 'fft_l1']
METRIC_KEYS = WAVEFORM_METRIC_KEYS + FREQUENCY_METRIC_KEYS + SEMANTIC_METRIC_KEYS

METRIC_COLUMNS: List[tuple] = [
    ('test_mse', 'MSE ↓', 4, True),
    ('test_corr', 'Pearson r ↑ (per-ch mean)', 3, False),
    ('bandpower_corr', 'Bandpower corr ↑', 3, False),
    ('fft_l1', 'FFT L1 ↓', 4, True),
    ('test_semantic_cosine', 'Semantic cos ↑', 3, False),
]


def extended_ckpt_path(condition: str, sub: int) -> str:
    meta = EXTENDED_CONDITIONS[condition]
    return os.path.join(str(meta['ckpt_dir']), f'sub-{sub:02d}', 'last.pt')


def extended_metrics_path(condition: str, sub: int) -> str:
    meta = EXTENDED_CONDITIONS[condition]
    return os.path.join(str(meta['result_dir']), f'sub-{sub:02d}', 'metrics.json')


def extended_eval_raw_dir(condition: str, sub: int) -> str:
    return os.path.join(EXTENDED_RAW_DIR, condition, f'sub-{sub:02d}')
