"""Paths and constants for generator quality evaluation (manuscript Results)."""

from __future__ import annotations

import os
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(REPO_ROOT, 'data/things-eeg/Preprocessed_data_250Hz_whiten')
GENERATOR_ARCH = 'full'
OUT_ROOT = os.path.join(REPO_ROOT, 'analysis/eeg_gen_eval')

# Predictor (image→EEG) checkpoints
FINAL_CKPT_DIR = os.path.join(
    REPO_ROOT, 'checkpoints/predictor/THINGSEEG2/Ours/full',
)
FINAL_RESULT_DIR = os.path.join(REPO_ROOT, 'results/generator_final/full')

GENERATOR_N_LAYERS = 4
DEFAULT_BASELINE = 'final'

CKPT_DIR = FINAL_CKPT_DIR
RAW_DIR = os.path.join(OUT_ROOT, 'raw')
FIG_DIR = os.path.join(OUT_ROOT, 'figures')

# Encoder checkpoint tree: checkpoints/encoder/{THINGSEEG2,Alljoined}/sub-XX_seed0/
ENCODER_DATASET = 'THINGSEEG2'
BRAIN_ENCODER_EXP = ENCODER_DATASET  # alias used by older call sites
BASELINE_NAME = 'final'
BASELINE_LABEL = 'Final (semantic pretrain + spectral refinement)'

BASELINE_PROFILES: Dict[str, Dict[str, str]] = {
    'final': {
        'ckpt_dir': FINAL_CKPT_DIR,
        'result_dir': FINAL_RESULT_DIR,
        'raw_dir': os.path.join(OUT_ROOT, 'raw'),
        'fig_dir': os.path.join(OUT_ROOT, 'figures'),
        'brain_encoder_exp': 'THINGSEEG2',
        'label': 'Final (semantic pretrain + spectral refinement)',
    },
}

BASELINE_ALIASES: Dict[str, str] = {
    'ours': 'final',
}


def set_active_baseline(name: str) -> Dict[str, str]:
    """Switch generator eval paths (checkpoint, raw, figures, brain encoder)."""
    global CKPT_DIR, RAW_DIR, FIG_DIR, BRAIN_ENCODER_EXP, ENCODER_DATASET
    global BASELINE_NAME, BASELINE_LABEL
    name = BASELINE_ALIASES.get(name, name)
    if name not in BASELINE_PROFILES:
        raise ValueError(f'Unknown baseline {name!r}; choose from {list(BASELINE_PROFILES)}')
    profile = BASELINE_PROFILES[name]
    BASELINE_NAME = name
    BASELINE_LABEL = profile['label']
    if 'things' in DATASET_PROFILES:
        DATASET_PROFILES['things']['raw_dir'] = profile['raw_dir']
        DATASET_PROFILES['things']['fig_dir'] = profile['fig_dir']
        DATASET_PROFILES['things']['ckpt_dir'] = profile['ckpt_dir']
        DATASET_PROFILES['things']['brain_encoder_exp'] = profile['brain_encoder_exp']
    if _ACTIVE_DATASET == 'things':
        CKPT_DIR = profile['ckpt_dir']
        RAW_DIR = profile['raw_dir']
        FIG_DIR = profile['fig_dir']
        BRAIN_ENCODER_EXP = profile['brain_encoder_exp']
        ENCODER_DATASET = profile['brain_encoder_exp']
    return profile


def eval_brain_ckpt_path(sub: int, seed: int = 0) -> str:
    """Frozen EEG encoder used for retrieval / similarity figures."""
    root = ENCODER_DATASET
    return os.path.join(
        REPO_ROOT, 'checkpoints', 'encoder', root,
        f'sub-{sub:02d}_seed{seed}', 'checkpoints', 'last.ckpt',
    )

SFREQ = 250.0
N_TEST_TRIALS = 80
TIME_WINDOWS_MS = [
    ('0-100 ms', 0, 100),
    ('100-200 ms', 100, 200),
    ('200-250 ms', 200, 250),
]

ERP_COMPONENT_WINDOWS_MS = [
    ('C1', 50, 90),
    ('P1', 80, 130),
    ('N1', 140, 200),
]

EEG_BANDS = [
    ('delta', 0.5, 4.0),
    ('theta', 4.0, 8.0),
    ('alpha', 8.0, 13.0),
    ('beta', 13.0, 30.0),
    ('gamma', 30.0, 45.0),
]

CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]

CHANNELS_O = [
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]

CHANNELS_OP17 = [
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2',
]

IMAGE_ROOT = os.path.join(REPO_ROOT, 'data/things-eeg/Image_set_Resize')

GENERATOR_RESULTS_DIR = os.path.join(REPO_ROOT, 'results/generator_runs')
GENERATOR_EVAL_CKPT = 'last.pt'


def generator_eval_ckpt_path(ckpt_dir: str, sub: int) -> str:
    return os.path.join(ckpt_dir, f'sub-{sub:02d}', GENERATOR_EVAL_CKPT)


N_TEST_CLASSES = 200
CHANCE_TOP1 = 1.0 / N_TEST_CLASSES
CHANCE_TOP5 = 5.0 / N_TEST_CLASSES

try:
    from encoder.data import ALLJOINED_CHANNELS as ALLJOINED_CHANNELS  # noqa: F401
except ImportError:
    ALLJOINED_CHANNELS = []

ALLJOINED_DATA_DIR = '/home/ubuntu/dataset/EEG/alljoined-1.6M/ubp_preprocessed'
ALLJOINED_FEATURE_DIR = '/home/ubuntu/dataset/EEG/alljoined-1.6M/clip_features/FoveaBlur'
ALLJOINED_CKPT_DIR = os.path.join(
    REPO_ROOT, 'checkpoints/predictor/Alljoined/Ours/full',
)
ALLJOINED_RESULTS_DIR = os.path.join(
    REPO_ROOT, 'results/alljoined_eeg/generator/final_two_stage/full',
)
ALLJOINED_BRAIN_ENCODER_EXP = 'Alljoined'
ALLJOINED_STRONG_SUBJECTS = [6, 12, 13, 14, 18]

DATASET_PROFILES: Dict[str, Dict] = {
    'things': {
        'data_dir': DATA_DIR,
        'image_root': IMAGE_ROOT,
        'raw_dir': RAW_DIR,
        'fig_dir': FIG_DIR,
        'ckpt_dir': CKPT_DIR,
        'brain_encoder_exp': 'THINGSEEG2',
        'subjects': list(range(1, 11)),
        'n_test_classes': N_TEST_CLASSES,
        'channels': CHANNELS,
    },
    'alljoined': {
        'data_dir': ALLJOINED_DATA_DIR,
        'image_root': IMAGE_ROOT,
        'feature_dir': ALLJOINED_FEATURE_DIR,
        'raw_dir': os.path.join(OUT_ROOT, 'raw_alljoined'),
        'fig_dir': os.path.join(OUT_ROOT, 'figures_alljoined'),
        'ckpt_dir': ALLJOINED_CKPT_DIR,
        'results_dir': ALLJOINED_RESULTS_DIR,
        'brain_encoder_exp': ALLJOINED_BRAIN_ENCODER_EXP,
        'subjects': list(ALLJOINED_STRONG_SUBJECTS),
        'n_test_classes': N_TEST_CLASSES,
        'channels': ALLJOINED_CHANNELS,
        'example_subject': 6,
    },
}

_ACTIVE_DATASET = 'things'


def set_active_dataset(name: str) -> Dict:
    """Switch eval paths (THINGS / Alljoined)."""
    global _ACTIVE_DATASET, DATA_DIR, IMAGE_ROOT, RAW_DIR, FIG_DIR, CKPT_DIR
    global BRAIN_ENCODER_EXP, ENCODER_DATASET, N_TEST_CLASSES, CHANCE_TOP1, CHANCE_TOP5
    if name not in DATASET_PROFILES:
        raise ValueError(f'Unknown dataset {name!r}; choose from {list(DATASET_PROFILES)}')
    profile = DATASET_PROFILES[name]
    _ACTIVE_DATASET = name
    DATA_DIR = profile['data_dir']
    IMAGE_ROOT = profile.get('image_root', IMAGE_ROOT)
    RAW_DIR = profile['raw_dir']
    FIG_DIR = profile['fig_dir']
    CKPT_DIR = profile['ckpt_dir']
    BRAIN_ENCODER_EXP = profile['brain_encoder_exp']
    ENCODER_DATASET = profile['brain_encoder_exp']
    N_TEST_CLASSES = profile['n_test_classes']
    CHANCE_TOP1 = 1.0 / N_TEST_CLASSES
    CHANCE_TOP5 = 5.0 / N_TEST_CLASSES
    if profile.get('channels') is not None:
        CHANNELS[:] = list(profile['channels'])
    return profile


def active_dataset() -> str:
    return _ACTIVE_DATASET


set_active_baseline(DEFAULT_BASELINE)
