"""Command-line options for image-to-EEG training."""

import argparse
import os

from predictor.data import CLIP_INPUT_CHOICES
from predictor.losses import (
    GAMMA_FMAX,
    LAMBDA_BAND,
    LAMBDA_BAND_CORR,
    LAMBDA_CORR,
    LAMBDA_DIV,
    LAMBDA_EEG_NCE,
    LAMBDA_ENC,
    LAMBDA_FREQ,
    LAMBDA_MARGIN,
    LAMBDA_SEM,
    LAMBDA_TIME,
    LAMBDA_UBP,
    SEMANTIC_MODES,
    SEMANTIC_TEMPERATURE,
)
from predictor.model import GENERATOR_ARCH_CHOICES

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train Image-to-EEG predictor (CLIP → EEG)')
    p.add_argument('--sub', type=int, default=None)
    p.add_argument('--num_sub', type=int, default=10)
    p.add_argument('--sub_start', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--batch_size_test', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=2023)
    p.add_argument('--n_val', type=int, default=740)
    p.add_argument('--early_stop_patience', type=int, default=15)
    p.add_argument(
        '--early_stop_metric', type=str, default='loss',
        choices=('loss', 'semantic_guard', 'corr', 'enc'),
        help='Validation scalar for early stopping / LR schedule (lower is better).',
    )
    p.add_argument('--resume', action='store_true')
    p.add_argument(
        '--init_ckpt', default=None,
        help='Load model weights only before training (no optimizer resume; for stage-2 fine-tune).',
    )
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--vision_backbone', type=str, default='RN50')
    p.add_argument('--z_dim', type=int, default=1024)
    p.add_argument('--blur_kernel_size', type=int, default=51,
                   help='Fovea blur kernel size (ignored for sharp)')
    p.add_argument(
        '--clip_input', type=str, default='fovea', choices=CLIP_INPUT_CHOICES,
        help='CLIP visual prior: sharp | fovea',
    )
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--n_layers', type=int, default=4)
    p.add_argument(
        '--generator_arch', type=str, default='full', choices=GENERATOR_ARCH_CHOICES,
        help='Generator architecture: full | no_dconv | tcn_only | no_self_attn.',
    )
    p.add_argument(
        '--no_per_channel_heads', action='store_true',
        help='Use shared linear output projection instead of per-channel heads.',
    )
    p.add_argument(
        '--tcn_dilations', type=str, default='',
        help='Comma-separated TCN dilations, e.g. 1,2,4 (default: 1,2,4,8,16).',
    )
    p.add_argument(
        '--tcn_no_residual', action='store_true',
        help='Disable residual skip connections inside dilated TCN blocks.',
    )
    p.add_argument(
        '--n_spatial', type=int, default=0,
        help='If >0, decode via rank-K spatial maps (K, C) instead of per-channel heads.',
    )
    p.add_argument('--lambda_time', type=float, default=LAMBDA_TIME)
    p.add_argument('--lambda_freq', type=float, default=LAMBDA_FREQ)
    p.add_argument('--lambda_band', type=float, default=LAMBDA_BAND)
    p.add_argument('--lambda_corr', type=float, default=LAMBDA_CORR)
    p.add_argument('--lambda_div', type=float, default=LAMBDA_DIV)
    p.add_argument('--lambda_sem', type=float, default=LAMBDA_SEM)
    p.add_argument('--lambda_ubp', type=float, default=LAMBDA_UBP,
                   help='UBP anchor loss weight (semantic_mode=ubp_margin).')
    p.add_argument('--lambda_eeg_nce', type=float, default=LAMBDA_EEG_NCE,
                   help='EEG–EEG InfoNCE weight: brain(y_hat_i) vs brain(y_*) in-batch.')
    p.add_argument('--lambda_margin', type=float, default=LAMBDA_MARGIN,
                   help='CLIP InfoNCE weight (semantic_mode=ubp_margin).')
    p.add_argument('--lambda_band_corr', type=float, default=LAMBDA_BAND_CORR,
                   help='Band-limited waveform correlation loss (delta/beta emphasis via --band_weights).')
    p.add_argument('--lambda_enc', type=float, default=LAMBDA_ENC,
                   help='Across-image Pearson at each channel×time (encoding / %%NC analogue).')
    p.add_argument(
        '--channel_region_weights', type=str, default='',
        help='Region weights for MSE/corr/enc, e.g. occipital=2,temporal=1.5,frontal=0.5',
    )
    p.add_argument(
        '--time_window_weights', type=str, default='',
        help='Time-window weights in ms, e.g. 80-400:2.5,0-80:0.5 (unspecified samples=1).',
    )
    p.add_argument('--gamma_fmax', type=float, default=GAMMA_FMAX,
                   help='Upper Hz bound for gamma band in bandpower loss.')
    p.add_argument(
        '--sfreq', type=float, default=250.0,
        help='Sampling rate in Hz for spectral losses (default 250 for EEG).',
    )
    p.add_argument('--band_weights', type=str, default='',
                   help='Per-band weights, e.g. delta=2,beta=1.5,gamma=0.4')
    p.add_argument(
        '--semantic_mode', type=str, default='ubp_margin', choices=SEMANTIC_MODES,
        help='clip: cos(brain(y_hat), clip) | ubp_margin: UBP anchor + InfoNCE.',
    )
    p.add_argument(
        '--semantic_temperature', type=float, default=SEMANTIC_TEMPERATURE,
        help='Temperature for margin InfoNCE (ubp_margin mode).',
    )
    p.add_argument(
        '--data_dir',
        default=os.path.join(_REPO_ROOT, 'data/processed/things_eeg2/preprocessed'),
    )
    p.add_argument('--image_root', default=None, help='Override image root for CLIP encoding')
    p.add_argument('--feature_dir', default=None, help='Override CLIP feature cache directory')
    p.add_argument('--encoder_exp', default=None, help='Brain encoder experiment name (e.g. nod_intra-subject_ubp_...)')
    p.add_argument('--no_eeg_avg', action='store_true',
                   help='Keep each trial separate (NOD: one unique image per trial, no rep averaging)')
    p.add_argument('--skip-test', action='store_true',
                   help='Do not read or score the test split during training.')
    p.add_argument('--ubp_ckpt', default=None)
    p.add_argument('--ubp_seed', type=int, default=0)
    p.add_argument('--ckpt_dir', default=os.path.join(_REPO_ROOT, 'checkpoints/reproductions/generator'))
    p.add_argument('--result_dir', default=os.path.join(_REPO_ROOT, 'results/generator_runs'))
    return p.parse_args(argv)
