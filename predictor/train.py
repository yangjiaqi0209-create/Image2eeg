"""
Per-subject Image-to-EEG generator (63ch, single-stage, multi-blur train).

  python -m predictor.train --sub 1 --gpu 0
  python -m predictor.train --num_sub 10 --gpu 0

Default outputs go to checkpoints/predictor/... (see scripts/train_*.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from encoder.models import clip_z_dim
from predictor.data import (
    CLIP_INPUT_CHOICES,
    _uses_multi_blur_train,
    load_subject_splits,
    make_loaders,
)
from predictor.evaluate import evaluate_subject
from predictor.losses import (
    GeneratorLoss,
    LAMBDA_BAND,
    LAMBDA_CORR,
    LAMBDA_DIV,
    LAMBDA_FREQ,
    LAMBDA_EEG_NCE,
    LAMBDA_HF,
    LAMBDA_BAND_CORR,
    LAMBDA_MARGIN,
    LAMBDA_SEM,
    LAMBDA_TIME,
    LAMBDA_UBP,
    GAMMA_FMAX,
    SEMANTIC_MODES,
    SEMANTIC_TEMPERATURE,
)
from predictor.model import GENERATOR_ARCH_CHOICES, build_generator
from predictor.data import (
    default_ubp_ckpt_path,
    load_frozen_ubp_brain,
    resolve_encoder_ckpt_path,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _brain_config(c_num: int, z_dim: int, timesteps: list) -> dict:
    return {
        'target': 'encoder.models.EEGProjectLayer',
        'params': {
            'c_num': c_num,
            'z_dim': z_dim,
            'timesteps': timesteps,
        },
    }


def _parse_band_weights(spec: str) -> dict:
    """Parse 'delta=2,beta=1.5,gamma=0.4' into a weight dict."""
    if not spec:
        return {}
    out = {}
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        name, val = part.split('=', 1)
        out[name.strip()] = float(val.strip())
    return out


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    sums = {
        'loss': 0.0, 'time': 0.0, 'freq': 0.0, 'band': 0.0,
        'corr': 0.0, 'diversity': 0.0, 'semantic': 0.0,
        'ubp': 0.0, 'eeg_nce': 0.0, 'margin': 0.0,
    }
    n = 0
    for clip_feat, eeg in loader:
        clip_feat = clip_feat.to(device)
        eeg = eeg.to(device)
        optimizer.zero_grad()
        y_hat = model(clip_feat)
        loss, details = criterion(y_hat, eeg, clip_feat)
        loss.backward()
        optimizer.step()
        bs = clip_feat.size(0)
        n += bs
        for k in sums:
            sums[k] += details.get(k, 0.0) * bs
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    sums = {
        'loss': 0.0, 'time': 0.0, 'freq': 0.0, 'band': 0.0,
        'corr': 0.0, 'diversity': 0.0, 'semantic': 0.0,
        'ubp': 0.0, 'eeg_nce': 0.0, 'margin': 0.0,
    }
    n = 0
    for clip_feat, eeg in loader:
        clip_feat = clip_feat.to(device)
        eeg = eeg.to(device)
        y_hat = model(clip_feat)
        _, details = criterion(y_hat, eeg, clip_feat)
        bs = clip_feat.size(0)
        n += bs
        for k in sums:
            sums[k] += details.get(k, 0.0) * bs
    return {k: v / max(n, 1) for k, v in sums.items()}


def _save_checkpoint(path, *, epoch, model, optimizer, scheduler, best_val, best_epoch,
                     epochs_no_improve, seq_len, n_ch, args):
    save_args = dict(vars(args))
    save_args.update(model.config_dict())
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val': best_val,
        'best_epoch': best_epoch,
        'epochs_no_improve': epochs_no_improve,
        'val_loss': best_val,
        'seq_len': seq_len,
        'n_channels': n_ch,
        'args': save_args,
    }, path)


def _try_resume(ckpt_dir, args, model, optimizer, scheduler):
    last_path = os.path.join(ckpt_dir, 'last.pt')
    best_path = os.path.join(ckpt_dir, 'best.pt')
    ckpt_path = last_path if args.resume and os.path.isfile(last_path) else None
    if ckpt_path is None and args.resume and os.path.isfile(best_path):
        ckpt_path = best_path
    if ckpt_path is None:
        return 0, float('inf'), 0, 0, False

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    start_epoch = int(ckpt['epoch']) + 1
    best_val = float(ckpt.get('best_val', ckpt.get('val_loss', float('inf'))))
    best_epoch = int(ckpt.get('best_epoch', ckpt['epoch']))
    epochs_no_improve = int(ckpt.get('epochs_no_improve', 0))
    print(
        f'resume from {ckpt_path}: next epoch {start_epoch}, '
        f'best epoch {best_epoch}, val={best_val:.4f}'
    )
    return start_epoch, best_val, best_epoch, epochs_no_improve, True


def _early_stop_score(val_m: dict, metric: str) -> float:
    """Scalar for early stopping / LR schedule (lower is better)."""
    if metric == 'loss':
        return val_m['loss']
    if metric == 'mse':
        return val_m['time']
    if metric == 'waveform':
        return val_m['time'] + 0.5 * val_m['corr']
    if metric == 'semantic_guard':
        return val_m['time'] + val_m.get('ubp', 0.0) + val_m.get('eeg_nce', 0.0)
    raise ValueError(
        f'Unknown early_stop_metric {metric!r}; '
        f'choose from loss, mse, waveform, semantic_guard'
    )


def _log_line(epoch, dt, train_m, val_m):
    sem_part = (
        f'ubp={train_m.get("ubp", 0.0):.4f} marg={train_m.get("margin", 0.0):.4f} '
        f'| val ubp={val_m.get("ubp", 0.0):.4f} marg={val_m.get("margin", 0.0):.4f} '
        if train_m.get('ubp', 0.0) or train_m.get('margin', 0.0)
        or val_m.get('ubp', 0.0) or val_m.get('margin', 0.0)
        else f'sem={train_m["semantic"]:.4f} | val sem={val_m["semantic"]:.4f} '
    )
    return (
        f'epoch {epoch:03d} ({dt:.1f}s) '
        f'train loss={train_m["loss"]:.4f} mse={train_m["time"]:.4f} '
        f'freq={train_m["freq"]:.4f} band={train_m["band"]:.4f} '
        f'corr={train_m["corr"]:.4f} '
        f'div={train_m["diversity"]:.4f} '
        f'{sem_part}'
        f'| '
        f'val loss={val_m["loss"]:.4f} mse={val_m["time"]:.4f} '
        f'freq={val_m["freq"]:.4f} band={val_m["band"]:.4f} '
        f'corr={val_m["corr"]:.4f} '
        f'div={val_m["diversity"]:.4f}\n'
    )


def train_subject(args, sub: int, device: torch.device):
    sub_tag = f'sub-{sub:02d}'
    ckpt_dir = os.path.join(args.ckpt_dir, sub_tag)
    result_dir = os.path.join(args.result_dir, sub_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, 'log.txt')

    (
        clip_tr_med, clip_tr_high, eeg_tr,
        clip_va, eeg_va,
        clip_test, eeg_test,
    ) = load_subject_splits(
        sub=sub,
        data_dir=args.data_dir,
        model_type=args.vision_backbone,
        clip_input=args.clip_input,
        blur_kernel_size=args.blur_kernel_size,
        blur_delta=args.blur_delta,
        seed=args.seed,
        n_val=args.n_val,
        device=device,
        image_root=args.image_root,
        feature_dir=args.feature_dir,
        avg=not args.no_eeg_avg,
    )

    multi_blur = _uses_multi_blur_train(args.clip_input, args.blur_delta)

    seq_len = eeg_tr.shape[-1]
    n_ch = eeg_tr.shape[1]
    brain_cfg = _brain_config(n_ch, args.z_dim, [0, seq_len])

    train_loader, val_loader = make_loaders(
        clip_tr_med, clip_tr_high, eeg_tr, clip_va, eeg_va,
        multi_blur_train=multi_blur,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    if args.ubp_ckpt:
        ubp_ckpt = args.ubp_ckpt
    elif args.encoder_exp:
        ubp_ckpt = resolve_encoder_ckpt_path(
            _REPO_ROOT, sub, args.ubp_seed, exp_name=args.encoder_exp,
        )
    else:
        ubp_ckpt = default_ubp_ckpt_path(_REPO_ROOT, sub, args.ubp_seed)
    brain = load_frozen_ubp_brain(brain_cfg, ubp_ckpt, device)
    if brain is None:
        print(f'{sub_tag} no UBP checkpoint at {ubp_ckpt}; semantic loss disabled')

    model = build_generator(
        args.generator_arch,
        img_dim=args.z_dim,
        n_channels=n_ch,
        seq_len=seq_len,
        hidden=args.hidden,
        n_layers=args.n_layers,
        per_channel_heads=not args.no_per_channel_heads,
        tcn_dilations=args.tcn_dilations or None,
        tcn_residual=not args.tcn_no_residual,
    ).to(device)

    criterion = GeneratorLoss(
        brain=brain,
        lambda_time=args.lambda_time,
        lambda_freq=args.lambda_freq,
        lambda_band=args.lambda_band,
        lambda_corr=args.lambda_corr,
        lambda_div=args.lambda_div,
        lambda_sem=args.lambda_sem if brain else 0.0,
        lambda_ubp=args.lambda_ubp,
        lambda_margin=args.lambda_margin,
        lambda_eeg_nce=args.lambda_eeg_nce,
        lambda_hf=args.lambda_hf,
        lambda_band_corr=args.lambda_band_corr,
        semantic_mode=args.semantic_mode,
        semantic_temperature=args.semantic_temperature,
        gamma_fmax=args.gamma_fmax,
        band_weights=_parse_band_weights(args.band_weights) or None,
        hf_fmin=args.hf_fmin,
        hf_fmax=args.hf_fmax,
        low_freq_emphasis=args.low_freq_emphasis,
        waveform_only=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    best_val = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    ckpt_path = os.path.join(ckpt_dir, 'best.pt')
    last_path = os.path.join(ckpt_dir, 'last.pt')
    stopped_early = False

    start_epoch, best_val, best_epoch, epochs_no_improve, resumed = _try_resume(
        ckpt_dir, args, model, optimizer, scheduler,
    )
    if args.init_ckpt and os.path.isfile(args.init_ckpt) and not resumed:
        init_ckpt = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(init_ckpt['model_state_dict'], strict=True)
        print(f'{sub_tag} init weights from {args.init_ckpt} (epoch {init_ckpt.get("epoch", "?")})')
    log_mode = 'a' if resumed else 'w'
    with open(log_path, log_mode) as log_f:
        if resumed:
            log_f.write(f'\n--- resumed {time.strftime("%Y-%m-%d %H:%M:%S")} ---\n')
        else:
            log_f.write(
                f'{sub_tag} generator training ({n_ch}ch, clip_input={args.clip_input}, '
                f'multi_blur={multi_blur}, arch={args.generator_arch})\n'
            )
            log_f.write(f'args: {vars(args)}\n\n')

        for epoch in range(start_epoch, args.epochs):
            t0 = time.time()
            train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_m = eval_epoch(model, val_loader, criterion, device)
            stop_score = _early_stop_score(val_m, args.early_stop_metric)
            scheduler.step(stop_score)

            line = _log_line(epoch, time.time() - t0, train_m, val_m)
            print(sub_tag, line.strip())
            log_f.write(line)
            log_f.flush()

            if stop_score < best_val:
                best_val = stop_score
                best_epoch = epoch
                epochs_no_improve = 0
                _save_checkpoint(
                    ckpt_path, epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, best_val=best_val, best_epoch=best_epoch,
                    epochs_no_improve=epochs_no_improve, seq_len=seq_len, n_ch=n_ch, args=args,
                )
            else:
                epochs_no_improve += 1

            _save_checkpoint(
                last_path, epoch=epoch, model=model, optimizer=optimizer,
                scheduler=scheduler, best_val=best_val, best_epoch=best_epoch,
                epochs_no_improve=epochs_no_improve, seq_len=seq_len, n_ch=n_ch, args=args,
            )

            if epochs_no_improve >= args.early_stop_patience:
                stop_line = (
                    f'early stopping at epoch {epoch}: '
                    f'best epoch {best_epoch}, val={best_val:.4f}\n'
                )
                print(sub_tag, stop_line.strip())
                log_f.write(stop_line)
                stopped_early = True
                break

    print(
        f'{sub_tag} best val={best_val:.4f} at epoch {best_epoch} '
        f'{"(early stopped) " if stopped_early else ""}-> {ckpt_path}'
    )

    if clip_test is not None and eeg_test is not None:
        eval_path = ckpt_path if (stopped_early and os.path.isfile(ckpt_path)) else last_path
        metrics = evaluate_subject(
            model_path=eval_path,
            clip_test=clip_test,
            eeg_test=eeg_test,
            device=device,
            result_dir=result_dir,
            brain_config=brain_cfg,
            ubp_ckpt=ubp_ckpt if brain is not None else None,
            batch_size=args.batch_size_test,
        )
        eval_ckpt = torch.load(eval_path, map_location='cpu', weights_only=False)
        metrics.update({
            'best_epoch': best_epoch,
            'best_val_loss': best_val,
            'stopped_early': stopped_early,
            'checkpoint': os.path.basename(eval_path),
            'eval_epoch': int(eval_ckpt.get('epoch', -1)),
        })
        with open(os.path.join(result_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    return best_val


def parse_args():
    p = argparse.ArgumentParser(description='Train 63ch Image-to-EEG generator (UBP blur prior)')
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
        choices=('loss', 'mse', 'waveform', 'semantic_guard'),
        help='Validation scalar for early stopping / LR schedule (mse ignores extra freq terms).',
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
                   help='Primary blur kernel (inference uses this level)')
    p.add_argument('--blur_delta', type=int, default=6,
                   help='High blur = blur_kernel_size + blur_delta (ubp train aug only)')
    p.add_argument(
        '--clip_input', type=str, default='ubp', choices=CLIP_INPUT_CHOICES,
        help='CLIP visual prior: sharp | uniform | fovea | ubp (medium+high aug)',
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
    p.add_argument('--lambda_hf', type=float, default=LAMBDA_HF,
                   help='High-frequency excess penalty (38–45 Hz artifacts).')
    p.add_argument('--lambda_band_corr', type=float, default=LAMBDA_BAND_CORR,
                   help='Band-limited waveform correlation loss (delta/beta emphasis via --band_weights).')
    p.add_argument('--gamma_fmax', type=float, default=GAMMA_FMAX,
                   help='Upper Hz bound for gamma band in bandpower loss.')
    p.add_argument('--band_weights', type=str, default='',
                   help='Per-band weights, e.g. delta=2,beta=1.5,gamma=0.4')
    p.add_argument('--hf_fmin', type=float, default=35.0)
    p.add_argument('--hf_fmax', type=float, default=45.0)
    p.add_argument('--low_freq_emphasis', type=float, default=0.0,
                   help='Emphasize low-frequency bins in spectral L1 (e.g. 8.0 for delta).')
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
        default=os.path.join(_REPO_ROOT, 'data/things-eeg/Preprocessed_data_250Hz_whiten'),
    )
    p.add_argument('--image_root', default=None, help='Override image root for CLIP encoding')
    p.add_argument('--feature_dir', default=None, help='Override CLIP feature cache directory')
    p.add_argument('--encoder_exp', default=None, help='Brain encoder experiment name (e.g. nod_intra-subject_ubp_...)')
    p.add_argument('--no_eeg_avg', action='store_true',
                   help='Keep each trial separate (NOD: one unique image per trial, no rep averaging)')
    p.add_argument('--ubp_ckpt', default=None)
    p.add_argument('--ubp_seed', type=int, default=0)
    p.add_argument('--ckpt_dir', default=os.path.join(_REPO_ROOT, 'checkpoints/predictor/THINGSEEG2/Ours/full'))
    p.add_argument('--result_dir', default=os.path.join(_REPO_ROOT, 'results/generator_runs'))
    return p.parse_args()


def main():
    args = parse_args()
    args.z_dim = clip_z_dim(args.vision_backbone)
    set_seed(args.seed)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    subs = [args.sub] if args.sub is not None else list(range(args.sub_start, args.num_sub + 1))
    for sub in subs:
        train_subject(args, sub, device)


if __name__ == '__main__':
    main()
