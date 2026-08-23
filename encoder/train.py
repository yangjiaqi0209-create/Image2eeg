import argparse
import json
import os
import shutil

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.optim import AdamW

from encoder.data import load_eeg_data
from encoder.models import CLIP_BACKBONES, clip_z_dim
from encoder.registry import apply_profile_to_config, get_dataset
from encoder.utils import ClipLoss, get_device, instantiate_from_config, update_config

device = get_device('auto')


def load_model(config, train_loader):
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)
    return PLModel(model, config, train_loader)


class PLModel(pl.LightningModule):
    def __init__(self, model, config, train_loader):
        super().__init__()
        self.config = config
        for key, value in model.items():
            setattr(self, key, value)
        self.criterion = ClipLoss()
        self.all_predicted_classes = []
        self.all_true_labels = []
        self.mAP_total = 0
        self.match_similarities = []

    def forward(self, batch):
        eeg = batch['eeg']
        img_z = batch['img_features']
        eeg_z = self.brain(eeg)
        img_z = img_z / img_z.norm(dim=-1, keepdim=True)
        logit_scale = self.brain.softplus(self.brain.logit_scale)
        eeg_loss, img_loss, _ = self.criterion(eeg_z, img_z, logit_scale)
        loss = (eeg_loss.mean() + img_loss.mean()) / 2
        return eeg_z, img_z, loss

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log(
            'train_loss', loss, on_step=True, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size,
        )

        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = eeg_z @ img_z.T
        _, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_correct = all_predicted_classes[:, 0] == all_true_labels
            top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
            top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
            top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log(
            'val_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size,
        )
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = eeg_z @ img_z.T
        _, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_correct = all_predicted_classes[:, 0] == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, loss = self(batch)
        self.log(
            'test_loss', loss, on_step=False, on_epoch=True,
            prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size,
        )
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)
        similarity = eeg_z @ img_z.T
        _, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())
        for i in range(similarity.shape[0]):
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == i).nonzero()[0][0] + 1
            self.mAP_total += 1 / rank
        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_correct = all_predicted_classes[:, 0] == all_true_labels
        top_1_accuracy = sum(top_1_correct) / len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct) / len(top_k_correct)

        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0

        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)

        self.all_predicted_classes = []
        self.all_true_labels = []
        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {
            'test_loss': avg_test_loss.item(),
            'test_top1_acc': top_1_accuracy.item(),
            'test_top5_acc': top_k_accuracy.item(),
            'mAP': self.mAP,
            'similarity': self.match_similarities,
        }

    def configure_optimizers(self):
        weight_decay = float(self.config.get('train', {}).get('weight_decay', 1e-4))
        name = self.config['train']['optimizer']
        if name != 'AdamW':
            raise ValueError(f'Only AdamW is supported, got {name!r}')
        return [AdamW(self.parameters(), lr=self.config['train']['lr'], weight_decay=weight_decay)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/eeg/fixed_fovea.yaml")
    parser.add_argument(
        "--dataset", type=str, default="eeg",
        help="Legacy tag written into config (THINGS: eeg; Alljoined: alljoined)",
    )
    parser.add_argument(
        "--dataset-profile", type=str, default=None,
        help="Registry id (e.g. alljoined_eeg). Omit for THINGS-EEG.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=str, default='sub-08')
    parser.add_argument(
        "--exp_setting", type=str, default='intra-subject',
        help="Only intra-subject is supported.",
    )
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--brain_backbone", type=str)
    parser.add_argument("--vision_backbone", type=str)

    opt = parser.parse_args()
    if opt.exp_setting != 'intra-subject':
        raise ValueError('Only exp_setting=intra-subject is supported')

    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    if opt.dataset_profile:
        config = apply_profile_to_config(config, opt.dataset_profile)
        prof = get_dataset(opt.dataset_profile)
        print(f"Dataset profile: {prof.display_name} ({prof.id})")
        print(f"  brain input: {prof.c_num} ch x {prof.seq_len} t = {prof.brain_input_dim}")
    config['data']['subjects'] = [opt.subjects]
    if opt.vision_backbone not in CLIP_BACKBONES:
        raise KeyError(
            f'Unknown vision_backbone {opt.vision_backbone!r}; '
            f'choose from {list(CLIP_BACKBONES)}'
        )
    config['z_dim'] = clip_z_dim(opt.vision_backbone)
    OmegaConf.resolve(config)
    print(config)

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = TensorBoardLogger(
        config['save_dir'], name=config['name'],
        version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}",
    )
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, val_loader, test_loader = load_eeg_data(config)
    print(
        f"train num: {len(train_loader.dataset)},"
        f"val num: {len(val_loader.dataset)}, test num: {len(test_loader.dataset)}"
    )
    pl_model = load_model(config, train_loader)

    monitor_val = (
        config.get('dataset') == 'alljoined'
        or bool(config.get('train', {}).get('monitor_val', False))
        or bool(config.get('data', {}).get('separate_val', False))
    )
    es_patience = int(config.get('train', {}).get('early_stop_patience', 15 if monitor_val else 5))
    if monitor_val:
        checkpoint_callback = ModelCheckpoint(
            save_last=True, monitor='val_top1_acc', mode='max', save_top_k=1,
        )
        early_stop_callback = EarlyStopping(
            monitor='val_top1_acc', min_delta=0.001, patience=es_patience,
            verbose=True, mode='max',
        )
        test_ckpt = 'best'
    else:
        checkpoint_callback = ModelCheckpoint(save_last=True)
        early_stop_callback = EarlyStopping(
            monitor='train_loss', min_delta=0.001, patience=es_patience,
            verbose=False, mode='min',
        )
        test_ckpt = 'last'

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
        devices = [device]
    else:
        strategy = "auto"
        devices = 1

    trainer = Trainer(
        log_every_n_steps=10, strategy=strategy,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=config['train']['epoch'], devices=devices,
        accelerator='cuda', logger=logger,
    )
    print(trainer.logger.log_dir)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path='last')
    test_results = trainer.test(ckpt_path=test_ckpt, dataloaders=test_loader)

    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()
