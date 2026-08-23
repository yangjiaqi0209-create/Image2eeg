"""EEG datasets, channel layouts, and CLIP feature loading."""

from __future__ import annotations

import gc
import logging
import os

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from encoder.models import CLIP_BACKBONES
from encoder.utils import get_device, instantiate_from_config

# Alljoined-1.6M (32-ch Emotiv Flex 2); order matches official flat .npy ch_names.
ALLJOINED_CHANNELS = [
    'Cz', 'FCz', 'AFz', 'Fp1', 'F5', 'F1', 'CP5', 'CP3', 'CP1', 'P1',
    'P3', 'P5', 'P7', 'PO7', 'PO3', 'O1', 'Pz', 'POz', 'Oz', 'O2',
    'PO4', 'PO8', 'P8', 'P6', 'P4', 'P2', 'CP2', 'CP4', 'CP6', 'F2', 'F6', 'Fp2',
]

THINGS_CHANNELS = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]


def resolve_clip_pretrained(model_type, default_tag):
    local_root = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
    local_files = {
        'RN50': os.path.join(local_root, 'RN50', 'open_clip_pytorch_model.bin'),
    }
    local_path = local_files.get(model_type)
    if local_path and os.path.isfile(local_path):
        logging.info(f"Using local CLIP weights: {local_path}")
        return local_path
    return default_tag


def load_eeg_data(config):
    """Intra-subject loaders: train / val (or test-as-val) / test."""
    test_dataset = EEGDataset(config, mode='test')
    print('init test_dataset success')
    train_dataset = EEGDataset(config, mode='train')
    print('init train_dataset success')
    test_loader = DataLoader(
        test_dataset, batch_size=config['data']['test_batch_size'],
        shuffle=False, drop_last=False, num_workers=25, pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['data']['train_batch_size'],
        shuffle=True, drop_last=False, num_workers=32, pin_memory=True,
    )
    val_path = os.path.join(
        config['data']['data_dir'],
        config['data']['subjects'][0],
        'val.pt',
    )
    if os.path.isfile(val_path):
        val_dataset = EEGDataset(config, mode='val')
        print('init val_dataset success')
        val_loader = DataLoader(
            val_dataset, batch_size=config['data']['val_batch_size'],
            shuffle=False, drop_last=False, num_workers=25, pin_memory=True,
        )
    else:
        val_loader = test_loader
    return train_loader, val_loader, test_loader


class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = config['data'].get('channels') or THINGS_CHANNELS
        # "None" means keep every channel present in the .pt.
        if self.selected_ch == "None":
            self.selected_ch = None

        self.avg = config['data'][f'{mode}_avg'] if mode != 'val' else config['data'].get(
            'val_avg', config['data'].get('test_avg', False),
        )
        self.blur_type = config['data']['blur_type']
        self.timesteps = config['data']['timesteps']
        data_cfg = config.get('data', {})
        default_per_trials = 4 if self.mode == 'train' else 80
        self.per_trials = int(data_cfg.get('per_trials', default_per_trials))

        self.data_paths = [
            os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects
        ]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]
        if self.loaded_data and self.loaded_data[0].get('ch_names') is not None:
            self.channels = list(self.loaded_data[0]['ch_names'])
            if self.selected_ch is None or self.selected_ch == self.channels:
                self.selected_ch = self.channels

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        feature_parent = data_cfg.get('img_feature_dir')
        if feature_parent:
            data_dir = os.path.join(
                feature_parent, f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}",
            )
        else:
            data_dir = os.path.join(
                self.data_dir, '../Image_feature',
                f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}",
            )
        os.makedirs(data_dir, exist_ok=True)

        features_filename = os.path.join(data_dir, f"{self.name}_{mode}.pt")
        self.image_root = data_cfg.get('image_root') or os.path.join(
            self.data_dir, '../Image_set_Resize',
        )

        self.blur_transform = instantiate_from_config(config['data']['blur_type'])
        self.process_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename, weights_only=False)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            device = get_device('auto')
            pretrained = resolve_clip_pretrained(
                self.model_type, CLIP_BACKBONES[self.model_type]['pretrained'],
            )
            self.vlmodel, _, _ = open_clip.create_model_and_transforms(
                self.model_type, device=f"cuda:{device}", pretrained=pretrained,
            )
            for param in self.vlmodel.parameters():
                param.requires_grad = False
            self.vlmodel.eval()
            _raw = torch.load(self.data_paths[0], weights_only=False)
            clip_imgs = np.unique(np.asarray(_raw['img']).reshape(-1))
            clip_texts = np.unique(np.asarray(_raw['text']).reshape(-1))
            self.img_features = self.ImageEncoder(clip_imgs)
            self.text_features = self.TextEncoder(clip_texts)
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, features_filename)

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path, weights_only=False)
        eeg = loaded_data['eeg']
        if torch.is_tensor(eeg):
            loaded_data['eeg'] = eeg
        else:
            loaded_data['eeg'] = torch.from_numpy(np.asarray(eeg))
        if 'ch_names' in loaded_data and loaded_data['ch_names'] is not None:
            file_channels = list(loaded_data['ch_names'])
        else:
            file_channels = self.channels

        if self.selected_ch:
            selected_idx = [file_channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]
            loaded_data['ch_names'] = [file_channels[i] for i in selected_idx]
        else:
            loaded_data['ch_names'] = file_channels

        if self.avg:
            avg_data = {
                'eeg': loaded_data['eeg'].mean(axis=1),
                'label': loaded_data['label'][:, 0],
                'img': loaded_data['img'][:, 0],
                'text': loaded_data['text'][:, 0],
                'session': loaded_data['session'],
                'times': loaded_data['times'],
            }
            loaded_data = avg_data
        else:
            trial_dim = loaded_data['eeg'].shape[1]
            per_image = loaded_data.get('trial_format') == 'per_image' or trial_dim == 1
            _data = {
                'eeg': loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:]),
                'per_image': per_image,
                'label': loaded_data['label'].reshape(-1),
                'img': loaded_data['img'].reshape(-1),
                'text': loaded_data['text'].reshape(-1),
                'session': loaded_data['session'].reshape(-1),
                'times': loaded_data['times'],
            }
            if per_image:
                _data['eeg_avg'] = _data['eeg']
            else:
                _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            loaded_data = _data

        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None):
        if blur_transform is None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()

        set_images = sorted(set(images))
        batch_size = 128
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]
            device = next(self.vlmodel.parameters()).device
            ele = [
                self.process_transform(
                    blur_transform(
                        Image.open(os.path.join(self.image_root, img)).convert("RGB"),
                    ),
                )
                for img in batch_images
            ]
            image_inputs = torch.stack(ele).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features / batch_image_features.norm(
                dim=-1, keepdim=True,
            )
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        return {
            set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))
        }

    @torch.no_grad()
    def TextEncoder(self, text):
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text])
        device = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return {set_text[i]: text_features[i].float().cpu() for i in range(len(set_text))}

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg or self.loaded_data[subject].get('per_image'):
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        if isinstance(img_path, (np.ndarray, list, tuple)):
            img_path = str(np.asarray(img_path).reshape(-1)[0])
        else:
            img_path = str(img_path)
        text_key = self.loaded_data[subject]['text'][trial_index]
        if isinstance(text_key, (np.ndarray, list, tuple)):
            text_key = str(np.asarray(text_key).reshape(-1)[0])
        else:
            text_key = str(text_key)

        img_features = self.img_features[img_path]
        text = f"This is a {text_key}."
        text_features = self.text_features[text_key]
        session = self.loaded_data[subject]['session'][trial_index]

        return {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': label,
            'img_path': img_path,
            'img': 'None',
            'img_features': img_features,
            'text': text,
            'text_features': text_features,
            'session': session,
            'subject': subject,
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }

    def __len__(self):
        return self.trial_all_subjects
