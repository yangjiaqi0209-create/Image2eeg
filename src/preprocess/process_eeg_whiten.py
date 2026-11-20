"""Epoch, whiten, and package THINGS-EEG into train.pt and test.pt."""

import argparse
import os

import mne
import numpy as np
import scipy.linalg
import torch
from sklearn.discriminant_analysis import _cov
from sklearn.utils import shuffle
from tqdm import tqdm

N_SESSIONS = 4
SEED = 20200220
RESAMPLE_FREQ = 250
TMIN = -0.2
TMAX = 1.0
PROJECT_DIR = 'data/things-eeg'

CHANNEL_ORDER = [
    'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
    'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
    'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2',
]


def get_args_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', type=int, required=True)
    parser.add_argument('--raw_data_dir', type=str, default=None,
                        help='Path to Raw_data directory (default: data/things-eeg/Raw_data)')
    parser.add_argument('--img_dir', type=str, default=None,
                        help='Path to Image_set_Resize directory (default: data/things-eeg/Image_set_Resize)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to save preprocessed .pt files (default: data/things-eeg/Preprocessed_data_250Hz_whiten)')
    return parser.parse_args(argv)


def resolve_raw_eeg_path(mode, sub, ses, raw_data_dir):
    sub_dir = os.path.join(raw_data_dir, f'sub-{sub:02d}', f'ses-{ses:02d}')
    candidates = [f'raw_eeg_{mode}.npy']
    if mode == 'train':
        candidates.append('raw_eeg_training.npy')
    for name in candidates:
        path = os.path.join(sub_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f'No raw EEG file found in {sub_dir} (tried {candidates})')


def resolve_train_img_dir(img_dir):
    for name in ('train_images', 'training_images'):
        path = os.path.join(img_dir, name)
        if os.path.isdir(path):
            return path, name
    raise FileNotFoundError(f'No train image directory found under {img_dir}')


def collect_image_metadata(img_directory, img_dir_root):
    folders = sorted(
        name for name in os.listdir(img_directory)
        if os.path.isdir(os.path.join(img_directory, name))
    )
    images, labels, texts = [], [], []
    for label, folder in enumerate(folders):
        folder_path = os.path.join(img_directory, folder)
        filenames = sorted(
            name for name in os.listdir(folder_path)
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        for name in filenames:
            rel_path = os.path.relpath(os.path.join(folder_path, name), img_dir_root)
            if rel_path.startswith('training_images' + os.sep):
                rel_path = 'train_images' + rel_path[len('training_images'):]
            images.append(rel_path)
            labels.append(label)
            texts.append(name.rsplit('_', 1)[0])
    return images, labels, texts


def _whiten_partition(data, sigma_inv):
    trials = data.reshape(-1, data.shape[2], data.shape[3])
    return (trials.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2).reshape(data.shape)


def mvnn(epoched_test, epoched_train, mvnn_dim='epochs'):
    """Whiten each session using training covariance only."""
    whitened_test, whitened_train = [], []
    for session in range(N_SESSIONS):
        session_data = [epoched_test[session], epoched_train[session]]
        n_channels = session_data[0].shape[2]
        sigma_part = np.empty((len(session_data), n_channels, n_channels))
        for partition, data in enumerate(session_data):
            sigma_cond = np.empty((data.shape[0], n_channels, n_channels))
            for condition in tqdm(range(data.shape[0])):
                cond_data = data[condition]
                if mvnn_dim == 'time':
                    sigma_cond[condition] = np.mean([
                        _cov(cond_data[:, :, t], shrinkage='auto')
                        for t in range(cond_data.shape[2])
                    ], axis=0)
                elif mvnn_dim == 'epochs':
                    sigma_cond[condition] = np.mean([
                        _cov(np.transpose(cond_data[epoch]), shrinkage='auto')
                        for epoch in range(cond_data.shape[0])
                    ], axis=0)
            sigma_part[partition] = sigma_cond.mean(axis=0)

        sigma_inv = scipy.linalg.fractional_matrix_power(sigma_part[1], -0.5)
        whitened_test.append(_whiten_partition(session_data[0], sigma_inv))
        whitened_train.append(_whiten_partition(session_data[1], sigma_inv))
    return whitened_test, whitened_train


def epoch_data(mode, sub, raw_data_dir):
    """Epoch each session and keep the original seeded repetition selection."""
    epoched_data, img_conditions = [], []
    for session in range(N_SESSIONS):
        eeg_path = resolve_raw_eeg_path(mode, sub, session + 1, raw_data_dir)
        eeg_data = np.load(eeg_path, allow_pickle=True).item()
        info = mne.create_info(eeg_data['ch_names'], eeg_data['sfreq'], eeg_data['ch_types'])
        raw = mne.io.RawArray(eeg_data['raw_eeg_data'], info)
        events = mne.find_events(raw, stim_channel='stim')
        raw.pick_channels(CHANNEL_ORDER, ordered=True)
        idx_target = np.where(events[:, 2] == 99999)[0]
        events = np.delete(events, idx_target, 0)

        epochs = mne.Epochs(
            raw, events, tmin=TMIN, tmax=TMAX, baseline=(None, 0), preload=True,
        )
        if RESAMPLE_FREQ < 1000:
            epochs.resample(RESAMPLE_FREQ)
        ch_names = epochs.info['ch_names']
        times = epochs.times
        data = epochs.get_data()
        events = epochs.events[:, 2]
        img_cond = np.unique(events)
        max_rep = 20 if mode == 'test' else 2
        sorted_data = np.zeros((len(img_cond), max_rep, data.shape[1], data.shape[2]))
        for condition, image_id in enumerate(img_cond):
            idx = np.where(events == image_id)[0]
            idx = shuffle(idx, random_state=SEED, n_samples=max_rep)
            sorted_data[condition] = data[idx]
        print(sorted_data[:, :, :, -RESAMPLE_FREQ:].shape)
        epoched_data.append(sorted_data[:, :, :, -RESAMPLE_FREQ:])
        img_conditions.append(img_cond)
    return epoched_data, img_conditions, ch_names, times


def merge_test_sessions(whitened_test):
    """Concatenate repetitions in session order, preserving float64 session IDs."""
    merged = np.concatenate(whitened_test, axis=1)
    sessions = np.zeros((200, 80))
    offset = 0
    for session, data in enumerate(whitened_test):
        end = offset + data.shape[1]
        sessions[:, offset:end] = session
        offset = end
    return merged, sessions


def merge_train_sessions(whitened_train, img_conditions_train):
    """Group session trials by image, preserving the original repetition order."""
    data = np.concatenate(whitened_train, axis=0)
    conditions = np.concatenate(img_conditions_train, axis=0)
    sessions = np.zeros((33080, 2))
    offset = 0
    for session, session_data in enumerate(whitened_train):
        end = offset + session_data.shape[0]
        sessions[offset:end] = session
        offset = end

    merged = np.zeros((len(np.unique(conditions)), data.shape[1] * 2, *data.shape[2:]))
    sorted_sessions = np.zeros((16540, 4))
    for image in range(len(merged)):
        idx = np.where(conditions == image + 1)[0]
        for repetition, row in enumerate(idx):
            sorted_sessions[image, repetition * 2:repetition * 2 + 2] = sessions[row]
        merged[image] = np.concatenate([data[row] for row in idx], axis=0)
    return merged, sorted_sessions


def save_split(path, eeg, sessions, img_directory, img_dir_root, repeats, ch_names, times):
    """Save the shared THINGS payload schema without changing dtypes or metadata."""
    images, labels, texts = collect_image_metadata(img_directory, img_dir_root)
    payload = {
        'eeg': eeg.astype(np.float16),
        'label': np.tile(np.array(labels)[:, None], (1, repeats)),
        'img': np.tile(np.array(images)[:, None], (1, repeats)),
        'text': np.tile(np.array(texts)[:, None], (1, repeats)),
        'session': sessions,
        'ch_names': ch_names,
        'times': times,
    }
    torch.save(payload, path, pickle_protocol=5)
    print(f'Wrote {path}  eeg={payload["eeg"].shape}  dtype={payload["eeg"].dtype}')


def main(argv=None):
    args = get_args_parser(argv)
    raw_data_dir = args.raw_data_dir or os.path.join(PROJECT_DIR, 'Raw_data')
    img_dir = args.img_dir or os.path.join(PROJECT_DIR, 'Image_set_Resize')
    output_base = args.output_dir or os.path.join(
        PROJECT_DIR, f'Preprocessed_data_{RESAMPLE_FREQ}Hz_whiten',
    )
    save_dir = os.path.join(output_base, f'sub-{args.subject:02d}')
    os.makedirs(save_dir, exist_ok=True)

    eeg_test, _, ch_names, times = epoch_data('test', args.subject, raw_data_dir)
    eeg_train, conditions, _, _ = epoch_data('train', args.subject, raw_data_dir)
    whitened_test, whitened_train = mvnn(eeg_test, eeg_train)
    del eeg_test, eeg_train

    merged_test, test_sessions = merge_test_sessions(whitened_test)
    del whitened_test
    save_split(
        os.path.join(save_dir, 'test.pt'), merged_test, test_sessions,
        os.path.join(img_dir, 'test_images'), img_dir, 80, ch_names, times,
    )

    merged_train, train_sessions = merge_train_sessions(whitened_train, conditions)
    del whitened_train
    train_images, _ = resolve_train_img_dir(img_dir)
    save_split(
        os.path.join(save_dir, 'train.pt'), merged_train, train_sessions,
        train_images, img_dir, 4, ch_names, times,
    )


if __name__ == '__main__':
    main()
