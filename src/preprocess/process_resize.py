"""Resize THINGS stimuli to 224 × 224 while preserving relative paths."""

import argparse
import os

from PIL import Image
from torchvision import transforms


def get_args_parser(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--type', type=str, default='eeg')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Source image directory')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory for resized images')
    return parser.parse_args(argv)


def resize_images(data_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    image_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(data_dir)
        for name in files
        if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    resize = transforms.Resize((224, 224))
    for path in image_paths:
        with Image.open(path) as image:
            image = resize(image)
            rel_path = os.path.relpath(path, data_dir)
            if rel_path.startswith('training_images' + os.sep):
                rel_path = os.path.join(
                    'train_images',
                    os.path.relpath(path, os.path.join(data_dir, 'training_images')),
                )
            save_path = os.path.join(save_dir, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)


def main(argv=None):
    args = get_args_parser(argv)
    if args.data_dir and args.save_dir:
        data_dir, save_dir = args.data_dir, args.save_dir
    elif args.type == 'eeg':
        data_dir = 'data/things-eeg/Image_set'
        save_dir = 'data/things-eeg/Image_set_Resize'
    else:
        raise ValueError('Provide --data_dir and --save_dir, or set --type to eeg')
    resize_images(data_dir, save_dir)


if __name__ == '__main__':
    main()
