
import os
from PIL import Image
from torchvision import transforms
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--type', type=str, default='eeg')
    parser.add_argument('--data_dir', type=str, default=None,
        help='Source image directory')
    parser.add_argument('--save_dir', type=str, default=None,
        help='Output directory for resized images')
    return parser.parse_args()

args = get_args_parser()
if args.data_dir and args.save_dir:
    data_dir = args.data_dir
    save_dir = args.save_dir
elif args.type == 'eeg':
    data_dir = 'data/things-eeg/Image_set'
    save_dir = 'data/things-eeg/Image_set_Resize'
else:
    raise ValueError('Provide --data_dir and --save_dir, or set --type to eeg')

os.makedirs(save_dir, exist_ok=True)
image_paths = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_paths.append(os.path.join(root, file))

t1 = transforms.Resize((224, 224))

for path in image_paths:
    img = Image.open(path)
    img = t1(img)

    rel_path = os.path.relpath(path, data_dir)
    if rel_path.startswith('training_images' + os.sep):
        rel_path = os.path.join('train_images', os.path.relpath(path, os.path.join(data_dir, 'training_images')))
    save_path = os.path.join(save_dir, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)