import importlib
import math
import subprocess

import torch
from torch import nn
from torch.nn import functional as F


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    params = config.get("params") or {}
    return get_obj_from_str(config["target"])(**params)


def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
    for key in args.__dict__.keys():
        config[key] = getattr(args, key)
    return config


def get_device(gpu_ids):
    """Return a CUDA device index (int). ``gpu_ids='auto'`` picks a free GPU."""
    if gpu_ids == 'auto':
        try:
            nvidia_smi_output = subprocess.check_output(
                [
                    'nvidia-smi',
                    '--query-gpu=index,memory.free,temperature.gpu',
                    '--format=csv,noheader,nounits',
                ]
            )
            gpu_info_lines = nvidia_smi_output.decode('utf-8').strip().split('\n')
            gpu_info = []
            for line in gpu_info_lines:
                gpu_data = line.strip().split(', ')
                index, memory_free, temperature = map(int, gpu_data)
                gpu_info.append((index, memory_free, temperature))
            gpu_info.sort(key=lambda x: x[1], reverse=True)

            memory_rank_num = math.ceil(0.4 * len(gpu_info))
            selected_gpus = gpu_info[:memory_rank_num]
            selected_gpus.sort(key=lambda x: x[2])
            return selected_gpus[0][0]
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return 0
    if gpu_ids == "cpu":
        return 0
    return list(map(int, gpu_ids.split(",")))[0]


class ClipLoss(nn.Module):
    def forward(self, image_features, text_features, logit_scale):
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        num_logits = logits_per_image.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)

        image_loss = F.cross_entropy(logits_per_image, labels, reduction='none')
        text_loss = F.cross_entropy(logits_per_text, labels, reduction='none')

        return image_loss, text_loss, logits_per_image
