# Uncertainty-aware Blur Prior (UBP)

This repository is derived from the open-source [Uncertainty-Aware Blur Prior (UBP)](https://arxiv.org/abs/2503.04207) code by Wu et al. (CVPR 2025). It keeps the UBP encoder stack and adds extensions for image-to-EEG prediction, dataset registry / Alljoined support, and training entry scripts.

Upstream official releases remain authoritative for the original UBP paper code. This tree is a derivative extension.

## License & Attribution

| Component | Attribution |
|-----------|-------------|
| Upstream UBP core (encoder / original training and data pipeline skeleton) | © Wu, Haitao et al.; Apache License 2.0 (see root `LICENSE`) |
| New or substantially rewritten code in this fork (`predictor/`, dataset registry, Alljoined adapters, training scripts, and related local tooling) | © Jiaqi Yang; also under Apache License 2.0 |

Keep the root `LICENSE` when redistributing, and cite the upstream UBP paper (see Citation below).

## What is in this repository

Tracked for sharing:

| Path | Role |
|------|------|
| `encoder/` + `configs/eeg/` | EEG encoder (`python -m encoder.train`) |
| `predictor/` | Image → EEG generator (`python -m predictor.train`) |
| `preprocess/` | Convert raw / downloaded data into training `.pt` |
| `scripts/` | One-shot training and preprocess entry points |
| `data/registry.yaml`, `data/profiles/`, `data/env.example` | Dataset index and path templates |

Local-only (gitignored; not uploaded): datasets, checkpoints, pretrained CLIP weights, `results/`, `analysis/`, `manuscript/`, and `data/env.local`.

## Environment

```bash
pip install -r requirements.txt
cp data/env.example data/env.local   # edit local roots; scripts source this file
# Default conda env name used by scripts: EEG  (override with CONDA_ENV=...)
```

## Data

Small registry files live under `data/`. Large payloads stay outside git.

```bash
python -m encoder.registry list
python -m encoder.registry show things_eeg
python -m encoder.registry show alljoined_eeg
```

### THINGS-EEG

Place preprocessed data under `data/things-eeg/` (gitignored), or download from Hugging Face: [Haitao999/things-eeg](https://huggingface.co/datasets/Haitao999/things-eeg).

If you only have raw stimuli / EEG and need preprocessing:

```bash
python preprocess/process_resize.py --type eeg
for s in $(seq 1 10); do
  python preprocess/process_eeg_whiten.py --subject "$s"
done
```

### Alljoined-1.6M

Source: [Alljoined/Alljoined-1.6M](https://huggingface.co/datasets/Alljoined/Alljoined-1.6M) → UBP `.pt`.

| Item | Value |
|------|-------|
| Registry id | `alljoined_eeg` |
| Shape | train `(16540, 4, 32, 250)` / test `(200, 80, 32, 250)` @ 250 Hz |
| Images | Reuse `data/things-eeg/Image_set_Resize` |
| Channels | `encoder.data.ALLJOINED_CHANNELS` (32) |

```text
$UBP_EEG_DATA_ROOT/alljoined-1.6M/          # default: $HOME/datasets/EEG
├── raw_hf/preprocessed_eeg/               # download cache (optional after preprocess)
└── ubp_preprocessed/sub-XX/{train,test}.pt
```

```bash
# Set UBP_EEG_DATA_ROOT in data/env.local, or:
export UBP_EEG_DATA_ROOT="$HOME/datasets/EEG"
# optional: export HF_ENDPOINT=https://hf-mirror.com
bash scripts/preprocess_alljoined_all.sh
python -m encoder.registry show alljoined_eeg
```

Notes: official MVNN whitening need not match THINGS `process_eeg_whiten.py` numerically; the test set keeps the first 80 trials per image.

## Training

```bash
# THINGS-EEG: encoder → two-stage generator
bash scripts/train_things_encoder.sh
bash scripts/train_final_two_stage.sh

# Fig.4-style ablations (structural | extended | loss | all)
bash scripts/train_ablation.sh structural
bash scripts/train_ablation.sh extended
bash scripts/train_ablation.sh loss
AGGREGATE=1 bash scripts/train_ablation.sh structural

# Alljoined: encoder (20 subjects) → generator (strong-5)
bash scripts/train_alljoined.sh
bash scripts/train_alljoined_final_two_stage.sh
```

Weights are written under `checkpoints/` (gitignored). See `COMMIT_CHECKLIST.md` for the upload allowlist.

## Citation

```bibtex
@inproceedings{wu2025bridging,
  title={Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior},
  author={Wu, Haitao and Li, Qing and Zhang, Changqing and He, Zhen and Ying, Xiaomin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={2246--2257},
  year={2025}
}

@misc{xu2025alljoined16mmilliontrialeegimagedataset,
  title={Alljoined-1.6M: A Million-Trial EEG-Image Dataset for Evaluating Affordable Brain-Computer Interfaces},
  author={Jonathan Xu and others},
  year={2025},
  eprint={2508.18571},
}
```

## Contact

yangjiaqi_bme2026@163.com
