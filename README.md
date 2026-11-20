# Image2EEG

Leak-free image-to-EEG revision of the Uncertainty-Aware Blur Prior (UBP) pipeline. The release code trains and evaluates a Brain Encoder and an EEG Generator on THINGS-EEG2 (10 subjects) and the buffered Alljoined cohort (20 subjects). Model architecture, training hyperparameters, and reported metrics are those of the revised manuscript.

Raw EEG, processed arrays, manuscript files, and final figures stay outside git. The official `.ckpt` and `.pt` weights also stay outside git because each file is larger than GitHub's 100 MB limit. They are published with the repository Release. See `.gitignore`.

## Layout

```text
src/            encoder, predictor (EEG Generator), preprocess
configs/        training configs and read-only checkpoint parameter views
scripts/        environment wrapper, THINGS training entry, revision checks
experiments/    subject training, baselines, ablations, statistics
figures/        plotting code and the figure reproduction entry
data/           dataset registry, profiles, env template
archive/        earlier paper code, weights, and the 2026-11-17 migration record
checkpoints/    selected weights (local; not in git)
results/        logs, selection records, metrics (local; not in git)
manuscript/     paper sources (local; not in git)
```

Run commands from the repository root. `src/project_paths.py` resolves paths that were stored before the directory move. The constant `OLD_REPO` in that file is only a prefix for those stored strings.

## Environment

Python 3.10, PyTorch 2.4.1, torchvision 0.19.1. Pins are in `requirements.txt`. The freeze used for the revision is `scripts/restructure/environment_freeze.txt`.

```bash
conda create -n EEG python=3.10
conda activate EEG
pip install -r requirements.txt
pip install -e . --no-deps
cp data/env.example data/env.local   # first setup only
```

`bash scripts/python.sh` loads `data/env.local`, puts `src/` on `PYTHONPATH`, and runs that environment. Optional variables, also listed in `data/env.example`:

| Variable | Used when |
| --- | --- |
| `UBP_EEG_DATA_ROOT` | Alljoined and other external EEG roots. Default `$HOME/datasets/EEG`. |
| `THINGS_EEG_RAW_ROOT` | Rebuilding THINGS epochs from raw sessions. |
| `THINGS_IMAGE_METADATA` | Official `image_metadata.npy` for the cross-subject image-id check. |
| `ALEXNET_WEIGHTS` | ImageNet AlexNet file for the Gifford baseline. Default is the torch-hub checkpoint path. |
| `PYTHON` | Interpreter for child jobs. Default is the current interpreter. |

## Data protocol

THINGS-EEG2 uses seed 2023 and a shared image split of 15,800 train / 740 validation / 200 test images. The index is `data/processed/things_eeg2/split/image_split.npz` (`image_split.json` records the rule). MVNN is fit on training-image trials only. The encoder is selected by validation top-1. Generator stage 1 monitors validation loss. Stage 2 starts from stage-1 `last.pt` and monitors validation `semantic_guard`. The deployed generator weight is stage-2 `best.pt`. Both `best.pt` and `last.pt` are kept.

The config copied beside each encoder run is the training-time file. `configs/eeg/fixed_fovea_leakfree.yaml` matches that file except for three path lines: the runs stored `results/things_eeg2_leakfree/...`, and the repo config stores the equivalent `data/processed/things_eeg2/...` and `checkpoints/things_eeg2/...` locations. Those results paths are links to the same directories. Per-run `hparams.yaml` files are empty; the copied yaml is the record.

`configs/alljoined/encoder.yaml` and `configs/alljoined/source_hashes.json` are the frozen training-time bytes. Do not edit them. The active Alljoined config is `configs/alljoined/encoder_final.yaml`, with paths relative to the repository root. `configs/things_eeg2/generator_stage{1,2}_reference.json` are read-only views of the selected sub-01 checkpoints, not a generic multi-subject config.

## Train

THINGS, one new run directory per subject. The entry refuses to overwrite an existing run.

```bash
EEG_RUN_ROOT=results/reproductions/things_sub01 GPU=0 \
  bash scripts/train_things.sh 1
```

This calls `experiments/leakfree_things/run_subject.sh`: image split and train-only MVNN, encoder, both generator stages, then frozen selection and test. Raw THINGS EEG must be available through `THINGS_EEG_RAW_ROOT`.

Alljoined training lives under `experiments/alljoined/` and reads `configs/alljoined/encoder_final.yaml`. Launch child jobs with the same interpreter (`scripts/python.sh` or `PYTHON`).

## Evaluate

Strict load of the 30-subject encoder and two-stage generator checkpoints, plus a sub-01 test-split inference check against the saved pooled Pearson (tolerance `1e-4`):

```bash
bash scripts/python.sh -m encoder.registry list
bash scripts/python.sh scripts/validate_revision.py --report /tmp/eeg_validation_new.json
```

The report path must be a new file. What that script covers is described in `scripts/restructure/VALIDATION.md`.

Figures are redrawn into a new directory. This does not replace `figures/final/`.

```bash
bash scripts/python.sh -m figures.reproduce --figure all \
  --output-dir figures/reproduced/my_revision_check
```

`--figure` also accepts `fig1` … `fig5` and `sfig1` … `sfig5`. Fig. 2 is re-exported from the retained final SVG. Rebuilding that panel from arrays is not part of this entry.

## Where the paper run is recorded

These files are local and gitignored. They are the records a code release should ship beside the weights.

| Record | Path |
| --- | --- |
| Encoder and generator weights | GitHub Release `v1.0.0`, asset archive of the 180 official `.ckpt` and `.pt` files. Unpack it so the paths below exist. |
| Encoder config used at training | `checkpoints/things_eeg2/encoder/THINGSEEG2_leakfree/sub-XX_seed0/fixed_fovea_leakfree.yaml` |
| Encoder selection | `results/things_eeg2_leakfree/selection/sub-XX_encoder.json` |
| Encoder logs | TensorBoard `events.out.tfevents*` in each `sub-XX_seed0` directory. No per-epoch text log; `save_top_k=1`. |
| Image split | `data/processed/things_eeg2/split/image_split.npz` |
| Generator weights | `checkpoints/things_eeg2/generator/stage{1,2}/sub-XX/{best,last}.pt` |
| Generator logs | `results/things_eeg2_leakfree/generator/stage{1,2}/sub-XX/log.txt` |
| Independent encoder selection | `results/things_eeg2_leakfree/external_baselines/independent_encoder/jobs/sub-XX/selection.json` |

The written protocol is `results/things_eeg2_leakfree/PROTOCOL.md`. Figure sources are mapped in `manuscript/FIGURE_MAP.md`. The file-by-file move from the previous tree is `scripts/restructure/file_migrations.csv`.

## Provenance

The encoder core extends Wu et al., *Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior*, CVPR 2025, pp. 2246–2257. The image-to-EEG generator and dataset adapters are additional work by Jiaqi Yang. Both are Apache-2.0 (`LICENSE`). Attribution and the CC BY-NC 4.0 exception for `experiments/external_baselines/upstream/` are in `NOTICE`.

Earlier paper weights and scripts are under `archive/legacy_paper/`. They are not the default models. The Chinese README from the 2026-11-17 layout pass is kept at `archive/project_readme_zh_20261117.md`.
