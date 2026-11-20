#!/usr/bin/env bash
# Leak-free THINGS-EEG2 retrain for one subject. Does not touch historical
# checkpoints, whitened archives, or the unified-MVNN experiment.
set -euo pipefail

SUB="${1:?subject number}"
GPU="${GPU:-1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/_env.sh"

printf -v TAG "sub-%02d" "$SUB"
BASE="results/things_eeg2_leakfree"
if [[ -n "${EEG_RUN_ROOT:-}" ]]; then
  BASE="$EEG_RUN_ROOT"
fi
LOG_DIR="$BASE/logs"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/${TAG}.log"

DATA="data/processed/things_eeg2/preprocessed"
CKPT_S1="checkpoints/things_eeg2/generator/stage1"
CKPT_S2="checkpoints/things_eeg2/generator/stage2"
RESULT_S1="$BASE/generator/stage1"
RESULT_S2="$BASE/generator/stage2"
FEAT="data/processed/things_eeg2/generator_features"
ENC_ROOT="checkpoints/things_eeg2/encoder/THINGSEEG2_leakfree/${TAG}_seed0/checkpoints"
if [[ -n "${EEG_RUN_ROOT:-}" ]]; then
  DATA="$BASE/preprocessed"
  CKPT_S1="$BASE/checkpoints/generator/stage1"
  CKPT_S2="$BASE/checkpoints/generator/stage2"
  FEAT="$BASE/generator_features"
  ENC_ROOT="$BASE/checkpoints/encoder/THINGSEEG2_leakfree/${TAG}_seed0/checkpoints"
fi
if [[ -d "$ENC_ROOT" || -d "$CKPT_S1/$TAG" || -d "$CKPT_S2/$TAG" ]]; then
  echo "Training output already exists. Set EEG_RUN_ROOT to a new run directory."
  exit 1
fi

{
  echo "===== leak-free ${TAG} gpu=${GPU} ====="
  echo "start: $(date -Is)"
  echo "encoder selection: best validation top-1 over a fixed 50-epoch budget; test unread"
  echo "stage1 selection: validation loss, patience 15 (explicit; historical checkpoint field was empty)"
  echo "stage2 selection: validation semantic_guard, patience 8; init from stage1 last.pt; best.pt is not copied over last.pt"
} | tee -a "$MASTER"

python -m experiments.leakfree_things.prepare_subject --subject "$SUB" 2>&1 | tee -a "$MASTER"

ENC_CONFIG="configs/eeg/fixed_fovea_leakfree.yaml"
if [[ -n "${EEG_RUN_ROOT:-}" ]]; then
  ENC_CONFIG="$BASE/configs/encoder.yaml"
  python - "$ENC_CONFIG" "$DATA" "$BASE" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf
config = OmegaConf.load('configs/eeg/fixed_fovea_leakfree.yaml')
config.data.data_dir = sys.argv[2]
config.data.img_feature_dir = str(Path(sys.argv[3])/'encoder_features')
config.save_dir = str(Path(sys.argv[3])/'checkpoints/encoder')
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
OmegaConf.save(config, sys.argv[1])
PY
fi

echo "=== encoder ${TAG} $(date -Is) ===" | tee -a "$MASTER"
CUDA_VISIBLE_DEVICES="$GPU" python -m encoder.train \
  --config "$ENC_CONFIG" \
  --subjects "$TAG" \
  --seed 0 \
  --exp_setting intra-subject \
  --brain_backbone EEGProjectLayer \
  --vision_backbone RN50 \
  --epoch 50 \
  --lr 1e-4 \
  --skip-test \
  2>&1 | tee -a "$MASTER"

python -m experiments.leakfree_things.evaluate record-encoder --subject "$SUB" 2>&1 | tee -a "$MASTER"
UBP_CKPT="$(find "$ENC_ROOT" -maxdepth 1 -name 'epoch=*.ckpt' | sort)"
if [[ "$(echo "$UBP_CKPT" | wc -l)" -ne 1 || ! -f "$UBP_CKPT" ]]; then
  echo "expected one encoder checkpoint, got: $UBP_CKPT" | tee -a "$MASTER"
  exit 1
fi

echo "=== generator stage1 ${TAG} $(date -Is) encoder=${UBP_CKPT} ===" | tee -a "$MASTER"
python -m predictor.train \
  --sub "$SUB" --gpu "$GPU" --seed 2023 \
  --epochs 100 --early_stop_patience 15 --early_stop_metric loss \
  --lr 1e-4 \
  --n_layers 4 --generator_arch full \
  --clip_input fovea \
  --semantic_mode ubp_margin --lambda_sem 0 \
  --lambda_time 1.0 --lambda_corr 0.8 --lambda_freq 0.2 --lambda_band 0.2 \
  --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
  --data_dir "$DATA" \
  --image_root data/things-eeg/Image_set_Resize \
  --feature_dir "$FEAT" \
  --ubp_ckpt "$UBP_CKPT" \
  --ckpt_dir "$CKPT_S1" --result_dir "$RESULT_S1" \
  --skip-test \
  2>&1 | tee -a "$MASTER"

echo "=== generator stage2 ${TAG} $(date -Is) ===" | tee -a "$MASTER"
python -m predictor.train \
  --sub "$SUB" --gpu "$GPU" --seed 2023 \
  --epochs 25 --early_stop_patience 8 --early_stop_metric semantic_guard \
  --lr 1.5e-5 \
  --n_layers 4 --generator_arch full \
  --clip_input fovea \
  --semantic_mode ubp_margin --lambda_sem 0 \
  --lambda_time 1.0 --lambda_corr 0.8 \
  --lambda_freq 0 --lambda_band 0 \
  --lambda_band_corr 0.32 \
  --band_weights "delta=2.0,beta=1.5,gamma=1.0" \
  --gamma_fmax 45 \
  --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
  --data_dir "$DATA" \
  --image_root data/things-eeg/Image_set_Resize \
  --feature_dir "$FEAT" \
  --ubp_ckpt "$UBP_CKPT" \
  --init_ckpt "$CKPT_S1/${TAG}/last.pt" \
  --ckpt_dir "$CKPT_S2" --result_dir "$RESULT_S2" \
  --skip-test \
  2>&1 | tee -a "$MASTER"

echo "=== final held-out test ${TAG} $(date -Is) ===" | tee -a "$MASTER"
python -m experiments.leakfree_things.evaluate final --subject "$SUB" --gpu "$GPU" 2>&1 | tee -a "$MASTER"
echo "done ${TAG} $(date -Is)" | tee -a "$MASTER"
