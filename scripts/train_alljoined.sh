#!/usr/bin/env bash
# Train Alljoined-1.6M UBP encoder (lr5e5 recipe, val-based early stop).
#
# Prerequisites:
#   bash scripts/download_alljoined.sh
#   bash scripts/preprocess_alljoined_all.sh
#
# Usage:
#   GPU=0 bash scripts/train_alljoined.sh
#   SUBJECT_LIST="1 2 3" GPU=0 bash scripts/train_alljoined.sh
#   SMOKE=1 GPU=0 bash scripts/train_alljoined.sh
set -euo pipefail

# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

DATA_DIR="${UBP_EEG_DATA_ROOT}/alljoined-1.6M/ubp_preprocessed"
ENCODER_CFG="configs/eeg/alljoined_ubp.yaml"
ENCODER_EXP="Alljoined"
GPU="${GPU:-0}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-5e-5}"
SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"

if [[ -n "${SUBJECT_LIST:-}" ]]; then
  read -ra SUBJECTS <<< "${SUBJECT_LIST}"
else
  SUBJECTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi
if [[ "${SMOKE}" == "1" ]]; then
  SUBJECTS=(1)
  EPOCHS=5
fi

mkdir -p results/alljoined_eeg/logs

if [[ ! -f "${DATA_DIR}/sub-01/train.pt" ]]; then
  echo "ERROR: missing ${DATA_DIR}/sub-01/train.pt — run scripts/preprocess_alljoined_all.sh first"
  exit 1
fi

for sub in "${SUBJECTS[@]}"; do
  sub_tag="sub-$(printf '%02d' "$sub")"
  enc_dir="checkpoints/encoder/Alljoined/${sub_tag}_seed${SEED}"
  if [[ -f "${enc_dir}/test_results.json" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP encoder ${sub_tag}: ${enc_dir}/test_results.json exists"
    continue
  fi

  echo "==> Encoder ${sub_tag} (lr5e5, bs256, val early stop)"
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. python -m encoder.train \
    --config "${ENCODER_CFG}" \
    --dataset-profile alljoined_eeg \
    --dataset alljoined \
    --subjects "${sub_tag}" \
    --seed "${SEED}" \
    --exp_setting intra-subject \
    --brain_backbone EEGProjectLayer \
    --vision_backbone RN50 \
    --epoch "${EPOCHS}" \
    --lr "${LR}" \
    2>&1 | tee -a "results/alljoined_eeg/logs/encoder_${sub_tag}.log"
done

echo "==> Alljoined encoder training complete (exp: ${ENCODER_EXP})."
