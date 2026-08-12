#!/usr/bin/env bash
# Alljoined-1.6M Final two-stage generator (same recipe as train_final_two_stage.sh).
#
#   Stage 1 — Semantic pretraining (from scratch)
#     L^(1) = L_wave + L_sem
#     L_wave = MSE + 0.8·(1 − Pearson r)
#     L_sem  = 0.5·L_ubp + 0.1·L_EEG-InfoNCE
#     (Implementation also uses λ_freq=λ_band=0.2 as mild spectral regularizers.)
#
#   Stage 2 — Spectral refinement (init from Stage 1)
#     L^(2) = L_wave + λ_spec·L_band-corr + L_sem
#     λ_spec=0.32, band weights delta/beta emphasized, lr=1.5e-5
#     Early stop on semantic_guard; deploy best checkpoint.
#
# Default subjects: strong-5 (Top-5 ≥ 20%): 6 12 13 14 18
#
# Usage:
#   GPU=0 bash scripts/train_alljoined_final_two_stage.sh
#   SUBJECT_LIST="6 14" GPU=0 bash scripts/train_alljoined_final_two_stage.sh
#   STAGE=2 bash scripts/train_alljoined_final_two_stage.sh   # S2 only
#   RESUME=1 bash scripts/train_alljoined_final_two_stage.sh
#   SMOKE=1 GPU=0 bash scripts/train_alljoined_final_two_stage.sh

set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

export UBP_EEG_DATA_ROOT="${UBP_EEG_DATA_ROOT:-/home/ubuntu/dataset/EEG}"
DATA_DIR="${UBP_EEG_DATA_ROOT}/alljoined-1.6M/ubp_preprocessed"
IMAGE_ROOT="${UBP_REPO_ROOT:-$(pwd)}/data/things-eeg/Image_set_Resize"
FEATURE_DIR="${UBP_EEG_DATA_ROOT}/alljoined-1.6M/clip_features/FoveaBlur"
ENCODER_EXP="Alljoined"
ENCODER_ROOT="checkpoints/encoder/Alljoined"

GPU="${GPU:-0}"
STAGE="${STAGE:-both}"
RESUME="${RESUME:-0}"
SMOKE="${SMOKE:-0}"
N_LAYERS=4
CLIP_INPUT=fovea
N_VAL="${N_VAL:-740}"

CKPT_S1="${CKPT_S1:-checkpoints/predictor/Alljoined/Ours/stage1}"
CKPT_FINAL="${CKPT_FINAL:-checkpoints/predictor/Alljoined/Ours/full}"
RESULT_S1="${RESULT_S1:-results/alljoined_eeg/generator/final_two_stage/stage1}"
RESULT_FINAL="${RESULT_FINAL:-results/alljoined_eeg/generator/final_two_stage/full}"
LOG_DIR="${RESULT_FINAL}/_logs"
mkdir -p "$CKPT_S1" "$CKPT_FINAL" "$RESULT_S1" "$RESULT_FINAL" "$LOG_DIR" results/alljoined_eeg/logs

if [[ -n "${SUBJECT_LIST:-}" ]]; then
  read -ra SUBJECTS <<< "${SUBJECT_LIST}"
else
  SUBJECTS=(6 12 13 14 18)
fi
if [[ "${SMOKE}" == "1" ]]; then
  SUBJECTS=(6)
  N_VAL=200
fi

MASTER_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
{
  echo "===== Alljoined Final two-stage generator ====="
  echo "stage=${STAGE} subjects=${SUBJECTS[*]} gpu=${GPU} smoke=${SMOKE}"
  echo "data: ${DATA_DIR}"
  echo "encoder: ${ENCODER_ROOT}"
  echo "S1: ${CKPT_S1}"
  echo "S2: ${CKPT_FINAL}"
  echo "start: $(date -Is)"
} | tee "$MASTER_LOG"

if [[ ! -f "${DATA_DIR}/sub-06/train.pt" ]]; then
  echo "ERROR: missing ${DATA_DIR}/sub-06/train.pt" | tee -a "$MASTER_LOG"
  exit 1
fi

common_data_args=(
  --data_dir "${DATA_DIR}"
  --image_root "${IMAGE_ROOT}"
  --feature_dir "${FEATURE_DIR}"
  --clip_input "${CLIP_INPUT}"
  --blur_delta 0
  --n_val "${N_VAL}"
  --n_layers "${N_LAYERS}"
  --generator_arch full
)

run_stage1() {
  local sub="$1"
  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local METRICS="${RESULT_S1}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"

  if [[ ! -f "$UBP_CKPT" ]]; then
    echo "[skip S1] ${SUBTAG}: missing encoder ${UBP_CKPT}" | tee -a "$MASTER_LOG"
    return 1
  fi
  if [[ -f "$METRICS" && "$RESUME" != "1" ]]; then
    echo "[skip S1] ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  fi

  local epoch_args=()
  if [[ "${SMOKE}" == "1" ]]; then
    epoch_args=(--epochs 2 --early_stop_patience 2)
  fi

  echo "=== S1 ${SUBTAG} $(date -Is) ===" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m predictor.train \
    --sub "${sub}" --gpu 0 --seed 2023 \
    --ubp_ckpt "${UBP_CKPT}" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "${CKPT_S1}" --result_dir "${RESULT_S1}" \
    "${common_data_args[@]}" \
    "${epoch_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${RESULT_S1}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"
}

run_stage2() {
  local sub="$1"
  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local INIT_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"
  local METRICS="${RESULT_FINAL}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_FINAL}/${SUBTAG}/last.pt"

  if [[ ! -f "$UBP_CKPT" ]]; then
    echo "[skip S2] ${SUBTAG}: missing encoder ${UBP_CKPT}" | tee -a "$MASTER_LOG"
    return 1
  fi
  [[ -f "$INIT_CKPT" ]] || {
    echo "[error] S1 missing: $INIT_CKPT" | tee -a "$MASTER_LOG"
    return 1
  }

  if [[ -f "$METRICS" && "$RESUME" != "1" ]]; then
    echo "[skip S2] ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  else
    resume_args=(--init_ckpt "$INIT_CKPT")
  fi

  local epoch_args=(--epochs 25 --early_stop_patience 8)
  if [[ "${SMOKE}" == "1" ]]; then
    epoch_args=(--epochs 2 --early_stop_patience 2)
  fi

  echo "=== S2 ${SUBTAG} $(date -Is) init=${INIT_CKPT} ===" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m predictor.train \
    --sub "${sub}" --gpu 0 --seed 2023 \
    "${epoch_args[@]}" \
    --early_stop_metric semantic_guard \
    --lr 1.5e-5 \
    --ubp_ckpt "${UBP_CKPT}" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_time 1.0 --lambda_corr 0.8 \
    --lambda_freq 0 --lambda_band 0 --lambda_hf 0 \
    --lambda_band_corr 0.32 \
    --band_weights "delta=2.0,beta=1.5,gamma=1.0" \
    --gamma_fmax 45 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "${CKPT_FINAL}" --result_dir "${RESULT_FINAL}" \
    "${common_data_args[@]}" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${RESULT_FINAL}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"

  if [[ -f "${CKPT_FINAL}/${SUBTAG}/best.pt" ]]; then
    cp "${CKPT_FINAL}/${SUBTAG}/best.pt" "${CKPT_FINAL}/${SUBTAG}/last.pt"
  fi
}

for sub in "${SUBJECTS[@]}"; do
  if [[ "$STAGE" == "both" || "$STAGE" == "1" ]]; then
    run_stage1 "$sub"
  fi
  if [[ "$STAGE" == "both" || "$STAGE" == "2" ]]; then
    run_stage2 "$sub"
  fi
done

{
  echo "Done $(date -Is)"
  echo "ckpt S1: ${CKPT_S1}"
  echo "ckpt S2: ${CKPT_FINAL}"
  echo "results: ${RESULT_FINAL}"
} | tee -a "$MASTER_LOG"
