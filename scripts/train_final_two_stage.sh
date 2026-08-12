#!/usr/bin/env bash
# Final two-stage generator (paper-ready)
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
# Usage:
#   bash scripts/train_final_two_stage.sh              # both stages, sub 1-10
#   bash scripts/train_final_two_stage.sh 1             # sub-01 only
#   STAGE=2 bash scripts/train_final_two_stage.sh 1     # stage-2 only
#   REUSE_EXP_A_S1=1 bash scripts/train_final_two_stage.sh 1  # copy S1 from Exp A if present
#   RESUME=1 bash scripts/train_final_two_stage.sh

set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

SUBS="${1:-1-10}"
GPU="${GPU:-0}"
STAGE="${STAGE:-both}"
RESUME="${RESUME:-0}"
REUSE_EXP_A_S1="${REUSE_EXP_A_S1:-0}"
N_LAYERS=4
CLIP_INPUT=fovea
ENCODER_ROOT="checkpoints/encoder/THINGSEEG2"
CKPT_S1="checkpoints/predictor/THINGSEEG2/Ours/stage1"
CKPT_FINAL="checkpoints/predictor/THINGSEEG2/Ours/full"
RESULT_S1="results/generator_final/stage1"
RESULT_FINAL="results/generator_final/full"
LOG_DIR="${RESULT_FINAL}/_logs"
mkdir -p "$CKPT_S1" "$CKPT_FINAL" "$RESULT_S1" "$RESULT_FINAL" "$LOG_DIR"

if [[ "$SUBS" == *-* ]]; then
  IFS='-' read -r S0 S1 <<< "$SUBS"
  SUB_LIST=$(seq "$S0" "$S1")
else
  SUB_LIST="$SUBS"
fi

MASTER_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
{
  echo "===== Final two-stage generator ====="
  echo "stage=${STAGE} subs=${SUB_LIST} gpu=${GPU} reuse_exp_a_s1=${REUSE_EXP_A_S1}"
  echo "S1: ${CKPT_S1}"
  echo "S2: ${CKPT_FINAL}"
  echo "start: $(date -Is)"
} | tee "$MASTER_LOG"

maybe_reuse_exp_a_s1() {
  local sub="$1"
  printf -v SUBTAG "sub-%02d" "$sub"
  local dst="${CKPT_S1}/${SUBTAG}/last.pt"
  local src="${EXP_A_CKPT}/${SUBTAG}/last.pt"
  if [[ "$REUSE_EXP_A_S1" == "1" && ! -f "$dst" && -f "$src" ]]; then
    mkdir -p "${CKPT_S1}/${SUBTAG}"
    cp "$src" "$dst"
    if [[ -f "${EXP_A_CKPT}/${SUBTAG}/best.pt" ]]; then
      cp "${EXP_A_CKPT}/${SUBTAG}/best.pt" "${CKPT_S1}/${SUBTAG}/best.pt"
    fi
    echo "[reuse S1] copied Exp A -> ${dst}" | tee -a "$MASTER_LOG"
  fi
}

run_stage1() {
  local sub="$1"
  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local METRICS="${RESULT_S1}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"

  maybe_reuse_exp_a_s1 "$sub"

  if [[ -f "$METRICS" && "$RESUME" != "1" ]]; then
    echo "[skip S1] ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi
  if [[ -f "$LAST_CKPT" && "$REUSE_EXP_A_S1" == "1" && ! -f "$METRICS" ]]; then
    echo "[reuse S1 only] ${SUBTAG} ckpt exists, run eval via stage1 train skip" | tee -a "$MASTER_LOG"
    # Run quick eval if metrics missing but ckpt copied
  fi
  if [[ -f "$METRICS" && "$RESUME" != "1" ]]; then
    return 0
  fi
  if [[ -f "$LAST_CKPT" && "$REUSE_EXP_A_S1" == "1" ]]; then
    echo "[skip S1 train] ${SUBTAG} (reused ckpt)" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  fi

  echo "=== S1 ${SUBTAG} $(date -Is) ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --n_layers "$N_LAYERS" --generator_arch full \
    --clip_input "$CLIP_INPUT" --blur_delta 0 \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_S1" --result_dir "$RESULT_S1" \
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

  maybe_reuse_exp_a_s1 "$sub"
  [[ -f "$INIT_CKPT" ]] || { echo "[error] S1 missing: $INIT_CKPT" | tee -a "$MASTER_LOG"; return 1; }

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

  echo "=== S2 ${SUBTAG} $(date -Is) init=${INIT_CKPT} ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --epochs 25 --early_stop_patience 8 --early_stop_metric semantic_guard \
    --lr 1.5e-5 \
    --n_layers "$N_LAYERS" --generator_arch full \
    --clip_input "$CLIP_INPUT" --blur_delta 0 \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_time 1.0 --lambda_corr 0.8 \
    --lambda_freq 0 --lambda_band 0 --lambda_hf 0 \
    --lambda_band_corr 0.32 \
    --band_weights "delta=2.0,beta=1.5,gamma=1.0" \
    --gamma_fmax 45 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_FINAL" --result_dir "$RESULT_FINAL" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${RESULT_FINAL}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"

  if [[ -f "${CKPT_FINAL}/${SUBTAG}/best.pt" ]]; then
    cp "${CKPT_FINAL}/${SUBTAG}/best.pt" "${CKPT_FINAL}/${SUBTAG}/last.pt"
  fi
}

for sub in $SUB_LIST; do
  if [[ "$STAGE" == "both" || "$STAGE" == "1" ]]; then
    run_stage1 "$sub"
  fi
  if [[ "$STAGE" == "both" || "$STAGE" == "2" ]]; then
    run_stage2 "$sub"
  fi
done

echo "Done $(date -Is)" | tee -a "$MASTER_LOG"
