#!/usr/bin/env bash
# Extended generator ablation (Final two-stage recipe).
#
# Manuscript Fig4a extended variants kept in-repo: no_FoveaBlur no_self_attn h128 h512
# Control (E0): reuse checkpoints/predictor/THINGSEEG2/Ours/full (not trained here).
#
# Usage:
#   bash scripts/train_extended_ablation.sh              # paper variants, sub 1-10
#   bash scripts/train_extended_ablation.sh 1            # sub-01 only
#   VARIANTS="no_FoveaBlur h128" bash scripts/train_extended_ablation.sh 1
#   STAGE=2 bash scripts/train_extended_ablation.sh 1
#   FORCE=1 bash scripts/train_extended_ablation.sh

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
FORCE="${FORCE:-0}"
VARIANTS="${VARIANTS:-no_FoveaBlur no_self_attn h128 h512}"

ENCODER_ROOT="checkpoints/encoder/THINGSEEG2"
CKPT_ROOT="checkpoints/predictor/THINGSEEG2/architecture_ablation"
RESULT_ROOT="results/generator_extended_ablation"
LOG_DIR="${RESULT_ROOT}/_logs"
mkdir -p "$LOG_DIR"

if [[ "$SUBS" == *-* ]]; then
  IFS='-' read -r S0 S1 <<< "$SUBS"
  SUB_LIST=$(seq "$S0" "$S1")
else
  SUB_LIST="$SUBS"
fi
SUB_LIST_PRINT=$(echo "$SUB_LIST" | tr '\n' ' ')

MASTER_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
{
  echo "===== Extended ablation (Final two-stage) ====="
  echo "variants=${VARIANTS} stage=${STAGE} subs=${SUB_LIST_PRINT} gpu=${GPU}"
  echo "start: $(date -Is)"
} | tee "$MASTER_LOG"

# Per-variant: CLIP_INPUT N_LAYERS HIDDEN GEN_ARCH EXTRA_ARGS
variant_config() {
  local variant="$1"
  CLIP_INPUT=fovea
  N_LAYERS=4
  HIDDEN=256
  GEN_ARCH=full
  EXTRA_ARGS=()
  case "$variant" in
    no_FoveaBlur|sharp)
      CLIP_INPUT=sharp
      ;;
    no_self_attn) GEN_ARCH=no_self_attn ;;
    h128) HIDDEN=128 ;;
    h512) HIDDEN=512 ;;
    *)
      echo "[error] unknown variant: $variant" | tee -a "$MASTER_LOG"
      return 1
      ;;
  esac
}

# Legacy results/ folder names (metrics JSON already stored there).
variant_result_name() {
  local variant="$1"
  case "$variant" in
    no_FoveaBlur) echo "sharp" ;;
    *) echo "$variant" ;;
  esac
}

run_stage1() {
  local variant="$1"
  local sub="$2"
  variant_config "$variant"
  local res_name
  res_name="$(variant_result_name "$variant")"

  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${CKPT_ROOT}/${variant}/stage1"
  local RESULT_S1="${RESULT_ROOT}/${res_name}/stage1"
  local METRICS="${RESULT_S1}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"

  mkdir -p "$CKPT_S1" "$RESULT_S1"

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S1] ${variant} ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  fi

  echo "=== S1 ${variant} ${SUBTAG} clip=${CLIP_INPUT} L=${N_LAYERS} H=${HIDDEN} arch=${GEN_ARCH} $(date -Is) ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --n_layers "$N_LAYERS" --hidden "$HIDDEN" --generator_arch "$GEN_ARCH" \
    "${EXTRA_ARGS[@]}" \
    --clip_input "$CLIP_INPUT" --blur_delta 0 \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_S1" --result_dir "$RESULT_S1" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${RESULT_S1}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"
}

run_stage2() {
  local variant="$1"
  local sub="$2"
  variant_config "$variant"
  local res_name
  res_name="$(variant_result_name "$variant")"

  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${CKPT_ROOT}/${variant}/stage1"
  local CKPT_FINAL="${CKPT_ROOT}/${variant}/full"
  local RESULT_FINAL="${RESULT_ROOT}/${res_name}/full"
  local INIT_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"
  local METRICS="${RESULT_FINAL}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_FINAL}/${SUBTAG}/last.pt"

  mkdir -p "$CKPT_FINAL" "$RESULT_FINAL"

  [[ -f "$INIT_CKPT" ]] || { echo "[error] S1 missing: $INIT_CKPT" | tee -a "$MASTER_LOG"; return 1; }

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S2] ${variant} ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  else
    resume_args=(--init_ckpt "$INIT_CKPT")
  fi

  echo "=== S2 ${variant} ${SUBTAG} clip=${CLIP_INPUT} L=${N_LAYERS} H=${HIDDEN} arch=${GEN_ARCH} init=${INIT_CKPT} $(date -Is) ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --epochs 25 --early_stop_patience 8 --early_stop_metric semantic_guard \
    --lr 1.5e-5 \
    --n_layers "$N_LAYERS" --hidden "$HIDDEN" --generator_arch "$GEN_ARCH" \
    "${EXTRA_ARGS[@]}" \
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

for variant in $VARIANTS; do
  for sub in $SUB_LIST; do
    if [[ "$STAGE" == "both" || "$STAGE" == "1" ]]; then
      run_stage1 "$variant" "$sub"
    fi
    if [[ "$STAGE" == "both" || "$STAGE" == "2" ]]; then
      run_stage2 "$variant" "$sub"
    fi
  done
done

echo "Done $(date -Is)" | tee -a "$MASTER_LOG"
