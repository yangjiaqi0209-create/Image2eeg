#!/usr/bin/env bash
# Generator structural ablation (Final two-stage recipe).
#
# Manuscript Fig4a structural variants kept in-repo: no_Dilated, no_Transformer
# Control (S0): reuse checkpoints/predictor/THINGSEEG2/Ours/full (not trained here).
#
# Usage:
#   bash scripts/train_structural_ablation.sh              # paper variants, sub 1-10
#   bash scripts/train_structural_ablation.sh 1            # sub-01 only
#   ARCHS="no_Dilated" bash scripts/train_structural_ablation.sh 1
#   STAGE=2 bash scripts/train_structural_ablation.sh 1    # stage-2 only
#   RESUME=1 bash scripts/train_structural_ablation.sh

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
ARCHS="${ARCHS:-no_Dilated no_Transformer}"
N_LAYERS=4
CLIP_INPUT=fovea
ENCODER_ROOT="checkpoints/encoder/THINGSEEG2"
CKPT_ROOT="checkpoints/predictor/THINGSEEG2/architecture_ablation"
RESULT_ROOT="results/generator_structural_ablation"
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
  echo "===== Structural ablation (Final two-stage) ====="
  echo "archs=${ARCHS} stage=${STAGE} subs=${SUB_LIST_PRINT} gpu=${GPU}"
  echo "start: $(date -Is)"
} | tee "$MASTER_LOG"

arch_generator_flag() {
  local arch="$1"
  case "$arch" in
    no_Dilated|no_dconv) echo "no_dconv" ;;
    no_Transformer|tcn_only) echo "tcn_only" ;;
    *) echo "$arch" ;;
  esac
}

# Legacy results/ folder names (metrics JSON already stored there).
arch_result_name() {
  local arch="$1"
  case "$arch" in
    no_Dilated) echo "no_dconv" ;;
    no_Transformer) echo "tcn_only" ;;
    *) echo "$arch" ;;
  esac
}

run_stage1() {
  local arch="$1"
  local sub="$2"
  local gen_arch
  gen_arch="$(arch_generator_flag "$arch")"
  local res_name
  res_name="$(arch_result_name "$arch")"

  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${CKPT_ROOT}/${arch}/stage1"
  local RESULT_S1="${RESULT_ROOT}/${res_name}/stage1"
  local METRICS="${RESULT_S1}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"

  mkdir -p "$CKPT_S1" "$RESULT_S1"

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S1] ${arch} ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  fi

  echo "=== S1 ${arch} ${SUBTAG} gen_arch=${gen_arch} $(date -Is) ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --n_layers "$N_LAYERS" --generator_arch "$gen_arch" \
    --clip_input "$CLIP_INPUT" --blur_delta 0 \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_S1" --result_dir "$RESULT_S1" \
    "${resume_args[@]}" \
    2>&1 | tee -a "${RESULT_S1}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"
}

run_stage2() {
  local arch="$1"
  local sub="$2"
  local gen_arch
  gen_arch="$(arch_generator_flag "$arch")"
  local res_name
  res_name="$(arch_result_name "$arch")"

  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${CKPT_ROOT}/${arch}/stage1"
  local CKPT_FINAL="${CKPT_ROOT}/${arch}/full"
  local RESULT_FINAL="${RESULT_ROOT}/${res_name}/full"
  local INIT_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"
  local METRICS="${RESULT_FINAL}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_FINAL}/${SUBTAG}/last.pt"

  mkdir -p "$CKPT_FINAL" "$RESULT_FINAL"

  [[ -f "$INIT_CKPT" ]] || { echo "[error] S1 missing: $INIT_CKPT" | tee -a "$MASTER_LOG"; return 1; }

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S2] ${arch} ${SUBTAG}" | tee -a "$MASTER_LOG"
    return 0
  fi

  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  else
    resume_args=(--init_ckpt "$INIT_CKPT")
  fi

  echo "=== S2 ${arch} ${SUBTAG} gen_arch=${gen_arch} init=${INIT_CKPT} $(date -Is) ===" | tee -a "$MASTER_LOG"
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --epochs 25 --early_stop_patience 8 --early_stop_metric semantic_guard \
    --lr 1.5e-5 \
    --n_layers "$N_LAYERS" --generator_arch "$gen_arch" \
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

for arch in $ARCHS; do
  for sub in $SUB_LIST; do
    if [[ "$STAGE" == "both" || "$STAGE" == "1" ]]; then
      run_stage1 "$arch" "$sub"
    fi
    if [[ "$STAGE" == "both" || "$STAGE" == "2" ]]; then
      run_stage2 "$arch" "$sub"
    fi
  done
done

echo "Done $(date -Is)" | tee -a "$MASTER_LOG"
