#!/usr/bin/env bash
# Single-stage loss-group ablation (2+2+2):
#   L = L_time + L_freq + L_sem
#   L_time = λ_time·MSE + λ_corr·(1−r)
#   L_freq = λ_freq·|FFT| + λ_band·bandpower
#   L_sem  = λ_ubp·UBP-anchor + λ_eeg_nce·EEG-InfoNCE
#
# Variants:
#   full         — all three groups
#   no_time      — drop time losses
#   no_freq      — drop frequency losses
#   no_semantic  — drop semantic losses
#
# Usage:
#   bash scripts/train_loss_group_ablation.sh 1-10
#   VARIANTS="no_time no_freq no_semantic" bash scripts/train_loss_group_ablation.sh 1-10
#   RESUME=1 bash scripts/train_loss_group_ablation.sh 2-10

set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

SUBS="${1:-1-10}"
GPU="${GPU:-0}"
RESUME="${RESUME:-0}"
FORCE="${FORCE:-0}"
N_LAYERS=4
CLIP_INPUT=fovea
ENCODER_ROOT="checkpoints/encoder/THINGSEEG2"
CKPT_ROOT="checkpoints/predictor/THINGSEEG2/loss_ablation"
RESULT_ROOT="results/generator_loss_ablation"
LOG_DIR="${RESULT_ROOT}/_logs"
mkdir -p "$LOG_DIR"

VARIANTS="${VARIANTS:-full no_time no_freq no_semantic}"

if [[ "$SUBS" == *-* ]]; then
  IFS='-' read -r S0 S1 <<< "$SUBS"
  SUB_LIST=$(seq "$S0" "$S1")
else
  SUB_LIST="$SUBS"
fi

MASTER_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
{
  echo "===== Single-stage loss-group ablation ====="
  echo "variants=${VARIANTS}"
  echo "subs=${SUB_LIST} gpu=${GPU} resume=${RESUME}"
  echo "L_time=λ_time·MSE+λ_corr·(1-r) | L_freq=λ_freq·FFT+λ_band·band | L_sem=λ_ubp+λ_eeg_nce"
  echo "start: $(date -Is)"
} | tee "$MASTER_LOG"

# Returns: time corr freq band ubp nce
loss_weights() {
  local variant="$1"
  case "$variant" in
    full)
      echo "1.0 0.8 0.35 0.35 0.5 0.1"
      ;;
    no_time)
      echo "0.0 0.0 0.35 0.35 0.5 0.1"
      ;;
    no_freq)
      echo "1.0 0.8 0.0 0.0 0.5 0.1"
      ;;
    no_semantic)
      echo "1.0 0.8 0.35 0.35 0.0 0.0"
      ;;
    *)
      echo "[error] unknown variant: $variant" | tee -a "$MASTER_LOG"
      exit 1
      ;;
  esac
}

for variant in $VARIANTS; do
  read -r LT LC LF LB LU LN <<< "$(loss_weights "$variant")"
  # Checkpoint folder: full → single_stage; metrics JSON keep legacy "full" key via RESULT_DIR
  CKPT_VARIANT="$variant"
  if [[ "$variant" == "full" ]]; then
    CKPT_VARIANT="single_stage"
  fi
  CKPT_DIR="${CKPT_ROOT}/${CKPT_VARIANT}"
  RESULT_DIR="${RESULT_ROOT}/${variant}/full"
  mkdir -p "$CKPT_DIR" "$RESULT_DIR"

  echo "===== variant=${variant}  λ_time=${LT} λ_corr=${LC} λ_freq=${LF} λ_band=${LB} λ_ubp=${LU} λ_nce=${LN} =====" \
    | tee -a "$MASTER_LOG"

  for sub in $SUB_LIST; do
    printf -v SUBTAG "sub-%02d" "$sub"
    UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
    METRICS="${RESULT_DIR}/${SUBTAG}/metrics.json"
    LAST_CKPT="${CKPT_DIR}/${SUBTAG}/last.pt"

    if [[ -f "$METRICS" && "$FORCE" != "1" && "$RESUME" != "1" ]]; then
      echo "[skip] ${variant} ${SUBTAG}" | tee -a "$MASTER_LOG"
      continue
    fi
    if [[ -f "$METRICS" && "$FORCE" != "1" && "$RESUME" == "1" ]]; then
      echo "[skip] ${variant} ${SUBTAG} (already finished)" | tee -a "$MASTER_LOG"
      continue
    fi

    resume_args=()
    if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
      resume_args=(--resume)
    fi

    echo "=== ${variant} ${SUBTAG} $(date -Is) ===" | tee -a "$MASTER_LOG"
    python -m predictor.train \
      --sub "$sub" --gpu "$GPU" --seed 2023 \
      --epochs 100 --early_stop_patience 15 --early_stop_metric loss \
      --lr 1e-4 \
      --n_layers "$N_LAYERS" --generator_arch full \
      --clip_input "$CLIP_INPUT" --blur_delta 0 \
      --ubp_ckpt "$UBP_CKPT" \
      --semantic_mode ubp_margin --lambda_sem 0 \
      --lambda_time "$LT" --lambda_corr "$LC" \
      --lambda_freq "$LF" --lambda_band "$LB" --lambda_hf 0 \
      --lambda_band_corr 0 \
      --gamma_fmax 45 \
      --lambda_ubp "$LU" --lambda_eeg_nce "$LN" --lambda_margin 0 --lambda_div 0 \
      --ckpt_dir "$CKPT_DIR" --result_dir "$RESULT_DIR" \
      "${resume_args[@]}" \
      2>&1 | tee -a "${RESULT_DIR}/${SUBTAG}_train.log" | tee -a "$MASTER_LOG"

    if [[ -f "${CKPT_DIR}/${SUBTAG}/best.pt" ]]; then
      cp "${CKPT_DIR}/${SUBTAG}/best.pt" "${CKPT_DIR}/${SUBTAG}/last.pt"
    fi
  done
done

echo "Done $(date -Is)" | tee -a "$MASTER_LOG"
