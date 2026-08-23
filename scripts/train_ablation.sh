#!/usr/bin/env bash
# THINGS predictor ablations for manuscript Fig.4 — one entrypoint.
#
# Groups:
#   structural  — two-stage; no_Dilated no_Transformer
#   extended    — two-stage; no_FoveaBlur no_self_attn h128 h512
#   loss        — single-stage loss groups; full no_time no_freq no_semantic
#   all         — structural + extended + loss
#
# Usage:
#   bash scripts/train_ablation.sh structural              # sub 1-10
#   bash scripts/train_ablation.sh extended 1
#   VARIANTS="h128" bash scripts/train_ablation.sh extended 1
#   STAGE=2 bash scripts/train_ablation.sh structural 1
#   AGGREGATE=1 bash scripts/train_ablation.sh structural  # train + metrics + redraw fig4
#   SKIP_TRAIN=1 AGGREGATE=1 bash scripts/train_ablation.sh extended

set -euo pipefail
# shellcheck disable=SC1091
source "$(dirname "$0")/_env.sh"

GROUP="${1:-}"
SUBS="${2:-1-10}"
GPU="${GPU:-0}"
STAGE="${STAGE:-both}"
RESUME="${RESUME:-0}"
FORCE="${FORCE:-0}"
AGGREGATE="${AGGREGATE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

ENCODER_ROOT="checkpoints/encoder/THINGSEEG2"
ARCH_CKPT_ROOT="checkpoints/predictor/THINGSEEG2/architecture_ablation"
LOSS_CKPT_ROOT="checkpoints/predictor/THINGSEEG2/loss_ablation"

usage() {
  cat <<'EOF'
Usage: bash scripts/train_ablation.sh <structural|extended|loss|all> [subs]

Env: GPU STAGE RESUME FORCE VARIANTS AGGREGATE=1 SKIP_TRAIN=1
EOF
  exit 1
}

[[ -n "$GROUP" ]] || usage

if [[ "$SUBS" == *-* ]]; then
  IFS='-' read -r S0 S1 <<< "$SUBS"
  SUB_LIST=$(seq "$S0" "$S1")
else
  SUB_LIST="$SUBS"
fi
SUB_LIST_PRINT=$(echo "$SUB_LIST" | tr '\n' ' ')

# --- two-stage helpers (structural / extended) ---

two_stage_variant_config() {
  local variant="$1"
  CLIP_INPUT=fovea
  N_LAYERS=4
  HIDDEN=256
  GEN_ARCH=full
  RESULT_NAME="$variant"
  case "$variant" in
    no_Dilated|no_dconv)
      GEN_ARCH=no_dconv
      RESULT_NAME=no_dconv
      ;;
    no_Transformer|tcn_only)
      GEN_ARCH=tcn_only
      RESULT_NAME=tcn_only
      ;;
    no_FoveaBlur|sharp)
      CLIP_INPUT=sharp
      RESULT_NAME=sharp
      ;;
    no_self_attn) GEN_ARCH=no_self_attn ;;
    h128) HIDDEN=128 ;;
    h512) HIDDEN=512 ;;
    *)
      echo "[error] unknown two-stage variant: $variant"
      return 1
      ;;
  esac
}

run_two_stage_s1() {
  local variant="$1" sub="$2" result_root="$3"
  two_stage_variant_config "$variant"
  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${ARCH_CKPT_ROOT}/${variant}/stage1"
  local RESULT_S1="${result_root}/${RESULT_NAME}/stage1"
  local METRICS="${RESULT_S1}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"
  mkdir -p "$CKPT_S1" "$RESULT_S1"

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S1] ${variant} ${SUBTAG}"
    return 0
  fi
  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  fi

  echo "=== S1 ${variant} ${SUBTAG} arch=${GEN_ARCH} H=${HIDDEN} clip=${CLIP_INPUT} ==="
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --n_layers "$N_LAYERS" --hidden "$HIDDEN" --generator_arch "$GEN_ARCH" \
    --clip_input "$CLIP_INPUT" \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_S1" --result_dir "$RESULT_S1" \
    "${resume_args[@]}"
}

run_two_stage_s2() {
  local variant="$1" sub="$2" result_root="$3"
  two_stage_variant_config "$variant"
  printf -v SUBTAG "sub-%02d" "$sub"
  local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
  local CKPT_S1="${ARCH_CKPT_ROOT}/${variant}/stage1"
  local CKPT_FINAL="${ARCH_CKPT_ROOT}/${variant}/full"
  local RESULT_FINAL="${result_root}/${RESULT_NAME}/full"
  local INIT_CKPT="${CKPT_S1}/${SUBTAG}/last.pt"
  local METRICS="${RESULT_FINAL}/${SUBTAG}/metrics.json"
  local LAST_CKPT="${CKPT_FINAL}/${SUBTAG}/last.pt"
  mkdir -p "$CKPT_FINAL" "$RESULT_FINAL"
  [[ -f "$INIT_CKPT" ]] || { echo "[error] S1 missing: $INIT_CKPT"; return 1; }

  if [[ -f "$METRICS" && "$RESUME" != "1" && "$FORCE" != "1" ]]; then
    echo "[skip S2] ${variant} ${SUBTAG}"
    return 0
  fi
  local resume_args=()
  if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
    resume_args=(--resume)
  else
    resume_args=(--init_ckpt "$INIT_CKPT")
  fi

  echo "=== S2 ${variant} ${SUBTAG} arch=${GEN_ARCH} H=${HIDDEN} clip=${CLIP_INPUT} ==="
  python -m predictor.train \
    --sub "$sub" --gpu "$GPU" --seed 2023 \
    --epochs 25 --early_stop_patience 8 --early_stop_metric semantic_guard \
    --lr 1.5e-5 \
    --n_layers "$N_LAYERS" --hidden "$HIDDEN" --generator_arch "$GEN_ARCH" \
    --clip_input "$CLIP_INPUT" \
    --ubp_ckpt "$UBP_CKPT" \
    --semantic_mode ubp_margin --lambda_sem 0 \
    --lambda_time 1.0 --lambda_corr 0.8 \
    --lambda_freq 0 --lambda_band 0 \
    --lambda_band_corr 0.32 \
    --band_weights "delta=2.0,beta=1.5,gamma=1.0" \
    --gamma_fmax 45 \
    --lambda_ubp 0.5 --lambda_eeg_nce 0.1 --lambda_margin 0 --lambda_div 0 \
    --ckpt_dir "$CKPT_FINAL" --result_dir "$RESULT_FINAL" \
    "${resume_args[@]}"

  if [[ -f "${CKPT_FINAL}/${SUBTAG}/best.pt" ]]; then
    cp "${CKPT_FINAL}/${SUBTAG}/best.pt" "${CKPT_FINAL}/${SUBTAG}/last.pt"
  fi
}

train_two_stage_group() {
  local label="$1" result_root="$2"
  shift 2
  local variants=("$@")
  echo "===== ${label} ablation ====="
  echo "variants=${variants[*]} stage=${STAGE} subs=${SUB_LIST_PRINT} gpu=${GPU}"
  for variant in "${variants[@]}"; do
    for sub in $SUB_LIST; do
      if [[ "$STAGE" == "both" || "$STAGE" == "1" ]]; then
        run_two_stage_s1 "$variant" "$sub" "$result_root"
      fi
      if [[ "$STAGE" == "both" || "$STAGE" == "2" ]]; then
        run_two_stage_s2 "$variant" "$sub" "$result_root"
      fi
    done
  done
}

# --- loss-group (single stage) ---

loss_weights() {
  case "$1" in
    full) echo "1.0 0.8 0.35 0.35 0.5 0.1" ;;
    no_time) echo "0.0 0.0 0.35 0.35 0.5 0.1" ;;
    no_freq) echo "1.0 0.8 0.0 0.0 0.5 0.1" ;;
    no_semantic) echo "1.0 0.8 0.35 0.35 0.0 0.0" ;;
    *) echo "[error] unknown loss variant: $1"; return 1 ;;
  esac
}

train_loss_group() {
  local variants=()
  if [[ -n "${VARIANTS:-}" ]]; then
    read -ra variants <<< "${VARIANTS}"
  else
    variants=(full no_time no_freq no_semantic)
  fi
  local result_root="results/generator_loss_ablation"
  echo "===== loss-group ablation ====="
  echo "variants=${variants[*]} subs=${SUB_LIST_PRINT} gpu=${GPU}"

  for variant in "${variants[@]}"; do
    read -r LT LC LF LB LU LN <<< "$(loss_weights "$variant")"
    local ckpt_variant="$variant"
    [[ "$variant" == "full" ]] && ckpt_variant="single_stage"
    local CKPT_DIR="${LOSS_CKPT_ROOT}/${ckpt_variant}"
    local RESULT_DIR="${result_root}/${variant}/full"
    mkdir -p "$CKPT_DIR" "$RESULT_DIR"

    for sub in $SUB_LIST; do
      printf -v SUBTAG "sub-%02d" "$sub"
      local UBP_CKPT="${ENCODER_ROOT}/${SUBTAG}_seed0/checkpoints/last.ckpt"
      local METRICS="${RESULT_DIR}/${SUBTAG}/metrics.json"
      local LAST_CKPT="${CKPT_DIR}/${SUBTAG}/last.pt"

      if [[ -f "$METRICS" && "$FORCE" != "1" ]]; then
        echo "[skip] ${variant} ${SUBTAG}"
        continue
      fi
      local resume_args=()
      if [[ "$RESUME" == "1" && -f "$LAST_CKPT" && ! -f "$METRICS" ]]; then
        resume_args=(--resume)
      fi

      echo "=== loss ${variant} ${SUBTAG} ==="
      python -m predictor.train \
        --sub "$sub" --gpu "$GPU" --seed 2023 \
        --epochs 100 --early_stop_patience 15 --early_stop_metric loss \
        --lr 1e-4 \
        --n_layers 4 --generator_arch full \
        --clip_input fovea \
        --ubp_ckpt "$UBP_CKPT" \
        --semantic_mode ubp_margin --lambda_sem 0 \
        --lambda_time "$LT" --lambda_corr "$LC" \
        --lambda_freq "$LF" --lambda_band "$LB" \
        --lambda_band_corr 0 \
        --gamma_fmax 45 \
        --lambda_ubp "$LU" --lambda_eeg_nce "$LN" --lambda_margin 0 --lambda_div 0 \
        --ckpt_dir "$CKPT_DIR" --result_dir "$RESULT_DIR" \
        "${resume_args[@]}"

      if [[ -f "${CKPT_DIR}/${SUBTAG}/best.pt" ]]; then
        cp "${CKPT_DIR}/${SUBTAG}/best.pt" "${CKPT_DIR}/${SUBTAG}/last.pt"
      fi
    done
  done
}

run_group() {
  local g="$1"
  case "$g" in
    structural)
      local vars=()
      if [[ -n "${VARIANTS:-${ARCHS:-}}" ]]; then
        read -ra vars <<< "${VARIANTS:-${ARCHS}}"
      else
        vars=(no_Dilated no_Transformer)
      fi
      train_two_stage_group "structural" "results/generator_structural_ablation" "${vars[@]}"
      ;;
    extended)
      local vars=()
      if [[ -n "${VARIANTS:-}" ]]; then
        read -ra vars <<< "${VARIANTS}"
      else
        vars=(no_FoveaBlur no_self_attn h128 h512)
      fi
      train_two_stage_group "extended" "results/generator_extended_ablation" "${vars[@]}"
      ;;
    loss)
      train_loss_group
      ;;
    *)
      usage
      ;;
  esac
}

aggregate_group() {
  local g="$1"
  case "$g" in
    structural)
      PYTHONPATH=. python -m analysis.eeg_gen_eval.compute.compute_structural_ablation --all --force_freq
      ;;
    extended)
      PYTHONPATH=. python -m analysis.eeg_gen_eval.compute.compute_extended_ablation --all --force_freq
      ;;
    loss)
      echo "[note] loss-group has no separate aggregate module; skip"
      return 0
      ;;
  esac
  MPLBACKEND=Agg PYTHONPATH=. python -m analysis.eeg_gen_eval.redraw_all --only final_fig4
}

GROUPS=()
case "$GROUP" in
  all) GROUPS=(structural extended loss) ;;
  structural|extended|loss) GROUPS=("$GROUP") ;;
  *) usage ;;
esac

if [[ "$SKIP_TRAIN" != "1" ]]; then
  for g in "${GROUPS[@]}"; do
    run_group "$g"
  done
fi

if [[ "$AGGREGATE" == "1" ]]; then
  for g in "${GROUPS[@]}"; do
    [[ "$g" == "loss" ]] && continue
    aggregate_group "$g"
  done
fi

echo "Done $(date -Is) group=${GROUP}"
