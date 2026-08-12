#!/usr/bin/env bash
# Train THINGS-EEG frozen brain encoders used by the manuscript generator pipeline.
# Config: configs/eeg/fixed_fovea.yaml  →  checkpoints/encoder/THINGSEEG2
#
# Usage:
#   bash scripts/train_things_encoder.sh           # sub-01..10
#   bash scripts/train_things_encoder.sh 1         # sub-01 only
#   bash scripts/train_things_encoder.sh 1-3

set -euo pipefail
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

SUBS="${1:-1-10}"
SEED="${SEED:-0}"
EPOCH="${EPOCH:-50}"
LR="${LR:-1e-4}"
CONFIG="configs/eeg/fixed_fovea.yaml"
BRAIN="EEGProjectLayer"
VISION="RN50"

if [[ "$SUBS" == *-* ]]; then
  IFS='-' read -r S0 S1 <<< "$SUBS"
  SUB_LIST=$(seq "$S0" "$S1")
else
  SUB_LIST="$SUBS"
fi

for i in $SUB_LIST; do
  tag=$(printf 'sub-%02d' "$i")
  echo "===== encoder $tag ====="
  python -m encoder.train \
    --config "$CONFIG" \
    --subjects "$tag" \
    --seed "$SEED" \
    --exp_setting intra-subject \
    --brain_backbone "$BRAIN" \
    --vision_backbone "$VISION" \
    --epoch "$EPOCH" \
    --lr "$LR"
done
