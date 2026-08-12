#!/usr/bin/env bash
# Train + aggregate manuscript Fig4 structural ablations, then redraw final_fig4.
#
# Usage:
#   bash scripts/run_structural_ablation.sh
#   SKIP_TRAIN=1 bash scripts/run_structural_ablation.sh
#   ARCHS="no_Dilated" bash scripts/run_structural_ablation.sh 1

set -euo pipefail
cd "$(dirname "$0")/.."

SUBS="${1:-1-10}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  bash scripts/train_structural_ablation.sh "$SUBS"
fi

PYTHONPATH=. python -m analysis.eeg_gen_eval.compute.compute_structural_ablation --all --force_freq
MPLBACKEND=Agg PYTHONPATH=. python -m analysis.eeg_gen_eval.redraw_all --only final_fig4

echo "Done. See analysis/eeg_gen_eval/raw/structural_ablation/ and figures/final_fig4_*"
