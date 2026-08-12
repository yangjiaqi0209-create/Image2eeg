#!/usr/bin/env bash
# Convert Alljoined-1.6M HF preprocessed data to UBP .pt format.
#
# Usage:
#   bash scripts/preprocess_alljoined_all.sh
#   SMOKE=1 bash scripts/preprocess_alljoined_all.sh   # sub-01 only
#   SUBJECTS="1 2 3" bash scripts/preprocess_alljoined_all.sh
#   FORCE=1 bash scripts/preprocess_alljoined_all.sh     # overwrite existing
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP

EEG_ROOT="${UBP_EEG_DATA_ROOT:-/home/ubuntu/dataset/EEG}"
HF_ROOT="${EEG_ROOT}/alljoined-1.6M/raw_hf"
OUT_DIR="${EEG_ROOT}/alljoined-1.6M/ubp_preprocessed"
IMAGE_ROOT="${UBP_REPO_ROOT:-$(pwd)}/data/things-eeg/Image_set_Resize"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"

if [[ -n "${ALLJOINED_SUBJECTS:-}" ]]; then
  read -r -a SUBJECTS <<< "${ALLJOINED_SUBJECTS}"
elif [[ "${SMOKE}" == "1" ]]; then
  SUBJECTS=(1)
else
  SUBJECTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

EXTRA=()
[[ "${FORCE}" == "1" ]] && EXTRA+=(--force)

mkdir -p "${OUT_DIR}"
LOG="${EEG_ROOT}/alljoined-1.6M/logs/preprocess.log"

for sub in "${SUBJECTS[@]}"; do
  echo "==> sub-$(printf '%02d' "$sub")" | tee -a "${LOG}"
  PYTHONPATH=. python preprocess/convert_alljoined.py \
    --subject "${sub}" \
    --hf_root "${HF_ROOT}" \
    --output_dir "${OUT_DIR}" \
    --image_root "${IMAGE_ROOT}" \
    --verify \
    "${EXTRA[@]}" 2>&1 | tee -a "${LOG}"
done

echo "Done. UBP .pt under ${OUT_DIR}/"
