#!/usr/bin/env bash
# Download Alljoined-1.6M preprocessed EEG from HuggingFace (no raw EDF / stimuli.zip).
#
# Usage:
#   bash scripts/download_alljoined.sh              # all 20 subjects
#   SMOKE=1 bash scripts/download_alljoined.sh      # sub-01 only
#   SUBJECTS="1 2 3" bash scripts/download_alljoined.sh
#
# Uses hf-mirror.com when HF_ENDPOINT is unset (helps in CN networks).
set -euo pipefail

cd "$(dirname "$0")/.."

EEG_ROOT="${UBP_EEG_DATA_ROOT:-/home/ubuntu/dataset/EEG}"
OUT_DIR="${EEG_ROOT}/alljoined-1.6M/raw_hf"
SMOKE="${SMOKE:-0}"

if [[ -n "${ALLJOINED_SUBJECTS:-}" ]]; then
  read -r -a SUBJECTS <<< "${ALLJOINED_SUBJECTS}"
elif [[ "${SMOKE}" == "1" ]]; then
  SUBJECTS=(1)
else
  SUBJECTS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate UBP

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${OUT_DIR}"
LOG_DIR="${EEG_ROOT}/alljoined-1.6M/logs"
mkdir -p "${LOG_DIR}"

echo "==> Downloading experiment_metadata_categories.parquet"
OUT_DIR="${OUT_DIR}" python - <<'PY'
import os
from huggingface_hub import hf_hub_download

out = os.environ["OUT_DIR"]
endpoint = os.environ.get("HF_ENDPOINT")
hf_hub_download(
    "Alljoined/Alljoined-1.6M",
    "preprocessed_eeg/experiment_metadata_categories.parquet",
    repo_type="dataset",
    local_dir=out,
    endpoint=endpoint,
)
print("OK categories parquet")
PY

for sub in "${SUBJECTS[@]}"; do
  sub_tag="sub-$(printf '%02d' "$sub")"
  echo "==> ${sub_tag}"
  OUT_DIR="${OUT_DIR}" SUB="${sub}" python - <<'PY'
import os
from huggingface_hub import hf_hub_download

out = os.environ["OUT_DIR"]
sub = int(os.environ["SUB"])
endpoint = os.environ.get("HF_ENDPOINT")
prefix = f"preprocessed_eeg/sub-{sub:02d}"
files = [
    f"{prefix}/stim_order.parquet",
    f"{prefix}/preprocessed_eeg_training_flat.npy",
    f"{prefix}/preprocessed_eeg_test_flat.npy",
]
for f in files:
    p = hf_hub_download(
        "Alljoined/Alljoined-1.6M",
        f,
        repo_type="dataset",
        local_dir=out,
        endpoint=endpoint,
    )
    print("  ", f, os.path.getsize(p))
PY
done

echo "Done. Raw files under ${OUT_DIR}/preprocessed_eeg/"
