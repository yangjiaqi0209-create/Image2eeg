#!/usr/bin/env bash
# Shared bootstrap for scripts/*.sh — source from repo scripts only.
# shellcheck disable=SC2034
_UBP_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_UBP_SCRIPTS_DIR}/.." && pwd)"
cd "$REPO_ROOT"

# Optional machine-local overrides (never commit data/env.local).
if [[ -f "$REPO_ROOT/data/env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/data/env.local"
  set +a
elif [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

export UBP_REPO_ROOT="${UBP_REPO_ROOT:-$REPO_ROOT}"
export UBP_EEG_DATA_ROOT="${UBP_EEG_DATA_ROOT:-$HOME/datasets/EEG}"

CONDA_ENV="${CONDA_ENV:-EEG}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"
