#!/usr/bin/env bash
# Run a Python module/script with the project's src layout and existing EEG env.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec python "$@"
