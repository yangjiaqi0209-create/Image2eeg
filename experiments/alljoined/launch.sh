#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
BASE="${ALLJOINED_RUN_ROOT:-results/alljoined}"
mkdir -p "$BASE/logs" "$BASE/status"
if [[ -f "$BASE/status/supervisor.pid" ]]; then
  read -r previous_pid < "$BASE/status/supervisor.pid"
  if kill -0 "$previous_pid" 2>/dev/null; then
    echo "Supervisor already running: $previous_pid"
    exit 0
  fi
fi
export NUMBA_CACHE_DIR=/tmp/numba_cache
export MPLCONFIGDIR=/tmp/eeg_cross_subject_mpl
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
# Preserve the verified download environment. Recovery waits on transient
# network failures instead of exhausting the queue while the proxy is down.
nohup setsid "${PYTHON:-python}" "$ROOT/experiments/alljoined/supervise.py" \
  "$@" >> "$BASE/logs/nohup_supervisor.log" 2>&1 < /dev/null &
supervisor_pid=$!
echo "$supervisor_pid" > "$BASE/status/supervisor.pid"
for attempt in $(seq 1 20); do
  if "${PYTHON:-python}" - "$BASE/status/supervisor.json" "$supervisor_pid" <<'PY'
import json,sys
try:sys.exit(0 if json.load(open(sys.argv[1]))['pid']==int(sys.argv[2]) else 1)
except (OSError,ValueError,KeyError):sys.exit(1)
PY
  then
    echo "Independent-session supervisor initialized: $supervisor_pid"
    exit 0
  fi
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    echo "Supervisor failed during startup; inspect $BASE/logs/nohup_supervisor.log"
    exit 1
  fi
  sleep 1
done
echo "nohup supervisor PID: $supervisor_pid"

