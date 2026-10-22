#!/usr/bin/env bash
# Full local gate: every pytest including gated torch bridge tests.
# Requires torch and a single-process gloo group; sets env itself.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY="${PYTHON:-python3}"
if command -v ruff >/dev/null 2>&1; then RUFF="ruff"; else RUFF="$PY -m ruff"; fi

$RUFF check --select F401 --no-cache gdtensor gdtensor_sim tests
$PY -m compileall -q gdtensor gdtensor_sim tests

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export RANK=0
export WORLD_SIZE=1
export GDTENSOR_TORCH_TESTS=1

$PY -m pytest -q -p no:cacheprovider

echo "[full-check] OK"
