#!/usr/bin/env bash
# Fast pre-commit gate. No torch required: torch tests skip automatically.
# Runs the checks that catch almost every regression in a few seconds.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PY="${PYTHON:-python3}"
if command -v ruff >/dev/null 2>&1; then RUFF="ruff"; else RUFF="$PY -m ruff"; fi

echo "[precommit] 1/4 ruff (unused imports / syntax)"
$RUFF check --select F401 --no-cache gdtensor gdtensor_sim tests

echo "[precommit] 2/4 byte-compile all sources"
$PY -m compileall -q gdtensor gdtensor_sim tests

echo "[precommit] 3/4 public + core package import smoke"
$PY - <<'PY'
import gdtensor, gdtensor.io, gdtensor.builder, gdtensor.session, gdtensor.access, gdtensor.tag
import gdtensor.internal.api, gdtensor.internal.plan, gdtensor.internal.torch_bridge
import gdtensor_sim.models, gdtensor_sim.runner
print("imports ok")
PY

echo "[precommit] 4/4 core + API pytest (no torch; torch cases skip)"
$PY -m pytest -q -p no:cacheprovider \
  tests/test_api_conformance.py \
  tests/test_builder_api.py \
  tests/test_spec_alignment.py \
  tests/test_public_api.py \
  tests/test_interval_math.py \
  tests/test_interval_properties.py

echo "[precommit] OK"
