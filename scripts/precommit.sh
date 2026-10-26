#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Forward to the digest host when we are on a macOS dev machine and not already
# running on the remote (or when explicitly forced). On Linux CI / on the digest
# host itself, run checks directly -- CI must never ssh to a private host.
IS_MAC="$(uname -s)"; IS_MAC="${IS_MAC:-Linux}"
if [ "${GDTENSOR_REMOTE_TEST:-0}" != "1" ] && { [ "$IS_MAC" = "Darwin" ] || [ "${GDTENSOR_FORCE_REMOTE:-0}" = "1" ]; }; then
  exec scripts/remote_gate.sh precommit
fi

# On digest / Linux CI, uv selects the interpreter and locked dependencies.
# PYTHONPATH must NOT inherit any base-env path -- import only THIS snapshot.
export PYTHONPATH="."

echo "[precommit] 1/5 fmt check"
uv run --locked --extra torch-cpu ruff format --check gdtensor gdtensor_sim tests

echo "[precommit] 2/5 lint"
uv run --locked --extra torch-cpu ruff check --no-cache gdtensor gdtensor_sim tests

echo "[precommit] 3/5 byte-compile"
uv run --locked --extra torch-cpu python -m compileall -q gdtensor gdtensor_sim tests

echo "[precommit] 4/5 package imports"
uv run --locked --extra torch-cpu python - <<'PY'
import gdtensor, gdtensor.io, gdtensor.builder, gdtensor.session, gdtensor.access, gdtensor.tag
import gdtensor_sim.models, gdtensor_sim.runner
print("imports ok")
PY

echo "[precommit] 5/5 API, IO contract, constraints and indentation"
uv run --locked --extra torch-cpu python -m pytest -v --capture=tee-sys --log-cli-level=INFO -p no:cacheprovider \
  tests/core tests/io tests/checkpoint/test_parallel_modes.py tests/sim/test_real_file_accessors.py

# The basic path (fmt/lint/compile/imports/contract pytest) is self-contained.
# The advanced formal gate is a separate opt-in step, independent of basic:
#   GDTENSOR_ADVANCED_FORMAL=1 scripts/precommit.sh
if [ "${GDTENSOR_ADVANCED_FORMAL:-0}" = "1" ]; then
  echo "[precommit] ADVANCED formal gate (pinned Lean v4.34.0 + sha256-verified TLC v1.8.0)"
  bash scripts/formal_gate.sh
else
  echo "[precommit] skip advanced formal gate (run it explicitly with GDTENSOR_ADVANCED_FORMAL=1)"
fi
