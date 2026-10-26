#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# On a macOS dev machine forward to the digest host (no local checks). On Linux
# CI / on the digest host run directly. Same routing as precommit.sh.
IS_MAC="$(uname -s)"; IS_MAC="${IS_MAC:-Linux}"
if [ "${GDTENSOR_REMOTE_TEST:-0}" != "1" ] && { [ "$IS_MAC" = "Darwin" ] || [ "${GDTENSOR_FORCE_REMOTE:-0}" = "1" ]; }; then
  exec scripts/remote_gate.sh full
fi
unset PYTHONPATH; export PYTHONPATH="."

# Keep the same locked CPU torch environment for every full-check step.
echo "[full-check] 1/5 fmt check"
uv run --locked --extra torch-cpu ruff format --check gdtensor gdtensor_sim tests

echo "[full-check] 2/5 lint"
uv run --locked --extra torch-cpu ruff check --no-cache gdtensor gdtensor_sim tests

echo "[full-check] 3/5 type check"
uv run --locked --extra torch-cpu ty check --error-on-warning
uv run --locked --extra torch-cpu pyrefly check --min-severity=warn

echo "[full-check] 4/5 byte-compile"
uv run --locked --extra torch-cpu python -m compileall -q gdtensor gdtensor_sim tests

# The basic path (fmt/lint/types/compile/pytest) is self-contained. The
# advanced formal gate is a separate opt-in step, independent of basic:
#   GDTENSOR_ADVANCED_FORMAL=1 scripts/full_check.sh
if [ "${GDTENSOR_ADVANCED_FORMAL:-0}" = "1" ]; then
  echo "[full-check] ADVANCED formal gate (pinned Lean v4.34.0 + sha256-verified TLC v1.8.0)"
  bash scripts/formal_gate.sh
else
  echo "[full-check] skip advanced formal gate (run it explicitly with GDTENSOR_ADVANCED_FORMAL=1)"
fi

echo "[full-check] 5/5 all pytest files; real torch rank tests enabled"
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch-cpu python -m pytest \
  -v --capture=tee-sys --log-cli-level=INFO -p no:cacheprovider
