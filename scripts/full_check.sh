#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "[full-check] 1/5 fmt check"
uv run --locked ruff format --check gdtensor gdtensor_sim tests

echo "[full-check] 2/5 lint"
uv run --locked ruff check --no-cache gdtensor gdtensor_sim tests

echo "[full-check] 3/5 type check"
uv run --locked --extra torch ty check --error-on-warning
uv run --locked --extra torch pyrefly check --min-severity=warn

echo "[full-check] 4/5 byte-compile"
uv run --locked python -m compileall -q gdtensor gdtensor_sim tests

echo "[full-check] 5/5 all pytest files; real torch rank tests enabled"
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch pytest \
  -v --capture=tee-sys --log-cli-level=INFO -p no:cacheprovider
