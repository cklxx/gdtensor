#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

echo "[precommit] 1/5 fmt check"
uv run --locked ruff format --check gdtensor gdtensor_sim tests

echo "[precommit] 2/5 lint"
uv run --locked ruff check --no-cache gdtensor gdtensor_sim tests

echo "[precommit] 3/5 byte-compile"
uv run --locked python -m compileall -q gdtensor gdtensor_sim tests

echo "[precommit] 4/5 package imports"
uv run --locked python - <<'PY'
import gdtensor, gdtensor.io, gdtensor.builder, gdtensor.session, gdtensor.access, gdtensor.tag
import gdtensor_sim.models, gdtensor_sim.runner
print("imports ok")
PY

echo "[precommit] 5/5 API, IO contract, constraints and indentation"
uv run --locked pytest -v --capture=tee-sys --log-cli-level=INFO -p no:cacheprovider \
  tests/test_api_conformance.py tests/test_api.py \
  tests/test_io_contract.py tests/test_constraints.py
