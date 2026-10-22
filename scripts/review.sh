#!/usr/bin/env bash
# Small visible datasets for reviewing IO and LAWA.
set -euxo pipefail
cd "$(git rev-parse --show-toplevel)"

printf '\n[review 1/4] fmt, lint, exports, signatures and <=3 indentation levels\n'
uv run --locked ruff format --check gdtensor gdtensor_sim tests
uv run --locked ruff check gdtensor gdtensor_sim tests
uv run --locked pytest -v -p no:cacheprovider tests/test_api_conformance.py \
  -k 'code_blocks or exports or function_signatures'

printf '\n[CRC+ARANGE][review 2/4] arange 0..11; one 3x4 tensor; CRC and 2->3 rank load/reshard\n'
uv run --locked pytest -v -s --log-cli-level=INFO -p no:cacheprovider tests/test_walkthrough.py

printf '\n[review 3/4] LAWA; 2 real ranks; 3 tensors (3x4, 2x3, 1); naive/Welford; k=1/2/3/5\n'
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch pytest -v -s -p no:cacheprovider \
  tests/test_lawa.py -k 'safetensors-dp-False-2-3 or distcp-tp-False-2-3'

printf '\n[CRC+ARANGE][review 4/4] checkpoint merge; 3->5->2 simulated ranks; 16 small tensors; mixed formats; file CRC and first 12 values\n'
uv run --locked pytest -v -s -p no:cacheprovider tests/test_checkpoint_merge_crc.py \
  -k 'source_p0 and weighted and safetensors'
