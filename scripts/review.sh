#!/usr/bin/env bash
# Small visible datasets for reviewing IO and LAWA.
set -euxo pipefail
cd "$(git rev-parse --show-toplevel)"

printf '\n[review 1/7] fmt, lint, exports, signatures and <=3 indentation levels\n'
uv run --locked ruff format --check gdtensor gdtensor_sim tests
uv run --locked ruff check gdtensor gdtensor_sim tests
uv run --locked pytest -v -p no:cacheprovider tests/core/test_api_conformance.py \
  -k 'code_blocks or exports or function_signatures'

printf '\n[CRC+ARANGE][review 2/7] arange 0..11; one 3x4 tensor; CRC and 2->3 rank load/save\n'
uv run --locked pytest -v -s --log-cli-level=INFO -p no:cacheprovider tests/demo/test_walkthrough.py

printf '\n[review 3/7] LAWA; 2 real ranks; 3 tensors (3x4, 2x3, 1); naive/Welford; k=1/2/3/5\n'
GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch pytest -v -s -p no:cacheprovider \
  tests/torch/test_lawa.py -k 'safetensors-dp-False-2-3 or distcp-tp-False-2-3'

printf '\n[CRC+ARANGE][review 4/7] checkpoint merge; 3->5->2 simulated ranks; 16 small tensors; mixed formats; file CRC and first 12 values\n'
uv run --locked pytest -v -s -p no:cacheprovider tests/checkpoint/test_checkpoint_merge_crc.py \
  -k 'source_p0 and weighted and safetensors'


printf '\n[CRC+ARANGE][review 5/7] serial baseline vs 3 parallel thread ranks; 16 tensors; exact assembled bytes, canonical files and read schedules; no numeric averaging\n'
uv run --locked pytest -v -s -p no:cacheprovider tests/checkpoint/test_serial_parallel_crc.py \
  -k 'parallel0 and safetensors'

printf '\n[PLAN+CONSTRAINTS][review 6/7] public IO range traces; hand-calculated 2x5 I16; shuffled and empty ranks; packed offsets; 5 replicated tensors; failures before begin\n'
uv run --locked pytest -v -s -p no:cacheprovider \
  tests/io/test_plan_behavior.py tests/core/test_constraints.py

printf '\n[CRC+ARANGE][PARALLEL_MODES][review 7/7] PP+DP+EP+TP; 16->12 thread ranks; 11 tensors, at most 15 elements each; expert owners and local values; canonical files equal serial baseline\n'
uv run --locked pytest -v -s -p no:cacheprovider tests/checkpoint/test_parallel_modes.py \
  -k 'PP_DP_EP_TP and rank_multi and safetensors'
