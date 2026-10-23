#!/usr/bin/env bash
# Parallel-axis demo: LAWA, CRC and arange under PP / DP / EP / TP.
#
#   scripts/demo_axes.sh          run once in the current terminal
#   scripts/demo_axes.sh --tmux   (re)start tmux session "gdt" and attach
#   scripts/demo_axes.sh --loop   run forever (used inside the tmux session)
#
# Each stage prints its own tag. "PASS" lines show the check that held;
# corruption cases print PASS when the detector caught the fault.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
SESSION=gdt

if [[ "${1:-}" == "--tmux" ]]; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" -n demo -c "$PWD" "scripts/demo_axes.sh --loop; exec bash"
  echo "tmux session '$SESSION' started; attach with: tmux attach -t $SESSION"
  [[ -t 1 ]] && exec tmux attach -t "$SESSION"
  exit 0
fi

banner() { printf '\n\033[1;36m==== %s ====\033[0m\n' "$*"; }

run_once() {
  local started=$SECONDS
  banner "[1/4] axis matrix guard: 15 PP/DP/EP/TP combinations, LAWA cases cross PP and EP"
  uv run --locked pytest -q -p no:cacheprovider \
    tests/checkpoint/test_parallel_modes.py::test_parallel_mode_matrix_covers_every_nonempty_axis_combination \
    tests/checkpoint/test_merge_crc_axes.py::test_merge_cases_cross_ep_with_other_axes \
    tests/torch/test_lawa_axes.py::test_lawa_cases_cover_pp_ep_and_topology_changes

  banner "[2/4] CRC+ARANGE: 4-axis checkpoint, 16 thread ranks -> 12, expert owners and local values"
  uv run --locked pytest -v -s -p no:cacheprovider tests/checkpoint/test_parallel_modes.py \
    -k 'PP_DP_EP_TP and rank_multi and safetensors' 2>&1 | grep -E 'PARALLEL_MODES|PASSED|FAILED|passed|failed'

  banner "[3/4] CRC+ARANGE: 3 checkpoints A, A+3, A+9 merged across EP grids; expert payload swap must fail"
  uv run --locked pytest -v -s -p no:cacheprovider tests/checkpoint/test_merge_crc_axes.py \
    -k '(PP_DP_EP_TP and weighted and safetensors) or (swapped and safetensors)' \
    2>&1 | grep -E '\[AXES\]|PASSED|FAILED|passed|failed'

  banner "[4/4] LAWA on real torch ranks: PP x DP x EP x TP (16 processes) and a topology change"
  GDTENSOR_TORCH_TESTS=1 uv run --locked --extra torch pytest -v -s -p no:cacheprovider \
    tests/torch/test_lawa_axes.py -k 'PP_DP_EP_TP or PP_EP_TP_to_PP_DP_EP' \
    2>&1 | grep -E 'lawa-axes|PASSED|FAILED|passed|failed'

  banner "round finished in $((SECONDS - started))s"
}

if [[ "${1:-}" == "--loop" ]]; then
  round=1
  while true; do
    banner "gdtensor parallel-axis demo, round $round, $(date '+%F %T')"
    run_once || banner "round $round FAILED (see above); retrying in 30s"
    round=$((round + 1))
    sleep 30
  done
fi

run_once
