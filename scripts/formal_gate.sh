#!/usr/bin/env bash
# Real formal gate (used by CI and the local precommit/full-check hooks).
#
# Runs the two formal proofs with FIXED tool versions and blocks (non-zero
# exit) on any failure. This is a real gate, not an echo:
#   1. Lean  proof -- elan selects formal/lean/lean-toolchain
#      (leanprover/lean4:v4.34.0); formal/lean/check.sh fails on any
#      sorry/admit/axiom or compile error.
#   2. TLC    -- scripts/get_tla_tools.sh provisions the pinned, sha256-verified
#      tla2tools.jar (v1.8.0) and formal/tla/run_tlc.sh model-checks the real
#      CopyPlan / AtomicPublish specs over their S/N grids, the
#      formal/tla/SlotLifecycle.tla slot-state model (per-ready-slot success,
#      invariant violations, peer-down and out-of-generation negatives), and the
#      formal/tla/CachePropagation.tla node-level propagation-plan model (byte
#      budget, parent/child order, owed-rank/edge release, fill error, and the
#      bounded absent-consumer liveness boundary). Both the Lean (5 positive
#      files + 6 negative false lemmas, all proof-level rejected) and TLA
#      (30 cases) suites have passed on the digest host.
#
# It does NOT modify the owners' entrypoints (formal/lean/check.sh,
# formal/tla/run_tlc.sh); it only invokes them with the pinned toolchain / jar.
set -euo pipefail
# Preflight: ROOT must be THIS script's repo. Compare git's worktree toplevel
# (physical path) against the directory implied by this script; require a
# resolvable git-dir and the formal entrypoints. Outside any git tree, or a
# nested/wrong ancestor checkout -> fail nonzero BEFORE any Lean/TLC tool runs.
EXPECTED_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "[formal] FATAL: not inside a git worktree (expected $EXPECTED_ROOT)" >&2
  exit 1
}
[ "$(cd "$ROOT" && pwd -P)" = "$EXPECTED_ROOT" ] || {
  echo "[formal] FATAL: worktree $ROOT != script repo $EXPECTED_ROOT (nested/wrong ancestor)" >&2
  exit 1
}
git rev-parse --git-dir >/dev/null 2>&1 || { echo "[formal] FATAL: no git-dir" >&2; exit 1; }
# Require a real committed HEAD: a fresh `git init` with no commit must not pass
# off "HEAD: none" as source provenance.
HEAD_SHA="$(git rev-parse --verify HEAD 2>/dev/null)" || {
  echo "[formal] FATAL: repository has no committed HEAD; refusing fake provenance" >&2
  exit 1
}
HEAD_TREE="$(git rev-parse --verify "HEAD^{tree}" 2>/dev/null)" || {
  echo "[formal] FATAL: cannot resolve HEAD tree" >&2; exit 1
}
cd "$ROOT"
test -f formal/lean/check.sh && test -f formal/tla/run_tlc.sh || {
  echo "[formal] FATAL: formal entrypoints missing under $ROOT" >&2; exit 1;
}
# Live provenance manifest: committed HEAD + its tree vs staged model tree.
mkdir -p log/fix-df1/formal
{
  echo "HEAD: $HEAD_SHA"
  echo "HEAD_TREE: $HEAD_TREE"
  echo "index-tree: $(git write-tree)"
  echo "worktree: $(git rev-parse --show-toplevel)"
  find formal scripts -name '*.sh' -o -name '*.lean' -o -name '*.tla' -o -name '*.cfg' \
    | sort | xargs shasum -a 256
} > log/fix-df1/formal/manifest.txt
echo "[formal] root=$ROOT HEAD=$HEAD_SHA; manifest: log/fix-df1/formal/manifest.txt"

# elan auto-installs the toolchain pinned in formal/lean/lean-toolchain.
export PATH="$HOME/.elan/bin:$PATH"

echo "[formal] Lean proof (toolchain pin: $(cat formal/lean/lean-toolchain))"
bash formal/lean/check.sh

echo "[formal] TLC model checking (provision pinned tla2tools jar)"
JAR="$(bash scripts/get_tla_tools.sh)"
echo "[formal] using pinned jar: $JAR"
TLA2TOOLS_JAR="$JAR" bash formal/tla/run_tlc.sh

echo "[formal] OK: Lean proof compiles (no sorry/admit/axiom); TLC models pass"
