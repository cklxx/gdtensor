#!/usr/bin/env bash
# Reproducible check of the gdtensor shard-placement + mapping proofs.
#
# The exact Lean toolchain is pinned in ./lean-toolchain. elan reads that file
# from this directory; no project lakefile or dependency is needed -- every
# file below only uses the Lean core Init library (no mathlib). A non-zero
# exit means a proof no longer compiles, a proof hole slipped in, or the
# intentionally-broken negative fixtures stopped being rejected for the right
# reason. There is no machine-specific cached state.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$HERE"

LEAN_BIN=""
if command -v lean >/dev/null 2>&1; then
  LEAN_BIN="$(command -v lean)"
elif [[ -x "$HOME/.elan/bin/lean" ]]; then
  LEAN_BIN="$HOME/.elan/bin/lean"
else
  echo "lean/elan not found. Install elan (https://lean-lang.org/); it will" >&2
  echo "auto-select the toolchain pinned in ./lean-toolchain." >&2
  exit 1
fi

# Positive proofs, in dependency order (each imports the earlier ones).
POSITIVES=(ShardAlgebra RowMajor ChunkRun Coverage Leader)

# Hard gate: forbid proof holes and admitted axioms in the POSITIVE sources.
# Match standalone tactic/declaration tokens; ignore header comment prose that
# merely describes the ban.
hole_grep () {
  grep -nE '(^|[[:space:]])(sorry|admit)([[:space:]]|$)|(^|[[:space:]])axiom[[:space:]]' "$1" \
    | grep -vE '^[0-9]+:[[:space:]]*(--|/-!)'
}
for f in "${POSITIVES[@]}"; do
  if hole_grep "$f.lean"; then
    echo "ERROR: $f.lean must not contain sorry/admit/axiom" >&2
    exit 1
  fi
done
# Negative fixtures must also be hole-free; their rejection comes from a false
# decidable claim, not from an admitted proof.
if hole_grep NegativeFixtures.lean; then
  echo "ERROR: NegativeFixtures.lean must not contain sorry/admit/axiom" >&2
  exit 1
fi

echo "lean: $("$LEAN_BIN" --version)"
echo "toolchain pin: $(cat lean-toolchain)"
# Emit .olean into the git-ignored log/ tree so this repo is not polluted.
# warningAsError turns any 'declaration uses sorry' style warning into failure.
OUT="$ROOT/log/lean"
mkdir -p "$OUT"
export LEAN_PATH="$OUT"

for f in "${POSITIVES[@]}"; do
  "$LEAN_BIN" -DwarningAsError=true -R "$HERE" -o "$OUT/$f.olean" "$f.lean"
  echo "$f.lean: OK (compiles, no sorry/admit/axiom); olean -> $OUT"
done

# --- Intentionally broken negative fixtures ---------------------------------
# This file MUST NOT compile. We expect a proof-level rejection (decide /
# tactic failed on a false claim), NOT a syntax or unbound-name error.
NEGOUT="$OUT/neg"
mkdir -p "$NEGOUT"
NEGLOG="$NEGOUT/log.txt"
if LEAN_PATH="$OUT" "$LEAN_BIN" -DwarningAsError=true -R "$HERE" \
     -o "$NEGOUT/NegativeFixtures.olean" NegativeFixtures.lean >"$NEGLOG" 2>&1; then
  echo "ERROR: NegativeFixtures.lean compiled, but it must be REJECTED" >&2
  exit 1
fi
if grep -qE 'unknown identifier|parse error|invalid|unbound' "$NEGLOG"; then
  echo "ERROR: negative failure looks like a syntax/unbound error, not a proof rejection:" >&2
  cat "$NEGLOG" >&2
  exit 1
fi
if ! grep -qE 'failed|not true|motive|decide|type mismatch' "$NEGLOG"; then
  echo "ERROR: negative failure does not look proof-level:" >&2
  cat "$NEGLOG" >&2
  exit 1
fi
echo "NegativeFixtures.lean: correctly REJECTED (proof-level; see $NEGLOG)"

echo "ALL POSITIVE PROOFS COMPILE, NO HOLES; NEGATIVE FIXTURES REJECTED AS DESIGNED."
