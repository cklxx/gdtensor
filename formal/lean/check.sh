#!/usr/bin/env bash
# Reproducible check of the shard-placement proof.
#
# The exact Lean toolchain is pinned in ./lean-toolchain. elan (the Lean
# version manager) reads that file from this directory; no project-level
# lakefile or dependency is needed because ShardAlgebra.lean only uses the
# Lean core Init library. A non-zero exit means the proof no longer compiles
# (e.g. a sorry or a broken theorem). There is no cached / machine-specific
# state: run this after any edit to ShardAlgebra.lean.
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

# Hard gate: forbid proof holes and admitted axioms in the source.
if grep -nE '(^|[^A-Za-z0-9_])(sorry|admit)([^A-Za-z0-9_]|$)|(^|[[:space:]])axiom[[:space:]]' ShardAlgebra.lean; then
  echo "ERROR: ShardAlgebra.lean must not contain sorry/admit/axiom" >&2
  exit 1
fi

echo "lean: $("$LEAN_BIN" --version)"
echo "toolchain pin: $(cat lean-toolchain)"
# Emit the .olean into the git-ignored log/ tree so this repo is not polluted.
# warningAsError turns any 'declaration uses sorry' style warning into failure.
OUT="$ROOT/log/lean"
mkdir -p "$OUT"
"$LEAN_BIN" -DwarningAsError=true -o "$OUT/ShardAlgebra.olean" ShardAlgebra.lean
echo "ShardAlgebra.lean: OK (compiles, no sorry/admit/axiom); olean -> $OUT"
