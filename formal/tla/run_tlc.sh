#!/usr/bin/env bash
# Run TLC on a small grid of models: S in {2,3}, N covering divisible,
# non-divisible, and boundary cases.
#
# All generated artifacts (TLA configs, TLC state dir, full logs) stay under
# the repository's git-ignored log/ tree so evidence is self-contained and
# never written to /tmp. The tla2tools jar is read-only input; override its
# location with TLA2TOOLS_JAR.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/log/tla"
mkdir -p "$OUT/cfgs" "$OUT/metadir"
: > "$OUT/copyplan.log"
: > "$OUT/atomic_publish.log"
JAR="${TLA2TOOLS_JAR:-/tmp/tla2tools.jar}"

run() {  # $1=model tla, $2=config, $3=log
  echo "--- $1 ($(basename "$2")) ---"
  java -cp "$JAR" tlc2.TLC -deadlock -config "$2" -cleanup -metadir "$OUT/metadir" "$HERE/$1" 2>&1 \
    | tee -a "$3" \
    | grep -E "Model checking|states, distinct|Invariant|Error|finished" | tail -4
}

for S in 2 3; do
  case $S in
    2) RANKS='{0,1}';;
    3) RANKS='{0,1,2}';;
  esac
  for N in 1 2 3 5 6; do
    CFG="$OUT/cfgs/cp_${S}_${N}.cfg"
    cat > "$CFG" <<CFG
SPECIFICATION Spec
CONSTANTS
N = $N
Ranks = $RANKS
INVARIANTS
SaveDstDisjoint
SaveCovering
SavePairCorrect
LoadPairCorrect
LoadShardedDisjoint
LoadShardedCovering
ReplicatedSingleWriter
BoundsOK
CFG
    run CopyPlan.tla "$CFG" "$OUT/copyplan.log"
  done
done

APCFG="$OUT/cfgs/ap.cfg"
cat > "$APCFG" <<CFG
SPECIFICATION Spec
INVARIANTS
VersionComplete
CrashBeforeRenameKeepsOld
NewOnlyAfterRename
CFG
run AtomicPublish.tla "$APCFG" "$OUT/atomic_publish.log"
echo "full logs: $OUT/copyplan.log , $OUT/atomic_publish.log"
