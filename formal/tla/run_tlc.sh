#!/bin/bash
# Run TLC on a small grid of models: S in {2,3}, N covering divisible,
# non-divisible, and boundary cases.
set -e
cd "$(dirname "$0")"
JAR=/tmp/tla2tools.jar
mkdir -p /tmp/tla_out
for S in 2 3; do
  case $S in
    2) RANKS='{0,1}';;
    3) RANKS='{0,1,2}';;
  esac
  for N in 1 2 3 5 6; do
    CFG=/tmp/tla_cfgs/cp_${S}_${N}.cfg
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
    echo "=== CopyPlan S=$S N=$N ==="
    java -cp "$JAR" tlc2.TLC -deadlock -config "$CFG" -cleanup -metadir /tmp/tla_out "$(pwd)/CopyPlan.tla" 2>&1 | grep -E "Model checking|states, distinct|Invariant|Error|finished" | tail -4
  done
done
echo "=== AtomicPublish ==="
cat > /tmp/tla_cfgs/ap.cfg <<CFG
SPECIFICATION Spec
INVARIANTS
VersionComplete
CrashBeforeRenameKeepsOld
NewOnlyAfterRename
CFG
java -cp "$JAR" tlc2.TLC -deadlock -config /tmp/tla_cfgs/ap.cfg -cleanup -metadir /tmp/tla_out "$(pwd)/AtomicPublish.tla" 2>&1 | grep -E "Model checking|states, distinct|Invariant|Error|finished" | tail -4
