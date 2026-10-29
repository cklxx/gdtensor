#!/usr/bin/env bash
# TLC gate for gdtensor formal models.
#
# Layers:
#   1. The two original, read-only design models: CopyPlan.tla and AtomicPublish.tla.
#   2. SlotLifecycle.tla: faithful projection of gdtensor/internal/api.py
#      _NodeState/_SlotTable/executeChain. A small grid of POSITIVE configs must
#      satisfy all five safety invariants; a grid of deliberately-broken MUTANT
#      configs must be CAUGHT by exactly the invariant they target.
#   3. Bounded liveness: checked only under explicit assumptions (complete cohort,
#      external eventual-ACK, weak fairness, fills succeed). A partial-peer-down
#      config is EXPECTED to produce a liveness counterexample -- that boundary is
#      reported, not papered over.
#
# Evidence (generated cfgs, full stdout logs, metadir) stays under the repo's
# git-ignored log/ tree, never in /tmp and never in formal/tla/. The tla2tools
# jar is read-only input; override its location with TLA2TOOLS_JAR.
#
# Exit status:
#   0  every positive config passed and every negative config was caught by its
#      expected invariant.
#   1  a positive config violated an invariant, a negative mutant was NOT caught,
#      or an unexpected TLC failure occurred.
set -euo pipefail
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
OUT="$ROOT/log/fix-df1/formal/tla"
mkdir -p "$OUT/cfgs" "$OUT/metadir"
JAR="${TLA2TOOLS_JAR:-/tmp/tla2tools.jar}"
[ -r "$JAR" ] || { echo "FATAL: tla2tools jar not readable at $JAR" >&2; exit 1; }
WORKERS="${TLC_WORKERS:-2}"
HEAP="${TLC_HEAP:--Xmx4g}"

POS_LOG="$OUT/slot_pos.log"
NEG_LOG="$OUT/slot_neg.log"
LIVE_LOG="$OUT/slot_live.log"
: > "$POS_LOG"; : > "$NEG_LOG"; : > "$LIVE_LOG"

PASS=0; FAIL=0

# tlc <tla> <cfg> <logfile> -> captures full output; returns raw TLC exit code
# (0 = no invariant violation / property holds). Use `|| rc=$?` at call sites so
# we never toggle errexit globally (which clobbered the caller's set +e).
tlc() {
  local tla="$1" cfg="$2" log="$3" rc=0
  echo "=== $tla ($(basename "$cfg")) ===" | tee -a "$log"
  java -XX:+UseParallelGC "$HEAP" -cp "$JAR" tlc2.TLC -deadlock -nowarning -workers "$WORKERS" \
       -config "$cfg" -cleanup -metadir "$OUT/metadir" "$HERE/$tla" > "$log.run" 2>&1 \
       || rc=$?
  cat "$log.run" >> "$log"
  rm -f "$HERE"/SlotLifecycle_TTrace_*.bin "$HERE"/SlotLifecycle_TTrace_*.tla \
        "$HERE"/CopyPlan_TTrace_*.bin    "$HERE"/CopyPlan_TTrace_*.tla \
        "$HERE"/AtomicPublish_TTrace_*.bin "$HERE"/AtomicPublish_TTrace_*.tla || true
  return $rc
}

# ============================================================================
# 1. Original design models (unchanged behavior)
# ============================================================================
echo "########## original CopyPlan grid ##########" | tee -a "$POS_LOG"
for S in 2 3; do
  case $S in 2) RANKS='{0,1}';; 3) RANKS='{0,1,2}';; esac
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
    if tlc CopyPlan.tla "$CFG" "$POS_LOG"; then
      echo "  ok CopyPlan S=$S N=$N" | tee -a "$POS_LOG"; PASS=$((PASS+1))
    else
      echo "  FAIL CopyPlan S=$S N=$N" | tee -a "$POS_LOG"; FAIL=$((FAIL+1))
    fi
  done
done

echo "########## original AtomicPublish ##########" | tee -a "$POS_LOG"
ACFG="$OUT/cfgs/ap.cfg"
cat > "$ACFG" <<'CFG'
SPECIFICATION Spec
INVARIANTS
VersionComplete
CrashBeforeRenameKeepsOld
NewOnlyAfterRename
CFG
if tlc AtomicPublish.tla "$ACFG" "$POS_LOG"; then
  echo "  ok AtomicPublish" | tee -a "$POS_LOG"; PASS=$((PASS+1))
else
  echo "  FAIL AtomicPublish" | tee -a "$POS_LOG"; FAIL=$((FAIL+1))
fi

# ============================================================================
# 2a. POSITIVE SlotLifecycle matrix: must satisfy ALL safety invariants.
#     Coverage: 1-2 nodes, 1-2 consumers, C1/C2, 1-2 slots, 1-2 intervals,
#     success + failure + next-full-cohort, delayed ack.
# ============================================================================
echo "########## SlotLifecycle POSITIVE ##########" | tee -a "$POS_LOG"
POS_INV="CapacityBound
NoEarlyRelease
NodeIsolation
GenerationIsolation
FailureBroadcast"

pos_cfg() { # $1=name $2 nodes $3 consumers $4 cap $5 keys $6 ivs $7 maxgen $8 ackranks $9 allowfail
  cat > "$OUT/cfgs/$1.cfg" <<CFG
SPECIFICATION Spec
CONSTANTS
Nodes = {$2}
Consumers = $3
Cap = $4
Keys = {$5}
Ivs = {$6}
MaxGen = $7
MUTANT = "none"
AckRanks = {$8}
AllowFail = $9
INVARIANTS
$POS_INV
CFG
}

# name            nodes  cons cap keys  ivs    maxgen ackranks allowfail
pos_cfg pos_2n1c  "0,1" 1    1   "0"   "0"    2       "0"       TRUE
pos_cfg pos_1n2c  "0"   2    2   "0"   "0"    1       "0,1"     TRUE
pos_cfg pos_1n1c2 "0"   1    2   "0,1" "0,1"  1       "0"       FALSE
pos_cfg pos_1n2c1 "0"   2    2   "0"   "0"    1       "0,1"     FALSE

for c in pos_2n1c pos_1n2c pos_1n1c2 pos_1n2c1; do
  if tlc SlotLifecycle.tla "$OUT/cfgs/$c.cfg" "$POS_LOG"; then
    echo "  ok $c (invariants hold)" | tee -a "$POS_LOG"; PASS=$((PASS+1))
  else
    echo "  FAIL $c (expected invariants to HOLD)" | tee -a "$POS_LOG"; FAIL=$((FAIL+1))
  fi
done

# ============================================================================
# 2b. NEGATIVE mutants: TLC MUST exit nonzero AND the violation must name the
#     invariant the mutant targets. Any other outcome is a failure.
# ============================================================================
echo "########## SlotLifecycle NEGATIVE mutants ##########" | tee -a "$NEG_LOG"

neg_cfg() { # $1=name $2=mutant $3 nodes $4 cons $5 cap $6 keys $7 ivs $8 maxgen
  cat > "$OUT/cfgs/$1.cfg" <<CFG
SPECIFICATION Spec
CONSTANTS
Nodes = {$3}
Consumers = $4
Cap = $5
Keys = {$6}
Ivs = {$7}
MaxGen = $8
MUTANT = "$2"
AckRanks = {0}
AllowFail = TRUE
INVARIANTS
$POS_INV
CFG
}

# name          mutant         nodes cons cap keys  ivs maxgen target
neg_cfg neg_nocap   no_cap         "0,1" 1    1   "0,1" "0" 1      # CapacityBound
neg_cfg neg_early   early_release  "0"   2    2   "0"   "0" 1      # NoEarlyRelease
neg_cfg neg_alias   node_alias     "0,1" 2    2   "0"   "0" 1      # NodeIsolation
neg_cfg neg_stale   stale_gen      "0,1" 1    1   "0"   "0" 2      # GenerationIsolation
neg_cfg neg_bcast   no_broadcast   "0"   1    2   "0"   "0" 1      # FailureBroadcast

check_negative() { # $1=name $2=expected invariant
  local name="$1" want="$2" log="$NEG_LOG" rc=0
  tlc SlotLifecycle.tla "$OUT/cfgs/$name.cfg" "$log" || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  BAD  $name : mutant NOT caught (TLC exit 0, wanted $want)" | tee -a "$log"
    FAIL=$((FAIL+1)); return
  fi
  if grep -qE "Invariant $want is violated" "$log.run"; then
    echo "  ok   $name : caught by $want (exit $rc)" | tee -a "$log"; PASS=$((PASS+1))
  else
    echo "  BAD  $name : nonzero exit but $want NOT named; log:" | tee -a "$log"
    grep -E "Invariant|Error" "$log.run" | head -5 | tee -a "$log"
    FAIL=$((FAIL+1))
  fi
}

check_negative neg_nocap CapacityBound
check_negative neg_early NoEarlyRelease
check_negative neg_alias NodeIsolation
check_negative neg_stale GenerationIsolation
check_negative neg_bcast FailureBroadcast

# ============================================================================
# 3. Bounded liveness. FairSpec + LiveGoal.
#    live_ok        : complete cohort, all consumers ack -> expect PASS (exit 0).
#    live_boundary  : rank1 required but silent (partial peer down) -> expect
#                     TLC counterexample on LiveGoal (exit nonzero). This is the
#                     documented boundary; we do NOT claim liveness here.
# ============================================================================
echo "########## SlotLifecycle LIVENESS ##########" | tee -a "$LIVE_LOG"

live_cfg() { # $1=name $2=ackranks
  cat > "$OUT/cfgs/$1.cfg" <<CFG
SPECIFICATION FairSpec
CONSTANTS
Nodes = {0}
Consumers = 2
Cap = 2
Keys = {0}
Ivs = {0,1}
MaxGen = 1
MUTANT = "none"
AckRanks = {$2}
AllowFail = FALSE
PROPERTIES
LiveGoal
CFG
}

live_cfg live_ok "0,1"
live_cfg live_boundary "0"

rc_ok=0; tlc SlotLifecycle.tla "$OUT/cfgs/live_ok.cfg" "$LIVE_LOG" || rc_ok=$?
if [ "$rc_ok" -eq 0 ]; then
  echo "  ok   live_ok : leads-to holds under complete-cohort fairness" | tee -a "$LIVE_LOG"; PASS=$((PASS+1))
else
  echo "  BAD  live_ok : expected liveness to hold, TLC exit $rc_ok" | tee -a "$LIVE_LOG"; FAIL=$((FAIL+1))
fi

rc_b=0; tlc SlotLifecycle.tla "$OUT/cfgs/live_boundary.cfg" "$LIVE_LOG" || rc_b=$?
if [ "$rc_b" -ne 0 ] && grep -qE "violated|Not yet validated|error" "$LIVE_LOG.run"; then
  echo "  ok   live_boundary : partial peer-down boundary reproduces (exit $rc_b)" | tee -a "$LIVE_LOG"; PASS=$((PASS+1))
else
  echo "  BAD  live_boundary : expected a liveness counterexample, exit $rc_b" | tee -a "$LIVE_LOG"; FAIL=$((FAIL+1))
fi

# ============================================================================
# ============================================================================
# 4. CachePropagation: node-level plan executor (internal/propagation.py +
#    internal/api.py). Two-layer fan-out instance, byte budgets, parent/child
#    ordering, owed-rank and owed-edge release, fill failure, and the bounded
#    absence-of-consumer liveness boundary.
# ============================================================================
echo "########## CachePropagation ##########" | tee -a "$POS_LOG"
CP_LOG="$OUT/cache_propagation.log"; : > "$CP_LOG"

# $1 cfg name, $2 mutant model value, $3 cfg tail (SPECIFICATION/INVARIANTS).
cp_cfg() {
  cat > "$OUT/cfgs/$1.cfg" <<CFG
CONSTANTS
MUTANT = "$2"
$3
CFG
}

cp_cfg cp_pos none "SPECIFICATION Spec
INVARIANTS
CACHE_INV"
if tlc CachePropagation.tla "$OUT/cfgs/cp_pos.cfg" "$CP_LOG"; then
  echo "  ok cp_pos (all plan invariants hold)" | tee -a "$CP_LOG"; PASS=$((PASS+1))
else
  echo "  FAIL cp_pos" | tee -a "$CP_LOG"; FAIL=$((FAIL+1))
fi

cp_cfg cp_live_ok none "SPECIFICATION FairSpec
PROPERTIES
LiveGoal"
rc=0; tlc CachePropagation.tla "$OUT/cfgs/cp_live_ok.cfg" "$CP_LOG" || rc=$?
if [ "$rc" -eq 0 ]; then
  echo "  ok cp_live_ok (segments drain under complete-cohort fairness)" | tee -a "$CP_LOG"
  PASS=$((PASS+1))
else
  echo "  BAD cp_live_ok (exit $rc)" | tee -a "$CP_LOG"; FAIL=$((FAIL+1))
fi

# Negative mutants: each must be caught by exactly one named invariant.
cp_neg() { # $1=name $2=mutant $3=target invariant
  cp_cfg "$1" "$2" "SPECIFICATION Spec
INVARIANTS
$3"
  local rc=0
  tlc CachePropagation.tla "$OUT/cfgs/$1.cfg" "$CP_LOG" || rc=$?
  if [ "$rc" -ne 0 ] && grep -qE "Invariant $3 is violated" "$CP_LOG.run"; then
    echo "  ok   $1 : caught by $3 (exit $rc)" | tee -a "$CP_LOG"; PASS=$((PASS+1))
  else
    echo "  BAD  $1 : $3 did not name the violation (exit $rc)" | tee -a "$CP_LOG"
    FAIL=$((FAIL+1))
  fi
}

cp_neg cp_no_budget no_budget ByteBudget
cp_neg cp_early_rank early_rank NoEarlyRankRelease
cp_neg cp_early_parent early_parent NoEarlyParentRelease
cp_neg cp_no_order no_parent_order ParentReadyBeforeChild
cp_neg cp_silent_error silent_error ErrorRecorded

# Peer down: node 2 never acks its terminal segment. Under FairSpec LiveGoal
# must return a temporal counterexample (exit nonzero). The runtime reports
# the same boundary as cache_wait.
cp_cfg cp_peer_down peer_down "SPECIFICATION FairSpec
PROPERTIES
LiveGoal"
rc=0; tlc CachePropagation.tla "$OUT/cfgs/cp_peer_down.cfg" "$CP_LOG" || rc=$?
if [ "$rc" -ne 0 ] && grep -qE "LiveGoal (is|was) violated" "$CP_LOG.run"; then
  echo "  ok cp_peer_down : missing-consumer boundary reproduces (exit $rc)" | tee -a "$CP_LOG"
  PASS=$((PASS+1))
else
  echo "  BAD cp_peer_down : expected a LiveGoal counterexample (exit $rc)" | tee -a "$CP_LOG"
  FAIL=$((FAIL+1))
fi

echo "########## SUMMARY: PASS=$PASS FAIL=$FAIL ##########" | tee -a "$POS_LOG"
echo "full logs: $POS_LOG"; echo "           $NEG_LOG"; echo "           $LIVE_LOG"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
