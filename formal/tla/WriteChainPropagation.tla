---- MODULE WriteChainPropagation ----
(* ===========================================================================
   Model of the node-level cached write propagation plan
   (internal/write_propagation.py) and executor (internal/write_chain.py).

   Bounded projection, not a proof of the Python runtime. Dual two-layer
   fan-in instance to CachePropagation.tla:

     layer 0 (first_cache, spatial: Shard)  : nodes 1 and 2, each with one 4-byte
                              segment staged by local compute rank ("b" on 1, "c" on 2)
     layer 1 (second_cache, spatial: Shard) : convergence node 0 with one 8-byte
                              segment "a", consuming edge pieces <<1, 0, 4>> and
                              <<2, 4, 4>> and flushing to storage once all producers stage.

   Segment state: "gone" | "admitted" | "ready" | "flushing" | "done" | "error".
   res[n]     = resident bytes on node n (`admitted`, `ready`, or `flushing`).
   owe[n]     = exact remaining expected (producer, offset, size) pieces.
   ready0[u]  = latch set once L0 node u finished all rank stages.
   flusher[n] = rank that flushed segment n (3 = unset).

   Constants:
     MUTANT     \in {"none", "no_budget", "early_stage_flush",
                     "early_parent_flush", "no_upstream_order",
                     "silent_error", "remote_flush", "peer_down"}
     ALLOW_FAIL \in BOOLEAN (TRUE exercises failure propagation in Spec; FALSE
                    restricts FairSpec to non-crashing runs).
   =========================================================================== *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS MUTANT, ALLOW_FAIL

Nodes == {0, 1, 2}
L0 == {1, 2}
L1 == {0}
SegSize == [n \in Nodes |-> IF n = 0 THEN 8 ELSE 4]
RankPiece(u) == <<"rank", u, 0, SegSize[u]>>
RankStagePieces(u) == {RankPiece(u), <<"rank", u + 10, 99, 99>>}
EdgePiece(u) == IF u = 1 THEN <<1, 0, 4>> ELSE <<2, 4, 4>>
EdgePieces == {EdgePiece(1), EdgePiece(2)}
EdgeStagePieces == EdgePieces \cup {<<99, 99, 99>>}
Budget == [n \in Nodes |-> IF MUTANT = "no_budget" THEN 4 ELSE 8]

VARIABLES st, res, owe, ready0, nerr, flusher

Init ==
  /\ st = [n \in Nodes |-> "gone"]
  /\ res = [n \in Nodes |-> 0]
  /\ owe = [n \in Nodes |-> IF n \in L0 THEN {RankPiece(n)} ELSE EdgePieces]
  /\ ready0 = [u \in L0 |-> FALSE]
  /\ nerr = [n \in Nodes |-> "none"]
  /\ flusher = [n \in Nodes |-> 3]

IsUp(n) == st[n] \in {"admitted", "ready", "flushing"}

(* ---- L0 admission jointly reserves L1 (no L0/L1 hold-and-wait cycle) ---- *)

AdmitL0(u) ==
  /\ u \in L0
  /\ st[u] = "gone"
  /\ nerr[u] = "none"
  /\ nerr[0] = "none"
  /\ (MUTANT = "no_budget" \/ res[u] + SegSize[u] <= Budget[u])
  /\ (st[0] = "admitted" \/ (st[0] = "gone" /\ (MUTANT = "no_budget" \/ res[0] + SegSize[0] <= Budget[0])))
  /\ st' = [st EXCEPT ![u] = "admitted",
                      ![0] = IF st[0] = "gone" THEN "admitted" ELSE st[0]]
  /\ res' = [res EXCEPT ![u] = @ + SegSize[u],
                        ![0] = IF st[0] = "gone" THEN @ + SegSize[0] ELSE @]
  /\ UNCHANGED <<owe, ready0, nerr, flusher>>

RankStage(u, piece) ==
  /\ u \in L0
  /\ st[u] = "admitted"
  /\ piece \in RankStagePieces(u)
  /\ (piece # RankPiece(u) \/ MUTANT # "peer_down" \/ u # 2)
  /\ owe' = [owe EXCEPT ![u] = @ \ {piece}]
  /\ UNCHANGED <<st, res, ready0, nerr, flusher>>

SealL0(u) ==
  /\ u \in L0
  /\ st[u] = "admitted"
  /\ (MUTANT = "early_stage_flush" \/ owe[u] = {})
  /\ st' = [st EXCEPT ![u] = "ready"]
  /\ ready0' = [ready0 EXCEPT ![u] = TRUE]
  /\ UNCHANGED <<res, owe, nerr, flusher>>

(* ---- L0 -> L1 interlayer admission & flush ------------------------------ *)

AdmitL1Early ==
  /\ MUTANT \in {"no_budget", "early_parent_flush", "no_upstream_order"}
  /\ st[0] = "gone"
  /\ nerr[0] = "none"
  /\ (MUTANT = "no_budget" \/ res[0] + SegSize[0] <= Budget[0])
  /\ st' = [st EXCEPT ![0] = "admitted"]
  /\ res' = [res EXCEPT ![0] = @ + SegSize[0]]
  /\ UNCHANGED <<owe, ready0, nerr, flusher>>

ClaimInterlayer(u, rank) ==
  /\ u \in L0
  /\ st[u] = "ready"
  /\ nerr[0] = "none"
  /\ (rank = u \/ MUTANT = "remote_flush")
  /\ st[0] \in {"gone", "admitted"}
  /\ (st[0] = "admitted" \/ MUTANT = "no_budget" \/ res[0] + SegSize[0] <= Budget[0])
  /\ st' = [st EXCEPT ![u] = "flushing", ![0] = IF st[0] = "gone" THEN "admitted" ELSE st[0]]
  /\ res' = [res EXCEPT ![0] = IF st[0] = "gone" THEN @ + SegSize[0] ELSE @]
  /\ flusher' = [flusher EXCEPT ![u] = rank]
  /\ UNCHANGED <<owe, ready0, nerr>>

BogusEdgeStage(piece) ==
  /\ st[0] = "admitted"
  /\ piece \in (EdgeStagePieces \ EdgePieces)
  /\ owe' = [owe EXCEPT ![0] = @ \ {piece}]
  /\ UNCHANGED <<st, res, ready0, nerr, flusher>>

FinishInterlayer(u) ==
  /\ u \in L0
  /\ st[u] = "flushing"
  /\ st[0] \in {"admitted", "ready"}
  /\ LET rem0 == owe[0] \ {EdgePiece(u)} IN
     /\ owe' = [owe EXCEPT ![0] = rem0]
     /\ st' = [st EXCEPT ![u] = "done", ![0] = IF rem0 = {} THEN "ready" ELSE st[0]]
     /\ res' = [res EXCEPT ![u] = @ - SegSize[u]]
  /\ UNCHANGED <<ready0, nerr, flusher>>

FailL0(u) ==
  /\ ALLOW_FAIL
  /\ u \in L0
  /\ st[u] \in {"admitted", "ready", "flushing"}
  /\ st' = [st EXCEPT ![u] = "error",
                      ![0] = IF IsUp(0) THEN "error" ELSE st[0]]
  /\ res' = [res EXCEPT ![u] = 0, ![0] = 0]
  /\ IF MUTANT = "silent_error"
     THEN nerr' = [nerr EXCEPT ![u] = "e"]
     ELSE nerr' = [nerr EXCEPT ![u] = "e", ![0] = "e"]
  /\ UNCHANGED <<owe, ready0, flusher>>

(* ---- L1 terminal flush to storage --------------------------------------- *)

FlushL1(rank) ==
  /\ rank \in Nodes
  /\ (rank = 0 \/ MUTANT = "remote_flush")
  /\ nerr[0] = "none"
  /\ (
       \/ st[0] = "ready"
       \/ (MUTANT = "early_parent_flush" /\ st[0] = "admitted" /\ \A u \in L0 : ready0[u])
       \/ (MUTANT = "no_upstream_order" /\ st[0] = "admitted" /\ ~\A u \in L0 : ready0[u])
     )
  /\ st' = [st EXCEPT ![0] = "done"]
  /\ res' = [res EXCEPT ![0] = 0]
  /\ flusher' = [flusher EXCEPT ![0] = rank]
  /\ UNCHANGED <<owe, ready0, nerr>>

Next ==
  \/ \E u \in L0 : AdmitL0(u)
  \/ \E u \in L0 : \E piece \in RankStagePieces(u) : RankStage(u, piece)
  \/ \E u \in L0 : SealL0(u)
  \/ AdmitL1Early
  \/ \E u \in L0, rank \in Nodes : ClaimInterlayer(u, rank)
  \/ \E piece \in (EdgeStagePieces \ EdgePieces) : BogusEdgeStage(piece)
  \/ \E u \in L0 : FinishInterlayer(u)
  \/ \E u \in L0 : FailL0(u)
  \/ \E rank \in Nodes : FlushL1(rank)

vars == <<st, res, owe, ready0, nerr, flusher>>
Spec == Init /\ [][Next]_vars

(* ===========================================================================
   Safety invariants
   =========================================================================== *)

ByteBudget == \A n \in Nodes : res[n] <= Budget[n]

ResidentExact ==
  \A n \in Nodes : res[n] = IF IsUp(n) THEN SegSize[n] ELSE 0

NoEarlyStageFlush ==
  \A u \in L0 : st[u] \in {"ready", "flushing", "done"} => owe[u] = {}

NoEarlyParentFlush ==
  st[0] \in {"flushing", "done"} => owe[0] = {}

UpstreamReadyBeforeFlush ==
  st[0] \in {"flushing", "done"} => \A u \in L0 : ready0[u]

EdgesArePlanEdges == owe[0] \subseteq EdgePieces

ErrorPropagated ==
  (\E u \in L0 : st[u] = "error") => nerr[0] = "e"

ErrorStopsAdmit ==
  \A n \in Nodes : nerr[n] = "e" => ~IsUp(n)

FlushOnDN ==
  \A n \in Nodes : flusher[n] = 3 \/ flusher[n] = n

WRITE_CACHE_INV ==
  /\ ByteBudget
  /\ ResidentExact
  /\ NoEarlyStageFlush
  /\ NoEarlyParentFlush
  /\ UpstreamReadyBeforeFlush
  /\ EdgesArePlanEdges
  /\ ErrorPropagated
  /\ ErrorStopsAdmit
  /\ FlushOnDN

(* ===========================================================================
   Bounded liveness under complete cohort + ALLOW_FAIL = FALSE
   =========================================================================== *)

FairSpec ==
  Spec
  /\ (\A u \in L0 : WF_vars(AdmitL0(u)))
  /\ (\A u \in L0 : WF_vars(RankStage(u, RankPiece(u))))
  /\ (\A u \in L0 : WF_vars(SealL0(u)))
  /\ (\A u \in L0 : WF_vars(ClaimInterlayer(u, u)))
  /\ (\A u \in L0 : WF_vars(FinishInterlayer(u)))
  /\ WF_vars(FlushL1(0))

LiveGoal ==
  \A n \in Nodes : <>(st[n] = "done")
====
