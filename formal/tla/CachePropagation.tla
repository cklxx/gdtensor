---- MODULE CachePropagation ----
(* ===========================================================================
   Model of the node-level cache propagation plan (internal/propagation.py)
   and its executor (internal/api.py).

   Bounded projection, not a proof of the Python. The fixed instance is the
   two-layer fan-out defect fixed by this change:

     layer 0  : node 0, one 8-byte segment "a"
     terminal : node 1 with one 4-byte segment "b", node 2 with "c"
     edges    : both terminal segments read node 0's "a"
     need     : each terminal segment is drained by one rank

   Segment state: "gone" | "admitted" | "ready" | "error".
   res[n] = bytes of admitted and ready segments on node n.
   owe[n]  = remaining rank acks on the terminal segment of node n.
   rem[n]  = unacked parent edges on the segment of node n (subset of Edges).

   MUTANT is the only constant, assigned by the cfg:
     none / no_budget / early_rank / early_parent / no_parent_order /
     silent_error / peer_down.
   =========================================================================== *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANT MUTANT

Nodes == {0, 1, 2}
L0 == {0}
Terminal == {1, 2}
A == <<0, "a">>
Edges == [n \in Nodes |-> IF n = 0 THEN {} ELSE {A}]
SegSize == [n \in Nodes |-> IF n = 0 THEN 8 ELSE 4]
(* The no_budget injection shrinks every budget below the segment size: a
   correct executor must block; the mutant admits and exceeds the budget. *)
Budget ==
  [n \in Nodes |-> IF MUTANT = "no_budget" THEN 4 ELSE 8]

VARIABLES st, res, owe, rem, nerr, seen0, seen, p0ready

Init ==
  /\ st  = [n \in Nodes |-> "gone"]
  /\ res = [n \in Nodes |-> 0]
  /\ owe = [n \in Nodes |-> IF n = 0 THEN 0 ELSE 1]
  /\ rem = Edges
  /\ nerr = [n \in Nodes |-> "none"]
  /\ seen0 = FALSE
  /\ seen = [n \in Nodes |-> FALSE]
  /\ p0ready = FALSE

IsUp(n) == st[n] \in {"admitted", "ready"}

(* ---- admission and fill ------------------------------------------------- *)

ReadyParents(n) ==
  \/ MUTANT = "no_parent_order"
  \/ \A e \in Edges[n] : st[e[1]] = "ready"

Admit(n) ==
  /\ st[n] = "gone"
  /\ nerr[n] = "none"
  /\ ReadyParents(n)
  /\ (MUTANT = "no_budget" \/ res[n] + SegSize[n] <= Budget[n])
  /\ st' = [st EXCEPT ![n] = "admitted"]
  /\ res' = [res EXCEPT ![n] = @ + SegSize[n]]
  /\ seen0' = (seen0 \/ n = 0)
  /\ seen' = [seen EXCEPT ![n] = TRUE]
  /\ UNCHANGED <<p0ready, owe, rem, nerr>>

(* A failed fill is reachable only in the silent-error injection; every other
   configuration has success-only fills, which is the executor's healthy
   contract (a real failure is broadcast and aborts the pass). *)
Fill(n, ok) ==
  /\ st[n] = "admitted"
  /\ (ok = FALSE => MUTANT = "silent_error")
  /\ st' = [st EXCEPT ![n] = IF ok THEN "ready" ELSE "error"]
  /\ p0ready' = (p0ready \/ (ok /\ n = 0))
  /\ res' = IF ok THEN res
            ELSE [res EXCEPT ![n] = @ - SegSize[n]]
  /\ UNCHANGED <<seen0, seen, owe, rem, nerr>>

(* ---- terminal rank drain and release ------------------------------------ *)

RankAck(n) ==
  /\ n \in Terminal /\ st[n] = "ready" /\ owe[n] > 0
  /\ (MUTANT # "peer_down" \/ n # 2)
  /\ owe' = [owe EXCEPT ![n] = @ - 1]
  /\ UNCHANGED <<seen0, seen, p0ready, st, res, rem, nerr>>

ReleaseTerminal(n) ==
  /\ n \in Terminal /\ st[n] = "ready"
  /\ rem[n] = {}
  /\ (MUTANT = "early_rank" \/ owe[n] = 0)
  /\ st' = [st EXCEPT ![n] = "gone"]
  /\ res' = [res EXCEPT ![n] = @ - SegSize[n]]
  /\ UNCHANGED <<seen0, seen, p0ready, owe, rem, nerr>>

(* ---- child ack of a parent edge and parent release ---------------------- *)

AckEdge(n) ==
  /\ n \in Terminal
  /\ st[n] = "ready"
  /\ A \in rem[n] /\ st[0] = "ready"
  /\ rem' = [rem EXCEPT ![n] = @ \ {A}]
  /\ UNCHANGED <<seen0, seen, p0ready, st, res, owe, nerr>>

ChildrenStillOwing ==
  {n \in Terminal : A \in rem[n]}

ReleaseParent ==
  /\ st[0] = "ready"
  /\ (MUTANT = "early_parent" \/ ChildrenStillOwing = {})
  /\ st' = [st EXCEPT ![0] = "gone"]
  /\ res' = [res EXCEPT ![0] = @ - SegSize[0]]
  /\ UNCHANGED <<seen0, seen, p0ready, owe, rem, nerr>>

Next ==
  \/ \E n \in Nodes : Admit(n)
  \/ \E n \in Nodes, ok \in BOOLEAN : Fill(n, ok)
  \/ \E n \in Terminal : RankAck(n)
  \/ \E n \in Terminal : ReleaseTerminal(n)
  \/ \E n \in Terminal : AckEdge(n)
  \/ ReleaseParent

vars == <<st, res, owe, rem, nerr, seen0, seen, p0ready>>
Spec == Init /\ [][Next]_vars

(* ===========================================================================
   Safety invariants
   =========================================================================== *)

(* Resident bytes never exceed the declared byte budget. *)
ByteBudget == \A n \in Nodes : res[n] <= Budget[n]

(* res[n] is exactly the segment size while the segment occupies bytes. *)
ResidentExact ==
  \A n \in Nodes :
    res[n] = IF IsUp(n) THEN SegSize[n] ELSE 0

(* A terminal segment becomes gone only after every rank acked. *)
NoEarlyRankRelease ==
  \A n \in Terminal :
    seen[n] /\ st[n] = "gone" => owe[n] = 0

(* The layer-0 parent becomes gone only after every child acked its edge.
   seen0 excludes the pre-pass state, where the parent has never been
   admitted and plan edges are still unacked. *)
NoEarlyParentRelease ==
  seen0 /\ st[0] = "gone" => ChildrenStillOwing = {}

(* A child occupies bytes only after the parent reached ready at least once.
   The parent may be released while the child stays resident: the child has
   its own copied bytes. no_parent_order admits the child first and violates
   this. *)
ParentReadyBeforeChild ==
  \A n \in Terminal : IsUp(n) => p0ready

(* A remaining child edge always references the plan edge. *)
EdgesArePlanEdges ==
  \A n \in Terminal : rem[n] \subseteq Edges[n]

(* A fill failure must be recorded on the node that failed. *)
ErrorRecorded ==
  \A n \in Nodes : st[n] = "error" => nerr[n] = "e"

(* A recorded fill failure blocks new admission in that pass. *)
ErrorStopsAdmit ==
  \A n \in Nodes : nerr[n] = "e" => st[n] # "admitted"

CACHE_INV ==
  /\ ByteBudget /\ ResidentExact /\ NoEarlyRankRelease
  /\ NoEarlyParentRelease /\ ParentReadyBeforeChild
  /\ EdgesArePlanEdges /\ ErrorRecorded /\ ErrorStopsAdmit

(* ===========================================================================
   Bounded liveness under explicit assumptions: fills succeed, every rank ack
   happens, every child acks its parent edge. MUTANT="peer_down" disables the
   rank ack on node 2 and is expected to refute LiveGoal: that is the
   documented absence-of-consumer boundary (the executor reports it at
   runtime as cache_wait after GDTENSOR_CACHE_WAIT_SECONDS).
   =========================================================================== *)

FairSpec ==
  Spec
  /\ (\A n1 \in Nodes : WF_vars(Admit(n1)))
  /\ (\A n2 \in Nodes : WF_vars(Fill(n2, TRUE)))
  /\ (\A n3 \in Terminal : WF_vars(RankAck(n3)))
  /\ (\A n4 \in Terminal : WF_vars(ReleaseTerminal(n4)))
  /\ (\A n5 \in Terminal : WF_vars(AckEdge(n5)))
  /\ WF_vars(ReleaseParent)

LiveGoal ==
  \A n \in Nodes : [](st[n] # "gone" => <>(st[n] = "gone"))
====
