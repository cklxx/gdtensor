---- MODULE CopyPlan ----
(* Formal model of the gdtensor CopyPlan protocol (design sections 5, B.1).

   A global tensor has N elements (linearized global indices 0..N-1). S ranks
   each build a local save/load plan from their local view. The model checks the
   B.1.3 invariants under all rank interleavings and both storage layouts:
     - save: designated writers' dst ranges are pairwise disjoint and, once
       every rank finished, cover exactly the global tensor;
     - load: every pair keeps src = dst (same global element); sharded storage
       also requires disjoint, covering reads;
     - replicated storage: exactly one leader writes.
   Remainder handling (N not divisible by S) is included: first N%S ranks get
   one extra element. *)
EXTENDS Naturals, FiniteSets, TLC

CONSTANTS N, Ranks

S == Cardinality(Ranks)
Elems == 0 .. (N - 1)

ASSUME S > 0
ASSUME N >= 1

(* ---- shard ownership (5.3) ---- *)

SizeOf(r) ==
  (N \div S) + IF r \in 0 .. ((N % S) - 1) THEN 1 ELSE 0

RECURSIVE StartOf(_)
StartOf(r) ==
  IF r = 0 THEN 0 ELSE StartOf(r - 1) + SizeOf(r - 1)

ShardOwned(r) ==
  { StartOf(r) + k : k \in 0 .. (SizeOf(r) - 1) }

(* ---- state ---- *)

VARIABLES layout, savePlan, saveDone, loadPlan, loadDone

(* layout: "sharded" = one slice per rank; "replicated" = one full copy. *)

WriterElems(r) ==
  IF layout = "sharded"
  THEN ShardOwned(r)
  ELSE IF r = 0 THEN Elems ELSE {}

ReaderElems(r) ==
  IF layout = "sharded" THEN ShardOwned(r) ELSE Elems

SavePairs(r) ==
  { [src |-> e, dst |-> e] : e \in WriterElems(r) }

LoadPairs(r) ==
  { [src |-> e, dst |-> e] : e \in ReaderElems(r) }

Init ==
  /\ layout \in {"sharded", "replicated"}
  /\ savePlan = [r \in Ranks |-> {}]
  /\ saveDone = [r \in Ranks |-> FALSE]
  /\ loadPlan = [r \in Ranks |-> {}]
  /\ loadDone = [r \in Ranks |-> FALSE]

AllSaveDone == \A r \in Ranks : saveDone[r]
AllLoadDone == \A r \in Ranks : loadDone[r]

RankSave(r) ==
  /\ ~saveDone[r]
  /\ savePlan' = [savePlan EXCEPT ![r] = savePlan[r] \cup SavePairs(r)]
  /\ saveDone' = [saveDone EXCEPT ![r] = TRUE]
  /\ UNCHANGED <<layout, loadPlan, loadDone>>

RankLoad(r) ==
  /\ AllSaveDone
  /\ ~loadDone[r]
  /\ loadPlan' = [loadPlan EXCEPT ![r] = loadPlan[r] \cup LoadPairs(r)]
  /\ loadDone' = [loadDone EXCEPT ![r] = TRUE]
  /\ UNCHANGED <<layout, savePlan, saveDone>>

Next ==
  \/ \E r \in Ranks : RankSave(r)
  \/ \E r \in Ranks : RankLoad(r)

Spec == Init /\ [][Next]_<<layout, savePlan, saveDone, loadPlan, loadDone>>

(* ---- invariants ---- *)

WrittenDst(r) == {p.dst : p \in savePlan[r]}
ReadDst(r) == {p.dst : p \in loadPlan[r]}

AllWritten == UNION {WrittenDst(r) : r \in Ranks}
AllRead == UNION {ReadDst(r) : r \in Ranks}

(* B.1.3: save dst ranges are pairwise disjoint across ranks. *)
SaveDstDisjoint ==
  \A r \in Ranks :
    \A e \in WrittenDst(r) :
      ~ \E r2 \in (Ranks \ {r}) : e \in WrittenDst(r2)

(* Once all ranks finish, every global element is written exactly once. *)
SaveCovering ==
  AllSaveDone => AllWritten = Elems

(* Every pair references the same global element on src and dst. *)
SavePairCorrect ==
  \A r \in Ranks : \A p \in savePlan[r] : p.src = p.dst

LoadPairCorrect ==
  \A r \in Ranks : \A p \in loadPlan[r] : p.src = p.dst

(* Sharded storage: reads are disjoint across ranks and cover the tensor. *)
LoadShardedDisjoint ==
  layout = "sharded" =>
    \A r \in Ranks :
      \A e \in ReadDst(r) :
        ~ \E r2 \in (Ranks \ {r}) : e \in ReadDst(r2)

LoadShardedCovering ==
  (layout = "sharded" /\ AllLoadDone) => AllRead = Elems

(* Replicated storage: exactly one leader (rank 0) writes. *)
ReplicatedSingleWriter ==
  layout = "replicated" =>
    /\ \A r \in (Ranks \ {0}) : savePlan[r] = {}
    /\ saveDone[0] => WrittenDst(0) = Elems

(* Elements never leave the tensor bounds. *)
BoundsOK ==
  /\ \A r \in Ranks : \A p \in savePlan[r] : p.dst \in Elems
  /\ \A r \in Ranks : \A p \in loadPlan[r] : p.dst \in Elems

====
