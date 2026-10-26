---- MODULE SlotLifecycle ----
(* ===========================================================================
   Model of the gdtensor cached-copy slot lifecycle in gdtensor/internal/api.py.

   This model is a faithful, *deliberately small* projection of _NodeState,
   _SlotTable and executeChain/_drain_cell/_recurse. It maps ONE implementation
   field to ONE model field and ONE implementation transition to ONE model step.
   It is NOT a proof that the Python is correct: it checks the named safety
   invariants over the bounded interleavings enumerated here, and it is paired
   with an independent public-API repro (not this model) for any bug found.

   Field map (api.py : SlotLifecycle):
     _NodeState.slots[key]            slots[n][k]            key=(tid,off,size)
     _NodeState.live                  live[n]                resident slot count
     _NodeState.cap                   Cap                    concurrency width C
     _NodeState.consumers              Consumers              ranks per node
     _NodeState.error / error_gen     nerr[n] / nerrgen[n]   node-wide primary error
     _NodeState.gen                   gen[n]                 cohort generation
     _NodeState.active                active[n]              ranks inside cohort
     _NodeState.cohort_joined         joined[n]              ranks joined this cohort
     slot.filled                      slots[..].filled
     slot.error / error_gen           slots[..].err / errgen
     slot.required  rank->intervals   slots[..].required     set of (rank,iv)
     slot.cov       rank->acked ivs   slots[..].cov          rank -> set of iv

   Transition map:
     generation_start/_cohort_join    JoinCohort
     generation_end                   EndCohort
     begin_fill/_begin_*              BeginFill
     finish_fill/_apply_finish        FinishFill(ok|fail)
     set_expected/_set_expected_locked SetExpected
     ack_drain/_covers                AckDrain

   MUTANT selects a deliberately broken variant used ONLY by the negative
   self-test cfgs; normal cfgs use MUTANT="none". Each mutant must be caught by
   exactly one named invariant (see runner expectations).
   =========================================================================== *)
EXTENDS Naturals, Integers, FiniteSets, TLC

CONSTANTS Nodes, Consumers, Cap, Keys, Ivs, MaxGen, MUTANT, AckRanks, AllowFail

ASSUME Consumers \in {1, 2}
ASSUME Cap >= 1

NodeRanks == 0 .. (Consumers - 1)

(* Required drain set for slot k on every node.
   k=0: rank0 must ack BOTH intervals (range overlap / multi-interval drain);
        when Consumers=2, rank1 must also ack interval 0 (replicated consumer).
   k=1: only rank0 must ack interval 0 (single narrow slot). *)
Required(k) ==
  IF k = 0
  THEN ( {[rank |-> 0, iv |-> 0], [rank |-> 0, iv |-> 1]}
         \union (IF Consumers = 2 THEN {[rank |-> 1, iv |-> 0]} ELSE {}) )
  ELSE {[rank |-> 0, iv |-> 0]}

VARIABLES slots, live, nerr, nerrgen, gen, active, joined, failed_gen

SlotInit ==
  [gone |-> TRUE, filled |-> FALSE, err |-> "none", errgen |-> -1,
   required |-> {}, cov |-> [r \in NodeRanks |-> {}]]

Init ==
  /\ slots  = [n \in Nodes |-> [k \in Keys |-> SlotInit]]
  /\ live   = [n \in Nodes |-> 0]
  /\ nerr   = [n \in Nodes |-> "none"]
  /\ nerrgen= [n \in Nodes |-> -1]
  /\ gen    = [n \in Nodes |-> 0]
  /\ active = [n \in Nodes |-> 0]
  /\ joined = [n \in Nodes |-> {}]
  /\ failed_gen = [n \in Nodes |-> -1]

(* ---- helpers ------------------------------------------------------------ *)

Fresh ==
  [gone |-> FALSE, filled |-> FALSE, err |-> "none", errgen |-> -1,
   required |-> {}, cov |-> [r \in NodeRanks |-> {}]]

(* Create a new resident slot. live counter:
   normal  -> node n's counter; node_alias mutant -> node 0's counter also bumps. *)
NewSlot(n, k) ==
  /\ slots' = [slots EXCEPT ![n] = [@ EXCEPT ![k] = Fresh]]
  /\ live' = [n2 \in Nodes |->
                live[n2]
                + IF n2 = n THEN 1 ELSE 0
                + IF MUTANT = "node_alias" /\ n2 = 0 /\ n # 0 THEN 1 ELSE 0]
  /\ UNCHANGED <<nerr, nerrgen, gen, active, joined, failed_gen>>

(* ---- cohort join / end (generation_start/_cohort_join, generation_end) ---- *)

JoinCohort(n, r) ==
  (* at most `consumers` ranks inside the cohort (one in-flight chain_load per
     consumer); this bounds the state space and mirrors the real active count. *)
  /\ active[n] < Consumers
  /\ \/ (* cohort in progress, OR previous cohort not fully joined:
           late same-cohort rank still sees the existing node-wide error. *)
        /\ active[n] # 0 \/ Cardinality(joined[n]) < Consumers
        /\ active' = [active EXCEPT ![n] = @ + 1]
        /\ joined' = [joined EXCEPT ![n] = @ \cup {r}]
        /\ UNCHANGED <<slots, live, gen, nerrgen, failed_gen>>
        /\ nerr' = nerr
     \/ (* previous cohort fully observed: open a fresh generation, clear error.
           stale_gen mutant forgets to clear nerr. *)
        /\ active[n] = 0
        /\ Cardinality(joined[n]) = Consumers
        /\ gen[n] < MaxGen
        /\ gen' = [gen EXCEPT ![n] = @ + 1]
        /\ nerr' = IF MUTANT = "stale_gen" THEN nerr
                   ELSE [nerr EXCEPT ![n] = "none"]
        /\ nerrgen' = IF MUTANT = "stale_gen" THEN nerrgen
                      ELSE [nerrgen EXCEPT ![n] = -1]
        /\ failed_gen' = IF MUTANT = "stale_gen" THEN failed_gen
                        ELSE [failed_gen EXCEPT ![n] = -1]
        /\ joined' = [joined EXCEPT ![n] = {r}]
        /\ active' = [active EXCEPT ![n] = 1]
        /\ UNCHANGED <<slots, live>>

EndCohort(n) ==
  /\ active[n] > 0
  /\ active' = [active EXCEPT ![n] = @ - 1]
  /\ UNCHANGED <<slots, live, nerr, nerrgen, gen, joined, failed_gen>>

(* ---- begin_fill --------------------------------------------------------- *)

BeginFill(n, k) ==
  (* node-wide error for THIS generation blocks the rank (raises). *)
  /\ ~(nerr[n] = "e" /\ nerrgen[n] = gen[n])
  /\ LET s == slots[n][k] IN
     \/ (* resident, already filled, healthy: reuse, do not rewrite. *)
        /\ ~s.gone /\ s.filled /\ s.err = "none"
        /\ UNCHANGED <<slots, live, nerr, nerrgen, gen, active, joined, failed_gen>>
     \/ (* filled with an error from a DIFFERENT (older) generation: drop and refill. *)
        /\ ~s.gone /\ s.filled /\ s.err = "e" /\ s.errgen # gen[n]
        /\ NewSlot(n, k)
     \/ (* no resident slot: create one, honour the capacity bound C.
           no_cap mutant ignores the bound. *)
        /\ s.gone
        /\ (MUTANT = "no_cap" \/ live[n] < Cap)
        /\ NewSlot(n, k)

(* ---- finish_fill -------------------------------------------------------- *)

FinishFill(n, k, ok) ==
  /\ ~slots[n][k].gone /\ ~slots[n][k].filled
  /\ \/ (* success: resident healthy, waits for every expected consumer to drain. *)
        /\ ok = TRUE
        /\ slots' = [slots EXCEPT ![n][k].filled = TRUE]
        /\ UNCHANGED <<live, nerr, nerrgen, gen, active, joined, failed_gen>>
     \/ (* failure: node-wide primary error for this generation; failed slot no
           longer counts against C. no_broadcast mutant leaves the node error unset. *)
        /\ ok = FALSE /\ AllowFail
        /\ slots' = [slots EXCEPT ![n][k].filled = TRUE,
                               ![n][k].err = "e",
                               ![n][k].errgen = gen[n]]
        /\ live' = [live EXCEPT ![n] = @ - 1]
        /\ nerr' = [n2 \in Nodes |->
                     IF n2 = n /\ MUTANT # "no_broadcast" THEN "e" ELSE nerr[n2]]
        /\ nerrgen' = [nerrgen EXCEPT ![n] = gen[n]]
        /\ failed_gen' = [failed_gen EXCEPT ![n] = gen[n]]
        /\ UNCHANGED <<gen, active, joined>>

(* ---- set_expected / ack_drain ------------------------------------------- *)

SetExpected(n, k) ==
  /\ ~slots[n][k].gone /\ slots[n][k].required = {}
  /\ slots' = [slots EXCEPT ![n][k].required = Required(k)]
  /\ UNCHANGED <<live, nerr, nerrgen, gen, active, joined, failed_gen>>

AckDrain(n, k, r, iv) ==
  (* a drain ACK happens only after a successful read of VALID bytes: a failed
     slot raises at wait_filled in-gen and is replaced by begin_fill next-gen,
     so ack_drain never runs against an error slot. AckRanks is the set of
     readers that can ever ack; omitting a required rank models a dead peer.
     In _drain_cell, set_expected runs BEFORE any read/ack, so an ack against a
     slot whose expected set was never published is impossible in the shared
     path: require s.required # {} here. *)
  /\ r \in AckRanks
  /\ LET s == slots[n][k] IN
     /\ ~s.gone /\ s.filled /\ s.err = "none" /\ s.required # {}
     /\ LET newcov == [s.cov EXCEPT ![r] = @ \cup {iv}]
            req == s.required
            done == \A rec \in req : rec.iv \in newcov[rec.rank]
            drop == (MUTANT = "early_release" \/ done)
         IN
         /\ slots' = [slots EXCEPT ![n][k].cov = newcov,
                                    ![n][k].gone = IF drop THEN TRUE ELSE s.gone]
         /\ live' = [live EXCEPT ![n] = @ - (IF drop THEN 1 ELSE 0)]
         /\ UNCHANGED <<nerr, nerrgen, gen, active, joined, failed_gen>>

Next ==
  \/ \E n \in Nodes, r \in NodeRanks : JoinCohort(n, r)
  \/ \E n \in Nodes : EndCohort(n)
  \/ \E n \in Nodes, k \in Keys : BeginFill(n, k)
  \/ \E n \in Nodes, k \in Keys : FinishFill(n, k, TRUE)
  \/ \E n \in Nodes, k \in Keys : FinishFill(n, k, FALSE)
  \/ \E n \in Nodes, k \in Keys : SetExpected(n, k)
  \/ \E n \in Nodes, k \in Keys, r \in NodeRanks, iv \in Ivs : AckDrain(n, k, r, iv)

vars == <<slots, live, nerr, nerrgen, gen, active, joined, failed_gen>>
Spec == Init /\ [][Next]_vars

(* ===========================================================================
   Safety invariants (normal cfgs must satisfy ALL).
   =========================================================================== *)

(* C: resident slots never exceed the mesh concurrency width. *)
CapacityBound == \A n \in Nodes : live[n] <= Cap

(* A slot is released (gone) only after every required (rank,interval) was acked.
   err="e" slots may be dropped by a later generation refill (no valid data). *)
NoEarlyRelease ==
  \A n \in Nodes : \A k \in Keys :
    LET s == slots[n][k] IN
    s.gone /\ s.err = "none" =>
      \A rec \in s.required : rec.iv \in s.cov[rec.rank]

(* A resident slot counts against C unless its fill failed: a failed slot lingers
   in the dict (to be drained or replaced next generation) but live already
   decremented at failure time (api.py _apply_finish). *)
CountsTowardLive(n, k) ==
  /\ ~slots[n][k].gone
  /\ ~(slots[n][k].filled /\ slots[n][k].err = "e")

(* Nodes are isolated: the live counter equals exactly this node's live-counting
   slots (a shared/aliased counter breaks this), and a node error is recorded
   on exactly the node that failed. *)
NodeIsolation ==
  /\ \A n \in Nodes : live[n] = Cardinality({k \in Keys : CountsTowardLive(n, k)})
  /\ \A n \in Nodes : nerr[n] = "e" => failed_gen[n] = nerrgen[n]

(* A node-wide error belongs to exactly the generation that raised it; a newer
   cohort must not observe it (generation advance clears the error). *)
GenerationIsolation ==
  \A n \in Nodes : nerr[n] = "none" \/ nerrgen[n] = gen[n]

(* A recorded node-wide failure actually broadcasts to the whole node. *)
FailureBroadcast ==
  \A n \in Nodes : failed_gen[n] # -1 =>
    nerr[n] = "e" /\ nerrgen[n] = failed_gen[n]

(* ===========================================================================
   Bounded liveness. Checked ONLY under explicit assumptions:
     - complete cohort (every consumer is present),
     - external reader eventually ack (WF restricted to AckRanks),
     - weak fairness on join/end and on fills finishing,
     - fills succeed (AllowFail=FALSE on liveness cfgs).
   In a partial-peer-down cfg (a required rank omitted from AckRanks) the goal
   is NOT claimed live and TLC is EXPECTED to return a counterexample trace —
   that boundary is documented, not papered over.
   =========================================================================== *)

FairSpec ==
  Spec
  /\ (\A n \in Nodes, r \in NodeRanks : WF_vars(JoinCohort(n, r)))
  /\ (\A n \in Nodes : WF_vars(EndCohort(n)))
  /\ (\A n \in Nodes : WF_vars(BeginFill(n, 0)))
  /\ (\A n \in Nodes, k \in Keys : WF_vars(SetExpected(n, k)))
  /\ (\A n \in Nodes, k \in Keys : WF_vars(FinishFill(n, k, TRUE)))
  /\ (\A n \in Nodes, k \in Keys, r \in AckRanks, iv \in Ivs :
        WF_vars(AckDrain(n, k, r, iv)))

(* Per-slot leads-to: once a slot is filled-healthy AND its expected set has been
   published (i.e. downstream readers are now allowed to drain it), it must
   eventually be released (gone) under complete-cohort fairness. A dead required
   peer keeps it occupied forever, which is the boundary TLC must flag. *)
Occupied(n, k) ==
  LET s == slots[n][k] IN
    ~s.gone /\ s.filled /\ s.err = "none" /\ s.required # {}

LiveGoal ==
  \A n \in Nodes, k \in Keys : []((Occupied(n, k)) => <>(slots[n][k].gone))

====
