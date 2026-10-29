# Formal TLA+ models — mapping to `gdtensor/internal/`

This document is the field-by-field and transition-by-transition correspondence
between the TLA+ models and the implementation, plus the explicit assumptions
under which the models are claimed. Each model is a *bounded projection*, not a
proof of the whole Python.

- `SlotLifecycle.tla` covers the cohort/error lifecycle of one cache node.
- `CachePropagation.tla` covers the node-level propagation plan: byte-budget
  admission, parent/child order, and release after every owed ack.

Pinned runtime baseline:
- `gdtensor/internal/api.py`
  `sha256 4c8e5a173549627feb994787f3600384383d5f98437a3482408df1c809c7c452`.
- `gdtensor/internal/propagation.py`
  `sha256 a1faec351e0e4c016e0efa221146030e941b64d37a0c13a9fc89210da8410506`.
Current source hashes are recorded by `scripts/formal_gate.sh` in its
generated manifest.

## Field map

| api.py (`_NodeState` / slot)            | model field                          |
|-----------------------------------------|--------------------------------------|
| `slots[key]`, key = `(tid,off,size)`    | `slots[n][k]` (k ∈ Keys)             |
| `live`                                  | `live[n]`                            |
| `cap`                                   | constant `Cap`                       |
| `consumers`                             | constant `Consumers`                 |
| `error` / `error_gen`                   | `nerr[n]` / `nerrgen[n]`             |
| `gen`                                   | `gen[n]`                             |
| `active`                                | `active[n]`                          |
| `cohort_joined`                         | `joined[n]`                          |
| slot `filled`                           | `slots[..].filled`                   |
| slot `error` / `error_gen`              | `slots[..].err` / `errgen`           |
| slot `required` (rank→intervals)        | `slots[..].required` (set of rec)    |
| slot `cov` (rank→acked intervals)       | `slots[..].cov` (rank→set of iv)     |
| (latch) recorded node failure            | `failed_gen[n]`                      |

## Transition map

| api.py                                            | model step                       |
|---------------------------------------------------|----------------------------------|
| `generation_start` / `_cohort_join`               | `JoinCohort(n,r)`                |
| `generation_end`                                   | `EndCohort(n)`                   |
| `begin_fill` / `_begin_decision/_begin_existing`  | `BeginFill(n,k)`                 |
| `finish_fill` / `_apply_finish`                   | `FinishFill(n,k,TRUE|FALSE)`      |
| `set_expected`                                     | `SetExpected(n,k)`                |
| `ack_drain` / `_covers`                           | `AckDrain(n,k,r,iv)`             |

Faithful notes:
- A failed slot stays in `slots` but `live` is decremented at failure time
  (`_apply_finish`); the model counts a slot toward `live` unless it is a
  filled-error slot (`CountsTowardLive`).
- `ack_drain` is guarded to a filled-healthy slot: in-gen a failed slot raises at
  `wait_filled`, and next-gen `begin_fill` replaces it before any drain, so the
  `ack_drain` code path never runs against an error slot.
- A cohort advances (gen+1, error cleared) only when `active=0` and every consumer
  has joined; a late same-cohort rank still observes the existing node error.
- `failed_gen` is a model latch, reset on cohort advance; it exists only to state
  `FailureBroadcast`.

## Safety invariants (normal cfgs must satisfy all)

- `CapacityBound`      — `live[n] <= Cap`.
- `NoEarlyRelease`      — a healthy slot is released only after every required
                         (rank,interval) is acked.
- `NodeIsolation`      — `live[n]` equals exactly n's live-counting slots; a node
                         error is recorded on the node that failed.
- `GenerationIsolation` — a node error belongs to exactly one generation; a newer
                         cohort does not observe it.
- `FailureBroadcast`    — a recorded node failure actually broadcasts node-wide.

## Negative mutants (expected to be caught)

| `MUTANT`      | models bug                                   | caught by            |
|---------------|----------------------------------------------|----------------------|
| `no_cap`      | begin_fill ignores the C bound               | `CapacityBound`      |
| `early_release` | delete slot on first ack, not all required | `NoEarlyRelease`     |
| `node_alias`  | a second node's fill bumps node 0's counter  | `NodeIsolation`      |
| `stale_gen`   | gen advance forgets to clear the node error  | `GenerationIsolation`|
| `no_broadcast`| failure only marks the slot, not the node    | `FailureBroadcast`    |

## Bounded liveness (NOT claimed unconditionally)

`LiveGoal == \A n,k : [](Occupied(n,k) => <>(slots[n][k].gone))` is a per-slot
leads-to: once a slot is filled-healthy AND its expected set has been published
(`required # {}`, mirroring `_drain_cell` calling `set_expected` before any
read/ack), it must eventually be released. The old global `<>(live=0)` was
vacuous at init and is removed.

Checked only under `FairSpec`, which assumes: weak fairness on cohort join/end,
on `BeginFill(n,0)`, on `SetExpected(n,k)`, on `FinishFill(n,k,TRUE)`, and on
every `AckDrain(n,k,r,iv)` for `r in AckRanks`; fills succeed (`AllowFail=FALSE`).
A config that omits a required rank from `AckRanks` (partial peer down) keeps
the slot occupied forever and TLC **does** return a temporal counterexample
(exit 13) — that boundary is reproduced, not papered over.

Scope note: the model's `cov`/`required` abstracts `_covers` as a set of
(rank,interval) pairs; it does NOT claim to model the real single-spanning
interval condition of the production path. It checks the cell-contract shape
(coverage set empties only when every expected pair is acked) under the bounded
grid here, not a proof of the Python.

---

# CachePropagation.tla — mapping to `propagation.py` + `api.py`

The fixed instance is the two-layer fan-out fixed by this change: layer 0 is
node 0 with one 8-byte segment; terminal nodes 1 and 2 each own one 4-byte
segment and both read the layer-0 segment.

## Plan field map (`internal/propagation.py`)

| plan / executor concept                  | model element                     |
|------------------------------------------|-----------------------------------|
| `ChainPlan.nodes` / segments per node    | `Nodes`, one segment per node     |
| segment byte size                        | `SegSize[n]`                      |
| `ChainPlan.budget` byte bound            | `Budget[n]`                       |
| parent edges (`Segment.parents`)         | `Edges[n]` (constant `A`)         |
| owed rank consumers (`segment.owed()`)   | `owe[n]`                          |
| unacked parent edges                     | `rem[n]`                          |
| `_NodeState.resident`                    | `res[n]`                          |
| `_NodeState.slots[k].filled`             | `st[n]`: admitted → ready         |
| node-wide generation error               | `nerr[n]`                         |
| fill admission (`state.resident+size<=budget`) | `Admit(n)` budget guard    |

Bookkeeping variables `seen`, `seen0`, and `p0ready` are model-only history
latches. They exclude the pre-pass state from "early release" statements and
remember that the parent reached `ready` at least once (the parent may be
released while a child keeps its own copied bytes). They have no runtime
counterpart.

## Transition map

| executor (`api.py`)                                  | model step          |
|------------------------------------------------------|---------------------|
| `_NodeState.admit` byte-budget check                 | `Admit(n)`          |
| fill job completes (`slots[k]["filled"]=True`)       | `Fill(n,TRUE)`      |
| `_SlotTable._ack` rank piece                         | `RankAck(n)`        |
| segment released after owed set empties              | `ReleaseTerminal(n)`|
| child ack of a parent edge                           | `AckEdge(n)`        |
| parent released after every child acked              | `ReleaseParent`     |
| `_fail_node` generation error                        | `Fill(n,FALSE)`     |

## Safety invariants (positive cfg must satisfy all)

- `ByteBudget`              — resident bytes never exceed `ChainPlan.budget`.
- `ResidentExact`           — `resident` equals exactly the admitted and ready
                              segment sizes.
- `NoEarlyRankRelease`      — a terminal segment leaves memory only after every
                              owed rank acked.
- `NoEarlyParentRelease`    — the layer-0 segment leaves memory only after
                              every child acked its edge.
- `ParentReadyBeforeChild`  — a child occupies bytes only after the parent
                              reached ready once.
- `EdgesArePlanEdges`       — a remaining ack edge is always a plan edge.
- `ErrorRecorded`           — a failed fill sets the node-wide generation error.
- `ErrorStopsAdmit`         — a recorded error blocks new admission in the pass.

## Negative mutants (each must be caught by one named invariant)

| `MUTANT`          | models bug                                       | caught by                  |
|-------------------|--------------------------------------------------|----------------------------|
| `no_budget`       | admission skips the byte-budget guard            | `ByteBudget`               |
| `early_rank`      | release a terminal segment before all rank acks  | `NoEarlyRankRelease`       |
| `early_parent`    | release the parent before all child acks         | `NoEarlyParentRelease`     |
| `no_parent_order` | admit a child before the parent is ready         | `ParentReadyBeforeChild`   |
| `silent_error`    | a failed fill is not recorded node-wide          | `ErrorRecorded`            |

## Bounded liveness (not unconditional)

`LiveGoal == \A n : [](st[n] # "gone" => <>(st[n] = "gone"))` holds under
`FairSpec`: weak fairness on admit/fill/ack/release, fills succeed, every rank
acks, and every child acks its parent edge. The `peer_down` config removes the
rank ack of node 2; TLC returns a temporal counterexample (exit 13). The
executor exposes the same boundary as `cache_wait` after
`GDTENSOR_CACHE_WAIT_SECONDS` (default 60 s): an absent consumer is reported,
not waited on forever.

Scope: one fixed fan-out instance, one segment per node, one byte size per
segment. Cross-process coordination is a declared replacement point
(`CACHE_COORDINATOR_FACTORY`) and is not modeled.
