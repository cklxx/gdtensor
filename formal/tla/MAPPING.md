# SlotLifecycle.tla — mapping to `gdtensor/internal/api.py`

This document is the field-by-field and transition-by-transition correspondence
between the TLA+ model `SlotLifecycle.tla` and the implementation, plus the explicit
assumptions under which the model is claimed. The model is a *bounded projection*,
not a proof of the whole Python.

Pinned slot-runtime baseline: `gdtensor/internal/api.py` at
`sha256 5ddec0b159b10fbb2a1eb2f0bcd4bbc4d5a7b89e8ccde983f0046aba083f4b3b`.
The file-range fix retains source paths and slot boundaries in fill jobs;
`_NodeState` transitions modeled here are unchanged. Current source hashes are
recorded by `scripts/formal_gate.sh` in its generated manifest.

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
