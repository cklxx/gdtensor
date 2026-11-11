# Formal models — runtime mapping and scope

Each model is a bounded projection, not a proof of the whole Python runtime.
Only `CachePropagation.tla` has a current runtime mapping in this document.
`SlotLifecycle.tla` and the Lean coverage predicate are legacy design models;
their scope is stated explicitly below.

## Runtime source pins

- `gdtensor/internal/api.py`
  `sha256 5ef2460bbcda6731ecaa120a20ef4c75ef8240e1711360af03b8d7511e4f0b61`.
- `gdtensor/internal/propagation.py`
  `sha256 db555b6f0d4333c59ba00d0bcad82414ae380c88627c45c7fcec8d1d1f8ff8ad`.

`scripts/formal_gate.sh` fails if either pin differs from the checked-out file.
It also records both hashes in its generated manifest. A hash match verifies
the documented baseline; it does not prove that the Python implements a model.

## Legacy bounded models

`SlotLifecycle.tla` checks an abstract capacity, cohort, node-error and ACK
lifecycle. It does not map field-by-field to the current `_NodeState`. Current
runtime tracks resident bytes, a completed set and exact outstanding `owed`
pieces. It has no `_covers` predicate, per-slot `required`/`cov` maps, or the
legacy `generation_start`, `begin_fill`, `finish_fill` and `ack_drain` methods.
The model's positive invariants, negative mutants and bounded liveness results
apply only to the abstract model.

`formal/lean/Coverage.lean` proves properties of a legacy single-spanning
interval predicate and ideal interval-union coverage. Current
`_NodeState.ack` discards exact `(consumer, offset, size)` pieces from `owed`.
The Lean predicate is not the current production ACK predicate.

These files remain checked as mathematical regression models. They are not
source-level evidence for the current slot and ACK implementation. Current
runtime behavior is covered by the Python contract, coordination, mutation and
fault-injection tests listed in `tests/README.md`.

---

# CachePropagation.tla — current runtime mapping

The fixed instance is a two-layer fan-out: layer 0 is node 0 with one 8-byte
segment; terminal nodes 1 and 2 each own one 4-byte segment and both read the
layer-0 segment.

## Plan field map (`internal/propagation.py`)

| plan / executor concept                  | model element                     |
|------------------------------------------|-----------------------------------|
| `ChainPlan.layer_nodes` / segments per node | `Nodes`, one segment per node  |
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
| `_NodeState.admit` / `reserve` byte-budget admission | `Admit(n)`          |
| `_RankRun._fill` marks a slot filled                 | `Fill(n,TRUE)`      |
| `_adopt` / `_seal_one` accept valid presence         | `Fill(n,TRUE)`      |
| `_RankRun._drain` / `_NodeState.ack` rank piece      | `RankAck(n)`        |
| `_NodeState.ack` releases when `owed` is empty       | `ReleaseTerminal(n)`|
| `_RankRun._ack_parent` child ACK of a parent piece   | `AckEdge(n)`        |
| parent released after every child ACK                | `ReleaseParent`     |
| `_RankRun._fail` / `_NodeState.fail` node error      | `Fill(n,FALSE)`     |

The model's `owe` and `rem` sets are bounded projections of exact runtime ACK
pieces. The model does not represent batched reads, presence probes, in-flight
owner accounting, file locks or fingerprint sharing. Those paths require the
runtime tests; the formal gate does not claim to verify them.

## File-identity and integrity scope

The finite models do not represent pathname/stat changes, checksum algorithms,
short writes, or physical storage. These are enforced by accessor code and
negative tests. `cache_identity` is backend-derived; callers supply no version.
Cache ranges bind `source_identity`. Unknown source identity disables cross-call
adoption. File mutation during load returns Err; destination rollback is not
claimed. Metadata/stat identity detects ordinary changes, not adversarial
metadata forgery. `DiskNodeCache` verifies SHA256 before consumer callbacks.
The mutation suite removes these checks to prove the corresponding tests fail.

Active slot admission is the modeled byte budget. Persistent valid file ranges
and callback temporary buffers are distinct metrics and are not that budget.

## DN fill ownership

`ChainPlan.fill_ranks` permits only local consumer ranks to fill a segment.
When no local consumer needs a segment, `services` assigns it to a local rank.
The helper remains in its execution closure until its service work completes.
The runtime's completed set prevents released segments from being filled again
within one generation. The public cache-subset constraint remains unchanged.

`CachePropagation.tla` records the rank that performs `Fill(n, rank, ok)`.
Its bounded instance has one rank per node, so `FillOnDN` requires `rank = n`.
The `remote_fill` mutant relaxes that guard and must violate `FillOnDN`.
This model verifies placement only for that bounded instance. Python service
queues, multiple ranks per node and separate processes are checked by tests.

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
segment. Cross-process coordination is a real implementation
(`_FileSlotTable`, selected by `GDTENSOR_CACHE_COORDINATOR`), but the model
projects it away: it checks the shared in-memory table contract shape, not file
locking. Related runtime constraints are:

- Cell budget sums every tensor of a (step, slot) cell: `_budget` keys cells by
  `(s, c)` and sums segment bytes over all tensors, so `ByteBudget` compares
  against the aggregated node budget that `ChainPlan.budget` publishes.
- Absent cohort ranks error, never hang: the executor raises `cache_wait` after
  `GDTENSOR_CACHE_WAIT_SECONDS` (modeled as the `peer_down` liveness boundary).
- In-flight fills count as progress: `_block` keeps waiting while `inflight > 0`
  even when the version is unchanged (a timeout detail, outside the model).
- The plan cache holds weak references only: `_slots` registers
  `weakref.finalize(view, _drop_table, ...)` so a `ChainPlan` never pins a cache
  view (lifetime, outside the model).
- `check_constraints` rejects mixed node sets and out-of-range ranks up front
  (`ConstraintViolationError` / `InvalidArgumentError`); `CacheDTensorsView` is
  registered with the constraint checker (construction-time, outside the model).
