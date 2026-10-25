# Decisions 2026-10-26

## Slot scheduling delta (leader-approved)

- Same slot: fill -> drain (cache read callback ACK) -> release/reuse.
- Different slots: cross-leg interleave permitted.
- Finish: all copies / reader ACK done.
- Public API unchanged; vendored cloud artifacts byte-exact.

## Cache observer results (14/14 pass)

- C1 capacity bound: peak <= C * ceil(shape/S) bytes per node.
- Node isolation: (node, tid, offset) keyed slots.
- Cohort failure: same-Error notify, no implicit retry; next generation explicit.
- Replicate all-consumer ACK: slot retained until every declared rank reads full coverage.

## Mixed session contract (5 pass)

- Each segment owns its source session; disjoint tensor IDs.
- B fail -> A bytes retained; both finish with same Error.

## DCP metadata offset + shape (5 cases pass)

- Per-dim chunk offset/shape mapping; col sub-slot row1 global[4,6) pread 8B == dense[1,cols].
- Existing tid / empty slot -> InvalidArg; unknown pos -> NotFound.
- Cross-file single base -> InvalidArg; public IO all reads correct.
- Multi-machine and packed-compression not yet tested.

## Boundary (not claimed)

- Multi-machine replication untested.
- Packed compression format rejected.
- cloud vendored artifacts unchanged (md5 verified).
