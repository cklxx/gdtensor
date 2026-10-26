# gdtensor — Lean formalization of the frozen DCP mapping semantics

This directory pins the **frozen** semantics of the DCP slot/owner mapping in
Lean 4 (toolchain pinned in `lean-toolchain`, currently `leanprover/lean4:v4.34.0`).
Everything here uses only the Lean core `Init` library — **no mathlib, no other
dependency**. Every positive file compiles with `-DwarningAsError=true` and
contains no `sorry` / `admit` / `axiom`.

Do **not** edit Python source from here; the Lean side mirrors the frozen
contract, it does not regenerate or refine it.

## How to run

```
./check.sh
```

It (1) greps every positive source for proof holes, (2) compiles the positive
chain in dependency order and writes `.olean` into the git-ignored `log/lean/`
tree, and (3) separately compiles `NegativeFixtures.lean`, which **must** be
rejected at the proof level (a false `decide` claim), and fails the gate if the
rejection is instead a syntax/unbound error.

## Files and what each maps to

| Lean file | Frozen Python source | What is stated |
|---|---|---|
| `ShardAlgebra.lean` | baseline shard algebra | pre-existing base proofs (unchanged). |
| `RowMajor.lean` | `gdtensor_sim/dcp_backend.py` `_owned_chunk_local_run` row-major decode | `shapeProd`, `coordAt`, `packedFlat`, `localFlat`; inverse lemma `packedFlat_coordAt` (global flat → coords → repack == global flat) for nonempty shapes; triangulation on shape `[2,4]`. |
| `ChunkRun.lean` | same function, tail `local[-1]-local[0]+1 != len → None` | insertion-sort criterion `isSingleRun`; membership/length preservation; soundness `singleRun_elem_in_window` (when accepted, every owned local lies in the pread window). Triangulation: column shard `{0,1,4,5}` rejected (`=false`), row-contiguous run `[2,3]` accepted (`=true`). |
| `Leader.lean` | `gdtensor_sim/models.py` grouping: `t0 = min(t for t,_ in cells)` | head-init `foldMin`/`leaderRow`; `leaderRow_le` (leader ≤ every cell row) and `leaderRow_mem` (leader achieved by some cell). `lexMinCell`/`leaderCell` model the lex-min `(row, rank)` entry `TensorComponentPos(ids[t0], leader.rank)`. Triangulation: leader of `[(5,0),(2,0),(7,0)]` is **2** (min, not max-member 7); row tie `[(2,5),(2,3)]` → `(2,3)` (min rank). Value equality: permuted cell lists give the same leader. |
| `Coverage.lean` | `gdtensor/internal/api.py` `NodeState._covers` | `prodCovers` = production "some single acked interval spans every required interval" (**no union merge**). `idealCoverInterval` = the ideal elementwise-union coverage. Lemma `prodSpanning_implies_ideal` (single-spanning ⇒ ideal holds); append/union lemmas for the ideal side only. |
| `NegativeFixtures.lean` | — | intentionally broken examples; the gate **expects** Lean to reject them (see below). |

## Preconditions stated explicitly

- `packedFlat_coordAt` requires a **nonempty** shape (`shape ≠ []`); empty
  shapes are never produced by the frozen layout and are out of scope.
- `leaderRow_mem` requires a **nonempty** group (production groups are built
  from cells).
- The soundness lemma takes `isSingleRun locals = true` as a hypothesis; it
  characterizes the accepted window, it does not claim acceptance is correct
  for every shard.

## What is deliberately NOT claimed

- The Lean side does **not** prove the whole Python file, nor that the Python
  runtime is correct. It states lemmas about the *modeled* functions and checks
  concrete triangulation instances by `decide`.
- Production `NodeState._covers` has **no multi-interval union merge**. The
  fixture `acks [(0,2),(2,4)]` covers `[0,4)` **ideally** but `prodCovers`
  returns **false** for it. The union lemmas live on the *ideal* predicate only;
  we do **not** fake a proof that production merges intervals.
- Row-major inverse and window lemmas are stated for finite concrete geometries
  plus the universally-quantified list/interval theorems; they are not a full
  refinement of the C++/Python implementation.
- **Leader model is a row projection only.** Production picks the lex-min
  `(row, rank)` cell to build `TensorComponentPos(ids[t0], leader.rank)`; the
  Lean side proves the row-minimum universally (`leaderRow_le`/`leaderRow_mem`)
  and pins the rank tie-break by concrete `decide` fixtures
  (`leaderCell [(2,5),(2,3)] = (2,3)`). The universal lemma "row projection of
  the lex-min cell equals `leaderRow`" is **not** proved inductively here; the
  missing true-leader-cell entry lookup (`file_paths[TensorComponentPos(...)] = path`)
  is **not** modeled. Do not read the row-min proofs as a full proof of the
  production `(row, rank)` selection or the label→pos equality.

## Component granularity

- **basic** (address mapping + labels): `RowMajor.lean`, `Leader.lean`,
  `ChunkRun.lean` — row-major global→chunk-local mapping, single-run criterion,
  non-contiguous rejection, leader = min row.
- **advanced** (ACK coverage): `Coverage.lean` — ideal union coverage vs the
  production single-spanning predicate.

## Negative fixtures (must be rejected, on purpose)

`NegativeFixtures.lean` asserts six falsehoods about the frozen semantics;
`check.sh` compiles it alone and requires a **proof-level** failure:

1. column shard `{0,1,4,5}` is a single contiguous run (false: gap 1..4);
2. leader row is the max member (false: it is the min);
3. production merges `[(0,2),(2,4)]` to cover `[0,4)` (false: no union merge);
4. chunk-local flat of `(1,0)` in chunk `(2,2)` is 4 (false: it is 2);
5. ideal union coverage implies production `_covers` (false converse);
6. on a row tie, max rank wins (false: production picks min rank).

## Proof scale

All proofs are built from Lean core tactics (`simp`, `rw`, `omega`, `decide`,
`by_cases`, structural recursion). No `sorry`/`admit`/`axiom` in any positive
file; the negative file also contains none — its failures come from false
decidable claims.
