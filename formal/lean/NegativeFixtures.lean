/-
INTENTIONALLY BROKEN negative-model fixtures.

This file MUST NOT compile. Each example states a false belief about the frozen
Python semantics; ``by decide`` fails on every one of them. The check script
(run from check.sh) compiles this file separately, EXPECTS failure, and
verifies the failure is a proof-level rejection (``decide`` / tactic failed)
rather than a syntax or unbound-name error.

What each fixture tries to assert (all false):
  neg1: the column shard {0,1,4,5} IS a single contiguous packed run
        (production _owned_chunk_local_run rejects it -- gap 1..4).
  neg2: the group leader row is the MAXIMUM member row (production uses min).
  neg3: acked [0,2),(2,4) satisfy production NodeState._covers for [0,4)
        (production has NO union merge; _covers needs one spanning interval).
  neg4: the chunk-local flat of (row1,col0) in chunk (2,2) is 4
        (row-major repack gives 2, not 4).
  neg5: ideal elementwise-union coverage would imply production done
        (the converse fallacy; false on the same fixture).

No sorry/admit/axiom anywhere.
-/
import RowMajor
import ChunkRun
import Coverage
import Leader

-- neg1: column shard {0,1,4,5} falsely claimed contiguous in packed order.
example : isSingleRun [0, 1, 4, 5] = true := by decide

-- neg2: leader row claimed to be the maximum member (must be minimum).
example : leaderRow [(5, 0), (2, 0), (7, 0)] = 7 := by decide

-- neg3: production falsely claimed to merge two acked intervals.
example : prodCovers [(0, 2), (2, 4)] [(0, 4)] = true := by decide

-- neg4: wrong chunk-local repack (row1 of chunk shape (2,2) is local flat 2).
example : localFlat [2, 2] [1, 0] = 4 := by decide

-- neg5: ideal union coverage falsely claimed to imply production done.
example : idealCoverInterval [(0, 2), (2, 4)] 0 4 →
    prodCovers [(0, 2), (2, 4)] [(0, 4)] = true := by
  intro h
  have h0 := h 0 (by simp) (by omega)
  have h1 := h 1 (by simp) (by omega)
  have h2 := h 2 (by simp) (by omega)
  have h3 := h 3 (by simp) (by omega)
  decide

-- neg6: on a row tie, max rank falsely claimed as leader (production picks min).
example : leaderCell [(2, 5), (2, 3)] = (2, 5) := by decide
