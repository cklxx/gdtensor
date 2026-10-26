/-
ACK interval coverage: ideal union vs the production single-spanning check.

Mirrors ``gdtensor/internal/api.py`` ``NodeState._covers`` (frozen source):

    @staticmethod
    def _covers(ivs, required):
        merged = set(ivs)
        return all(any(x <= a and y >= b for x, y in merged)
                   for a, b in required)

KEY FACT (must not be over-proved): ``merged`` is just the set of acked
intervals recorded FOR ONE RANK; the check requires SOME SINGLE acked interval
(x, y) to satisfy x <= a and y >= b for every required interval (a, b).
There is NO union/merge of several acked intervals: acks [0,2) and [2,4) do
NOT cover required [0,4) in production, because no single (x,y) spans it.

This file proves:
  1. the IDEAL elementwise-union coverage definition, and its union lemma
     (element covered by the union of two ACK lists iff covered by one of them);
  2. the production single-spanning predicate, and that single-spanning is a
     SUFFICIENT condition for ideal coverage (every n in [a,b) lies inside the
     spanning acked interval).

The converse (ideal union coverage => production done) is FALSE and is carried
as an intentionally-rejected fixture in NegativeFixtures.lean.
No sorry/admit/axiom.
-/
import Init.Tactics

/-- Does the acked interval list contain an interval satisfying p? -/
def anyInterval : List (Nat × Nat) → (Nat → Nat → Bool) → Bool
  | [], _ => false
  | (x, y) :: rest, p => p x y || anyInterval rest p

/-- PRODUCTION predicate (NodeState._covers): every required interval must be
    contained in SOME single acked interval. No union merge. -/
def prodCovers (acks required : List (Nat × Nat)) : Bool :=
  match required with
  | [] => true
  | (a, b) :: rest =>
    anyInterval acks (fun x y => x ≤ a && y ≥ b) && prodCovers acks rest

/-- IDEAL coverage of one element: n lies in at least one acked interval. -/
def idealCoverElem (acks : List (Nat × Nat)) (n : Nat) : Bool :=
  anyInterval acks (fun x y => x ≤ n && n < y)

/-- IDEAL coverage of a whole required half-open interval [a,b): every element
    of it lies in the union of the acked intervals. -/
def idealCoverInterval (acks : List (Nat × Nat)) (a b : Nat) : Prop :=
  ∀ n, a ≤ n → n < b → idealCoverElem acks n

/-- Helper: an interval satisfying p somewhere in the list makes anyInterval
    true. -/
private theorem anyInterval_of_mem (acks : List (Nat × Nat))
    (p : Nat → Nat → Bool) (x y : Nat)
    (hmem : (x, y) ∈ acks) (hp : p x y = true) :
    anyInterval acks p = true := by
  induction acks with
  | nil => simp at hmem
  | cons e rest ih =>
    by_cases he : e = (x, y)
    · subst he
      simp [anyInterval, hp]
    · have hmem' : (x, y) ∈ rest := by
        simp only [List.mem_cons] at hmem
        rcases hmem with heq | hrest
        · exfalso
          apply he
          exact heq.symm
        · exact hrest
      have := ih hmem'
      simp [anyInterval, this]

-- ---------------------------------------------------------------------------
-- Union lemma: ideal coverage through the union of two ACK lists is the
-- disjunction of ideal coverage through each list.
-- ---------------------------------------------------------------------------

theorem idealCoverElem_append : ∀ acks1 acks2 n,
    idealCoverElem (acks1 ++ acks2) n =
      (idealCoverElem acks1 n || idealCoverElem acks2 n) := by
  intro acks1 acks2 n
  induction acks1 with
  | nil => rfl
  | cons e rest ih =>
    rcases e with ⟨x, y⟩
    simp only [idealCoverElem, anyInterval, List.cons_append]
    have hrest : anyInterval (rest ++ acks2) (fun x y => x ≤ n && n < y) =
        (anyInterval rest (fun x y => x ≤ n && n < y) ||
         anyInterval acks2 (fun x y => x ≤ n && n < y)) := by
      simpa [idealCoverElem] using ih
    by_cases hxy : (x ≤ n && n < y) = true
    · rw [hxy, hrest] <;> simp
    · have hfalse : (x ≤ n && n < y) = false := by simpa using hxy
      rw [hfalse, hrest] <;> simp

/-- Union monotonicity: acking more intervals can only grow the covered set. -/
theorem idealCoverElem_monotone (acks : List (Nat × Nat)) (n : Nat)
    (h : idealCoverElem acks n = true) :
    ∀ extra : List (Nat × Nat), idealCoverElem (acks ++ extra) n = true := by
  intro extra
  rw [idealCoverElem_append, h]
  <;> simp

-- ---------------------------------------------------------------------------
-- Production sufficient condition: a single acked interval that spans the
-- required interval (x <= a and y >= b) makes every element of [a,b)
-- ideally covered.
-- ---------------------------------------------------------------------------

/-- Extraction from the production test: some acked interval (x,y) satisfies
    x <= a and y >= b. -/
private theorem anyInterval_spanning_exists (acks : List (Nat × Nat)) (a b : Nat)
    (h : anyInterval acks (fun x y => x ≤ a && y ≥ b) = true) :
    ∃ x y, (x, y) ∈ acks ∧ x ≤ a ∧ y ≥ b := by
  induction acks with
  | nil => simp [anyInterval] at h
  | cons e rest ih =>
    rcases e with ⟨x, y⟩
    simp only [anyInterval] at h
    by_cases hxy : (x ≤ a && y ≥ b) = true
    · exact ⟨x, y, by simp, by simpa using hxy⟩
    · have hfalse : (x ≤ a && y ≥ b) = false := by simpa using hxy
      rw [hfalse] at h
      rcases ih h with ⟨x', y', hm, hx', hy'⟩
      exact ⟨x', y', by simp [hm], hx', hy'⟩

/-- Single acked interval spanning [a,b) implies ideal coverage of [a,b). -/
theorem prodSpanning_implies_ideal (acks : List (Nat × Nat)) (a b : Nat)
    (h : prodCovers acks [(a, b)] = true) :
    idealCoverInterval acks a b := by
  have hspan : anyInterval acks (fun x y => x ≤ a && y ≥ b) = true := by
    simpa [prodCovers] using h
  rcases anyInterval_spanning_exists acks a b hspan with ⟨x, y, hmem, hx, hy⟩
  intro n hn1 hn2
  have hnx : x ≤ n := by omega
  have hny : n < y := by omega
  have hp : (x ≤ n && n < y) = true := by
    simp [hnx, hny]
  have : idealCoverElem acks n = true := by
    simpa [idealCoverElem] using anyInterval_of_mem acks (fun x y => x ≤ n && n < y) x y hmem hp
  exact this

-- ---------------------------------------------------------------------------
-- Concrete checks on the audit fixture:
--   acks = [(0,2),(2,4)] ideally cover [0,4) elementwise (0,1 in first;
--   2,3 in second), yet production prodCovers [(0,4)] is FALSE: no single
--   acked interval spans [0,4).
-- ---------------------------------------------------------------------------

example : idealCoverElem [(0, 2), (2, 4)] 0 = true := by decide
example : idealCoverElem [(0, 2), (2, 4)] 1 = true := by decide
example : idealCoverElem [(0, 2), (2, 4)] 2 = true := by decide
example : idealCoverElem [(0, 2), (2, 4)] 3 = true := by decide

example : prodCovers [(0, 2)] [(0, 2)] = true := by decide
example : prodCovers [(0, 2), (2, 4)] [(0, 4)] = false := by decide

/-- The sufficient condition in action: acked [0,4) spanning required [0,2)
    makes production done AND ideally covers every element. -/
example : prodCovers [(0, 4)] [(0, 2)] = true := by decide

/-- Union lemma on the fixture: union of [0,2) and [2,4) ideal-covers 1 and 2
    through different branches. -/
example : idealCoverElem ([(0, 2)] ++ [(2, 4)]) 1 = true := by decide
example : idealCoverElem ([(0, 2)] ++ [(2, 4)]) 2 = true := by decide
