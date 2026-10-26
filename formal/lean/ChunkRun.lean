/-
Single-contiguous-run criterion for the DCP slot mapping.

Mirrors the tail of ``gdtensor_sim/dcp_backend.py`` ``_owned_chunk_local_run``
(frozen source, audit df1):

    local.sort()
    if local[-1] - local[0] + 1 != len(local):
        return None          # gap between owned elements in chunk-packed order
    return local[0], len(local)

Only Lean core Init; no proof holes.
-/
import Init.Tactics
import Init.Data.Nat.Lemmas

/-- Insertion sort, ascending. Membership is preserved. -/
def insertSorted : Nat → List Nat → List Nat
  | n, [] => [n]
  | n, m :: ms => if n ≤ m then n :: m :: ms else m :: insertSorted n ms

def sortNats : List Nat → List Nat
  | [] => []
  | n :: ns => insertSorted n (sortNats ns)

/-- Ascending-sorted predicate (non-strict). -/
def SortedLe : List Nat → Bool
  | [] => true
  | [_] => true
  | a :: b :: xs => a ≤ b && SortedLe (b :: xs)

/-- Last element; fallback only used on empty lists. -/
def lastDef : List Nat → Nat → Nat
  | [], d => d
  | [x], _ => x
  | _ :: xs, d => lastDef xs d

/-- The production criterion: owned locals form ONE contiguous packed run.
    Empty input is rejected (Python returns None before sorting). -/
def isSingleRun (locals : List Nat) : Bool :=
  match locals with
  | [] => false
  | _ :: _ =>
    let s := sortNats locals
    match s with
    | [] => false
    | l0 :: rest => (lastDef rest l0) - l0 + 1 == locals.length

/-- Unpack a sorted-cons hypothesis into its two components. -/
private theorem sortedLe_cons : ∀ a b bs,
    SortedLe (a :: b :: bs) = true →
    a ≤ b ∧ SortedLe (b :: bs) = true := by
  intro a b bs h
  simpa [SortedLe, Bool.and_eq_true] using h

-- ---------------------------------------------------------------------------
-- Small facts about insertion sort.
-- ---------------------------------------------------------------------------

private theorem insertSorted_mem_iff : ∀ n s m,
    m ∈ insertSorted n s ↔ m = n ∨ m ∈ s := by
  intro n s
  induction s with
  | nil =>
    intro m
    simp [insertSorted]
  | cons a s ih =>
    intro m
    by_cases hna : n ≤ a
    · have hif : insertSorted n (a :: s) = n :: a :: s := by
        rw [insertSorted, ite_eq_left hna]
      rw [hif]
      simp [List.mem_cons]
    · have hif : insertSorted n (a :: s) = a :: insertSorted n s := by
        rw [insertSorted, ite_eq_right hna]
      rw [hif]
      simp only [List.mem_cons]
      rw [ih m]
      constructor
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)
      · intro h
        rcases h with h1 | h2 | h3
        · exact Or.inr (Or.inl h1)
        · exact Or.inl h2
        · exact Or.inr (Or.inr h3)

theorem sortNats_mem_iff : ∀ n l, n ∈ sortNats l ↔ n ∈ l := by
  intro n l
  induction l with
  | nil => rfl
  | cons h t ih =>
    simp only [sortNats, insertSorted_mem_iff, ih, List.mem_cons]

private theorem insertSorted_len : ∀ n s, (insertSorted n s).length = s.length + 1 := by
  intro n s
  induction s with
  | nil => rfl
  | cons a s ih =>
    simp only [insertSorted]
    split <;> simp [ih, List.length_cons] <;> omega

theorem sortNats_len : ∀ l, (sortNats l).length = l.length := by
  intro l
  induction l with
  | nil => rfl
  | cons h t ih =>
    simp only [sortNats, insertSorted_len, ih, List.length_cons]

/-- Inserting n into an ascending sorted list keeps it ascending. -/
private theorem insertSorted_sorted : ∀ n s, SortedLe s = true →
    SortedLe (insertSorted n s) = true := by
  intro n s
  induction s with
  | nil =>
    intro hs
    simp [insertSorted, SortedLe]
  | cons a s ih =>
    intro hs
    by_cases hna : n ≤ a
    · -- n goes to the front: n :: a :: s
      simp [insertSorted, hna, SortedLe, hs]
    · -- n goes into the tail: a :: insertSorted n s
      cases s with
      | nil =>
        simp [insertSorted, hna, SortedLe] <;> omega
      | cons b bs =>
        have ⟨hab, htail⟩ := sortedLe_cons a b bs hs
        have ih' := ih htail
        have hhead : a ≤ (insertSorted n (b :: bs)).head! := by
          by_cases hnb : n ≤ b
          · have hif : insertSorted n (b :: bs) = n :: b :: bs := by
              rw [insertSorted, ite_eq_left hnb]
            rw [hif]
            simp [List.head!]
            have hlt : a < n := by omega
            exact Nat.le_of_lt hlt
          · have hif : insertSorted n (b :: bs) = b :: insertSorted n bs := by
              rw [insertSorted, ite_eq_right hnb]
            rw [hif]
            simp [List.head!]
            exact hab
        have hif : insertSorted n (a :: (b :: bs)) = a :: insertSorted n (b :: bs) := by
          rw [insertSorted, ite_eq_right hna]
        rw [hif]
        have hne : insertSorted n (b :: bs) ≠ [] := by
          by_cases hnb : n ≤ b <;> simp [insertSorted, hnb]
        cases hsort : insertSorted n (b :: bs) with
        | nil => exact False.elim (hne hsort)
        | cons y ys =>
          rw [hsort] at hhead ih'
          simp [SortedLe]
          <;> exact ⟨hhead, ih'⟩

theorem sortNats_sorted : ∀ l, SortedLe (sortNats l) = true := by
  intro l
  induction l with
  | nil => rfl
  | cons h t ih => exact insertSorted_sorted h (sortNats t) ih

/-- In an ascending sorted list, the head is a lower bound of every member. -/
private theorem sorted_head_le : ∀ s, SortedLe s = true →
    ∀ n, n ∈ s → s.head! ≤ n := by
  intro s
  induction s with
  | nil =>
    intro hs n hn
    simp at hn
  | cons a s ih =>
    intro hs n hn
    simp only [List.mem_cons] at hn
    rcases hn with h0 | hrest
    · rw [show n = a from h0]
      exact Nat.le_refl a
    · cases s with
      | nil => simp at hrest
      | cons b bs =>
        have ⟨hab, htail⟩ := sortedLe_cons a b bs hs
        exact Nat.le_trans hab (ih htail n hrest)

/-- In an ascending sorted list, no member exceeds the last. -/
private theorem sorted_le_last : ∀ s f, SortedLe s = true →
    ∀ n, n ∈ s → n ≤ lastDef s f
  | [], f, _, n, hn => by simp at hn
  | a :: s, f, hs, n, hn => by
    simp only [List.mem_cons] at hn
    rcases hn with h0 | hrest
    · rw [show n = a from h0]
      cases s with
      | nil => exact Nat.le_refl a
      | cons b bs =>
        have ⟨hab, htail⟩ := sortedLe_cons a b bs hs
        have hb : b ≤ lastDef (b :: bs) f := by
          exact (sorted_le_last (b :: bs) f htail) b (by simp)
        exact Nat.le_trans hab hb
    · cases s with
      | nil => simp at hrest
      | cons b bs =>
        have ⟨_, htail⟩ := sortedLe_cons a b bs hs
        have ih' := sorted_le_last (b :: bs) f htail n hrest
        exact ih'

-- ---------------------------------------------------------------------------
-- Soundness: when the criterion holds, every owned local lies inside the
-- window [first, first + len) that one pread covers.
-- ---------------------------------------------------------------------------

/-- SOUNDNESS lemma: if the production criterion accepts, the owned locals
    are contained in the pread window. -/
theorem singleRun_elem_in_window (locals : List Nat) (h : isSingleRun locals = true) :
    ∀ n, n ∈ locals →
      (sortNats locals).head! ≤ n ∧ n < (sortNats locals).head! + locals.length := by
  have hsorted : SortedLe (sortNats locals) = true := sortNats_sorted locals
  have hne : locals ≠ [] := by
    intro h0; rw [h0] at h; simp [isSingleRun] at h
  have hlen : (sortNats locals).length = locals.length := sortNats_len locals
  have hcrit : lastDef (sortNats locals).tail (sortNats locals).head! - (sortNats locals).head! + 1 = locals.length := by
    cases locals with
    | nil => contradiction
    | cons x xs =>
      cases hsort : sortNats (x :: xs) with
      | nil =>
        have hl : (x :: xs).length = 0 := by rw [← hlen, hsort]; rfl
        simp at hl
      | cons l0 rest =>
        have hdef : isSingleRun (x :: xs) = (lastDef rest l0 - l0 + 1 == (x :: xs).length) := by
          simp [isSingleRun, hsort]
        rw [hdef] at h
        simp at h ⊢
        <;> exact h
  intro n hn
  have nin : n ∈ sortNats locals := (sortNats_mem_iff n locals).mpr hn
  have h1 : (sortNats locals).head! ≤ n := sorted_head_le (sortNats locals) hsorted n nin
  have h2 : n ≤ lastDef (sortNats locals) (sortNats locals).head! :=
    sorted_le_last (sortNats locals) (sortNats locals).head! hsorted n nin
  have h3 : lastDef (sortNats locals) (sortNats locals).head! = lastDef (sortNats locals).tail (sortNats locals).head! := by
    cases hsort : sortNats locals with
    | nil =>
      have hl : locals.length = 0 := by rw [← hlen, hsort]; rfl
      have : locals = [] := by simpa using hl
      contradiction
    | cons l0 rest =>
      cases rest with
      | nil => rfl
      | cons _ _ => rfl
  rw [h3] at h2
  omega

-- ---------------------------------------------------------------------------
-- Guard triangulation instance (audit fixture): shape (2,4), the column shard
-- owns global flat elements {0,1,4,5} = left two columns of both rows.
-- In the packed chunk order these locals are 0,1,4,5: gap between 1 and 4,
-- so the production criterion REJECTS this slot (returns None).
-- ---------------------------------------------------------------------------

/-- Column shard (2,4): owns global 0,1,4,5 -> locals 0,1,4,5 in full chunk. -/
example : isSingleRun [0, 1, 4, 5] = false := by decide

/-- The owned locals are distinct, yet the criterion rejects them. -/
example : [0, 1, 4, 5].Nodup := by decide

/-- Row-contiguous sub-run of the second row: locals 2,3 in the left-half
    chunk (global [4,6) -> chunk-local [2,4)); criterion ACCEPTS. -/
example : isSingleRun [2, 3] = true := by decide

/-- The accepted run is exactly the window [2,4): soundness application. -/
example :
    ∀ n ∈ ([2, 3] : List Nat),
      (sortNats [2, 3]).head! ≤ n ∧ n < (sortNats [2, 3]).head! + 2 := by
  intro n hn
  have h := singleRun_elem_in_window [2, 3] (by decide)
  exact h n ((sortNats_mem_iff n [2, 3]).mpr hn)
