/-
Group leader = minimum row index in full table order.

Mirrors ``gdtensor_sim/models.py`` layout grouping (frozen source):

    for t, nd in enumerate(table.named_d_tensors):     # t = ABSOLUTE row index
        ...
        groups[path].append((t, rank))
    for path in sorted(groups):
        cells = groups[path]
        leader = leader_tensor_component_pos_of(lst)  # canonical leader rank
        t0 = min(t for t, _ in cells)                # leader ROW in table order
        file_paths[TensorComponentPos(ids[t0], leader.rank)] = path

Facts pinned here:
  * the leader row ``t0`` is the MINIMUM row index among the group's cells
    (NOT a maximum / most-member guess);
  * the entry is looked up by VALUE (the group's list of (row, rank) pairs):
    two equal cell lists -- even permutations of each other, which are the same
    group -- compute the same leader row. No object identity is involved.

Only Lean core Init; no sorry/admit/axiom.
-/
import Init.Tactics

/-- Componentwise minimum of two rows. -/
def minNat (a b : Nat) : Nat := if a ≤ b then a else b

/-- Head-init fold: minimum over ``acc`` and every row in ``rest``. -/
def foldMin : Nat → List (Nat × Nat) → Nat
  | acc, [] => acc
  | acc, (t, _) :: rest => foldMin (minNat acc t) rest

/-- Leader row = minimum row index among the group's cells.
    Empty groups never occur in production (groups are built from cells). -/
def leaderRow : List (Nat × Nat) → Nat
  | [] => 0
  | (t, _) :: rest => foldMin t rest

/-- foldMin acc rest is ≤ acc and ≤ every row in rest (it is the minimum). -/
private theorem foldMin_le : ∀ acc rest,
    foldMin acc rest ≤ acc ∧ ∀ c : Nat × Nat, c ∈ rest → foldMin acc rest ≤ c.1
  | acc, [] => by
    simp [foldMin]
  | acc, (t, _) :: rest => by
    simp only [foldMin]
    have ih := foldMin_le (minNat acc t) rest
    have hmin_le_acc : minNat acc t ≤ acc := by
      unfold minNat; split <;> omega
    have hmin_le_t : minNat acc t ≤ t := by
      unfold minNat; split <;> omega
    constructor
    · have h1 : foldMin (minNat acc t) rest ≤ minNat acc t := ih.1
      exact Nat.le_trans h1 hmin_le_acc
    · intro c hc
      have h3 : foldMin (minNat acc t) rest ≤ c.1 := by
        simp only [List.mem_cons] at hc
        rcases hc with rfl | hrest
        · have h4 : foldMin (minNat acc t) rest ≤ minNat acc t := ih.1
          exact Nat.le_trans h4 hmin_le_t
        · exact ih.2 c hrest
      exact h3

/-- foldMin acc rest is achieved by some participant row (acc or a rest row). -/
private theorem foldMin_achieves : ∀ acc rest,
    ∃ r : Nat, (r = acc ∨ ∃ c : Nat × Nat, c ∈ rest ∧ r = c.1) ∧
      foldMin acc rest = r
  | acc, [] => by
    refine ⟨acc, Or.inl rfl, by simp [foldMin]⟩
  | acc, (t, s) :: rest => by
    simp only [foldMin]
    rcases foldMin_achieves (minNat acc t) rest with ⟨r, hr, heq⟩
    refine ⟨r, ?_, heq⟩
    rcases hr with hr | ⟨c, hc, rq⟩
    · have hmin : minNat acc t = acc ∨ minNat acc t = t := by
        unfold minNat; split <;> simp
      rcases hmin with hmin | hmin
      · left; rw [hr, hmin]
      · right; refine ⟨(t, s), by simp, ?_⟩; rw [hr, hmin]
    · right
      exact ⟨c, by simp [hc], rq⟩

/-- The leader row never exceeds any cell's row. -/
theorem leaderRow_le : ∀ cells t w, (t, w) ∈ cells → leaderRow cells ≤ t := by
  intro cells t w h
  induction cells with
  | nil => simp at h
  | cons cell rest ih =>
    simp only [List.mem_cons] at h
    dsimp only [leaderRow]
    rcases h with heq | hrest
    · have heqt : t = cell.1 := congrArg Prod.fst heq
      rw [heqt]
      exact (foldMin_le cell.1 rest).1
    · exact (foldMin_le cell.1 rest).2 (t, w) hrest

/-- On a nonempty group the leader row is actually achieved by some cell. -/
theorem leaderRow_mem : ∀ cells, cells ≠ [] →
    ∃ t w, (t, w) ∈ cells ∧ t = leaderRow cells := by
  intro cells hne
  cases cells with
  | nil => contradiction
  | cons cell rest =>
    dsimp only [leaderRow]
    rcases foldMin_achieves cell.1 rest with ⟨r, hr, heq⟩
    rcases hr with hr | ⟨c, hc, rq⟩
    · refine ⟨cell.1, cell.2, by simp, ?_⟩
      rw [heq, hr]
    · refine ⟨c.1, c.2, by simp [hc], ?_⟩
      rw [heq, rq]

-- ---------------------------------------------------------------------------
-- Guard triangulation fixtures:
--   cells [(0,0),(1,1)]       -> leader row 0  (min, not max)
--   cells [(5,0),(2,0),(7,0)] -> leader row 2 (not max-member 7)
-- ---------------------------------------------------------------------------

example : leaderRow [(0, 0), (1, 1)] = 0 := by decide
example : leaderRow [(5, 0), (2, 0), (7, 0)] = 2 := by decide

/-- Value equality: the same group expressed in another cell order yields the
    same leader row (entry lookup by value, not by object identity). -/
example : leaderRow [(1, 0), (0, 1)] = leaderRow [(0, 1), (1, 0)] := by decide

/-- The minimum-row property on the fixture: leader row 2 is ≤ every row. -/
example : leaderRow [(5, 0), (2, 0), (7, 0)] ≤ 2 ∧
    leaderRow [(5, 0), (2, 0), (7, 0)] ≤ 5 ∧
    leaderRow [(5, 0), (2, 0), (7, 0)] ≤ 7 := by
  have h := leaderRow_le [(5, 0), (2, 0), (7, 0)] 2 0 (by simp)
  have h2 := leaderRow_le [(5, 0), (2, 0), (7, 0)] 5 0 (by simp)
  have h3 := leaderRow_le [(5, 0), (2, 0), (7, 0)] 7 0 (by simp)
  exact ⟨h, h2, h3⟩

/-- The leader row is achieved by the (2,0) cell itself. -/
example : ∃ t w, (t, w) ∈ [(5, 0), (2, 0), (7, 0)] ∧
    t = leaderRow [(5, 0), (2, 0), (7, 0)] := by
  have h := leaderRow_mem [(5, 0), (2, 0), (7, 0)] (by simp)
  rcases h with ⟨t, w, hm, eq⟩
  exact ⟨t, w, hm, eq⟩

-- ---------------------------------------------------------------------------
-- Lexicographic leader cell: production t0 = min row, then min rank among
-- cells on that row (TensorComponentPos(ids[t0], leader.rank)). We only pin
-- the ROW projection by concrete decide here; the rank tie-break universal
-- lemma is NOT proved (see README boundary).
-- ---------------------------------------------------------------------------

/-- Lex-min (row, rank) head-init fold. -/
def lexMinCell : Nat × Nat → List (Nat × Nat) → Nat × Nat
  | acc, [] => acc
  | (ar, aw), (tr, tw) :: rest =>
    if tr < ar ∨ (tr = ar ∧ tw < aw)
      then lexMinCell (tr, tw) rest
      else lexMinCell (ar, aw) rest

/-- The production leader cell = lex-min (row, rank) of the group's cells. -/
def leaderCell : List (Nat × Nat) → Nat × Nat
  | [] => (0, 0)
  | c :: rest => lexMinCell c rest

/-- Concrete: leaderCell row projection equals leaderRow on the audit fixture. -/
example : (leaderCell [(5, 0), (2, 0), (7, 0)]).1 = 2 := by decide

/-- Concrete: on a row-tie, min rank wins (not max rank). -/
example : leaderCell [(2, 5), (2, 3)] = (2, 3) := by decide

