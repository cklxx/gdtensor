/-
Formal proof of the gdtensor shard-placement algebra (design 5.3, B.1).

For a global tensor of N elements split over S ranks, we prove:
  1. the per-rank sizes (quotient, plus one extra element for the first N % S
     ranks) sum to N;
  2. the owned intervals are adjacent: start (r + 1) = start r + size r;
  3. every global n < N is reached by exactly one (rank, local) pair:
     existence (surjectivity) and uniqueness (injectivity).
This is the disjoint-covering property of the placement, stated on the
address map. Only the Lean core Init library is used.
-/
import Init.Tactics
import Init.Data.Nat.Div
import Init.Data.Nat.Lemmas

open Nat

/-- Size of rank r: quotient plus one extra for the first N % S ranks. -/
def shardSize (N S r : Nat) : Nat :=
  N / S + if r < N % S then 1 else 0

/-- Start offset of rank r: sum of the sizes of the ranks before it. -/
def shardStart (N S : Nat) : Nat → Nat
  | 0 => 0
  | r + 1 => shardStart N S r + shardSize N S r

theorem shardStart_succ (N S r : Nat) :
    shardStart N S (r + 1) = shardStart N S r + shardSize N S r := by
  rfl

/-- start r ≤ start (r + 1): the next interval starts no earlier. -/
theorem shardStart_succ_le (N S r : Nat) :
    shardStart N S r ≤ shardStart N S (r + 1) := by
  rw [shardStart_succ N S]; exact Nat.le_add_right _ _

/-- Start offsets are monotone: a ≤ b → start a ≤ start b. -/
theorem shardStart_monotone (N S : Nat) {a b : Nat} (h : a ≤ b) :
    shardStart N S a ≤ shardStart N S b := by
  induction h with
  | refl => exact Nat.le_refl _
  | step h' ih => exact Nat.le_trans ih (shardStart_succ_le N S _)

/-- Sum of the sizes of the first r ranks equals the start of rank r. -/
theorem sum_size_eq_start (N S : Nat) : ∀ r : Nat,
    List.sum (List.map (shardSize N S) (List.range r)) = shardStart N S r
  | 0 => by rfl
  | r + 1 => by
    rw [List.range_succ, List.map_append, List.map_singleton,
        List.sum_append, List.sum_singleton,
        sum_size_eq_start N S r, shardStart_succ N S]

/-- Filtering range S by (_ < k) gives range k when k ≤ S. -/
private theorem filter_lt_range : ∀ (S k : Nat), k ≤ S →
    (List.range S).filter (fun r => r < k) = List.range k
  | 0, k, hk => by
    have hk0 : k = 0 := by omega
    rw [hk0]; rfl
  | S + 1, k, hk => by
    rw [List.range_succ, List.filter_append]
    by_cases hks : k ≤ S
    · rw [filter_lt_range S k hks]
      have h : List.filter (fun r => r < k) [S] = [] := by
        simp [Nat.not_lt.mpr (by omega)]
      rw [h, List.append_nil]
    · have hk1 : k = S + 1 := by omega
      rw [hk1]
      have h1 : List.filter (fun r => r < S + 1) (List.range S) = List.range S :=
        List.filter_eq_self.mpr
          (fun r hr => decide_eq_true
            (Nat.lt_succ_of_lt (List.mem_range.mp hr)))
      have h2 : List.filter (fun r => r < S + 1) [S] = [S] := by
        simp [Nat.lt_succ_self S]
      rw [h1, h2, List.range_succ]

theorem extra_count (N S : Nat) (hS : 0 < S) :
    (List.range S).filter (fun r => r < N % S) = List.range (N % S) :=
  filter_lt_range S (N % S) (Nat.le_of_lt (mod_lt N hS))

/-- Sum over a list of ranks: each rank contributes quotient plus 0/1. -/
private theorem sum_formula (N S : Nat) (r : List Nat) :
    List.sum (List.map (shardSize N S) r) =
      (N / S) * r.length + (r.filter (fun q => q < N % S)).length := by
  induction r with
  | nil => rfl
  | cons a r ih =>
    rw [List.map_cons, List.sum_cons, ih, List.length_cons, List.filter_cons]
    rw [Nat.mul_succ]
    by_cases h : a < N % S
    · have hsize : shardSize N S a = N / S + 1 := by
        unfold shardSize; simp [h]
      rw [hsize]; simp [h]; omega
    · have hsize : shardSize N S a = N / S := by
        unfold shardSize; simp [h]
      rw [hsize]; simp [h]; omega

/-- The per-rank sizes sum to N: S * (N / S) + (N % S) = N. -/
theorem sum_shardSize (N S : Nat) (hS : 0 < S) :
    List.sum (List.map (shardSize N S) (List.range S)) = N := by
  rw [sum_formula N S, extra_count N S hS]
  simp [List.length_range]
  exact Nat.div_add_mod' N S

/-- The start of rank S is N: all intervals together end at the tensor end. -/
theorem shardStart_S_eq_N (N S : Nat) (hS : 0 < S) :
    shardStart N S S = N := by
  rw [← sum_size_eq_start N S S, sum_shardSize N S hS]

/-- For every rank r < S, the end of its interval does not exceed N. -/
theorem shardEnd_le_N (N S : Nat) (hS : 0 < S) : ∀ r, r < S →
    shardStart N S r + shardSize N S r ≤ N
  | r, hr => by
    have h₁ : r + 1 ≤ S := Nat.succ_le_iff.mpr hr
    have h₂ := shardStart_monotone N S h₁
    rw [shardStart_succ N S] at h₂
    rw [shardStart_S_eq_N N S hS] at h₂
    exact h₂

/-- Existence: every global n < N is reached by some rank r at local offset i.
    Search ranks from 0: n below start (r + 1) but not below start r lies in
    the interval of rank r. -/
theorem shardAddress_exists (N S : Nat) (hS : 0 < S) : ∀ n, n < N →
    ∃ r i, r < S ∧ i < shardSize N S r ∧
      shardStart N S r + i = n
  | n, hn => by
    have hN : shardStart N S S = N := shardStart_S_eq_N N S hS
    have h : ∀ r : Nat, r ≤ S → n < shardStart N S r →
        ∃ r' i, r' < r ∧ i < shardSize N S r' ∧
          shardStart N S r' + i = n := by
      intro r
      induction r with
      | zero =>
        intro _ hc
        simp [shardStart] at hc
      | succ r ih =>
        intro hr hc
        have hsucc : shardStart N S (r + 1) =
            shardStart N S r + shardSize N S r := shardStart_succ N S r
        by_cases hb : n < shardStart N S r
        · rcases ih (by omega) hb with ⟨r', i, hrl, hsize, heq⟩
          exact ⟨r', i, by omega, hsize, heq⟩
        · have hle : shardStart N S r ≤ n := by omega
          have hbelow : n < shardStart N S r + shardSize N S r := by
            rw [← hsucc]; exact hc
          refine ⟨r, n - shardStart N S r, Nat.lt_succ_self r, ?_, ?_⟩
          · omega
          · exact Nat.add_sub_of_le hle
    exact h S (Nat.le_refl S) (by rw [hN]; exact hn)

/-- Uniqueness: two (rank, local) pairs mapping to the same index are equal.
    If r₁ < r₂, the interval of r₁ ends before r₂ starts. -/
theorem shardAddress_unique (N S : Nat) :
    ∀ r₁ i₁ r₂ i₂,
      r₁ < S → i₁ < shardSize N S r₁ →
      r₂ < S → i₂ < shardSize N S r₂ →
      shardStart N S r₁ + i₁ = shardStart N S r₂ + i₂ →
      r₁ = r₂ ∧ i₁ = i₂
  | r₁, i₁, r₂, i₂, hr₁, hi₁, hr₂, hi₂, h => by
    by_cases hlt : r₁ < r₂
    · have hb : shardStart N S r₁ + shardSize N S r₁ ≤ shardStart N S r₂ := by
        have h₁ : r₁ + 1 ≤ r₂ := Nat.succ_le_iff.mpr hlt
        have h₂ := shardStart_monotone N S h₁
        rw [shardStart_succ N S] at h₂
        exact h₂
      have h₄ : shardStart N S r₁ + i₁ < shardStart N S r₁ + shardSize N S r₁ :=
        Nat.add_lt_add_left hi₁ (shardStart N S r₁)
      have hbelow : shardStart N S r₁ + i₁ < shardStart N S r₂ :=
        Nat.lt_of_lt_of_le h₄ hb
      have habove : shardStart N S r₂ ≤ shardStart N S r₁ + i₁ := by
        rw [h]; exact Nat.le_add_right _ _
      exact absurd hbelow (Nat.not_lt.mpr habove)
    · by_cases hlt' : r₂ < r₁
      · have hb : shardStart N S r₂ + shardSize N S r₂ ≤ shardStart N S r₁ := by
          have h₁ : r₂ + 1 ≤ r₁ := Nat.succ_le_iff.mpr hlt'
          have h₂ := shardStart_monotone N S h₁
          rw [shardStart_succ N S] at h₂
          exact h₂
        have h₄ : shardStart N S r₂ + i₂ < shardStart N S r₂ + shardSize N S r₂ :=
          Nat.add_lt_add_left hi₂ (shardStart N S r₂)
        have hbelow : shardStart N S r₂ + i₂ < shardStart N S r₁ :=
          Nat.lt_of_lt_of_le h₄ hb
        have habove : shardStart N S r₁ ≤ shardStart N S r₂ + i₂ := by
          rw [← h]; exact Nat.le_add_right _ _
        exact absurd hbelow (Nat.not_lt.mpr habove)
      · have hr : r₁ = r₂ := by omega
        subst hr
        exact ⟨rfl, Nat.add_left_cancel h⟩
