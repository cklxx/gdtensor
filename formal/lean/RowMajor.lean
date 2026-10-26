/-
Row-major global -> chunk-local coordinate mapping for the DCP storage layout.

Mirrors the real algorithm in ``gdtensor_sim/dcp_backend.py``
``_owned_chunk_local_run`` (frozen source, audit df1), lines 246-260:

    for g in owned_flat:
        rem = g; coords = []
        for i in range(len(gshape)):
            stride = prod(gshape[i+1:])
            coords.append(rem // stride); rem = rem % stride
        rel = [coords[i] - ch_off[i]]
        if rel out of [0, ch_shape[i]): return None
        lf = sum(rel[i] * prod(ch_shape[i+1:]))

Only the Lean core Init library is used; no sorry/admit/axiom.
-/
import Init.Tactics
import Init.Data.Nat.Lemmas

/-- Volume of a row-major shape: product of all dims. Empty shape = 1. -/
def shapeProd : List Nat → Nat
  | [] => 1
  | d :: ds => d * shapeProd ds

/-- Row-major coordinate of global flat index ``g`` in shape ``shape``.
    dim i stride = product of shape[i+1:] (mirrors the Python loop). -/
def coordAt : List Nat → Nat → List Nat
  | [], _ => []
  | _ :: ds, g => (g / shapeProd ds) :: coordAt ds (g % shapeProd ds)

/-- Pack a row-major coordinate back into a flat index (dim i stride =
    product of shape[i+1:]). Requires coord length = shape length. -/
def packedFlat : List Nat → List Nat → Nat
  | [], _ => 0
  | _ :: _, [] => 0
  | _ :: ds, c :: cs => shapeProd ds * c + packedFlat ds cs

/-- Nonzero dims give a strictly positive tensor volume (precondition used
    by the boundedness reasoning). -/
theorem shapeProd_pos : ∀ {ds : List Nat}, (∀ d ∈ ds, 0 < d) → 0 < shapeProd ds
  | [], _ => by decide
  | d :: ds, hp => by
    have hd : 0 < d := hp d (by simp)
    have hr : 0 < shapeProd ds :=
      shapeProd_pos (fun x hx => hp x (by simp [hx]))
    exact Nat.mul_pos hd hr

-- ---------------------------------------------------------------------------
-- Inverse lemma: repacking a decomposition gives back the global index.
-- ---------------------------------------------------------------------------

/-- packedFlat shape (coordAt shape g) = g for nonempty shape. -/
theorem packedFlat_coordAt : ∀ (shape : List Nat), shape ≠ [] → ∀ g : Nat,
    packedFlat shape (coordAt shape g) = g
  | [d], h, g => by
    simp [coordAt, packedFlat, shapeProd, Nat.div_one]
  | d :: e :: es, h, g => by
    simp only [coordAt, packedFlat]
    by_cases hes : es = []
    · subst hes
      have h : e * (g / e) + g % e = g := Nat.div_add_mod g e
      simpa [shapeProd, coordAt, packedFlat, Nat.div_one] using h
    · have hrec : ∀ x, packedFlat es (coordAt es x) = x :=
        packedFlat_coordAt es hes
      rw [hrec (g % shapeProd (e :: es) % shapeProd es)]
      have hinner : shapeProd es * ((g % shapeProd (e :: es)) / shapeProd es) +
          (g % shapeProd (e :: es)) % shapeProd es = g % shapeProd (e :: es) :=
        Nat.div_add_mod _ _
      rw [hinner]
      exact Nat.div_add_mod g (shapeProd (e :: es))

-- ---------------------------------------------------------------------------
-- Chunk-local repacking (mirrors rel = coords - ch_off, then packed by chunk).
-- ---------------------------------------------------------------------------

/-- Componentwise chunk-relative coordinates: coords[i] - ch_off[i]. -/
def relCoord (chOff coord : List Nat) : List Nat :=
  List.zipWith (fun c o => c - o) coord chOff

/-- True iff every rel coordinate lies strictly inside the chunk:
    rel_i < ch_shape_i (rel_i ≥ 0 follows from coords ≥ ch_off in callers). -/
def relWithin (chShape chOff coord : List Nat) : Bool :=
  List.zipWith (fun c o => c - o) coord chOff |>.zip chShape |>.all
    (fun (r, cs) => r < cs)

/-- Chunk-local flat index of a global coordinate (row-major inside chunk). -/
def localFlat (chShape rel : List Nat) : Nat := packedFlat chShape rel

-- ---------------------------------------------------------------------------
-- Guard triangulation instance from the audit (shape (2,4)):
--   gshape = [2,4]  (2 rows, 4 cols)
--   g = 4 -> coords [1,0] ; g = 5 -> coords [1,1]          (row-major split)
--   left-half chunk: ch_off = [0,0], ch_shape = [2,2]
--     rel coords stay [1,0], [1,1]; local flat = 2, 3
--     i.e. global [4,6) repacks to chunk-local [2,4).
-- ---------------------------------------------------------------------------

example : coordAt [2, 4] 4 = [1, 0] := by decide
example : coordAt [2, 4] 5 = [1, 1] := by decide
example : relWithin [2, 2] [0, 0] [1, 0] = true := by decide
example : localFlat [2, 2] [1, 0] = 2 := by decide
example : localFlat [2, 2] [1, 1] = 3 := by decide

/-- The inverse lemma on the triangulation: repacking gives back the index. -/
example : packedFlat [2, 4] (coordAt [2, 4] 4) = 4 := by
  exact packedFlat_coordAt [2, 4] (by simp) 4

/-- Sanity: the full-tensor volume and the second-row base flat offset. -/
example : shapeProd [2, 4] = 8 := by decide
example : packedFlat [2, 4] [1, 0] = 4 := by decide
