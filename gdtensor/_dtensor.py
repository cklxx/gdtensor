"""DTensor entities (DTensorsView B.2.3.3 - B.2.3.10) and ownership math.

Ownership model (lite):
* A participant (compute rank or storage column) owns a box of the *naive*
  global shape. ``Shard(axis)`` splits that axis with torch.chunk rules
  (chunk = ceil(n / parts)); several mesh dims may shard one axis, nested in
  mesh order. ``Replicate`` keeps the full extent. ``Partial`` is not
  persistable.
* A box maps to sorted, disjoint element intervals of the flat row-major
  tensor. Row-major order of a box equals ascending global order, so a local
  buffer offset is a prefix sum over these intervals.
* ``ReshapedDTensor`` keeps element order. Ownership is computed on the naive
  shape; the reshaped ``shape`` is the logical (global meta) shape.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np

from ._mesh import (
    ComputeMesh,
    StorageMesh,
    mesh_coords,
    resolve_compute_mesh,
    resolve_storage_mesh,
)
from ._result import Result, ResultError, catch
from ._constraints import check_constraints as _cc
from ._builder_mixin import attach_builders as _attach


def _check(self):
    return _cc(self)

MeshT = TypeVar("MeshT")

# ---------------------------------------------------------------- B.2.3.6


@dataclass(frozen=True)
class GlobalTensorMeta:
    global_shape: Tuple[int, ...]
    dtype: str
    itemsize: int

    def check_constraints(self):
        return _check(self)

    @property
    def numel(self) -> int:
        return int(math.prod(self.global_shape))

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize


# ---------------------------------------------------------------- B.2.3.7


from .tag import Partial, Replicate, Shard

Placement = Union[Shard, Replicate, Partial]

# ---------------------------------------------------------------- B.2.3.5 / B.2.3.4


@dataclass(frozen=True)
class NaiveDTensor(Generic[MeshT]):
    global_tensor_meta: GlobalTensorMeta
    placements: Tuple[Placement, ...]
    mesh: MeshT
    # B.2.3.5: underlying torch.DTensor for compute-side processing; None
    # means metadata-only. Storage-side tensors keep it None.
    torch_d_tensor: object = None


@dataclass(frozen=True)
class ReshapedDTensor(Generic[MeshT]):
    """$reshaped (NaiveDTensor * $shape): element-order-preserving reshape."""

    naive: NaiveDTensor
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        if math.prod(self.shape) != self.naive.global_tensor_meta.numel:
            raise ResultError(
                f"reshape {self.naive.global_tensor_meta.global_shape} -> {self.shape} changes numel"
            )


GenericDTensor = Union[NaiveDTensor, ReshapedDTensor]

# ---------------------------------------------------------------- B.2.3.3 / B.2.3.8


@dataclass(frozen=True)
class NamedDTensor(Generic[MeshT]):
    global_tensor_id: str
    d_tensor: GenericDTensor
    alias: Tuple[str, ...] = ()


@dataclass
class DTensorTable(Generic[MeshT]):
    named_d_tensors: List[NamedDTensor] = field(default_factory=list)
    _order: Dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._order = {}
        for i, nd in enumerate(self.named_d_tensors):
            if nd.global_tensor_id in self._order:
                raise ResultError(f"duplicate tensor id {nd.global_tensor_id}")
            self._order[nd.global_tensor_id] = i

    def check_constraints(self):
        return _check(self)

    @property
    def tensor_ids(self) -> List[str]:
        return [nd.global_tensor_id for nd in self.named_d_tensors]


# ---------------------------------------------------------------- B.2.3.9 / B.2.3.10


def naive_of(nd: NamedDTensor) -> NaiveDTensor:
    dt = nd.d_tensor
    return dt.naive if isinstance(dt, ReshapedDTensor) else dt


def global_tensor_meta_of(named_d_tensor: NamedDTensor) -> GlobalTensorMeta:
    """Naive and reshaped branches map to one global meta (logical shape)."""
    dt = named_d_tensor.d_tensor
    if isinstance(dt, ReshapedDTensor):
        return replace(dt.naive.global_tensor_meta, global_shape=tuple(dt.shape))
    return dt.global_tensor_meta


@catch
def tensor_order_value_of(d_tensor_table: DTensorTable, tensor_id: str) -> int:
    try:
        return d_tensor_table._order[tensor_id]
    except KeyError:
        raise ResultError(f"tensor {tensor_id!r} not in DTensorTable") from None


def order_of(table: DTensorTable, tensor_id: str) -> int:
    return tensor_order_value_of(table, tensor_id).unwrap()


def named_of(table: DTensorTable, tensor_id: str) -> NamedDTensor:
    return table.named_d_tensors[order_of(table, tensor_id)]


# ---------------------------------------------------------------- intervals

EMPTY = np.zeros((0, 2), dtype=np.int64)


def iv_total(iv: np.ndarray) -> int:
    return int((iv[:, 1] - iv[:, 0]).sum()) if iv.size else 0


def iv_normalize(iv: np.ndarray) -> np.ndarray:
    """Sort, drop empties, merge adjacent / overlapping intervals."""
    if iv.size == 0:
        return EMPTY
    iv = iv[iv[:, 1] > iv[:, 0]]
    if iv.size == 0:
        return EMPTY
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    starts, ends = iv[:, 0], np.maximum.accumulate(iv[:, 1])
    new = np.ones(len(iv), dtype=bool)
    new[1:] = starts[1:] > ends[:-1]
    grp = np.cumsum(new) - 1
    out = np.empty((int(grp[-1]) + 1, 2), dtype=np.int64)
    out[:, 0] = starts[new]
    out[:, 1] = np.maximum.reduceat(ends, np.flatnonzero(new))
    return out


def iv_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection of two normalized interval lists (vectorized sweep)."""
    if a.size == 0 or b.size == 0:
        return EMPTY
    # For each a-interval, candidate b-intervals overlapping it.
    lo = np.searchsorted(b[:, 1], a[:, 0], side="right")
    hi = np.searchsorted(b[:, 0], a[:, 1], side="left")
    counts = np.maximum(hi - lo, 0)
    if counts.sum() == 0:
        return EMPTY
    ai = np.repeat(np.arange(len(a)), counts)
    offs = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    bi = np.repeat(lo, counts) + offs
    s = np.maximum(a[ai, 0], b[bi, 0])
    e = np.minimum(a[ai, 1], b[bi, 1])
    keep = e > s
    return np.stack([s[keep], e[keep]], axis=1).astype(np.int64)


def iv_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return a
    out: List[Tuple[int, int]] = []
    j = 0
    for s, e in a.tolist():
        cur = s
        while j < len(b) and b[j, 1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k, 0] < e:
            bs, be = int(b[k, 0]), int(b[k, 1])
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def iv_prefix(iv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Local offset (in units of iv) of each global point; each point must lie in iv."""
    lens = iv[:, 1] - iv[:, 0]
    cum = np.concatenate([[0], np.cumsum(lens)])
    k = np.searchsorted(iv[:, 0], points, side="right") - 1
    if np.any(k < 0) or np.any(points >= iv[np.maximum(k, 0), 1]):
        raise ResultError("point outside owned intervals")
    return cum[k] + (points - iv[k, 0])


def box_to_intervals(shape: Sequence[int], lo: Sequence[int], hi: Sequence[int]) -> np.ndarray:
    """Flat row-major element intervals covered by box [lo, hi)."""
    nd = len(shape)
    if nd == 0:
        return np.array([[0, 1]], dtype=np.int64)
    if any(h <= l for l, h in zip(lo, hi)):
        return EMPTY
    strides = [1] * nd
    for d in range(nd - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    k = nd - 1
    while k > 0 and lo[k] == 0 and hi[k] == shape[k]:
        k -= 1
    run = (hi[k] - lo[k]) * strides[k]
    base = lo[k] * strides[k]
    if k == 0:
        return np.array([[base, base + run]], dtype=np.int64)
    grids = np.meshgrid(*[np.arange(lo[d], hi[d], dtype=np.int64) for d in range(k)], indexing="ij")
    starts = base + sum(g.ravel() * strides[d] for d, g in enumerate(grids))
    return np.stack([starts, starts + run], axis=1).astype(np.int64)


# ---------------------------------------------------------------- ownership


def _chunk(n: int, parts: int, idx: int) -> Tuple[int, int]:
    c = -(-n // parts) if n else 0
    s = min(idx * c, n)
    return s, min(s + c, n)


def box_of(naive: NaiveDTensor, mesh_shape: Tuple[int, ...], coords: Tuple[int, ...]) -> Tuple[List[int], List[int]]:
    shape = list(naive.global_tensor_meta.global_shape)
    if len(naive.placements) != len(mesh_shape):
        raise ResultError(f"{len(naive.placements)} placements for mesh ndim {len(mesh_shape)}")
    lo, hi = [0] * len(shape), list(shape)
    for d, p in enumerate(naive.placements):
        if isinstance(p, Shard):
            if not 0 <= p.axis < len(shape):
                raise ResultError(f"Shard axis {p.axis} out of range for shape {shape}")
            s, e = _chunk(hi[p.axis] - lo[p.axis], mesh_shape[d], coords[d])
            lo[p.axis], hi[p.axis] = lo[p.axis] + s, lo[p.axis] + e
        elif isinstance(p, Partial):
            raise ResultError("Partial placement is not persistable")
    return lo, hi


def resolve_mesh(nd: NamedDTensor, side: str) -> np.ndarray:
    """Participant ndarray: compute ranks, or storage columns (row t=0 only).

    Empty literal cells (-1) stay -1; mesh_coords treats them as unowned.
    """
    naive = naive_of(nd)
    if side == "compute":
        return resolve_compute_mesh(naive.mesh)
    cells, w_size = resolve_storage_mesh(naive.mesh)
    valid = cells >= 0
    if np.any(valid & (cells // w_size != 0)):
        raise ResultError(f"{nd.global_tensor_id}: SingleTensorStorageMesh covers tensor row != 0")
    out = cells % w_size
    out[~valid] = -1
    return out


def participants_of(nd: NamedDTensor, side: str) -> List[int]:
    return sorted(int(x) for x in resolve_mesh(nd, side).ravel() if x >= 0)


def owned_box(nd: NamedDTensor, side: str, participant: int) -> Optional[Tuple[List[int], List[int]]]:
    """Box [lo, hi) of the naive shape owned by a participant, or None."""
    naive = naive_of(nd)
    cells = resolve_mesh(nd, side)
    coords = mesh_coords(cells, participant)
    if coords is None:
        return None
    return box_of(naive, cells.shape, coords)


def owned_intervals(nd: NamedDTensor, side: str, participant: int) -> np.ndarray:
    """Element intervals owned by a participant; EMPTY if not on the mesh."""
    box = owned_box(nd, side, participant)
    if box is None:
        return EMPTY
    return box_to_intervals(naive_of(nd).global_tensor_meta.global_shape, *box)


def participants_of(nd: NamedDTensor, side: str) -> List[int]:
    return sorted(int(x) for x in resolve_mesh(nd, side).ravel())


def is_designated_writer(nd: NamedDTensor, rank: int, tensor_order: int) -> bool:
    """Dedup rule for replicated compute dims: one coordinate per replicate dim
    writes, rotated by tensor order to balance writes across replicas."""
    naive = naive_of(nd)
    cells = resolve_mesh(nd, "compute")
    coords = mesh_coords(cells, rank)
    if coords is None:
        return False
    for d, p in enumerate(naive.placements):
        if isinstance(p, Partial):
            raise ResultError(f"{nd.global_tensor_id}: Partial placement is not persistable")
        if isinstance(p, Replicate) and cells.shape[d] > 1:
            if coords[d] != tensor_order % cells.shape[d]:
                return False
    return True


# C.1 builder registration
from ._builder_mixin import attach_builders as _ab
for _c in (GlobalTensorMeta, NaiveDTensor, ReshapedDTensor, NamedDTensor, DTensorTable):
    _ab(_c)
