"""DTensor entities (design B.2.3.3 - B.2.3.10) and ownership math (internal).

Ownership model:
* A participant (compute rank or storage column) owns a box of the naive
  global shape. ``Shard(axis)`` splits that axis with torch.chunk rules
  (chunk = ceil(n / parts)); several mesh dims may shard one axis, nested in
  mesh order. ``Replicate`` keeps the full extent. ``Partial`` is not
  persistable.
* A box maps to sorted, disjoint element intervals of the flat row-major
  tensor. Row-major order of a box equals ascending global order, so a local
  buffer offset is a prefix sum over these intervals.
* ``ReshapedDTensor`` keeps element order. Ownership is computed on the
  underlying naive shape; ``shape`` is the logical (global meta) shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import GenericAlias
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .builder_mixin import field_hash, make_builder_mixin
from .mesh import mesh_coords, resolve_compute_mesh, resolve_storage_mesh
from .result import Fail, catch, invalid, not_found
from .products import Partial, Replicate, Shard


def _cc(self):
    from .constraints import check_constraints

    return check_constraints(self)


Placement = Shard | Replicate | Partial

GlobalTensorMetaBuilderMixin = make_builder_mixin(
    "GlobalTensorMeta",
    [("global_shape", "list"), ("dtype", "scalar"), ("itemsize", "scalar")],
)
NaiveDTensorBuilderMixin = make_builder_mixin(
    "NaiveDTensor",
    [
        ("global_tensor_meta", "object"),
        ("placements", "list"),
        ("mesh", "oneof"),
        ("torch_d_tensor", "nullable"),
    ],
)
ReshapedDTensorBuilderMixin = make_builder_mixin(
    "ReshapedDTensor", [("underlying_d_tensor", "object"), ("shape", "list")]
)
NamedDTensorBuilderMixin = make_builder_mixin(
    "NamedDTensor",
    [("global_tensor_id", "scalar"), ("d_tensor", "oneof"), ("alias", "list")],
)
DTensorTableBuilderMixin = make_builder_mixin(
    "DTensorTable", [("named_d_tensors", "list")]
)


# ---------------------------------------------------------------- B.2.3.6


@dataclass(eq=True)
class GlobalTensorMeta(GlobalTensorMetaBuilderMixin):
    global_shape: List[int] = field(default_factory=list)
    dtype: str = ""
    itemsize: int = 0
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.global_shape = list(self.global_shape)

    def check_constraints(self):
        return _cc(self)


def numel_of(meta: GlobalTensorMeta) -> int:
    return int(math.prod(meta.global_shape))


def nbytes_of(meta: GlobalTensorMeta) -> int:
    return numel_of(meta) * meta.itemsize


# ---------------------------------------------------------------- B.2.3.5 / B.2.3.4


@dataclass(eq=False)
class NaiveDTensor(NaiveDTensorBuilderMixin):
    global_tensor_meta: GlobalTensorMeta = field(default_factory=GlobalTensorMeta)
    placements: List[Any] = field(default_factory=list)
    mesh: Any = None
    # torch.DTensor for compute-side processing; None means metadata-only.
    torch_d_tensor: Any = None

    def __eq__(self, other):
        if type(other) is not NaiveDTensor:
            return NotImplemented
        # Comparing opaque torch handles must not execute tensor operations.
        return (
            self.torch_d_tensor is other.torch_d_tensor
            and self.global_tensor_meta == other.global_tensor_meta
            and self.placements == other.placements
            and self.mesh == other.mesh
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.global_tensor_meta,
                tuple(self.placements),
                self.mesh,
                id(self.torch_d_tensor),
            )
        )

    def __post_init__(self) -> None:
        self.placements = list(self.placements)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class ReshapedDTensor(ReshapedDTensorBuilderMixin):
    underlying_d_tensor: NaiveDTensor = field(default_factory=NaiveDTensor)
    shape: List[int] = field(default_factory=list)
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.shape = list(self.shape)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


GenericDTensor = NaiveDTensor | ReshapedDTensor


# ---------------------------------------------------------------- B.2.3.3 / B.2.3.8


@dataclass(eq=True)
class NamedDTensor(NamedDTensorBuilderMixin):
    global_tensor_id: str = ""
    d_tensor: Any = None
    alias: List[str] = field(default_factory=list)
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.alias = list(self.alias)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


@dataclass(eq=True)
class DTensorTable(DTensorTableBuilderMixin):
    named_d_tensors: List[NamedDTensor] = field(default_factory=list)
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.named_d_tensors = list(self.named_d_tensors)

    def __class_getitem__(cls, item):
        return GenericAlias(cls, item)

    def check_constraints(self):
        return _cc(self)


def tensor_ids_of(table: DTensorTable) -> List[str]:
    return [nd.global_tensor_id for nd in table.named_d_tensors]


# ---------------------------------------------------------------- B.2.3.9 / B.2.3.10


def naive_of(nd: NamedDTensor) -> NaiveDTensor:
    dt = nd.d_tensor
    return dt.underlying_d_tensor if type(dt) is ReshapedDTensor else dt


def global_tensor_meta_of(named_d_tensor: NamedDTensor) -> GlobalTensorMeta:
    """Naive and reshaped branches map to one global meta (logical shape)."""
    dt = named_d_tensor.d_tensor
    if type(dt) is ReshapedDTensor:
        m = dt.underlying_d_tensor.global_tensor_meta
        return GlobalTensorMeta(list(dt.shape), m.dtype, m.itemsize)
    return dt.global_tensor_meta


@catch
def tensor_order_value_of(d_tensor_table: DTensorTable, tensor_id: str) -> int:
    for i, nd in enumerate(d_tensor_table.named_d_tensors):
        if nd.global_tensor_id == tensor_id:
            return i
    raise not_found("tensor_id", tensor_id, f"tensor {tensor_id!r} not in DTensorTable")


def order_of(table: DTensorTable, tensor_id: str) -> int:
    r = tensor_order_value_of(table, tensor_id)
    if type(r).__name__ != "Ok":
        raise Fail(r.value)
    return r.value


def named_of(table: DTensorTable, tensor_id: str) -> NamedDTensor:
    return table.named_d_tensors[order_of(table, tensor_id)]


# ---------------------------------------------------------------- intervals (internal)

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


def _subtract_interval(start, end, b, index):
    out = []
    current = start
    while index < len(b) and b[index, 0] < end:
        begin, stop = map(int, b[index])
        if begin > current:
            out.append((current, begin))
        current = max(current, stop)
        if current >= end:
            break
        index += 1
    if current < end:
        out.append((current, end))
    return out


def iv_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return a
    out = []
    index = 0
    for start, end in a.tolist():
        while index < len(b) and b[index, 1] <= start:
            index += 1
        out.extend(_subtract_interval(start, end, b, index))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def iv_prefix(iv: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Local offset (in units of iv) of each global point; each point must lie in iv."""
    lens = iv[:, 1] - iv[:, 0]
    cum = np.concatenate([[0], np.cumsum(lens)])
    k = np.searchsorted(iv[:, 0], points, side="right") - 1
    if np.any(k < 0) or np.any(points >= iv[np.maximum(k, 0), 1]):
        raise invalid("range", "point outside owned intervals")
    return cum[k] + (points - iv[k, 0])


def box_to_intervals(
    shape: Sequence[int], lo: Sequence[int], hi: Sequence[int]
) -> np.ndarray:
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
    grids = np.meshgrid(
        *[np.arange(lo[d], hi[d], dtype=np.int64) for d in range(k)], indexing="ij"
    )
    starts = base + sum(g.ravel() * strides[d] for d, g in enumerate(grids))
    return np.stack([starts, starts + run], axis=1).astype(np.int64)


# ---------------------------------------------------------------- ownership (internal)


def _chunk(n: int, parts: int, idx: int) -> Tuple[int, int]:
    c = -(-n // parts) if n else 0
    s = min(idx * c, n)
    return s, min(s + c, n)


def box_of(
    naive: NaiveDTensor, mesh_shape: Tuple[int, ...], coords: Tuple[int, ...]
) -> Tuple[List[int], List[int]]:
    shape = list(naive.global_tensor_meta.global_shape)
    if len(naive.placements) != len(mesh_shape):
        raise invalid(
            "placements",
            f"{len(naive.placements)} placements for mesh ndim {len(mesh_shape)}",
        )
    lo, hi = [0] * len(shape), list(shape)
    for d, p in enumerate(naive.placements):
        if type(p) is Partial:
            raise invalid("placements", "Partial placement is not persistable")
        if type(p) is not Shard:
            continue
        axis = p.axis
        if not 0 <= axis < len(shape):
            raise invalid(
                "placements", f"Shard axis {axis} out of range for shape {shape}"
            )
        start, end = _chunk(hi[axis] - lo[axis], mesh_shape[d], coords[d])
        lo[axis], hi[axis] = lo[axis] + start, lo[axis] + end
    return lo, hi


def resolve_mesh(nd: NamedDTensor, side: str) -> np.ndarray:
    """Participant ndarray: compute ranks, or storage columns (row t=0 only).

    Cells a LiteralMesh does not select stay -1; mesh_coords treats them as unowned.
    """
    naive = naive_of(nd)
    if side == "compute":
        return resolve_compute_mesh(naive.mesh)
    cells, w_size = resolve_storage_mesh(naive.mesh)
    valid = cells >= 0
    if np.any(valid & (cells // w_size != 0)):
        raise invalid(
            "mesh",
            f"{nd.global_tensor_id}: SingleTensorStorageMesh covers tensor row != 0",
        )
    out = cells % w_size
    out[~valid] = -1
    return out


def participants_of(nd: NamedDTensor, side: str) -> List[int]:
    return sorted(int(x) for x in resolve_mesh(nd, side).ravel() if x >= 0)


def owned_box(
    nd: NamedDTensor, side: str, participant: int
) -> Optional[Tuple[List[int], List[int]]]:
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


def is_designated_writer(nd: NamedDTensor, rank: int, tensor_order: int) -> bool:
    """Dedup rule for replicated compute dims: one coordinate per replicate dim
    writes, rotated by tensor order to balance writes across replicas."""
    naive = naive_of(nd)
    cells = resolve_mesh(nd, "compute")
    coords = mesh_coords(cells, rank)
    if coords is None:
        return False
    for d, p in enumerate(naive.placements):
        if type(p) is Partial:
            raise invalid(
                "placements",
                f"{nd.global_tensor_id}: Partial placement is not persistable",
            )
        if (
            type(p) is Replicate
            and cells.shape[d] > 1
            and coords[d] != tensor_order % cells.shape[d]
        ):
            return False
    return True
