"""Shared PP/DP/EP/TP geometry for checkpoint and LAWA scenarios.

Rank axes are (PP, DP, EP, TP), rank = row-major index of that grid.
Dense weights replicate over DP and EP. Expert e lives on EP coordinate
e % EP and may shard over TP. PP assigns whole tensors to stages by layer.
Optimizer state is flat, shards over DP and TP, and keeps its HF shape
through ReshapedDTensor. The oracle below slices a dense index array; it
does not use gdtensor ownership code.
"""

import itertools

import numpy as np

import gdtensor.builder as gb
import gdtensor.tag as gt

AXES = ("pp", "dp", "ep", "tp")

# Every non-empty axis combination, as (label, source grid, target grid).
CASES = [
    ("TP", (1, 1, 1, 3), (1, 1, 1, 2)),
    ("DP", (1, 3, 1, 1), (1, 2, 1, 1)),
    ("EP", (1, 1, 3, 1), (1, 1, 2, 1)),
    ("PP", (3, 1, 1, 1), (2, 1, 1, 1)),
    ("TP_DP", (1, 2, 1, 3), (1, 3, 1, 2)),
    ("TP_EP", (1, 1, 2, 3), (1, 1, 3, 2)),
    ("DP_EP", (1, 2, 3, 1), (1, 3, 2, 1)),
    ("PP_TP", (2, 1, 1, 3), (3, 1, 1, 2)),
    ("PP_DP", (2, 3, 1, 1), (3, 2, 1, 1)),
    ("PP_EP", (2, 1, 3, 1), (3, 1, 2, 1)),
    ("DP_EP_TP", (1, 2, 2, 2), (1, 2, 2, 3)),
    ("PP_DP_TP", (2, 2, 1, 2), (2, 3, 1, 2)),
    ("PP_DP_EP", (2, 2, 2, 1), (3, 1, 3, 1)),
    ("PP_EP_TP", (2, 1, 2, 2), (3, 1, 2, 2)),
    ("PP_DP_EP_TP", (2, 2, 2, 2), (2, 1, 3, 2)),
]

# (tensor id, HF shape, layer, expert or None, kind)
SPECS = (
    [
        ("layers.0.column", (3, 5), 0, None, "column"),
        ("layers.1.row", (3, 5), 1, None, "row"),
        ("layers.2.norm", (5,), 2, None, "replicate"),
        ("layers.3.optimizer", (3, 5), 3, None, "optimizer"),
    ]
    + [
        (f"layers.{e % 4}.experts.{e}.weight", (2, 3), e % 4, e, "row")
        for e in range(5)
    ]
    + [
        ("layers.0.tiny", (1,), 0, None, "column"),
        ("layers.1.empty", (0, 3), 1, None, "column"),
    ]
)


def nonempty_axis_sets():
    return {axes for axes in itertools.product((False, True), repeat=4) if any(axes)}


def coords_of(rank, topology):
    return tuple(int(i) for i in np.unravel_index(rank, topology))


def naive_shape(shape, kind):
    return (int(np.prod(shape)),) if kind == "optimizer" else tuple(shape)


def placements(kind, shape, transpose=False):
    """Placements over (PP, DP, EP, TP) for one tensor kind."""
    axis = (int(kind == "row") + int(transpose)) % len(shape)
    return [
        gb.Replicate(),
        gb.Shard(0) if kind == "optimizer" else gb.Replicate(),
        gb.Replicate(),
        gb.Replicate() if kind == "replicate" else gb.Shard(axis),
    ]


def stage_of(layer, topology):
    return layer % topology[0]


def expert_owner(expert, topology):
    return expert % topology[2]


def mesh_slices(topology, stage, expert, places, order, storage):
    """Slices of the (PP, DP, EP, TP) grid that own one tensor.

    Storage fixes every replicated axis to one writer coordinate so each
    element is stored exactly once.
    """
    slices = [gb.Slice(stage, stage + 1), None, None, None]
    if expert is not None:
        owner = expert_owner(expert, topology)
        slices[2] = gb.Slice(owner, owner + 1)
    if not storage:
        return slices
    for axis in (1, 2, 3):
        if slices[axis] is None and type(places[axis]) is gb.Replicate:
            slices[axis] = gb.Slice(order % topology[axis], order % topology[axis] + 1)
    return slices


def gd_mesh(topology, stage, expert, places, order, storage):
    world = int(np.prod(topology))
    base = (
        gb.SlotOuterProductMesh(
            gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
        )
        if storage
        else gb.ArrangedMesh(gt.WorldSize(world))
    )
    slices = mesh_slices(topology, stage, expert, places, order, storage)
    return gb.SlicedMesh(gb.ReshapedMesh(base, list(topology)), slices)


def dense_points(shape, topology, rank, stage, expert, places, order, storage):
    """Global flat indices one rank owns, from a dense index array."""
    coords = coords_of(rank, topology)
    if coords[0] != stage:
        return np.array([], dtype=np.int64)
    if expert is not None and coords[2] != expert_owner(expert, topology):
        return np.array([], dtype=np.int64)
    points = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    for axis in (1, 2, 3):
        fixed = axis == 2 and expert is not None
        size, coord = (1, 0) if fixed else (topology[axis], coords[axis])
        place = places[axis]
        if type(place) is gb.Replicate and storage and coord != order % size:
            return np.array([], dtype=np.int64)
        if type(place) is gb.Shard:
            points = _chunk(points, place.axis, size, coord)
    return points.ravel()


def _chunk(points, axis, size, coord):
    chunk = (points.shape[axis] + size - 1) // size
    selection = [slice(None)] * points.ndim
    selection[axis] = slice(coord * chunk, (coord + 1) * chunk)
    return points[tuple(selection)]


def tensor_values(order, shape, dtype=np.int64, scale=1, shift=0):
    base = np.arange(int(np.prod(shape)), dtype=np.float64) + order * 1000
    return (base * scale + shift).astype(dtype).reshape(shape)


def schema(topology, *, storage=False, transpose=False, dtype="I64"):
    """DTensorTable, per-rank global indices, arange values, and stages."""
    names = {"I64": (np.int64, 8), "F64": (np.float64, 8), "F32": (np.float32, 4)}
    np_dtype, itemsize = names[dtype]
    table, values, stages = [], {}, []
    points = {rank: {} for rank in range(int(np.prod(topology)))}
    for order, (tid, shape, layer, expert, kind) in enumerate(SPECS):
        flat = naive_shape(shape, kind)
        places = placements(kind, flat, transpose)
        stage = stage_of(layer, topology)
        mesh = gd_mesh(topology, stage, expert, places, order, storage)
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta(list(flat), dtype, itemsize), places, mesh
        )
        generic = gb.ReshapedDTensor(dt, list(shape)) if kind == "optimizer" else dt
        table.append(gb.NamedDTensor(tid, generic))
        values[tid] = tensor_values(order, shape, np_dtype)
        stages.append(stage)
        for rank in points:
            points[rank][tid] = dense_points(
                flat, topology, rank, stage, expert, places, order, storage
            )
    return gb.DTensorTable(table), points, values, stages


def flat_points(points):
    return {
        (tid, rank): indices
        for rank, tensors in points.items()
        for tid, indices in tensors.items()
    }
