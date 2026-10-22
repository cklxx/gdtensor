"""Interpret public mesh metadata for simulated memory and file headers."""

from itertools import product
from math import prod

import numpy as np

from gdtensor.builder import (
    ArrangedMesh,
    LiteralMesh,
    OuterProductMesh,
    PermutedMesh,
    ReshapedDTensor,
    ReshapedMesh,
    SlicedMesh,
)
from gdtensor.tag import Partial, Shard
from ._result import invalid


def naive_of(nd):
    dt = nd.d_tensor
    return dt.underlying_d_tensor if type(dt) is ReshapedDTensor else dt


def tensor_ids_of(table):
    return [nd.global_tensor_id for nd in table.named_d_tensors]


def named_of(table, tensor_id):
    return next(nd for nd in table.named_d_tensors if nd.global_tensor_id == tensor_id)


def numel_of(meta):
    return prod(meta.global_shape)


def nbytes_of(meta):
    return numel_of(meta) * meta.itemsize


def _mesh(mesh):
    if type(mesh) is ArrangedMesh:
        return np.arange(mesh.size.value, dtype=np.int64)
    if type(mesh) is LiteralMesh:
        return np.asarray(mesh.ranks, dtype=np.int64)
    if type(mesh) is OuterProductMesh:
        content = mesh.content_slots
        width = (
            content.max.value if type(content) is LiteralMesh else content.size.value
        )
        columns = np.full(width, -1, dtype=np.int64)
        ranks = content.ranks if type(content) is LiteralMesh else range(width)
        columns[list(ranks)] = list(ranks)
        return np.tile(columns, (mesh.tensor_slots.size.value, 1))
    inner = _mesh(mesh.underlying_mesh)
    if type(mesh) is ReshapedMesh:
        return inner.reshape(mesh.shape)
    if type(mesh) is PermutedMesh:
        return inner.transpose(mesh.permutation)
    if type(mesh) is SlicedMesh:
        indices = tuple(
            slice(None) if s is None else slice(s.start, s.stop, s.step)
            for s in mesh.slices
        )
        return inner[indices]
    raise invalid("mesh", f"unsupported mesh {type(mesh).__name__}")


def participants_of(nd, side):
    return sorted(int(rank) for rank in _mesh(naive_of(nd).mesh).ravel() if rank >= 0)


def owned_box(nd, side, participant):
    naive = naive_of(nd)
    cells = _mesh(naive.mesh)
    hit = np.argwhere(cells == participant)
    if not len(hit):
        return None
    if len(hit) != 1 or len(naive.placements) != cells.ndim:
        raise invalid(
            "mesh", "ambiguous rank or placements do not match mesh dimensions"
        )
    lo = [0] * len(naive.global_tensor_meta.global_shape)
    hi = list(naive.global_tensor_meta.global_shape)
    for placement, size, coordinate in zip(naive.placements, cells.shape, hit[0]):
        if type(placement) is Partial:
            raise invalid("placements", "Partial cannot be saved")
        if type(placement) is Shard:
            axis = placement.value
            chunk = -(-(hi[axis] - lo[axis]) // size)
            start = min(hi[axis], lo[axis] + int(coordinate) * chunk)
            lo[axis], hi[axis] = start, min(hi[axis], start + chunk)
    return lo, hi


def owned_intervals(nd, side, participant):
    box = owned_box(nd, side, participant)
    empty = np.empty((0, 2), dtype=np.int64)
    if box is None:
        return empty
    lo, hi = box
    shape = naive_of(nd).global_tensor_meta.global_shape
    if any(left == right for left, right in zip(lo, hi)):
        return empty
    if not shape:
        return np.array([[0, 1]], dtype=np.int64)
    axis = len(shape) - 1
    while axis and lo[axis] == 0 and hi[axis] == shape[axis]:
        axis -= 1
    strides = [prod(shape[d + 1 :]) for d in range(len(shape))]
    starts = [
        sum(c * stride for c, stride in zip(coords, strides)) + lo[axis] * strides[axis]
        for coords in product(*(range(lo[d], hi[d]) for d in range(axis)))
    ]
    start = np.array(starts, dtype=np.int64)
    return np.column_stack((start, start + (hi[axis] - lo[axis]) * strides[axis]))


def iv_total(intervals):
    return int((intervals[:, 1] - intervals[:, 0]).sum())


def iv_prefix(intervals, points):
    lengths = intervals[:, 1] - intervals[:, 0]
    prefix = np.concatenate(([0], np.cumsum(lengths)))
    index = np.searchsorted(intervals[:, 0], points, side="right") - 1
    return prefix[index] + points - intervals[index, 0]


def nonempty_columns(nd):
    columns = {}
    for rank in participants_of(nd, "storage"):
        intervals = owned_intervals(nd, "storage", rank)
        if iv_total(intervals):
            columns[rank] = intervals
    return columns
