"""Small checkpoint data and source-file fingerprints shared by scenarios."""

import numpy as np

import gdtensor.builder as gb
from tests.support.oracle import slice_points
from tests.support.schemas import tensor_schema


def small_schema(parallel, tensor_count, column=False):
    table, _, _ = tensor_schema(parallel, column=column, tensor_count=tensor_count)
    values, points = {}, {rank: {} for rank in range(int(np.prod(parallel)))}
    for order, nd in enumerate(table.named_d_tensors):
        shape, dtype = _small_tensor(nd)
        seed = 100 if dtype.itemsize == 2 else order * 100
        values[nd.global_tensor_id] = (
            np.arange(int(np.prod(shape)), dtype=dtype) + seed
        ).reshape(shape)
        _tensor_points(points, nd, parallel, order)
    return table, points, values


def _small_tensor(nd):
    dt = nd.d_tensor
    naive = dt.underlying_d_tensor if type(dt) is gb.ReshapedDTensor else dt
    original = naive.global_tensor_meta.global_shape
    shape = [3, 7] if type(dt) is gb.ReshapedDTensor else [5, 7]
    if 0 in original:
        shape = [0, 3]
    elif len(original) == 3:
        shape = [3, 4, 5]
    elif original == [1]:
        shape = [1]
    naive.global_tensor_meta.set_global_shape(
        [int(np.prod(shape))] if type(dt) is gb.ReshapedDTensor else shape
    )
    if type(dt) is gb.ReshapedDTensor:
        dt.set_shape(shape)
    dtype = np.dtype(
        {"I64": "<i8", "F64": "<f8", "F32": "<f4", "F16": "<f2"}[
            naive.global_tensor_meta.dtype
        ]
    )
    return shape, dtype


def _tensor_points(points, nd, parallel, order):
    dt = nd.d_tensor
    naive = dt.underlying_d_tensor if type(dt) is gb.ReshapedDTensor else dt
    for rank in points:
        points[rank][nd.global_tensor_id] = slice_points(
            naive.global_tensor_meta.global_shape,
            parallel,
            naive.placements,
            order % parallel[0],
            rank,
        )


def file_bytes(root):
    return {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
