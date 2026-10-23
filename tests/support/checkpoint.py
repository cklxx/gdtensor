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


def payload_file_bytes(root):
    """Whole-directory byte map of payload files, excluding the DCP container
    metadata.

    Real torch DCP embeds a fresh per-write ``storage_uuid`` in ``.metadata``;
    two saves of identical tensors therefore never produce byte-identical
    ``.metadata`` even though every payload shard is deterministic. Comparing
    payload files keeps the byte-level canonicalization assertion without
    pinning the non-deterministic container header.
    """
    return {
        path.name: path.read_bytes()
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != ".metadata"
    }


def corrupt_distcp_payload(root, tid_substring=None):
    """Inject a real on-disk byte fault into a real torch DCP checkpoint.

    Decodes each tensor through the independent oracle (torch's own reader,
    not gdtensor_sim), locates one tensor's raw bytes inside the shard by
    direct search, and flips a single byte there. The fault never goes through
    the backend's save path — the container is left byte-corrupted on disk,
    so the oracle re-decoding afterwards reads genuinely wrong payloads.
    """
    from tests.support.schemas import payload_regions

    shards = list(root.glob("*.distcp"))
    assert shards, "no distcp shard to corrupt"
    regions = list(payload_regions(root, "distcp"))
    if tid_substring is not None:
        region = next(r for r in regions if tid_substring in r[0])
    else:
        region = regions[0]
    data = region[3]
    raw = bytearray(shards[0].read_bytes())
    idx = raw.find(data[:32])
    assert idx >= 0, "tensor payload bytes not found in shard"
    raw[idx] ^= 0xFF
    shards[0].write_bytes(bytes(raw))


def repack_via_compute(world, src_read_view, dst_save_view, table, points, values=None):
    """Two-phase storage->storage repack (design §10.3, §419).

    The design has no storage->storage copy and no compute->compute reshard;
    both are cut on purpose. Redistribution through the public IO boundary is
    exactly two calls: rank_load into an intermediate compute fabric on the
    destination topology, then rank_save out.

    When ``values`` (the independent dense arange oracle) is given, the
    intermediate fabric is asserted per-rank and per-element right after the
    load, before the save phase — a bad load cannot be masked away by a later
    write. The fabric returns to the caller for further assertions.
    """
    import gdtensor.io as gio

    from tests.support.oracle import ArrayFabric
    from tests.support.schemas import run_ranks

    mid = ArrayFabric(table, points)
    run_ranks(world, lambda rank: gio.rank_load(rank, mid.view(rank), src_read_view))
    if values is not None:
        mid.assert_values(values)
    run_ranks(world, lambda rank: gio.rank_save(rank, mid.view(rank), dst_save_view))
    return mid
