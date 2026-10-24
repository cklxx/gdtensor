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

    Locates the target tensor through the public ``.metadata`` pickle:
    ``storage_data[MetadataIndex].StorageInfo`` gives the shard file
    (relative_path) and the segment (offset, length) holding that tensor's
    mini-zip. The tensor payload entry (``archive/data/N``) is found by
    parsing the per-entry local zip headers inside that segment — no content
    search, no fixed offset. One payload byte is flipped in place. The fault
    never goes through the backend's save path, and repeated payload bytes
    cannot mis-attribute the fault to the wrong tensor.
    """
    import pickle
    import re
    import struct

    meta = pickle.loads((root / ".metadata").read_bytes())
    entries = list(meta.storage_data.items())
    if tid_substring is not None:
        idx, info = next(e for e in entries if tid_substring in e[0].fqn)
    else:
        idx, info = entries[0]
    shard = root / info.relative_path
    raw = bytearray(shard.read_bytes())
    segment = bytes(raw[info.offset : info.offset + info.length])
    headers = [info.offset + m.start() for m in re.finditer(rb"PK\x03\x04", segment)]
    payload = None
    for pos in headers:
        fnlen = struct.unpack("<H", raw[pos + 26 : pos + 28])[0]
        extralen = struct.unpack("<H", raw[pos + 28 : pos + 30])[0]
        name = bytes(raw[pos + 30 : pos + 30 + fnlen]).decode()
        if re.fullmatch(r"archive/data/\d+", name):
            payload = pos + 30 + fnlen + extralen
            break
    assert payload is not None, f"no archive/data entry for {idx.fqn}"
    raw[payload] ^= 0xFF
    shard.write_bytes(bytes(raw))


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
