"""Torchrun worker for the format/cache Cartesian product."""

import argparse
import mmap
import os
from dataclasses import replace
from datetime import timedelta
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.distributed import ReduceOp, all_reduce, destroy_process_group
from torch.distributed import get_rank, get_world_size, init_process_group

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor import Result
from gdtensor_sim.access import buffer_address
from gdtensor_sim.coord import GlooCoordinator
from tests.support.checkpoint import file_bytes, small_schema
from tests.support.oracle import ArrayFabric
from tests.support.schemas import storage_view

FORMATS = ("safetensors", "distcp")
CACHE_MODES = ("direct", "one", "two")
PARALLELS = {2: (1, 1, 2), 4: (1, 2, 2)}
_BufferCallback = Callable[[int, int], Result[None]]


def parallel(world):
    return PARALLELS[world]


def _ok(result):
    return type(result) is gb.Ok


def _require(coord, ok, message):
    if not coord.vote(bool(ok)):
        raise RuntimeError(message)


def _bad_tensors(fabric, values, rank):
    bad = []
    for tid, indices in fabric.points[rank].items():
        want = values[tid].reshape(-1)[indices].copy().view(np.uint8).ravel()
        if not np.array_equal(fabric.buffers[rank][tid], want):
            bad.append(tid)
    return bad


def _range_parts(rng):
    offset = (
        rng.global_tensor_offset
        if hasattr(rng, "global_tensor_offset")
        else rng.contiguous_payload_offset
    )
    return rng.global_tensor_id, offset, rng.size


class _SourceCounter:
    def __init__(self):
        self.reads = 0


class _CountedAccessor(ga.DTensorDataAccessor):
    def __init__(self, inner, counter):
        self.inner, self.counter = inner, counter

    def cache_identity(self):
        return self.inner.cache_identity()

    def read_to_buffer(
        self, global_range: Any, read_to_buffer: _BufferCallback
    ) -> Result[None]:
        rng, callback = global_range, read_to_buffer
        self.counter.reads += rng.size
        return self.inner.read_to_buffer(rng, callback)

    def write_from_buffer(
        self, global_range: Any, write_from_buffer: _BufferCallback
    ) -> Result[None]:
        return self.inner.write_from_buffer(global_range, write_from_buffer)


class _CountedSession:
    def __init__(self, inner, counter):
        self.inner, self.counter = inner, counter

    def get_file_offset(self, pos):
        return self.inner.get_file_offset(pos)

    def init_data_accessor(self):
        result = self.inner.init_data_accessor()
        if not _ok(result):
            return result
        return gb.Ok(_CountedAccessor(result.value, self.counter))

    def finish(self):
        return self.inner.finish()


def _counted_source(view, counter):
    begin_load = view.begin_load

    def counted(view_arg, rank):
        result = begin_load(view_arg, rank)
        if not _ok(result):
            return result
        return gb.Ok(_CountedSession(result.value, counter))

    return replace(view, begin_load=counted)


class _MmapCache(ga.DTensorDataAccessor):
    """Shared test cache: publish bytes before their shared validity bits."""

    def __init__(self, table, root, namespace):
        self.root, self.namespace = Path(root), str(namespace)
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.reads = self.writes = 0
        for index, nd in enumerate(table.named_d_tensors):
            meta = gb.global_tensor_meta_of(nd)
            size = int(np.prod(meta.global_shape)) * meta.itemsize
            self.entries[nd.global_tensor_id] = self._map(index, size)

    def _map(self, index, size):
        data_path = self.root / f"{index}.data"
        valid_path = self.root / f"{index}.valid"
        data_map = self._open_map(data_path, size)
        valid_map = self._open_map(valid_path, size)
        data = self._array(data_map, size, np.uint8)
        valid = self._array(valid_map, size, np.bool_)
        return data, valid, data_map, valid_map

    @staticmethod
    def _open_map(path, size):
        if size == 0:
            return None
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            os.ftruncate(fd, size)
            return mmap.mmap(fd, size)
        finally:
            os.close(fd)

    @staticmethod
    def _array(mapping, size, dtype):
        if mapping is None:
            return np.zeros(size, dtype=dtype)
        return np.frombuffer(mapping, dtype=dtype)

    def cache_namespace(self):
        return self.namespace

    def has_bytes(self, global_range: Any) -> bool:
        rng = global_range
        tid, offset, size = _range_parts(rng)
        _data, valid, _dm, _vm = self.entries[tid]
        return bool(np.all(valid[offset : offset + size]))

    def read_to_buffer(
        self, global_range: Any, read_to_buffer: _BufferCallback
    ) -> Result[None]:
        rng, callback = global_range, read_to_buffer
        tid, offset, size = _range_parts(rng)
        data, valid, _dm, _vm = self.entries[tid]
        if not np.all(valid[offset : offset + size]):
            return gb.Err(
                gb.GDTensorIOError(operation="read", msg="cache bytes invalid")
            )
        buf = data[offset : offset + size]
        self.reads += size
        return callback(buffer_address(buf) if size else 0, size)

    def write_from_buffer(
        self, global_range: Any, write_from_buffer: _BufferCallback
    ) -> Result[None]:
        rng, callback = global_range, write_from_buffer
        tid, offset, size = _range_parts(rng)
        data, valid, data_map, valid_map = self.entries[tid]
        buf = data[offset : offset + size]
        result = callback(buffer_address(buf) if size else 0, size)
        if not _ok(result):
            return result
        if data_map is not None:
            data_map.flush()
        valid[offset : offset + size] = True
        if valid_map is not None:
            valid_map.flush()
        self.writes += size
        return result


def _cache_view(table, root, namespace, nodes):
    named = []
    for nd in table.named_d_tensors:
        dt = nd.d_tensor
        reshaped = dt.shape if type(dt) is gb.ReshapedDTensor else None
        if reshaped is not None:
            dt = dt.underlying_d_tensor
        mesh = gb.StCacheOuterProductMesh(
            temporal=gb.ArrangedMesh(gt.NumSteps(2)),
            spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(len(nodes))),
            concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
        )
        cached = gb.NaiveDTensor(
            dt.global_tensor_meta,
            [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
            mesh,
        )
        if reshaped is not None:
            cached = gb.ReshapedDTensor(cached, reshaped)
        named.append(gb.NamedDTensor(nd.global_tensor_id, cached, nd.alias))
    cache_table = gb.DTensorTable(named)
    accessor = _MmapCache(cache_table, root, namespace)
    ids = [nd.global_tensor_id for nd in named]
    return gb.CacheDTensorsView(ids, cache_table, accessor), accessor


def _chain(mode, source, destination, table, root, case, world):
    if mode == "direct":
        return gb.DirectDTensorsReadChain(src=source, dst=destination), []
    one, one_acc = _cache_view(table, root / "one", f"{case}-one", (0,))
    if mode == "one":
        chain = gb.OneLayerCachedDTensorsReadChain(
            src=source, cache=one, dst=destination, num_ranks_per_node=world
        )
        return chain, [one_acc]
    two, two_acc = _cache_view(table, root / "two", f"{case}-two", (0,))
    chain = gb.TwoLayerCachedDTensorsReadChain(
        src=source,
        small_cache=one,
        big_cache=two,
        dst=destination,
        num_ranks_per_node=world,
    )
    return chain, [one_acc, two_acc]


def _global_metrics(counter, caches):
    local = [counter.reads]
    local.extend(cache.reads for cache in caches)
    local.extend(cache.writes for cache in caches)
    values = torch.tensor(local, dtype=torch.int64, device="cpu")
    all_reduce(values, op=ReduceOp.SUM)
    return values.tolist()


def _load_once(coord, mode, source, table, points, values, rank, root, case, world):
    counter = _SourceCounter()
    counted = _counted_source(source, counter)
    mid = ArrayFabric(table, {rank: points[rank]})
    chain, caches = _chain(mode, counted, mid.view(rank), table, root, case, world)
    check = chain.check_constraints()
    _require(coord, _ok(check), f"{case}: invalid read chain: {check!r}")
    coord.wait()
    result = gio.chain_load(rank, chain)
    metrics = _global_metrics(counter, caches)
    bad = _bad_tensors(mid, values, rank) if _ok(result) else [repr(result.value)]
    _require(coord, not bad, f"{case}: load or value check failed: {bad}")
    return mid, metrics


def _save(coord, rank, fabric, view, values, case):
    before = _bad_tensors(fabric, values, rank)
    result = gio.rank_save(rank, fabric.view(rank), view)
    after = _bad_tensors(fabric, values, rank)
    ok = _ok(result) and not before and not after
    detail = f"before={before} after={after} result={result!r}"
    _require(coord, ok, f"{case}: save failed: {detail}")
    coord.wait()


def _roundtrip(coord, rank, view, table, points, values, case):
    fabric = ArrayFabric(table, {rank: points[rank]})
    result = gio.rank_load(rank, fabric.view(rank), view)
    bad = _bad_tensors(fabric, values, rank) if _ok(result) else [repr(result.value)]
    _require(coord, not bad, f"{case}: target roundtrip failed: {bad}")


def _traffic_ok(mode, first, second):
    if mode == "direct":
        return first[0] > 0 and second[0] == first[0]
    layers = (len(first) - 1) // 2
    reads = second[1 : 1 + layers]
    first_writes = first[1 + layers :]
    second_writes = second[1 + layers :]
    return (
        first[0] > 0
        and second[0] == 0
        and reads[-1] > 0
        and all(value > 0 for value in first_writes)
        and all(value == 0 for value in second_writes)
    )


def _run_case(
    coord, rank, world, table, points, values, source, src_fmt, dst_fmt, mode, root
):
    case = f"{src_fmt}-to-{dst_fmt}-{mode}"
    case_root = root / case
    mid1, first = _load_once(
        coord, mode, source, table, points, values, rank, case_root, case, world
    )
    mid2, second = _load_once(
        coord, mode, source, table, points, values, rank, case_root, case, world
    )
    _require(
        coord, _traffic_ok(mode, first, second), f"{case}: bad traffic {first} {second}"
    )
    target = storage_view(
        case_root / "target",
        table,
        parallel(world),
        "rank_multi",
        "mirror",
        dst_fmt,
        save=True,
        coord=coord,
    )
    _save(coord, rank, mid2, target, values, case)
    _roundtrip(coord, rank, target, table, points, values, case)
    if rank == 0:
        print(f"CASE PASS {case} first={first} second={second}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    init_process_group("gloo", init_method="env://", timeout=timedelta(seconds=120))
    try:
        rank, world = get_rank(), get_world_size()
        torch.set_num_threads(1)
        os.environ["GDTENSOR_CACHE_DIR"] = str(args.root / "coordination")
        os.environ["GDTENSOR_CACHE_WAIT_SECONDS"] = "30"
        (args.root / "coordination").mkdir(parents=True, exist_ok=True)
        coord = GlooCoordinator()
        table, points, values = small_schema(parallel(world), tensor_count=8)
        args.root.mkdir(parents=True, exist_ok=True)
        snapshots = {}
        sources = {}
        for fmt in FORMATS:
            source_root = args.root / f"source-{fmt}"
            source = storage_view(
                source_root,
                table,
                parallel(world),
                "rank_multi",
                "mirror",
                fmt,
                save=True,
                coord=coord,
            )
            fabric = ArrayFabric(table, {rank: points[rank]}, values)
            _save(coord, rank, fabric, source, values, f"source-{fmt}")
            _roundtrip(coord, rank, source, table, points, values, f"source-{fmt}")
            snapshots[fmt] = file_bytes(source_root) if rank == 0 else None
            sources[fmt] = source
        for src_fmt, dst_fmt, mode in product(FORMATS, FORMATS, CACHE_MODES):
            _run_case(
                coord,
                rank,
                world,
                table,
                points,
                values,
                sources[src_fmt],
                src_fmt,
                dst_fmt,
                mode,
                args.root,
            )
        unchanged = (
            all(
                file_bytes(args.root / f"source-{fmt}") == snapshots[fmt]
                for fmt in FORMATS
            )
            if rank == 0
            else True
        )
        _require(coord, unchanged, "source checkpoint changed during matrix")
        if rank == 0:
            print(f"MATRIX PASS world={world} cases=12", flush=True)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
    # ponytail: bypass native atexit after explicit Gloo teardown; test IO is complete.
    os._exit(0)
