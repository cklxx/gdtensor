"""Real-file cache strategy tests: memory peak, source-read traffic, and reuse.

The test writes a real checkpoint in two formats:
- safetensors: one ``model.safetensors`` file.
- DCP: a real torch.distributed.checkpoint directory (``.metadata`` + ``.distcp``).

It then reads the checkpoint through a one-layer cache chain and measures:
- cache memory peak per node (temporal steps S, concurrency width C);
- source-file read bytes (direct vs cached);
- concurrent reuse (one fill per node per cell).

The two formats take different source paths:
- safetensors reads payload bytes through the gdtensor FileIO (traced pread).
- DCP opens each rank's session with a full torch DCP load into memory; the cache
  fills from that in-memory copy, so the gdtensor FileIO records no payload pread.
"""

from __future__ import annotations

import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pytest

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor import Result
from gdtensor_sim.formats import make_storage_view
from gdtensor_sim.iobackend import FileIO
from tests.support.cache_oracle import ResidentCache, storage_schema

SIZE = 16
WORLD = 4
K = 2  # ranks per node
NODES = [0, 1]


# --------------------------------------------------------------------- helpers


class ReplicateComputeMemory(ga.DTensorDataAccessor):
    """One flat buffer per rank. Each rank wants the full tensor (Replicate).

    chain_load sets current_rank. Writes land in that rank's buffer at global
    offsets. This lets the test verify per-rank delivery.
    """

    def __init__(self, world: int, size: int):
        self.buffers = {r: np.zeros(size, dtype=np.uint8) for r in range(world)}
        self._tl = threading.local()
        self._tl.rank = 0

    @property
    def current_rank(self) -> int:
        return getattr(self._tl, "rank", 0)

    @current_rank.setter
    def current_rank(self, rank: int) -> None:
        self._tl.rank = rank

    def write_from_buffer(
        self, global_range, write_from_buffer: Callable[[int, int], Result[None]]
    ) -> Result[None]:
        rng = global_range
        data = np.zeros(rng.size, dtype=np.uint8)
        res = write_from_buffer(data.ctypes.data, data.nbytes)
        if type(res) is gb.Ok:
            buf = self.buffers[self.current_rank]
            buf[rng.global_tensor_offset : rng.global_tensor_offset + rng.size] = data
        return res

    def read_to_buffer(
        self, global_range, read_to_buffer: Callable[[int, int], Result[None]]
    ) -> Result[None]:
        raise AssertionError("compute dst must not be read during load")


def _save_real(root, fmt: str):
    """Write a real checkpoint with arange values. Return (table, layout, path)."""
    st, layout = storage_schema(["w"], SIZE)
    layout.container_format = fmt
    path = "model.safetensors" if fmt == "safetensors" else "__0_0.distcp"
    layout.set_file_paths({gb.TensorComponentPos("w", 0): path})
    sv = make_storage_view(["w"], st, layout, str(root), FileIO(), dcp_mode="process")
    src_ct = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "w",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([SIZE], "U8", 1),
                    [gb.Replicate()],
                    gb.ArrangedMesh(gt.WorldSize(1)),
                ),
            )
        ]
    )
    points = {0: {"w": np.arange(SIZE)}}
    from tests.support.oracle import ArrayFabric

    fabric = ArrayFabric(src_ct, points, {"w": np.arange(SIZE, dtype=np.uint8)})
    assert type(gio.rank_save(0, fabric.view(0), sv)) is gb.Ok
    return st, layout, path


def _load_view(root, st, layout, fmt: str):
    """A fresh storage view with a fresh FileIO for read counting."""
    io = FileIO()
    sv = make_storage_view(["w"], st, layout, str(root), io, dcp_mode="process")
    return sv, io


def _compute_dst():
    ct = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "w",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([SIZE], "U8", 1),
                    [gb.Replicate()],
                    gb.ArrangedMesh(gt.WorldSize(WORLD)),
                ),
            )
        ]
    )
    mem = ReplicateComputeMemory(WORLD, SIZE)
    return mem, gb.ComputeDTensorsView(["w"], ct, mem)


def _cache_view(cache_mem, steps: int, width: int):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(NODES), gt.NumNodes(len(NODES))),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([SIZE], "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    table = gb.DTensorTable([gb.NamedDTensor("w", dt)])
    return gb.CacheDTensorsView(["w"], table, cache_mem)


def _run_concurrent(chain, dst_mem, cache_mem):
    def work(rank):
        dst_mem.current_rank = rank
        if cache_mem is not None:
            cache_mem.current_rank = rank
        res = gio.chain_load(rank, chain)
        assert type(res) is gb.Ok, res

    with ThreadPoolExecutor(max_workers=WORLD) as ex:
        list(ex.map(work, range(WORLD)))


def _assert_delivery(mem):
    for rank in range(WORLD):
        np.testing.assert_array_equal(
            mem.buffers[rank], np.arange(SIZE, dtype=np.uint8), err_msg=f"rank={rank}"
        )


def _data_start(path) -> int:
    with open(path, "rb") as handle:
        n = struct.unpack("<Q", handle.read(8))[0]
    return 8 + n


def _payload_pread(io, path: str, data_start: int) -> int:
    return sum(
        r.size
        for r in io.tracer.records
        if r.op == "pread" and r.path.endswith(path) and r.offset >= data_start
    )


def _assert_memory(cache_mem, chain, steps, width):
    """Cache residency and fill traffic.

    The core fills lazily; it does not prefetch (design line 418). One cell is
    resident at a time, so peak per node = TOTAL/(S*C). The stat function is a
    conservative upper bound = TOTAL/S (the step), assuming full overlap.
    """
    cell_bytes = SIZE // (steps * width)
    step_bytes = SIZE // steps
    stat_mem = gio.chain_stat_mem_overhead(chain)
    for node in NODES:
        peak = cache_mem.peak_by_node[node]
        assert peak == cell_bytes, (node, peak, steps, width)
        assert stat_mem[node] == step_bytes, (node, stat_mem, steps, width)
        assert peak <= stat_mem[node]
    # Each node stages the full tensor once: fill bytes = N*SIZE.
    fill_bytes = sum(
        sz
        for name, op, _tid, _off, sz in cache_mem.events
        if name == "cache" and op == "write"
    )
    assert fill_bytes == len(NODES) * SIZE, fill_bytes


# ---------------------------------------------------------------- memory tests


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("width", [1, 2])
def test_safetensors_memory_strategy(tmp_path, steps, width):
    st, layout, path = _save_real(tmp_path, "safetensors")
    sv, _io = _load_view(tmp_path, st, layout, "safetensors")
    dst_mem, cv = _compute_dst()
    cache_mem = ResidentCache([], K)
    cvt = _cache_view(cache_mem, steps, width)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=K
    )
    assert type(chain.check_constraints()) is gb.Ok
    _run_concurrent(chain, dst_mem, cache_mem)
    _assert_memory(cache_mem, chain, steps, width)
    _assert_delivery(dst_mem)


@pytest.mark.parametrize("steps", [1, 2, 4])
@pytest.mark.parametrize("width", [1, 2])
def test_dcp_memory_strategy(tmp_path, steps, width):
    st, layout, path = _save_real(tmp_path, "distcp")
    sv, io = _load_view(tmp_path, st, layout, "distcp")
    dst_mem, cv = _compute_dst()
    cache_mem = ResidentCache([], K)
    cvt = _cache_view(cache_mem, steps, width)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=K
    )
    assert type(chain.check_constraints()) is gb.Ok
    _run_concurrent(chain, dst_mem, cache_mem)
    _assert_memory(cache_mem, chain, steps, width)
    _assert_delivery(dst_mem)
    # DCP legs read from in-memory torch load: no payload pread through gdtensor.
    assert sum(1 for r in io.tracer.records if r.op == "pread") == 0


# ------------------------------------------------------------- traffic tests


def test_safetensors_direct_vs_cached_source_reads(tmp_path):
    st, layout, path = _save_real(tmp_path, "safetensors")
    full = str(tmp_path / path)
    data_start = _data_start(full)

    # Direct: each rank reads the full tensor from the file.
    sv_direct, io_direct = _load_view(tmp_path, st, layout, "safetensors")
    dst_direct, cv_direct = _compute_dst()
    direct = gb.DirectDTensorsReadChain(src=sv_direct, dst=cv_direct)
    assert type(direct.check_constraints()) is gb.Ok
    _run_concurrent(direct, dst_direct, None)
    direct_bytes = _payload_pread(io_direct, path, data_start)
    assert direct_bytes == WORLD * SIZE

    # Cached: each node reads the full tensor once.
    sv_cached, io_cached = _load_view(tmp_path, st, layout, "safetensors")
    dst_cached, cv_cached = _compute_dst()
    cache_mem = ResidentCache([], K)
    cvt = _cache_view(cache_mem, 1, 1)
    cached = gb.OneLayerCachedDTensorsReadChain(
        src=sv_cached, cache=cvt, dst=cv_cached, num_ranks_per_node=K
    )
    assert type(cached.check_constraints()) is gb.Ok
    _run_concurrent(cached, dst_cached, cache_mem)
    cached_bytes = _payload_pread(io_cached, path, data_start)
    assert cached_bytes == len(NODES) * SIZE
    assert cached_bytes < direct_bytes


def test_dcp_session_open_full_reads_per_rank(tmp_path, monkeypatch):
    st, layout, _path = _save_real(tmp_path, "distcp")
    sv, io = _load_view(tmp_path, st, layout, "distcp")
    dst_mem, cv = _compute_dst()
    cache_mem = ResidentCache([], K)
    cvt = _cache_view(cache_mem, 1, 1)
    chain = gb.OneLayerCachedDTensorsReadChain(
        src=sv, cache=cvt, dst=cv, num_ranks_per_node=K
    )

    # Count real torch DCP loads: one per rank session open.
    import gdtensor_sim.dcp_backend as dcp_backend

    calls = {"n": 0}
    orig = dcp_backend._real_dcp_load

    def counting_load(table: gb.DTensorTable, root: str) -> dict[str, object]:
        calls["n"] += 1
        return orig(table, root)

    monkeypatch.setattr(dcp_backend, "_real_dcp_load", counting_load)
    _run_concurrent(chain, dst_mem, cache_mem)
    assert calls["n"] == WORLD  # full load per rank, not deduped by the cache
    _assert_delivery(dst_mem)
