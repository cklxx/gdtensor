"""Concurrency-width fixes: the declared slot count must be enforced at runtime
and memory stats must cover two-layer cache chains.

Only public APIs and independent in-memory accessors are used here. The forced
overlap distinguishes C=1 (a second fill waits) from C=2 (a second cell fills).
"""

from __future__ import annotations

import threading

import numpy as np

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.cache_oracle import ResidentCache
from tests.support.memory import Memory


def _storage_view(accessor, size, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)),
                gb.ArrangedMesh(gt.WorldSize(world)),
            )
        ],
        ["w"],
    )
    leader = gb.TensorComponentPos("w", 0)
    entries = {gb.TensorComponentPos("w", rank): group for rank in range(world)}
    layout = gb.PhysicalLayoutTable("safetensors", entries, {leader: "memory"})

    class _LoadSession:
        def get_file_offset(self, pos):
            return gb.Ok(0)

        def init_data_accessor(self):
            return gb.Ok(accessor)

        def finish(self):
            return gb.Ok(None)

    return gb.StorageDTensorsView(
        ["w"],
        gb.DTensorTable([gb.NamedDTensor("w", dt)]),
        Memory([0]),
        layout,
        lambda _v, _r: gb.Ok(_LoadSession()),
        lambda _v, _r: gb.Ok(_LoadSession()),
    )


def _compute_view(accessor, size, world):
    nd = gb.NamedDTensor(
        "w",
        gb.NaiveDTensor(
            gb.GlobalTensorMeta([size], "U8", 1),
            [gb.Shard(0)],
            gb.ArrangedMesh(gt.WorldSize(world)),
        ),
    )
    return gb.ComputeDTensorsView(["w"], gb.DTensorTable([nd]), accessor)


def _cache_view(accessor, size, steps, width):
    mesh = gb.StCacheOuterProductMesh(
        gb.ArrangedMesh(gt.NumSteps(steps)),
        gb.ArrangedMesh(gt.NumNodes(1)),
        gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    nd = gb.NamedDTensor(
        "w",
        gb.NaiveDTensor(
            gb.GlobalTensorMeta([size], "U8", 1),
            [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
            mesh,
        ),
    )
    return gb.CacheDTensorsView(["w"], gb.DTensorTable([nd]), accessor)


def _one_layer_chain(size, steps, width, cache_mem):
    return gb.OneLayerCachedDTensorsReadChain(
        src=_storage_view(Memory(np.arange(size, dtype=np.uint8)), size, 2),
        cache=_cache_view(cache_mem, size, steps, width),
        dst=_compute_view(Memory(np.zeros(size, dtype=np.uint8)), size, 2),
        num_ranks_per_node=2,
    )


def _run_with_rank0_drain_blocked(chain, cache_mem, hold_seconds):
    """Start rank 0, block its drain, then start rank 1; return whether rank 1
    finishes while rank 0 is held, and the final per-rank results."""
    release = threading.Event()
    rank0_draining = threading.Event()
    rank1_done = threading.Event()
    original_read = cache_mem.read_to_buffer
    results = {}

    def blocking_read(global_range, callback):
        if cache_mem.current_rank == 0:
            rank0_draining.set()
            assert release.wait(10), "rank0 drain barrier timed out"
        return original_read(global_range, callback)

    def run(rank):
        cache_mem.current_rank = rank
        results[rank] = gio.chain_load(rank, chain)
        if rank == 1:
            rank1_done.set()

    cache_mem.read_to_buffer = blocking_read
    t0 = threading.Thread(target=run, args=(0,), daemon=True)
    t0.start()
    assert rank0_draining.wait(10)
    t1 = threading.Thread(target=run, args=(1,), daemon=True)
    t1.start()
    coexist = rank1_done.wait(hold_seconds)
    release.set()
    t0.join(timeout=10)
    t1.join(timeout=10)
    assert not t0.is_alive() and not t1.is_alive()
    assert all(type(v) is gb.Ok for v in results.values()), results
    return coexist


def test_concurrency_width_one_serializes_and_two_pipelines():
    # Both chains use 4-byte cells (one rank's share). C=1 (S=2) keeps one cell
    # live, so rank 1 waits for rank 0. C=2 (S=1) keeps two cells live, so
    # rank 1 fills its own cell while rank 0 is held.
    blocked = ResidentCache([], 2)
    c1 = _one_layer_chain(8, steps=2, width=1, cache_mem=blocked)
    assert gio.chain_stat_mem_overhead(c1) == [4]
    assert _run_with_rank0_drain_blocked(c1, blocked, hold_seconds=2) is False

    pipelined = ResidentCache([], 2)
    c2 = _one_layer_chain(8, steps=1, width=2, cache_mem=pipelined)
    assert gio.chain_stat_mem_overhead(c2) == [8]
    assert _run_with_rank0_drain_blocked(c2, pipelined, hold_seconds=2) is True


def test_two_layer_mem_stat_sums_independent_layers():
    size, world, k = 8, 4, 2

    def cache2():
        mesh = gb.StCacheOuterProductMesh(
            gb.ArrangedMesh(gt.NumSteps(2)),
            gb.LiteralMesh([0, 1], gt.NumNodes(2)),
            gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
        )
        nd = gb.NamedDTensor(
            "w",
            gb.NaiveDTensor(
                gb.GlobalTensorMeta([size], "U8", 1),
                [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
                mesh,
            ),
        )
        return gb.CacheDTensorsView(
            ["w"], gb.DTensorTable([nd]), Memory(np.zeros(size, np.uint8))
        )

    chain = gb.TwoLayerCachedDTensorsReadChain(
        src=_storage_view(Memory(np.arange(size, dtype=np.uint8)), size, world),
        small_cache=cache2(),
        big_cache=cache2(),
        dst=_compute_view(Memory(np.zeros(size, np.uint8)), size, world),
        num_ranks_per_node=k,
    )
    assert type(chain.check_constraints()) is gb.Ok
    # Two independent cache layers; each stages the 8B step on a node.
    assert gio.chain_stat_mem_overhead(chain) == [8, 8]
