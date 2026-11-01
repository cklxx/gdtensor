"""Co-located deployment simulation: cache DN and ranks share each node.

Each node has its own cache memory and NIC (tests/support/cluster_sim.py). The
tests check that (1) every rank gets correct bytes, (2) every cache access
hits the node the plan says, (3) the stat traffic functions equal the bytes
the simulated NICs move, and (4) a slow NIC does not stall the executor.
"""

import threading

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from tests.support.cluster_sim import (
    Cluster,
    NodeCache,
    RankMem,
    StorageMem,
    layer_targets,
)


def _storage(acc, size, world):
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(world))
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
    )
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)), gb.ArrangedMesh(gt.WorldSize(world))
            )
        ],
        ["w"],
    )
    entries = {gb.TensorComponentPos("w", r): group for r in range(world)}
    layout = gb.PhysicalLayoutTable(
        "safetensors", entries, {gb.TensorComponentPos("w", 0): "mem"}
    )

    class _Load:
        def get_file_offset(self, pos):
            return gb.Ok(0)

        def init_data_accessor(self):
            return gb.Ok(acc)

        def finish(self):
            return gb.Ok(None)

    return gb.StorageDTensorsView(
        ["w"],
        gb.DTensorTable([gb.NamedDTensor("w", dt)]),
        acc,
        layout,
        lambda _v, _r: gb.Ok(_Load()),
        lambda _v, _r: gb.Ok(_Load()),
    )


def _cache(acc, size, steps, nodes, width):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(len(nodes))),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
        mesh,
    )
    return gb.CacheDTensorsView(["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc)


def _compute(acc, size, world):
    dt = gb.NaiveDTensor(
        gb.GlobalTensorMeta([size], "U8", 1),
        [gb.Shard(0)],
        gb.ArrangedMesh(gt.WorldSize(world)),
    )
    return gb.ComputeDTensorsView(
        ["w"], gb.DTensorTable([gb.NamedDTensor("w", dt)]), acc
    )


def _build(layers, size=16, world=8, k=2, steps=1, width=1, bandwidth=0.0):
    """layers: node tuples in data-flow order (one or two cache layers)."""
    cluster = Cluster(world // k, k, bandwidth)
    targets = layer_targets(cluster, [list(n) for n in layers])
    caches = [NodeCache(cluster, {"w": size}, t) for t in targets]
    src = StorageMem(cluster, np.arange(size, dtype=np.uint8))
    dst = RankMem(cluster, size)
    views = [_cache(c, size, steps, n, width) for c, n in zip(caches, layers)]
    common = dict(
        src=_storage(src, size, world),
        dst=_compute(dst, size, world),
        num_ranks_per_node=k,
    )
    if len(views) == 1:
        chain = gb.OneLayerCachedDTensorsReadChain(cache=views[0], **common)
    else:
        chain = gb.TwoLayerCachedDTensorsReadChain(
            small_cache=views[0], big_cache=views[1], **common
        )
    return chain, cluster, caches, dst


def _run(chain, cluster, world):
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, chain)

    threads = [threading.Thread(target=one, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "chain_load hung"
    assert set(results) == set(range(world)), results
    bad = {r: v.value.msg for r, v in results.items() if type(v) is not gb.Ok}
    assert not bad, bad


def _check(chain, cluster, caches, dst, size, world):
    """Correct bytes on every rank, and every cache access on the planned node."""
    assert type(chain.check_constraints()) is gb.Ok
    _run(chain, cluster, world)
    expected = np.arange(size, dtype=np.uint8)
    for rank, data in dst.buffers.items():
        start = rank * ((size + world - 1) // world)
        stop = min(size, start + (size + world - 1) // world)
        want = np.full(size, 0xA5, dtype=np.uint8)
        want[start:stop] = expected[start:stop]
        np.testing.assert_array_equal(data, want, err_msg=f"rank {rank}")
    assert [c.wrong_node for c in caches] == [0] * len(caches)
    return cluster.nic_bytes()


def _stat_vs_nic(chain, cluster):
    rx, tx = cluster.nic_bytes()
    nodes = sorted(cluster.nics)
    # Storage has no node id: the stat books a storage read as inbound of the
    # cache node it fills; the NIC books cache traffic between nodes.
    observed_in = [rx[n] + cluster.storage_in[n] for n in nodes]
    return (
        (gio.chain_stat_inbound_traffic(chain), observed_in),
        (gio.chain_stat_outbound_traffic(chain), tx),
    )


CASES = {
    "full_cover_one_layer": [(0, 1, 2, 3)],
    "partial_one_layer": [(0, 1)],
    "single_dn_one_layer": [(0,)],
    "two_layer_fanout": [(0,), (0, 1, 2, 3)],
    "two_layer_partial": [(0, 1), (0, 1, 2)],
}


REMOTE_FILL = "remote rank fills a cache node: storage bytes go to the rank node, then over the NIC to the DN"


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("steps,width", [(1, 1), (2, 2)])
def test_colocated_delivery_hits_planned_nodes(name, steps, width):
    size, world = 16, 8
    chain, cluster, caches, dst = _build(CASES[name], size, world, 2, steps, width)
    _check(chain, cluster, caches, dst, size, world)


@pytest.mark.parametrize(
    "name",
    [
        "full_cover_one_layer",
        pytest.param(
            "partial_one_layer",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=REMOTE_FILL
            ),
        ),
        pytest.param(
            "single_dn_one_layer",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=REMOTE_FILL
            ),
        ),
        pytest.param(
            "two_layer_fanout",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=REMOTE_FILL
            ),
        ),
        pytest.param(
            "two_layer_partial",
            marks=pytest.mark.xfail(
                strict=True, raises=AssertionError, reason=REMOTE_FILL
            ),
        ),
    ],
)
def test_colocated_nic_bytes_match_stats(name):
    chain, cluster, caches, dst = _build(CASES[name])
    _check(chain, cluster, caches, dst, 16, 8)
    (stat_in, obs_in), (stat_out, tx) = _stat_vs_nic(chain, cluster)
    assert stat_out == tx, (stat_out, tx)
    assert stat_in == obs_in, (stat_in, obs_in)


def test_full_cover_moves_no_network_bytes():
    chain, cluster, caches, dst = _build([(0, 1, 2, 3)])
    rx, tx = _check(chain, cluster, caches, dst, 16, 8)
    assert rx == tx == [0, 0, 0, 0]
    # Each node reads only its own 4 B from storage, once for both local ranks.
    assert [cluster.storage_in[n] for n in range(4)] == [4, 4, 4, 4]


def test_slow_nic_does_not_time_out(monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "5")
    # 64 B/s link: one 4 B remote piece takes about 0.06 s.
    chain, cluster, caches, dst = _build([(0,)], bandwidth=64.0)
    _check(chain, cluster, caches, dst, 16, 8)
