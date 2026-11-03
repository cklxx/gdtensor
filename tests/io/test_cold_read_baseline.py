"""No-cache cold-read baseline, randomized, multi-round.

Round 0 starts from guard-filled outputs. Every following round resets the
outputs again, so every round is cold: the chain must read the source bytes
again. This is the denominator for every cache placement: same shape, same
rounds, no cache.

Checks per round:

- every rank output equals the source bytes for its shard, protected bytes
  stay at the guard value;
- storage reads exactly the bytes the ranks want, once, with no dedupe
  across ranks;
- storage bytes are identical across rounds.

A companion case runs the same random shape with one or two cache layers and
does not reset the cache between rounds. It checks that round 0 pays the
storage read once per node and later rounds read no storage at all, while
outputs stay byte-exact.
"""

import threading
import time

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_shapes import (
    GUARD,
    FileStorage,
    RankBuffers,
    _build,
    _bytes,
    _compute_view,
    _demand,
    _storage_view,
    _wants,
)
from tests.support.cluster_sim import Cluster
from tests.support.crc import crc_of

SEEDS = 12
SEED_BASE = 0


def _draw(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(1, 5))
    k = int(rng.choice([2, 4, 8]))
    while nodes * k > 32:
        k = 1
    count = int(rng.integers(1, 4))
    sizes = [int(rng.choice([64, 128, 256, 512, 1024, 2048])) for _ in range(count)]
    return dict(
        seed=seed,
        nodes=nodes,
        k=k,
        sizes=sizes,
        replicate=[bool(rng.random() < 0.2) for _ in sizes],
        rounds=int(rng.integers(2, 5)),
    )


def _tables(case) -> dict:
    world = case["k"] * case["nodes"]
    tids = [f"t{i}" for i in range(len(case["sizes"]))]
    sizes = dict(zip(tids, case["sizes"]))
    replicate = dict(zip(tids, case["replicate"]))
    cluster = Cluster(case["nodes"], case["k"], 0.0)
    rng = np.random.default_rng(case["seed"] + 10_000)
    data = {tid: rng.integers(0, 256, sizes[tid], dtype=np.uint8) for tid in tids}
    return dict(
        world=world,
        tids=tids,
        sizes=sizes,
        replicate=replicate,
        cluster=cluster,
        data=data,
    )


def _build_direct(case) -> dict:
    built = _tables(case)
    src = FileStorage(built["cluster"], built["data"])
    dst = RankBuffers(built["cluster"], built["sizes"], built["world"])
    chain = gb.DirectDTensorsReadChain(
        src=_storage_view(src, built["tids"], built["sizes"], built["world"]),
        dst=_compute_view(
            dst, built["tids"], built["sizes"], built["world"], built["replicate"]
        ),
    )
    built.update(src=src, dst=dst, chain=chain, caches=[])
    return built


def _build_cached(case, layers) -> dict:
    shape = dict(
        case,
        layers=layers,
        steps=[2] * len(layers),
        width=[2] * len(layers),
        repeat=1,
        coordinator="file",
        bandwidth=0.0,
    )
    built = _build(shape)
    built["case"] = shape
    return built


def _reset_outputs(built) -> None:
    for tensors in built["dst"].buffers.values():
        for buf in tensors.values():
            buf[:] = GUARD
    built["src"].reads.clear()
    cluster = built["cluster"]
    for nic in cluster.nics.values():
        nic.rx = nic.tx = 0
    for node in cluster.storage_in:
        cluster.storage_in[node] = 0
    for cache in built["caches"]:
        cache.reads.clear()


def _run_round(built) -> None:
    world, cluster = built["world"], built["cluster"]
    results = {}

    def one(rank):
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True) for r in range(world)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 60
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), "chain_load hung"
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, bad


def _check_outputs(built) -> None:
    for rank, tensors in built["dst"].buffers.items():
        for tid, got in tensors.items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            assert crc_of(got.tobytes()) == crc_of(want.tobytes()), (rank, tid)
            np.testing.assert_array_equal(got, want, err_msg=f"{rank} {tid}")


def _wanted_bytes(built) -> int:
    return sum(
        hi - lo
        for tid, size in built["sizes"].items()
        for rank in range(built["world"])
        for lo, hi in [_wants(size, built["world"], built["replicate"][tid], rank)]
    )


def _storage_bytes(built) -> int:
    return sum(built["cluster"].storage_in.values())


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + SEEDS))
def test_cold_read_matches_source_every_round(seed):
    case = _draw(seed)
    built = _build_direct(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    want_bytes = _wanted_bytes(built)
    rounds = []
    for _ in range(case["rounds"]):
        _reset_outputs(built)
        start = time.monotonic()
        _run_round(built)
        rounds.append((time.monotonic() - start, _storage_bytes(built)))
        _check_outputs(built)
        assert rounds[-1][1] == want_bytes, (case, rounds[-1], want_bytes)
    assert [b for _w, b in rounds] == [want_bytes] * case["rounds"], (case, rounds)


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + SEEDS))
@pytest.mark.parametrize("layers", [1, 2])
def test_cache_refills_every_generation(seed, layers, tmp_path, monkeypatch):
    """Warm rounds reuse cache bytes instead of refilling them.

    Round 0 pays the storage read once per node. Later rounds ask the cache
    view whether the bytes are still there and adopt them, so storage stays
    at zero and fill records do not grow.
    """
    case = _draw(seed)
    nodes = list(range(case["nodes"]))
    case_layers = [nodes] * layers
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "file")
    built = _build_cached(case, case_layers)
    assert type(built["chain"].check_constraints()) is gb.Ok, case_layers
    shape = built["case"]
    cold_storage = sum(map(_bytes, _demand(shape, built)[0].values()))
    rounds = []
    for _index in range(case["rounds"]):
        _reset_outputs(built)
        start = time.monotonic()
        _run_round(built)
        rounds.append(
            (
                time.monotonic() - start,
                _storage_bytes(built),
                _cache_reads(built),
                _fill_count(built),
            )
        )
        _check_outputs(built)
    first = rounds[0]
    assert first[1] == cold_storage, (case_layers, first, cold_storage)
    per_round_fill = first[3]
    for index, (wall, storage, reads, fills) in enumerate(rounds[1:], start=1):
        assert storage == 0, (case_layers, index, storage)
        assert reads > 0, (case_layers, index, reads)
        assert fills == per_round_fill, (case_layers, index, fills)


def _fill_count(built) -> int:
    return sum(len(cache.fills) for cache in built["caches"])


def _cache_reads(built) -> int:
    return sum(len(cache.reads) for cache in built["caches"])
