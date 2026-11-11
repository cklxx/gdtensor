"""Randomized stress for the co-located cache read chain.

Each seed draws one legal deployment: ranks per node, node count, one or two
cache layers, cache node sets, steps, concurrency width, tensor count and
size, compute placement, coordinator, start order, start delay, NIC bandwidth
and repeated loads. Payloads are random bytes, so outputs use exact-byte and
CRC checks.

An independent oracle derives the wanted bytes of every rank and the demand
of every cache node. It does not call the gdtensor planner. The test checks:

* every rank receives exact bytes with a matching CRC; guard bytes stay intact;
* every cache read finds valid bytes on the addressed node;
* every cache write is a local fill: filler node equals the DN node;
* every drain reads the serve node the oracle derives for the rank;
* every rank drains exactly its wanted bytes from the last cache layer, so
  each output byte passed through the cache;
* every demanded cache byte is filled once per generation, no more, no less;
* storage bytes are read once per first-layer node, by a rank on that node;
* the stat inbound and outbound traffic equal the simulated NIC bytes.

Each generation starts from empty cache bytes and zero counters and is checked
on its own, so a 2+0 fill split across generations fails.

A second sampler keeps the common shape of a real hybrid deployment: 2 or 4
nodes, 4 or 8 ranks per node, 2 or 4 KiB-class tensors, 4 or 8 steps, one or
two layers, file coordinator. It guards the sizes the uniform sampler never
reaches. GDTENSOR_STRESS_COMMON_SEEDS sets its seed count (default 8).

GDTENSOR_STRESS_SEEDS sets the seed count; GDTENSOR_STRESS_BASE sets the
first seed. A failure message prints the seed and the drawn case.
GDTENSOR_STRESS_REPORT names a file; each passed generation appends one JSON
line with the observed values next to the oracle values.
"""

import json
import os
import threading
import time

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from tests.support.cache_shapes import (
    GUARD,
    _build,
    _bytes,
    _demand,
    _rank_mask,
    _wants,
)
from tests.support.crc import crc_of


SEEDS = int(os.environ.get("GDTENSOR_STRESS_SEEDS", "128"))
SEED_BASE = int(os.environ.get("GDTENSOR_STRESS_BASE", "0"))
COMMON_SEEDS = int(os.environ.get("GDTENSOR_STRESS_COMMON_SEEDS", "8"))
FANOUT_SEEDS = int(os.environ.get("GDTENSOR_STRESS_FANOUT_SEEDS", "24"))


# ---------------------------------------------------------------- draw


def _subset(rng, items) -> list:
    items = list(items)
    count = int(rng.integers(1, len(items) + 1))
    return sorted(int(x) for x in rng.choice(items, size=count, replace=False))


def _draw(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nodes = int(rng.integers(1, 5))
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.5 else [_subset(rng, big), big]
    sizes = [int(rng.integers(1, 33)) for _ in range(int(rng.integers(1, 4)))]
    return dict(
        seed=seed,
        k=int(rng.integers(1, 4)),
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([1, 2, 3, 4, 8])) for _ in layers],
        width=[int(rng.integers(1, 4)) for _ in layers],
        sizes=sizes,
        replicate=[rng.random() < 0.3 for _ in sizes],
        coordinator=str(rng.choice(["file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 16384.0])),
        repeat=int(rng.integers(1, 3)),
    )


def _draw_fanout(seed: int) -> dict:
    """Two-layer fan-out shape: every rank node is a layer-1 cache.

    Layer 1 covers every node, so each rank drains its own node (no remote
    drain). Layer 0 is a strict random subset, so more layer-1 nodes than
    layer-0 parents exist; at least one parent must serve several children
    ("one fans out to ten"). One rank per node keeps each node its own
    filler. The fan-out assignment itself is checked from observed reads,
    not from the planner.
    """
    rng = np.random.default_rng(seed + 200_000)
    nodes = int(rng.choice([2, 3, 4]))
    count = int(rng.integers(1, nodes))  # strict subset: 1 .. nodes-1
    big = _subset_size(rng, range(nodes), count)
    sizes = [
        int(rng.choice([64, 128, 256, 512])) for _ in range(int(rng.integers(2, 5)))
    ]
    return dict(
        seed=seed,
        k=1,
        nodes=nodes,
        layers=[big, list(range(nodes))],
        steps=[int(rng.choice([1, 2, 4])), int(rng.choice([1, 2, 4]))],
        width=[int(rng.choice([1, 2, 3])) for _ in range(2)],
        sizes=sizes,
        replicate=[False] * len(sizes),
        coordinator=str(rng.choice(["file", "thread"])),
        bandwidth=0.0,
        repeat=2,
    )


def _subset_size(rng, items, count: int) -> list:
    items = list(items)
    return sorted(int(x) for x in rng.choice(items, size=count, replace=False))


def _draw_common(seed: int) -> dict:
    """Draw a common deployment shape, above the uniform sampler's ceiling.

    Ranks and DNs share a node, so k stays above one. Tensors reach 256 to
    1024 bytes and steps reach 8, so one cache layer costs real bytes and
    the segment count passes the 32-byte, 8-step limit of `_draw`.
    """
    rng = np.random.default_rng(seed + 100_000)
    nodes = int(rng.choice([2, 4]))
    big = _subset(rng, range(nodes))
    layers = [big] if rng.random() < 0.15 else [_subset(rng, big), big]
    count = int(rng.integers(2, 5))
    sizes = [int(rng.choice([256, 512, 1024])) for _ in range(count)]
    return dict(
        seed=seed,
        k=int(rng.choice([4, 8])),
        nodes=nodes,
        layers=layers,
        steps=[int(rng.choice([4, 8])) for _ in layers],
        width=[int(rng.choice([1, 2, 3])) for _ in layers],
        sizes=sizes,
        replicate=[rng.random() < 0.2 for _ in sizes],
        coordinator=str(rng.choice(["file", "file", "file", "thread"])),
        bandwidth=float(rng.choice([0.0, 0.0, 0.0, 65536.0])),
        repeat=2,
    )


# ---------------------------------------------------------------- oracle


def _counts(records, nodes, sizes) -> dict:
    counts = {
        n: {tid: np.zeros(size, np.int64) for tid, size in sizes.items()} for n in nodes
    }
    for node, tid, off, size in records:
        counts[node][tid][off : off + size] += 1
    return counts


# ---------------------------------------------------------------- run + check


def _run(case, built) -> None:
    world, cluster = built["world"], built["cluster"]
    rng = np.random.default_rng(case["seed"] + 20_000)
    order = rng.permutation(world).tolist()
    delay = rng.uniform(0.0, 0.004, world)
    results = {}

    def one(rank):
        time.sleep(float(delay[rank]))
        cluster.rank = rank
        results[rank] = gio.chain_load(rank, built["chain"])

    threads = [threading.Thread(target=one, args=(r,), daemon=True) for r in order]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 60
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not any(t.is_alive() for t in threads), ("chain_load hung", case)
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, (bad, case)


def _check_outputs(case, built) -> None:
    for rank, tensors in built["dst"].buffers.items():
        for tid, got in tensors.items():
            lo, hi = _wants(got.size, built["world"], built["replicate"][tid], rank)
            want = np.full(got.size, GUARD, np.uint8)
            want[lo:hi] = built["data"][tid][lo:hi]
            assert crc_of(got.tobytes()) == crc_of(want.tobytes()), (rank, tid, case)
            np.testing.assert_array_equal(got, want, err_msg=f"{rank} {tid} {case}")


def _check_cache_fills(case, built, demand) -> None:
    serve = built["targets"][-1]
    drains = [(r, t) for r, t, _s in built["caches"][-1].reads if t != serve(r)]
    assert not drains, ("drain off serve node", drains[:4], case)
    got = _drain_bytes(built)
    want = [_bytes(_rank_mask(case, built, r)) for r in range(built["world"])]
    assert got == want, ("drain bytes", got, want, case)
    for layer, cache in enumerate(built["caches"]):
        assert cache.wrong_node == 0, (layer, case)
        remote = [f for f in cache.fills if f[0] != f[1]]
        assert not remote, ("remote fill", layer, remote[:4], case)
        records = [(t, tid, o, s) for _a, t, tid, o, s in cache.fills]
        counts = _counts(records, case["layers"][layer], built["sizes"])
        _check_once(counts, demand[layer], ("cache", layer, case))


def _check_once(counts, demand, label) -> None:
    for node, tensors in demand.items():
        for tid, mask in tensors.items():
            want = mask.astype(np.int64)
            np.testing.assert_array_equal(
                counts[node][tid], want, err_msg=f"{label} node={node} tid={tid}"
            )


def _check_storage(case, built, demand) -> None:
    reads = built["src"].reads
    first = case["layers"][0]
    assert all(node in first for node, *_ in reads), ("storage read off DN", case)
    counts = _counts(reads, first, built["sizes"])
    _check_once(counts, demand[0], ("storage", case))


def _check_fanout(case, built, demand) -> None:
    """Independent fan-out oracle for the every-node-L1 / subset-L0 shape.

    Derived from observed transfers, not from the planner:
    * layer-1 drain nodes: every rank drains its own colocated node only;
    * each layer-1 node takes every demanded byte from exactly one L0 node;
    * every L0 node serves at least one child, and at least one serves many;
    * every child's parent is an L0 node (cross-layer hops only).
    """
    l0, l1 = case["layers"]
    assert l1 == list(range(case["nodes"])) and len(l0) < len(l1)
    assert case["k"] == 1
    # Layer-0 read records are (reader_rank, target_L0_node, size). One rank
    # per node makes the reader the layer-1 child node; derive its parent from
    # observed L0 reads, never from the planner.
    parent_of_l1: dict[int, int] = {}
    for reader, target, _size in built["caches"][0].reads:
        assert target in l0, ("L1 read from a non-L0 node", target, case)
        prev = parent_of_l1.setdefault(reader, target)
        assert prev == target, (
            "L1 node used multiple parents",
            reader,
            prev,
            target,
            case,
        )
    assert set(parent_of_l1) == set(l1), (
        "not every node read from L0",
        parent_of_l1,
        case,
    )
    children: dict[int, set] = {}
    for child, parent in parent_of_l1.items():
        children.setdefault(parent, set()).add(child)
    assert set(children) == set(l0), ("idle L0 node", children, l0, case)
    assert max(len(v) for v in children.values()) >= 2, (
        "no fan-out parent",
        children,
        case,
    )
    # Every demanded L1 byte is filled once on the local L1 child node.
    l1_fills = built["caches"][1].fills
    assert not [f for f in l1_fills if f[0] != f[1]], ("remote L1 fill", case)
    counts = _counts(
        [(t, tid, off, size) for _a, t, tid, off, size in l1_fills],
        l1,
        built["sizes"],
    )
    _check_once(counts, demand[1], ("fanout-l1", case))
    # Drains are local: the serving node of each rank is its own node.
    serve = built["targets"][-1]
    for rank in range(built["world"]):
        assert serve(rank) == rank // case["k"], (
            "remote drain in full-L1 shape",
            rank,
            case,
        )


def _check_traffic(case, built) -> None:
    cluster = built["cluster"]
    rx, tx = cluster.nic_bytes()
    nodes = sorted(cluster.nics)
    observed_in = [rx[n] + cluster.storage_in[n] for n in nodes]
    stat_in = list(gio.chain_stat_inbound_traffic(built["chain"]))
    stat_out = list(gio.chain_stat_outbound_traffic(built["chain"]))
    assert stat_out == tx, (stat_out, tx, case)
    assert stat_in == observed_in, (stat_in, observed_in, case)


def _drain_bytes(built) -> list:
    out = [0] * built["world"]
    for rank, _target, size in built["caches"][-1].reads:
        out[rank] += size
    return out


def _report(case, built, demand, gen) -> None:
    """Append one evidence line: observed values next to oracle values."""
    path = os.environ.get("GDTENSOR_STRESS_REPORT")
    if not path:
        return
    caches, cluster = built["caches"], built["cluster"]
    rx, tx = cluster.nic_bytes()
    row = dict(
        seed=case["seed"],
        gen=gen,
        ranks=built["world"],
        layers=len(case["layers"]),
        coordinator=case["coordinator"],
        fill_bytes=[sum(f[4] for f in c.fills) for c in caches],
        oracle_fill_bytes=[sum(map(_bytes, d.values())) for d in demand],
        drain_bytes=sum(_drain_bytes(built)),
        output_bytes=sum(
            _bytes(_rank_mask(case, built, r)) for r in range(built["world"])
        ),
        storage_bytes=sum(r[3] for r in built["src"].reads),
        oracle_storage_bytes=sum(map(_bytes, demand[0].values())),
        stat_out=list(gio.chain_stat_outbound_traffic(built["chain"])),
        nic_out=tx,
    )
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")


def _reset_generation(built) -> None:
    """Clear outputs, cache bytes and counters, so each generation must refill."""
    for tensors in built["dst"].buffers.values():
        for buf in tensors.values():
            buf[:] = GUARD
    for cache in built["caches"]:
        cache.mem.clear()
        cache.fills.clear()
        cache.reads.clear()
    built["src"].reads.clear()
    cluster = built["cluster"]
    for nic in cluster.nics.values():
        nic.rx = nic.tx = 0
    for node in cluster.storage_in:
        cluster.storage_in[node] = 0


@pytest.mark.parametrize("seed", range(SEED_BASE, SEED_BASE + SEEDS))
def test_random_cache_strategy_is_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw(seed), tmp_path, monkeypatch)


@pytest.mark.parametrize("seed", range(COMMON_SEEDS))
def test_common_shape_stays_correct(seed, tmp_path, monkeypatch):
    _verify_case(_draw_common(seed), tmp_path, monkeypatch)


@pytest.mark.parametrize("seed", range(FANOUT_SEEDS))
def test_every_node_cache_fanout_is_correct(seed, tmp_path, monkeypatch):
    case = _draw_fanout(seed)
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", case["coordinator"])
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    demand = _demand(case, built)
    for gen in range(case["repeat"]):
        _reset_generation(built)
        _run(case, built)
        _check_outputs(case, built)
        _check_cache_fills(case, built, demand)
        _check_storage(case, built, demand)
        _check_fanout(case, built, demand)


def _verify_case(case, tmp_path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", case["coordinator"])
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    demand = _demand(case, built)
    for gen in range(case["repeat"]):
        _reset_generation(built)
        _run(case, built)
        _check_outputs(case, built)
        _check_cache_fills(case, built, demand)
        _check_storage(case, built, demand)
        _check_traffic(case, built)
        _report(case, built, demand, gen)


@pytest.fixture(
    params=[
        pytest.param(("file", [[0, 2]]), id="file-one-layer"),
        pytest.param(("thread", [[0, 2]]), id="thread-one-layer"),
        pytest.param(("file", [[0], [0, 2]]), id="file-two-layers"),
        pytest.param(("thread", [[0], [0, 2]]), id="thread-two-layers"),
    ]
)
def changed_payload_generation(request, tmp_path, monkeypatch):
    coordinator, layers = request.param
    case = dict(
        seed=0,
        k=2,
        nodes=3,
        layers=layers,
        steps=[3] * len(layers),
        width=[2] * len(layers),
        sizes=[17, 9],
        replicate=[False, True],
        coordinator=coordinator,
        bandwidth=0.0,
        repeat=2,
    )
    monkeypatch.setenv("GDTENSOR_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GDTENSOR_CACHE_WAIT_SECONDS", "10")
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", coordinator)
    built = _build(case)
    assert type(built["chain"].check_constraints()) is gb.Ok, case
    demand = _demand(case, built)
    _reset_generation(built)
    _run(case, built)
    _check_outputs(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)

    # Change every byte in storage and in the separate oracle, on the same chain.
    old_data = built["data"]
    built["src"].payload ^= np.uint8(0xFF)
    built["data"] = {tid: value ^ np.uint8(0xFF) for tid, value in old_data.items()}
    _reset_generation(built)
    return case, built, demand, old_data


def test_second_generation_reads_changed_payload(changed_payload_generation):
    case, built, demand, _old_data = changed_payload_generation
    _run(case, built)
    _check_outputs(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)


def test_second_generation_rejects_stale_last_layer(
    changed_payload_generation, monkeypatch
):
    case, built, demand, old_data = changed_payload_generation
    cache = built["caches"][-1]
    write = cache.write_from_buffer

    def stale_write(global_range, write_from_buffer):
        result = write(global_range, write_from_buffer)
        if type(result) is not gb.Ok:
            return result
        _actor, target = cache._ends(global_range)
        tid = global_range.global_tensor_id
        off, size = global_range.global_tensor_offset, global_range.size
        with cache.lock:
            data, _valid = cache._buf(target, tid)
            data[off : off + size] = old_data[tid][off : off + size]
        return result

    monkeypatch.setattr(cache, "write_from_buffer", stale_write)
    _run(case, built)
    _check_cache_fills(case, built, demand)
    _check_storage(case, built, demand)
    _check_traffic(case, built)
    with pytest.raises(AssertionError):
        _check_outputs(case, built)


def test_draw_covers_every_strategy_axis():
    cases = [_draw(s) for s in range(SEED_BASE, SEED_BASE + SEEDS)]
    assert {len(c["layers"]) for c in cases} == {1, 2}
    assert {c["coordinator"] for c in cases} == {"file", "thread"}
    assert {c["repeat"] for c in cases} == {1, 2}
    assert {c["k"] for c in cases} == {1, 2, 3}
    assert any(any(c["replicate"]) for c in cases)
    assert any(c["bandwidth"] > 0 for c in cases)
    assert any(len(c["layers"][-1]) < c["nodes"] for c in cases)
    assert any(max(c["steps"]) * max(c["width"]) > min(c["sizes"]) for c in cases)
