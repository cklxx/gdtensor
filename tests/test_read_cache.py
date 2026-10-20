"""omnifs-host read path: L1 / L2 (P2P) / L3, single-flight, amplification."""
from __future__ import annotations

import os
import threading

from conftest import assert_all_ok, verify_fabric
from gdtensor.iobackend import FileIO, IoTracer, MiB, P2PReadCache
from gdtensor.models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from gdtensor.runner import load_threads, save_threads
from gdtensor.views import ComputeFabric


def test_single_flight_and_p2p(tmp_path):
    path = str(tmp_path / "blob")
    with open(path, "wb") as f:
        f.write(os.urandom(4 * MiB))
    cache = P2PReadCache(ranks_per_node=2)
    io = FileIO(IoTracer(), cache)
    want = open(path, "rb").read()
    barrier = threading.Barrier(8)
    out = {}

    def reader(rank):
        barrier.wait()  # 8 ranks (4 nodes) read the same 3 MiB at once
        out[rank] = io.pread(rank, path, 512 * 1024, 3 * MiB)

    ts = [threading.Thread(target=reader, args=(r,)) for r in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert all(v == want[512 * 1024: 512 * 1024 + 3 * MiB] for v in out.values())
    c = io.tracer.counters
    assert c["L3_block_bytes"] == 4 * MiB  # each 1 MiB block left the store once
    assert c["L2_block_bytes"] > 0  # other nodes got it from peers
    assert c["L1_bytes"] + c["L2_bytes"] + c["L3_bytes"] == c["requested_bytes"] == 8 * 3 * MiB


def test_load_read_amplification_below_two(tmp_path):
    m = ModelConfig(layers=4, hidden=128, ffn=256, vocab=1024)
    p = ParallelConfig(tp=2, dp=4, pp=2)
    root = str(tmp_path)
    ct = compute_table(m, p)
    st = storage_table(ct, p, "hf")
    lay = layout_builder("hf", st, p, "safetensors", 2)
    res, _ = save_threads(ct, st, lay, root, ComputeFabric(ct).allocate(fill=fill_pattern), p.world)
    assert_all_ok(res)
    io = FileIO(IoTracer(), P2PReadCache(ranks_per_node=4))
    fab = ComputeFabric(ct).allocate()
    res, _ = load_threads(ct, st, lay, root, fab, p.world, io)
    assert_all_ok(res)
    verify_fabric(fab, ct)
    file_bytes = sum(os.path.getsize(os.path.join(root, f)) for f in os.listdir(root))
    amp = io.tracer.counters["L3_block_bytes"] / file_bytes
    assert amp < 2.0, amp
    assert io.tracer.counters["requested_bytes"] > file_bytes  # DP replicas read the same bytes
