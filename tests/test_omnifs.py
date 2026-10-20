"""OmniFS backend: snapshot protocol, chunk + KV store, sink, cache replicas.

Rules come from "OmniFS-Store 方案分析和选型" (2.1 EIC-KV ChunkStore)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import assert_all_ok, verify_fabric, verify_safetensors_dir
from gdtensor.coord import ThreadCoordinator
from gdtensor.iobackend import IoTracer, MiB, P2PReadCache, eic_replica_policy
from gdtensor.models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from gdtensor.omnifs import KvStore, Manifest, OmniFsConfig, OmniFsIO, sink_snapshot, split_by_chunk
from gdtensor.omnifs_view import export_hf_dir, make_omnifs_storage_view
from gdtensor.runner import _run_threads, compute_view
from gdtensor.api import d_tensors_view_rank_load, d_tensors_view_rank_save
from gdtensor.views import ComputeFabric

M = ModelConfig(layers=4, hidden=64, ffn=128, vocab=256, experts=4, optimizer=True)
P = ParallelConfig(tp=2, dp=2, pp=2)
SMALL_CHUNK = 64 * 1024  # small chunks so every file spans many chunks


def setup(tmp_path, chunk=SMALL_CHUNK, shards=3, m=M, p=P, aggregate_tp=False):
    cfg = OmniFsConfig(str(tmp_path / "omnifs"), "train", 1000, chunk, n_dn=4)
    ct = compute_table(m, p)
    st = storage_table(ct, p, "hf", aggregate_tp=aggregate_tp)
    lay = layout_builder("hf", st, p, "safetensors", shards)
    return cfg, ct, st, lay


def save(cfg, ct, st, lay, p=P, fail_rank=None):
    io = OmniFsIO(cfg)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io, ThreadCoordinator(p.world))
    fab = ComputeFabric(ct).allocate(fill=fill_pattern)

    def fn(r):
        if r == fail_rank:
            s = sv.begin_save(sv, r).unwrap()
            return s.abort()
        return d_tensors_view_rank_save(r, compute_view(ct, fab, r), sv)

    return _run_threads(p.world, fn), io


def load(cfg, st, lay, ct2, p2, cache=None):
    io = OmniFsIO(cfg, IoTracer(), cache)
    sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, io)
    fab = ComputeFabric(ct2).allocate()
    res = _run_threads(p2.world, lambda r: d_tensors_view_rank_load(r, compute_view(ct2, fab, r), sv))
    return res, fab, io


# ------------------------------------------------------------------ mapping


def test_split_by_chunk_rule():
    # chunk id = offset // chunk; a range crossing a boundary splits in two.
    assert list(split_by_chunk(100, 50, 128)) == [(0, 100, 0, 28), (1, 0, 28, 22)]
    assert list(split_by_chunk(256, 128, 128)) == [(2, 0, 0, 128)]


def test_roundtrip_hf_output_and_layout(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    res, io = save(cfg, ct, st, lay)
    assert_all_ok(res)
    m = Manifest.load(cfg.manifest_path)
    assert m.state == "committed"
    assert "model.safetensors.index.json" in m.files and m.files["model.safetensors.index.json"]["kind"] == "json"
    # Export through the store and check with the official parser.
    out = str(tmp_path / "export")
    export_hf_dir(cfg, out)
    verify_safetensors_dir(out, st)
    # Many writers per chunk; every KV range lies inside one chunk.
    lay_kv = KvStore(cfg).layout()
    writers = [len({e.rank for e in lst}) for lst in lay_kv.values()]
    assert max(writers) >= 4
    assert all(e.chunk_offset + e.length <= cfg.chunk_bytes for lst in lay_kv.values() for e in lst)
    # Non-affinity placement: KVs spread over all DNs.
    assert {e.dn for lst in lay_kv.values() for e in lst} == set(range(cfg.n_dn))
    # Elastic load with another parallelism.
    p2 = ParallelConfig(tp=4, dp=1, pp=2)
    ct2 = compute_table(M, p2)
    res2, fab2, _ = load(cfg, st, lay, ct2, p2)
    assert_all_ok(res2)
    verify_fabric(fab2, ct2)


def test_two_level_directory(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    assert os.path.exists(os.path.join(cfg.root, "meta", "train", "step-1000", "manifest.json"))


# ------------------------------------------------------------------ snapshot protocol


def test_uncommitted_is_invisible_and_rewritable(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    res, _ = save(cfg, ct, st, lay, fail_rank=3)
    assert res[3].ok and not res[0].ok
    assert Manifest.load(cfg.manifest_path).state == "prepared"
    res2, _, _ = load(cfg, st, lay, ct, P)
    assert not res2[0].ok and "not committed" in res2[0].error
    # Rewrite the same snapshot: succeeds and is complete.
    assert_all_ok(save(cfg, ct, st, lay)[0])
    out = str(tmp_path / "export")
    export_hf_dir(cfg, out)
    verify_safetensors_dir(out, st)


def test_committed_snapshot_is_immutable(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    res, _ = save(cfg, ct, st, lay)
    assert not res[0].ok and "already committed" in res[0].error


def test_retry_same_range_is_idempotent(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    kv = KvStore(cfg)
    e = next(iter(kv.layout().values()))[0]
    data = kv.read_chunk(0, e.blob_id, e.chunk, e.chunk_offset, e.length)
    kv.put(e.rank, e.blob_id, e.chunk, e.chunk_offset, data)  # retry: same range, same bytes
    kv.flush(e.rank)
    assert len(kv.layout(refresh=True)[(e.blob_id, e.chunk)]) == len(KvStore(cfg).layout()[(e.blob_id, e.chunk)])


def test_conflicting_rewrite_and_overlap_are_rejected(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    e = next(iter(KvStore(cfg).layout().values()))[0]
    kv = KvStore(cfg)
    kv.put(e.rank, e.blob_id, e.chunk, e.chunk_offset, b"\xff" * e.length)
    kv.flush(e.rank)
    with pytest.raises(Exception, match="KV conflict"):
        kv.layout(refresh=True)
    cfg2, ct2, st2, lay2 = setup(tmp_path / "b")
    assert_all_ok(save(cfg2, ct2, st2, lay2)[0])
    e2 = next(iter(KvStore(cfg2).layout().values()))[0]
    kv2 = KvStore(cfg2)
    kv2.put(e2.rank, e2.blob_id, e2.chunk, e2.chunk_offset + 1, b"\x00" * max(e2.length - 1, 1))
    kv2.flush(e2.rank)
    with pytest.raises(Exception, match="overlap|conflict"):
        kv2.layout(refresh=True)


def test_gap_is_detected_before_commit(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    m = Manifest.load(cfg.manifest_path)
    m.files["model-00001-of-00003.safetensors"]["size"] += 1  # one byte nobody writes
    with pytest.raises(Exception, match="covers|gap"):
        KvStore(cfg).check_full_coverage(m)


# ------------------------------------------------------------------ sink


def test_sink_to_hdd_keeps_manifest_and_routes_reads(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    before = open(cfg.manifest_path, "rb").read()
    hdfs = str(tmp_path / "hdfs")
    stats = sink_snapshot(cfg, hdfs)
    assert open(cfg.manifest_path, "rb").read() == before  # sink never edits the manifest
    assert not os.path.exists(cfg.store_dir)  # SSD reclaimed
    hf_dir = os.path.join(hdfs, "train", "step-1000")
    verify_safetensors_dir(hf_dir, st)  # plain HF directory on HDD
    assert stats["files"] == len(Manifest.load(cfg.manifest_path).files)
    p2 = ParallelConfig(tp=1, dp=4, pp=2)
    ct2 = compute_table(M, p2)
    res, fab, io = load(cfg, st, lay, ct2, p2)  # reads route by sink.json
    assert_all_ok(res)
    verify_fabric(fab, ct2)
    levels = {r.level for r in io.tracer.records if r.op == "pread"}
    assert levels == {"direct-hdd"}


def test_sink_requires_committed(tmp_path):
    cfg, ct, st, lay = setup(tmp_path)
    save(cfg, ct, st, lay, fail_rank=0)
    with pytest.raises(Exception, match="committed"):
        sink_snapshot(cfg, str(tmp_path / "hdfs"))


# ------------------------------------------------------------------ cache replicas


def test_eic_replica_cap(tmp_path):
    cfg, ct, st, lay = setup(tmp_path, chunk=4 * MiB)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    p2 = ParallelConfig(tp=1, dp=8, pp=1)  # 8 nodes, every node reads every byte
    ct2 = compute_table(M, p2)
    cache = P2PReadCache(ranks_per_node=1, max_replicas=eic_replica_policy)
    res, fab, io = load(cfg, st, lay, ct2, p2, cache)
    assert_all_ok(res)
    verify_fabric(fab, ct2)
    reps = [len(h) for (path, _), h in cache._holders.items() if path.endswith(".safetensors")]
    assert max(reps) <= 3 and max(reps) == 3
    m = Manifest.load(cfg.manifest_path)
    file_bytes = sum(f["size"] for f in m.files.values() if f["kind"] == "safetensors")
    assert io.tracer.counters["L3_block_bytes"] / file_bytes < 2.0


def test_ssd_miss_does_not_imply_sunk(tmp_path):
    """Without sink.json the reader must not fall back to HDD on an SSD miss."""
    cfg, ct, st, lay = setup(tmp_path)
    assert_all_ok(save(cfg, ct, st, lay)[0])
    sink_snapshot(cfg, str(tmp_path / "hdfs"))
    os.remove(cfg.sink_path)  # sink state not published
    res, _, _ = load(cfg, st, lay, ct, P)
    assert not res[0].ok
