"""gdtensor simulator CLI: save (concurrent pwrite) + load (L1/L2/L3 P2P).

Backends:
* file   : local files. safetensors (tmp+rename, index.json) or distcp
           (.metadata).
* omnifs : OmniFS-Store (design 2.1). manifest (prepared/committed) +
           64 MiB chunks + variable KVs on simulated DNs. --sink sinks the
           snapshot to an HF directory on simulated HDFS before load.

Example:
  python -m gdtensor.sim --root /tmp/ck --tp 4 --dp 4 --pp 2 --optimizer \
      --backend omnifs --sink --cache eic --ranks-per-node 8
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .iobackend import CHUNK_BYTES, FileIO, IoTracer, MiB, P2PReadCache, eic_replica_policy
from .layout import build_payload_index
from .models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from .omnifs import Manifest, OmniFsConfig, OmniFsIO, sink_snapshot
from .omnifs_view import make_omnifs_storage_view
from .plan import check_physical_layout
from .runner import load_threads, reshard_threads, run_processes, save_threads, compute_view
from .views import ComputeFabric

SIZE_BINS = [("<4KiB", 4 << 10), ("4-64KiB", 64 << 10), ("64KiB-1MiB", MiB), ("1-16MiB", 16 * MiB), (">=16MiB", 1 << 62)]


def size_hist(sizes: List[int]) -> Dict[str, int]:
    out = {name: 0 for name, _ in SIZE_BINS}
    for s in sizes:
        for name, hi in SIZE_BINS:
            if s < hi:
                out[name] += 1
                break
    return out


def write_report(records, root: str) -> dict:
    w = [r for r in records if r.op == "pwrite"]
    sizes = [r.size for r in w]
    per_file = defaultdict(set)
    chunk_writers = defaultdict(set)
    chunk_ops = 0
    unaligned = 0
    for r in w:
        per_file[r.path].add(r.rank)
        c0, c1 = r.offset // CHUNK_BYTES, (r.offset + r.size - 1) // CHUNK_BYTES
        chunk_ops += c1 - c0 + 1  # a write that crosses a chunk boundary splits
        for c in range(c0, c1 + 1):
            chunk_writers[(r.path, c)].add(r.rank)
        unaligned += r.offset % MiB != 0
    per_rank_ops = Counter(r.rank for r in w)
    return {
        "pwrite_ops": len(w),
        "pwrite_bytes": sum(sizes),
        "size_p50": int(np.percentile(sizes, 50)) if sizes else 0,
        "size_p99": int(np.percentile(sizes, 99)) if sizes else 0,
        "size_hist": size_hist(sizes),
        "frag_lt_64KiB_ratio": round(sum(s < (64 << 10) for s in sizes) / max(len(sizes), 1), 4),
        "unaligned_1MiB_ratio": round(unaligned / max(len(w), 1), 4),
        "chunk_ops_after_split": chunk_ops,
        "files": len(per_file),
        "max_writers_per_file": max((len(v) for v in per_file.values()), default=0),
        "max_writers_per_64MiB_chunk": max((len(v) for v in chunk_writers.values()), default=0),
        "ops_per_rank_min_max": [min(per_rank_ops.values(), default=0), max(per_rank_ops.values(), default=0)],
    }


def kv_report(records) -> dict:
    kv = [r for r in records if r.op == "kv_put"]
    dns = Counter(r.level for r in kv)
    return {
        "kv_put_ops": len(kv),
        "kv_size_p50": int(np.percentile([r.size for r in kv], 50)) if kv else 0,
        "kv_size_p99": int(np.percentile([r.size for r in kv], 99)) if kv else 0,
        "kv_per_dn": dict(sorted(dns.items())),
    }


def read_report(counters: dict, records, file_bytes: int) -> dict:
    l3 = [r for r in records if r.op == "pread"]
    l3_block = counters.get("L3_block_bytes", 0) or counters.get("L3_bytes", 0)
    return {
        "requested_bytes": counters["requested_bytes"],
        "L1_bytes": counters.get("L1_bytes", 0),
        "L2_p2p_bytes": counters.get("L2_bytes", 0),
        "L2_p2p_transfer_bytes": counters.get("L2_block_bytes", counters.get("L2_bytes", 0)),
        "L3_store_bytes": counters.get("L3_bytes", 0),
        "L3_store_read_bytes": l3_block,
        "L3_ops": len(l3),
        "unique_file_bytes": file_bytes,
        "store_read_amplification": round(l3_block / max(file_bytes, 1), 3),
        "served_ratio_L1_L2_L3": [round(counters.get(k, 0) / max(counters["requested_bytes"], 1), 4)
                                  for k in ("L1_bytes", "L2_bytes", "L3_bytes")],
    }


def dump_trace(records, path: str, t0: float) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_us", "rank", "op", "level", "file", "offset", "size", "chunk"])
        for r in sorted(records, key=lambda r: r.t):
            wr.writerow([int((r.t - t0) * 1e6), r.rank, r.op, r.level, os.path.basename(r.path).replace(".tmp", ""),
                         r.offset, r.size, r.offset // CHUNK_BYTES])


def _make_cache(kind: str, ranks_per_node: int):
    if kind == "none":
        return None
    if kind == "host":
        return P2PReadCache(ranks_per_node)  # replicas grow with readers
    if kind == "eic":
        return P2PReadCache(ranks_per_node, max_replicas=eic_replica_policy)
    raise ValueError(kind)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--ffn", type=int, default=1408)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--optimizer", action="store_true", help="add ZeRO-sharded Adam m/v (F32)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--load-tp", type=int)
    ap.add_argument("--load-dp", type=int)
    ap.add_argument("--load-pp", type=int)
    ap.add_argument("--layout", default="hf", choices=["hf", "single", "per_rank", "rank_multi", "per_stage"])
    ap.add_argument("--format", default="safetensors", choices=["safetensors", "distcp"])
    ap.add_argument("--hf-shards", type=int, default=2)
    ap.add_argument("--ranks-per-node", type=int, default=8)
    ap.add_argument("--backend", default="file", choices=["file", "omnifs"])
    ap.add_argument("--omnifs-chunk", type=int, default=CHUNK_BYTES, help="chunk bytes (default 64 MiB)")
    ap.add_argument("--n-dn", type=int, default=8, help="simulated DN count")
    ap.add_argument("--sink", action="store_true", help="[omnifs] sink SSD -> HDFS before load")
    ap.add_argument("--cache", default="host", choices=["host", "eic", "none"])
    ap.add_argument("--aggregate-tp", action="store_true",
                    help="gather TP dim1 shards inside a TP group, one pwrite per slot")
    ap.add_argument("--processes", action="store_true", help="[file] fork one process per rank for save")
    ap.add_argument("--report", default=None)
    a = ap.parse_args(argv)

    m = ModelConfig(a.layers, a.hidden, a.ffn, a.vocab, a.experts, optimizer=a.optimizer)
    p = ParallelConfig(a.tp, a.dp, a.pp)
    lp = ParallelConfig(a.load_tp or a.tp, a.load_dp or a.dp, a.load_pp or a.pp)
    out_dir = a.report or a.root + "_report"
    shutil.rmtree(a.root, ignore_errors=True)
    os.makedirs(a.root)
    os.makedirs(out_dir, exist_ok=True)

    t = time.perf_counter()
    ct = compute_table(m, p)
    mode = "hf" if a.layout in ("hf", "single", "per_stage") else "mirror"
    st = storage_table(ct, p, mode, aggregate_tp=(a.aggregate_tp and a.backend == "omnifs"))
    fmt = "safetensors" if a.backend == "omnifs" else a.format
    lay = layout_builder(a.layout, st, p, fmt, a.hf_shards)
    check_physical_layout(st, lay).unwrap()
    index = build_payload_index(st, lay)
    t_plan = time.perf_counter() - t

    extra_kv = {}
    # ---- save
    t0 = time.perf_counter()
    if a.backend == "file":
        if a.processes:
            res, wrecs, _ = run_processes("save", ct, st, lay, a.root, p.world, fill=fill_pattern)
        else:
            fab = ComputeFabric(ct).allocate(fill=fill_pattern)
            res, wio = save_threads(ct, st, lay, a.root, fab, p.world)
            wrecs = wio.tracer.records
    else:
        cfg = OmniFsConfig(a.root, "proj", 0, a.omnifs_chunk, a.n_dn)
        wio = OmniFsIO(cfg)
        from .coord import ThreadCoordinator
        sv = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, wio, ThreadCoordinator(p.world))
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        res = reshard_threads(p.world, lambda r: compute_view(ct, fab, r), lambda r: sv,
                              coalesce_storage=a.aggregate_tp)
        wrecs = wio.tracer.records
    t_save = time.perf_counter() - t0
    bad = {r: x.error for r, x in res.items() if not x.ok}
    if bad:
        raise SystemExit(f"save failed: {bad}")

    # ---- optional sink (omnifs only)
    sink_stats = None
    if a.backend == "omnifs" and a.sink:
        sink_stats = sink_snapshot(cfg, os.path.join(a.root, "hdfs"))

    # ---- load with read cache
    if a.backend == "file":
        data_bytes = sum(os.path.getsize(os.path.join(a.root, f)) for f in index.files)
    else:
        data_bytes = sum(f["size"] for f in Manifest.load(cfg.manifest_path).files.values()
                         if f["kind"] == "safetensors")
    cache = _make_cache(a.cache, a.ranks_per_node)
    ct2 = compute_table(m, lp)
    fab2 = ComputeFabric(ct2).allocate()
    t1 = time.perf_counter()
    if a.backend == "file":
        rio = FileIO(IoTracer(), cache)
        res2, _ = load_threads(ct2, st, lay, a.root, fab2, lp.world, rio)
    else:
        rio = OmniFsIO(cfg, IoTracer(), cache)
        sv2 = make_omnifs_storage_view(st.tensor_ids, st, lay, cfg, rio)
        res2 = reshard_threads(lp.world, lambda r: sv2, lambda r: compute_view(ct2, fab2, r))
    t_load = time.perf_counter() - t1
    bad = {r: x.error for r, x in res2.items() if not x.ok}
    if bad:
        raise SystemExit(f"load failed: {bad}")
    ok = all(np.array_equal(buf, fill_pattern(ct2.named_d_tensors[ct2._order[tid]], fab2.intervals(r, tid)))
             for r, bufs in fab2.buffers.items() for tid, buf in bufs.items())

    report = {
        "model": m.__dict__,
        "backend": a.backend,
        "save_parallel": {"tp": p.tp, "dp": p.dp, "pp": p.pp, "world": p.world},
        "load_parallel": {"tp": lp.tp, "dp": lp.dp, "pp": lp.pp, "world": lp.world},
        "layout": a.layout, "format": fmt, "tensors": len(st.named_d_tensors),
        "aggregate_tp": a.aggregate_tp,
        "cache": a.cache,
        "files": {path: fp.payload_size for path, fp in sorted(index.files.items())},
        "seconds": {"plan_layout": round(t_plan, 3), "save": round(t_save, 3), "load": round(t_load, 3)},
        "write": write_report(wrecs, a.root),
        "read": read_report(rio.tracer.counters, rio.tracer.records, data_bytes),
        "load_bytes_verified": bool(ok),
    }
    if a.backend == "omnifs":
        report["write"].update(kv_report(wrecs))
        if sink_stats is not None:
            report["sink"] = sink_stats
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    dump_trace(wrecs, os.path.join(out_dir, "write_trace.csv"), t0)
    dump_trace(rio.tracer.records, os.path.join(out_dir, "read_trace.csv"), t1)
    print(json.dumps({k: report[k] for k in ("backend", "save_parallel", "load_parallel", "layout", "format",
                                              "aggregate_tp", "cache", "seconds", "write", "read",
                                              "load_bytes_verified")}, indent=1, default=str))
    return report


if __name__ == "__main__":
    main()
