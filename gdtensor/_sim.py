"""gdtensor simulator CLI: save with concurrent pwrite, then load.

The store is a local directory. Formats: safetensors (tmp + rename,
index.json) or distcp (.metadata). The CLI reports write/read patterns and
verifies the loaded bytes.

Example:
  python -m gdtensor._sim --root /tmp/ck --tp 4 --dp 4 --pp 2 --optimizer
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

from ._iobackend import FileIO, IoTracer, MiB
from ._layout import build_payload_index
from ._dtensor import tensor_ids_of
from ._models import ModelConfig, ParallelConfig, compute_table, fill_pattern, layout_builder, storage_table
from ._plan import check_physical_layout
from ._runner import load_threads, run_processes, save_threads
from ._views import ComputeFabric

SIZE_BINS = [("<4KiB", 4 << 10), ("4-64KiB", 64 << 10), ("64KiB-1MiB", MiB), ("1-16MiB", 16 * MiB), (">=16MiB", 1 << 62)]


def size_hist(sizes: List[int]) -> Dict[str, int]:
    out = {name: 0 for name, _ in SIZE_BINS}
    for s in sizes:
        for name, hi in SIZE_BINS:
            if s < hi:
                out[name] += 1
                break
    return out


def write_report(records) -> dict:
    w = [r for r in records if r.op == "pwrite"]
    sizes = [r.size for r in w]
    per_file = defaultdict(set)
    unaligned = 0
    for r in w:
        per_file[r.path].add(r.rank)
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
        "files": len(per_file),
        "max_writers_per_file": max((len(v) for v in per_file.values()), default=0),
        "ops_per_rank_min_max": [min(per_rank_ops.values(), default=0), max(per_rank_ops.values(), default=0)],
    }


def read_report(records, file_bytes: int) -> dict:
    r = [x for x in records if x.op == "pread"]
    read_bytes = sum(x.size for x in r)
    return {
        "pread_ops": len(r),
        "pread_bytes": read_bytes,
        "unique_file_bytes": file_bytes,
        "read_amplification": round(read_bytes / max(file_bytes, 1), 3),
    }


def dump_trace(records, path: str, t0: float) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_us", "rank", "op", "file", "offset", "size"])
        for r in sorted(records, key=lambda r: r.t):
            wr.writerow([int((r.t - t0) * 1e6), r.rank, r.op, os.path.basename(r.path).replace(".tmp", ""),
                         r.offset, r.size])


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
    ap.add_argument("--aggregate-tp", action="store_true",
                    help="business layout: one TP rank stores each dim1-sharded tensor whole")
    ap.add_argument("--processes", action="store_true", help="fork one process per rank for save")
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
    st = storage_table(ct, p, mode, aggregate_tp=a.aggregate_tp)
    lay = layout_builder(a.layout, st, p, a.format, a.hf_shards)
    check_physical_layout(st, lay).unwrap()
    index = build_payload_index(st, lay)
    t_plan = time.perf_counter() - t

    # ---- save
    t0 = time.perf_counter()
    if a.processes:
        res, wrecs, _ = run_processes("save", ct, st, lay, a.root, p.world, fill=fill_pattern)
    else:
        fab = ComputeFabric(ct).allocate(fill=fill_pattern)
        res, wio = save_threads(ct, st, lay, a.root, fab, p.world)
        wrecs = wio.tracer.records
    t_save = time.perf_counter() - t0
    bad = {r: x.value for r, x in res.items() if not x.ok}
    if bad:
        raise SystemExit(f"save failed: {bad}")

    # ---- load
    data_bytes = sum(os.path.getsize(os.path.join(a.root, f)) for f in index.files)
    ct2 = compute_table(m, lp)
    fab2 = ComputeFabric(ct2).allocate()
    t1 = time.perf_counter()
    rio = FileIO(IoTracer())
    res2, _ = load_threads(ct2, st, lay, a.root, fab2, lp.world, rio)
    t_load = time.perf_counter() - t1
    bad = {r: x.value for r, x in res2.items() if not x.ok}
    if bad:
        raise SystemExit(f"load failed: {bad}")
    ok = all(np.array_equal(buf, fill_pattern(ct2.named_d_tensors[ct2._order[tid]], fab2.intervals(r, tid)))
             for r, bufs in fab2.buffers.items() for tid, buf in bufs.items())

    report = {
        "model": m.__dict__,
        "save_parallel": {"tp": p.tp, "dp": p.dp, "pp": p.pp, "world": p.world},
        "load_parallel": {"tp": lp.tp, "dp": lp.dp, "pp": lp.pp, "world": lp.world},
        "layout": a.layout, "format": a.format, "tensors": len(st.named_d_tensors),
        "aggregate_tp": a.aggregate_tp,
        "files": {path: fp.payload_size for path, fp in sorted(index.files.items())},
        "seconds": {"plan_layout": round(t_plan, 3), "save": round(t_save, 3), "load": round(t_load, 3)},
        "write": write_report(wrecs),
        "read": read_report(rio.tracer.records, data_bytes),
        "load_bytes_verified": bool(ok),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    dump_trace(wrecs, os.path.join(out_dir, "write_trace.csv"), t0)
    dump_trace(rio.tracer.records, os.path.join(out_dir, "read_trace.csv"), t1)
    print(json.dumps({k: report[k] for k in ("save_parallel", "load_parallel", "layout", "format",
                                              "aggregate_tp", "seconds", "write", "read",
                                              "load_bytes_verified")}, indent=1, default=str))
    return report


if __name__ == "__main__":
    main()
