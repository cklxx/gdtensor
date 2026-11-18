#!/usr/bin/env python3
"""Measure plan construction on real safetensors metadata without payload IO."""

import argparse
import gc
import hashlib
import json
import os
import resource
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gdtensor.builder as gb
import gdtensor_conn.file_cache as cache_impl
from gdtensor.internal.propagation import build_chain_plan, rank_copy_plans
from gdtensor_conn.formats import make_storage_view
from gdtensor_conn.iobackend import FileIO
from tests.support import cache_shapes as shapes
from tests.support.cluster_sim import Cluster, layer_targets

ITEMSIZE = {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "I64": 8, "I32": 4}


def header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        doc = json.loads(stream.read(size))
    doc.pop("__metadata__", None)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return doc, digest


def retag(table, entries):
    for named in table.named_d_tensors:
        entry = entries[named.global_tensor_id]
        named.d_tensor.global_tensor_meta = gb.GlobalTensorMeta(
            list(entry["shape"]), entry["dtype"], ITEMSIZE[entry["dtype"]]
        )


def build_chain(entries, checkpoint, cache_root, ranks):
    tids = list(entries)
    sizes = {
        tid: entry["data_offsets"][1] - entry["data_offsets"][0]
        for tid, entry in entries.items()
    }
    cluster = Cluster(ranks, 1, 0.0)
    storage = shapes._storage_view(None, tids, sizes, ranks)
    storage.physical_layout.file_paths = {
        gb.TensorComponentPos(tids[0], 0): checkpoint.name
    }
    retag(storage.d_tensor_table, entries)
    src = make_storage_view(
        tids,
        storage.d_tensor_table,
        storage.physical_layout,
        str(checkpoint.parent),
        FileIO(),
    )
    dst = cache_impl.SparseRankBuffers(cluster)
    compute = shapes._compute_view(dst, tids, sizes, ranks, dict.fromkeys(tids, False))
    retag(compute.d_tensor_table, entries)
    layer_nodes = [[0, min(1, ranks - 1)], list(range(ranks))]
    targets = layer_targets(cluster, layer_nodes)
    views = []
    for layer, nodes in enumerate(layer_nodes):
        cache = cache_impl.DiskNodeCache(
            cluster,
            sizes,
            targets[layer],
            str(Path(cache_root, f"layer{layer}")),
            cache_impl.Device(ranks, 0.0, 0.0),
        )
        view = shapes._cache_view(cache, tids, sizes, 1, nodes, 1)
        retag(view.d_tensor_table, entries)
        views.append(view)
    return gb.TwoLayerCachedDTensorsReadChain(
        src=src,
        small_cache=views[0],
        big_cache=views[1],
        dst=compute,
        num_ranks_per_node=1,
    )


def sample(chain):
    gc.collect()
    tracemalloc.start()
    start = time.perf_counter()
    plan = build_chain_plan(chain)
    planned = time.perf_counter()
    copies = [rank_copy_plans(plan, rank) for rank in plan.ranks]
    finished = time.perf_counter()
    _current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result = {
        "plan_build_seconds": planned - start,
        "copy_plans_seconds": finished - planned,
        "full_plan_seconds": finished - start,
        "python_peak_bytes": python_peak,
        "segments": sum(
            len(segments) for nodes in plan.segments for segments in nodes.values()
        ),
        "copy_pairs": sum(1 for legs in copies for leg in legs for _pair in leg),
    }
    del copies, plan
    return result


def summary(samples, key):
    values = [row[key] for row in samples]
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def memory_status():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in ("VmRSS", "VmHWM"):
            fields[f"{name}_bytes"] = int(value.strip().split()[0]) * 1024
    fields["ru_maxrss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    return fields


def _result(args, model_digest, samples):
    keys = (
        "plan_build_seconds",
        "copy_plans_seconds",
        "full_plan_seconds",
        "python_peak_bytes",
        "VmRSS_bytes",
        "VmHWM_bytes",
        "ru_maxrss_bytes",
    )
    return {
        "kind": "cache_plan_benchmark",
        "commit": os.environ.get("GDTENSOR_BENCH_COMMIT"),
        "tree": subprocess.check_output(["git", "write-tree"], text=True).strip(),
        "model_sha256": model_digest,
        "ranks": args.ranks,
        "coordinator": "not-used",
        "process_isolation": "one process per sample",
        "threads": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "samples": samples,
        "summary": {key: summary(samples, key) for key in keys},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ranks", type=int, choices=(20, 100), required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error("--repetitions must be between 1 and 20")
    if args.worker and args.repetitions != 1:
        parser.error("an isolated worker accepts exactly one repetition")
    if args.worker:
        entries, model_digest = header(args.checkpoint)
        with tempfile.TemporaryDirectory(prefix="gdtensor-plan-bench-") as cache_root:
            row = sample(build_chain(entries, args.checkpoint, cache_root, args.ranks))
        row.update(memory_status())
        row["pid"] = os.getpid()
        samples = [row]
    else:
        results = []
        with tempfile.TemporaryDirectory(prefix="gdtensor-plan-driver-") as directory:
            for index in range(args.repetitions):
                output = Path(directory, f"{index}.json")
                command = [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--ranks",
                    str(args.ranks),
                    "--repetitions",
                    "1",
                    "--output",
                    str(output),
                    "--worker",
                ]
                subprocess.run(
                    command, check=True, stdout=subprocess.DEVNULL, timeout=1800
                )
                results.append(json.loads(output.read_text()))
        samples = [result["samples"][0] for result in results]
        model_digest = results[0]["model_sha256"]
    result = _result(args, model_digest, samples)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
