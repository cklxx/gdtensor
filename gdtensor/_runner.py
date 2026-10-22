"""Multi-rank runner: one thread (or forked process) per rank.

Threads share one ComputeFabric (device memory of all ranks), so
compute->compute reshard works. Processes give real OS-level concurrency for
pwrite; they support save and load only.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from ._api import d_tensors_view_rank_load, d_tensors_view_rank_reshard, d_tensors_view_rank_save
from ._coord import ProcessCoordinator, ThreadCoordinator
from ._dtensor import DTensorTable
from ._formats import make_storage_view
from ._iobackend import FileIO
from ._layout import PhysicalLayoutTable
from ._result import Err, Ok
from ._views import ComputeDataAccessor, ComputeDTensorsView, ComputeFabric


def compute_view(table: DTensorTable, fabric: ComputeFabric, rank: int) -> ComputeDTensorsView:
    return ComputeDTensorsView(table.tensor_ids, table, ComputeDataAccessor(rank, fabric))


def _run_threads(world: int, fn: Callable[[int], object]) -> Dict[int, object]:
    with ThreadPoolExecutor(max_workers=world) as ex:
        futs = {r: ex.submit(fn, r) for r in range(world)}
    out = {}
    for r, f in futs.items():
        try:
            out[r] = f.result()
        except Exception as exc:  # noqa: BLE001
            out[r] = Err(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    return out


def save_threads(table, storage_table, layout, root, fabric, world, io: Optional[FileIO] = None):
    io = io or FileIO()
    sv = make_storage_view(storage_table.tensor_ids, storage_table, layout, root, io, ThreadCoordinator(world))
    res = _run_threads(world, lambda r: d_tensors_view_rank_save(r, compute_view(table, fabric, r), sv))
    return res, io


def load_threads(table, storage_table, layout, root, fabric, world, io: Optional[FileIO] = None):
    io = io or FileIO()
    sv = make_storage_view(storage_table.tensor_ids, storage_table, layout, root, io)
    res = _run_threads(world, lambda r: d_tensors_view_rank_load(r, compute_view(table, fabric, r), sv))
    return res, io


def reshard_threads(world: int, make_src: Callable[[int], object], make_dst: Callable[[int], object]):
    return _run_threads(world, lambda r: d_tensors_view_rank_reshard(r, make_src(r), make_dst(r)))


# ------------------------------------------------------------------ processes


def _proc_main(rank, op, table, storage_table, layout, root, coord, fill, q):
    try:
        io = FileIO()
        fabric = ComputeFabric(table).allocate([rank], fill=fill if op == "save" else None)
        sv = make_storage_view(storage_table.tensor_ids, storage_table, layout, root, io, coord)
        cv = compute_view(table, fabric, rank)
        if op == "save":
            res = d_tensors_view_rank_save(rank, cv, sv)
        else:
            res = d_tensors_view_rank_load(rank, cv, sv)
        q.put((rank, res.ok, getattr(res, "error", ""), io.tracer.records,
               fabric.buffers.get(rank, {}) if op == "load" else {}))
    except Exception as exc:  # noqa: BLE001
        q.put((rank, False, f"{exc}\n{traceback.format_exc()}", [], {}))


def run_processes(op: str, table, storage_table, layout, root, world, fill=None):
    """Fork one process per rank. Returns ({rank: Result}, records, buffers)."""
    ctx = mp.get_context("fork")
    coord = ProcessCoordinator(world, ctx) if op == "save" else None
    q = ctx.Queue()
    procs = [ctx.Process(target=_proc_main, args=(r, op, table, storage_table, layout, root, coord, fill, q))
             for r in range(world)]
    for p in procs:
        p.start()
    results, records, buffers = {}, [], {}
    for _ in procs:
        rank, ok, err, recs, bufs = q.get(timeout=600)
        results[rank] = Ok(None) if ok else Err(err)
        records.extend(recs)
        if bufs:
            buffers[rank] = bufs
    for p in procs:
        p.join(timeout=60)
    return results, records, buffers
