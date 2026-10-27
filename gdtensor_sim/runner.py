"""Multi-rank runner: one thread (or forked process) per rank.

Threads share one ComputeFabric (device memory of all ranks), so a simulated
collective can read another rank's shard. Processes give real OS-level
concurrency for pwrite; they support save and load only.
"""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Optional

from gdtensor import Result
from gdtensor.io import rank_load, rank_save
from .coord import ProcessCoordinator, ThreadCoordinator
from gdtensor.builder import ComputeDTensorsView, DTensorTable, GDTensorIOError
from .formats import make_storage_view
from .iobackend import FileIO
from gdtensor.builder import Err, Ok
from gdtensor_sim.fabric import ComputeDataAccessor, ComputeFabric


def compute_view(
    table: DTensorTable, fabric: ComputeFabric, rank: int
) -> ComputeDTensorsView:
    return ComputeDTensorsView(
        [nd.global_tensor_id for nd in table.named_d_tensors],
        table,
        ComputeDataAccessor(rank, fabric),
    )


def _run_threads(
    world: int, fn: Callable[[int], Result[None]]
) -> Dict[int, Result[None]]:
    with ThreadPoolExecutor(max_workers=world) as ex:
        futs = {r: ex.submit(fn, r) for r in range(world)}
    out = {}
    for r, f in futs.items():
        try:
            out[r] = f.result()
        except Exception as exc:  # noqa: BLE001
            out[r] = Err(
                GDTensorIOError(
                    operation="runner",
                    msg=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
    return out


def save_threads(
    table, storage_table, layout, root, fabric, world, io: Optional[FileIO] = None
):
    io = io or FileIO()
    sv = make_storage_view(
        [nd.global_tensor_id for nd in storage_table.named_d_tensors],
        storage_table,
        layout,
        root,
        io,
        ThreadCoordinator(world),
    )
    res = _run_threads(
        world, lambda r: rank_save(r, compute_view(table, fabric, r), sv)
    )
    return res, io


def load_threads(
    table, storage_table, layout, root, fabric, world, io: Optional[FileIO] = None
):
    io = io or FileIO()
    sv = make_storage_view(
        [nd.global_tensor_id for nd in storage_table.named_d_tensors],
        storage_table,
        layout,
        root,
        io,
    )
    res = _run_threads(
        world, lambda r: rank_load(r, compute_view(table, fabric, r), sv)
    )
    return res, io


# ------------------------------------------------------------------ processes


def _proc_main(rank, op, table, storage_table, layout, root, coord, fill, q):
    try:
        io = FileIO()
        fabric = ComputeFabric(table).allocate(
            [rank], fill=fill if op == "save" else None
        )
        sv = make_storage_view(
            [nd.global_tensor_id for nd in storage_table.named_d_tensors],
            storage_table,
            layout,
            root,
            io,
            coord,
            dcp_mode="process" if layout.container_format == "distcp" else "thread",
        )
        cv = compute_view(table, fabric, rank)
        if op == "save":
            res = rank_save(rank, cv, sv)
        else:
            res = rank_load(rank, cv, sv)
        q.put(
            (
                rank,
                type(res).__name__ == "Ok",
                getattr(res, "value", ""),
                io.tracer.records,
                fabric.buffers.get(rank, {}) if op == "load" else {},
            )
        )
    except Exception as exc:  # noqa: BLE001
        q.put(
            (
                rank,
                False,
                GDTensorIOError(
                    operation="runner", msg=f"{exc}\n{traceback.format_exc()}"
                ),
                [],
                {},
            )
        )


def run_processes(op: str, table, storage_table, layout, root, world, fill=None):
    """Fork one process per rank. Returns ({rank: Result}, records, buffers)."""
    ctx = mp.get_context("fork")
    coord = ProcessCoordinator(world, ctx) if op == "save" else None
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_proc_main,
            args=(r, op, table, storage_table, layout, root, coord, fill, q),
        )
        for r in range(world)
    ]
    started = []
    deadline = time.monotonic() + 600
    try:
        for p in procs:
            p.start()
            started.append(p)
        results, records, buffers = _collect_processes(q, procs, deadline)
        _join_processes(started, deadline)
        return results, records, buffers
    finally:
        _stop_processes(started)
        q.close()


def _join_processes(processes, deadline):
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f"rank process {process.pid} did not exit cleanly")


def _stop_processes(processes):
    for process in processes:
        if process.is_alive():
            process.kill()
        process.join()


def _collect_processes(queue, processes, deadline):
    results, records, buffers = {}, [], {}
    for _ in processes:
        rank, ok, err, recs, bufs = queue.get(
            timeout=max(0, deadline - time.monotonic())
        )
        results[rank] = Ok(None) if ok else Err(err)
        records.extend(recs)
        if bufs:
            buffers[rank] = bufs
    return results, records, buffers
