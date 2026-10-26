"""E2: a failed LAWA mean save is atomic and safely rerunnable.

Uses the real safetensors save session (gdtensor_sim.formats) and the same
fault injection as test_sim_formats.py: fsync of the mean shard file fails
on one fd. The LAWA mean is a brand-new checkpoint; the k input versions
must never be overwritten. Contract:

1. every rank returns an IO error;
2. the failing leader's shard is never published (its final file is absent);
3. the k input directories stay byte-for-byte unchanged;
4. after the fault clears, rerunning the mean save succeeds and independently
   decodes to the dense mean.

On a failed vote the session enters the "commit unknown" state and keeps
other ranks' already-published payloads rather than deleting them; this is
the snapshot protocol ("result unknown keeps payload"). The atomic unit
asserted here is therefore the faulted shard, not every peer shard.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gdtensor.builder as gb
import gdtensor.io as gio
from gdtensor_sim.coord import ThreadCoordinator
from gdtensor_sim.formats import make_storage_view
from gdtensor_sim.models import ParallelConfig, layout_builder, storage_table
from tests.support.checkpoint import file_bytes
from tests.support.oracle import ArrayFabric
from tests.support.schemas import SRC, run_ranks, tensor_schema, verify_crc

KIND = "per_rank"
MODE = "mirror"


def _storage(root, ct, save=False):
    parallel = ParallelConfig(pp=SRC[0], dp=SRC[1], tp=SRC[2])
    table = storage_table(ct, parallel, MODE)
    layout = layout_builder(KIND, table, parallel, "safetensors", 3)
    return make_storage_view(
        [nd.global_tensor_id for nd in table.named_d_tensors],
        table,
        layout,
        str(root),
        coord=ThreadCoordinator(parallel.world) if save else None,
    )


def _save(root, ct, points, values):
    source = ArrayFabric(ct, points, values)
    # One coordinator/view shared by all ranks. Building the view inside each
    # rank gives every rank its own Barrier(8) with one party, which deadlocks.
    view = _storage(root, ct, True)
    run_ranks(int(np.prod(SRC)), lambda r: gio.rank_save(r, source.view(r), view))


def _versions(values):
    return [
        {tid: arr + version * 1001 + 3 for tid, arr in values.items()}
        for version in range(2)
    ]


def test_lawa_mean_save_failure_is_atomic(tmp_path, monkeypatch):
    ct, points, base = tensor_schema(SRC)
    versions = _versions(base)

    for version, values in enumerate(versions):
        _save(tmp_path / f"input{version}", ct, points, values)
    fingerprints = {v: file_bytes(tmp_path / f"input{v}") for v in range(2)}

    mean_values = {}
    for tid, arr in base.items():
        if arr.dtype.kind == "f":
            mean_values[tid] = np.mean(
                [versions[i][tid].astype(np.float32) for i in range(2)], axis=0
            ).astype(arr.dtype)
        else:
            mean_values[tid] = versions[0][tid].copy()

    mean_root = tmp_path / "mean"
    world = int(np.prod(SRC))

    real_fsync = os.fsync
    staging = mean_root / "__0_0.safetensors.tmp"
    final_shard = mean_root / "__0_0.safetensors"

    def fail_mean_fsync(fd):
        if staging.exists() and os.fstat(fd).st_ino == staging.stat().st_ino:
            raise OSError("injected mean publication failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_mean_fsync)
    source = ArrayFabric(ct, points, mean_values)
    view = _storage(mean_root, ct, True)
    with ThreadPoolExecutor(max_workers=world) as pool:
        results = list(
            pool.map(lambda r: gio.rank_save(r, source.view(r), view), range(world))
        )
    assert all(type(r) is gb.Err for r in results), results

    # The faulted leader shard must never reach its final published name. Its
    # staging file may linger; peer shards are kept by the unknown-state rule.
    assert not final_shard.exists(), final_shard
    for version in range(2):
        assert file_bytes(tmp_path / f"input{version}") == fingerprints[version], (
            version
        )

    monkeypatch.undo()
    _save(mean_root, ct, points, mean_values)
    verify_crc(mean_root, "safetensors", ct, SRC, "mirror", mean_values)
