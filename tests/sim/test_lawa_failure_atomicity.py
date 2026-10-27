"""E2: a failed LAWA mean save is atomic at the checkpoint level and rerunnable.

Uses the real safetensors save session (gdtensor_sim.formats). Visibility is
gated by a single completion marker published by one writer after every
shard rename; load refuses a directory without a matching marker. The LAWA
mean is a brand-new checkpoint and must never overwrite its inputs.

Contract:
1. every rank returns an IO error when the faulted shard cannot publish;
2. the failed directory is invisible: a subsequent rank_load fails (no
   completion marker), even though some peer shards were already renamed;
3. the k input directories stay byte-for-byte unchanged;
4. after the fault clears, rerunning the mean save publishes the marker,
   rank_load succeeds, and verify_crc passes.

On a failed vote the session enters the "commit unknown" state and keeps
already-published peer payloads rather than deleting them; checkpoint-level
atomicity is provided by the marker gate, not by rolling peer shards back.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gdtensor.builder as gb
import gdtensor.io as gio
from gdtensor_sim.coord import ThreadCoordinator
from gdtensor_sim.formats import COMMIT_MARKER_NAME, make_storage_view
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
    marker = mean_root / COMMIT_MARKER_NAME

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

    # Checkpoint-level atomicity: no marker, so the directory is unloadable.
    assert not marker.exists(), marker
    # The faulted shard never reached its final name.
    assert not (mean_root / "__0_0.safetensors").exists()
    # A real rank_load against the half-published directory must be rejected.
    probe = ArrayFabric(ct, points)
    read_view = _storage(mean_root, ct, False)
    load_result = gio.rank_load(0, probe.view(0), read_view)
    assert type(load_result) is gb.Err, load_result
    # Inputs are untouched.
    for version in range(2):
        assert file_bytes(tmp_path / f"input{version}") == fingerprints[version], (
            version
        )

    # Clear the fault and rerun: marker appears, load + CRC verify pass.
    monkeypatch.undo()
    _save(mean_root, ct, points, mean_values)
    assert marker.exists(), marker
    verify_crc(mean_root, "safetensors", ct, SRC, "mirror", mean_values)
    reloaded = ArrayFabric(ct, points)
    rerun = gio.rank_load(0, reloaded.view(0), _storage(mean_root, ct, False))
    assert type(rerun) is gb.Ok, rerun
