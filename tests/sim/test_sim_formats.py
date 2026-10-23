"""Tests of simulation file-format implementations and their failure paths."""

import json
import struct
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import gdtensor.builder as gb
import gdtensor.io as gio
from gdtensor_sim.coord import ThreadCoordinator
from gdtensor_sim.formats import make_storage_view
from gdtensor_sim.models import ParallelConfig, layout_builder, storage_table
from tests.support.oracle import ArrayFabric
from tests.support.schemas import SRC, run_ranks, tensor_schema


def _storage(root, ct, p, kind, mode, fmt, *, save=False):
    parallel = ParallelConfig(pp=p[0], dp=p[1], tp=p[2])
    table = storage_table(ct, parallel, mode)
    layout = layout_builder(kind, table, parallel, fmt, 3)
    view = make_storage_view(
        [nd.global_tensor_id for nd in table.named_d_tensors],
        table,
        layout,
        str(root),
        coord=ThreadCoordinator(parallel.world) if save else None,
    )
    return view


def _require_real_dcp(fmt):
    """Real distcp is genuine torch DCP: gate only on that real dependency.

    Safetensors paths must run with no torch installed; distcp skips there.
    """
    if fmt == "distcp":
        pytest.importorskip("torch.distributed.checkpoint")


@pytest.mark.parametrize(
    "fmt,field",
    [
        ("safetensors", "dtype"),
        ("safetensors", "shape"),
        ("safetensors", "size"),
        # Real torch DCP does not re-validate the stored dtype / element
        # count; it trusts the caller's destination state dict. Exercise the
        # corruptions it actually rejects instead.
        ("distcp", "missing_metadata"),
        ("distcp", "missing_tensor"),
        ("distcp", "bigger_shape"),
    ],
)
def test_format_load_rejects_checkpoint_metadata_mismatch(tmp_path, fmt, field):
    _require_real_dcp(fmt)
    p = (1, 1, 1)
    ct, points, values = tensor_schema(p)
    source = ArrayFabric(ct, points, values)
    sv = _storage(tmp_path, ct, p, "single", "hf", fmt, save=True)
    run_ranks(1, lambda r: gio.rank_save(r, source.view(r), sv))
    if fmt == "safetensors":
        path = tmp_path / "model.safetensors"
        raw = path.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + n])
        entry = header["matrix"]
        if field == "dtype":
            entry["dtype"] = "F64"
        elif field == "shape":
            entry["shape"] = list(reversed(entry["shape"]))
        else:
            entry["data_offsets"][1] -= 8
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + n :])
    elif field == "missing_metadata":
        (tmp_path / ".metadata").unlink()
    elif field == "missing_tensor":
        import pickle

        path = tmp_path / ".metadata"
        meta = pickle.loads(path.read_bytes())
        tid = ct.named_d_tensors[0].global_tensor_id
        del meta.state_dict_metadata[tid]
        del meta.planner_data[tid]
        meta.storage_data = {
            index: info for index, info in meta.storage_data.items() if index.fqn != tid
        }
        path.write_bytes(pickle.dumps(meta))
    else:  # bigger_shape: demand more elements than the checkpoint stored
        import pickle

        torch = pytest.importorskip("torch")
        path = tmp_path / ".metadata"
        meta = pickle.loads(path.read_bytes())
        tid = ct.named_d_tensors[0].global_tensor_id
        tsm = meta.state_dict_metadata[tid]
        tsm.size = torch.Size(list(tsm.size[:-1]) + [tsm.size[-1] + 4])
        tsm.chunks[0].sizes = tsm.size
        path.write_bytes(pickle.dumps(meta))
    destination = ArrayFabric(ct, points)
    result = gio.rank_load(
        0, destination.view(0), _storage(tmp_path, ct, p, "single", "hf", fmt)
    )
    assert type(result) is gb.Err and type(result.value) is gb.GDTensorIOError, result
    for buf in destination.buffers[0].values():
        assert np.all(buf == 0xA5)


def test_format_distcp_save_returns_after_metadata_is_published(tmp_path, monkeypatch):
    _require_real_dcp("distcp")
    import os

    replace = os.replace

    def slow_metadata(src, dst):
        if str(dst).endswith(".metadata"):
            time.sleep(0.1)
        return replace(src, dst)

    monkeypatch.setattr(os, "replace", slow_metadata)
    ct, points, values = tensor_schema(SRC)
    source = ArrayFabric(ct, points, values)
    sv = _storage(tmp_path, ct, SRC, "single", "hf", "distcp", save=True)

    def save(rank):
        result = gio.rank_save(rank, source.view(rank), sv)
        assert (tmp_path / ".metadata").is_file()
        return result

    run_ranks(int(np.prod(SRC)), save)


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_format_save_publish_failure_reaches_all_ranks(tmp_path, monkeypatch, fmt):
    _require_real_dcp(fmt)
    import os

    fsync = os.fsync
    target = tmp_path / ("__0_0." + fmt + (".tmp" if fmt == "safetensors" else ""))

    def fail_one_file(fd):
        if target.exists() and os.fstat(fd).st_ino == target.stat().st_ino:
            raise OSError("injected publication failure")
        return fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_one_file)
    ct, points, values = tensor_schema(SRC)
    source = ArrayFabric(ct, points, values)
    sv = _storage(tmp_path, ct, SRC, "per_rank", "mirror", fmt, save=True)
    with ThreadPoolExecutor(max_workers=int(np.prod(SRC))) as pool:
        results = list(
            pool.map(
                lambda r: gio.rank_save(r, source.view(r), sv), range(int(np.prod(SRC)))
            )
        )
    assert all(
        type(r) is gb.Err and type(r.value) is gb.GDTensorIOError for r in results
    ), results
    assert not (tmp_path / ".metadata").exists()
    assert not (tmp_path / "model.safetensors.index.json").exists()


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
@pytest.mark.parametrize("terminal", ["commit", "abort"])
def test_save_session_rejects_calls_after_terminal_state(tmp_path, fmt, terminal):
    _require_real_dcp(fmt)
    p = (1, 1, 1)
    table, _, _ = tensor_schema(p)
    view = _storage(tmp_path, table, p, "single", "hf", fmt)
    assert view.begin_save is not None
    opened = view.begin_save(view, 0)
    assert type(opened) is gb.Ok, opened
    session = opened.value
    assert session is not None
    pos = gb.TensorComponentPos("matrix", 0)
    offset = session.get_file_offset(pos)
    # get_file_offset maps contiguous_payload_offset to the real file offset for an
    # existing pos; NotFound only when the pos has no physical slot. Both formats
    # return a real offset (>=0); the save-unpublished staging region is header-less.
    assert type(offset) is gb.Ok and offset.value is not None and offset.value >= 0, (
        offset
    )
    assert type(session.init_data_accessor()) is gb.Ok
    assert type(getattr(session, terminal)()) is gb.Ok
    for call in (
        lambda: session.get_file_offset(pos),
        session.init_data_accessor,
        session.commit,
        session.abort,
    ):
        result = call()
        assert (
            type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
        ), result


@pytest.mark.parametrize("fmt", ["safetensors", "distcp"])
def test_load_session_rejects_calls_after_finish(tmp_path, fmt):
    _require_real_dcp(fmt)
    p = (1, 1, 1)
    table, points, values = tensor_schema(p)
    source = ArrayFabric(table, points, values)
    view = _storage(tmp_path, table, p, "single", "hf", fmt)
    run_ranks(1, lambda rank: gio.rank_save(rank, source.view(rank), view))
    assert view.begin_load is not None
    opened = view.begin_load(view, 0)
    assert type(opened) is gb.Ok, opened
    session = opened.value
    assert session is not None
    pos = gb.TensorComponentPos("matrix", 0)
    off = session.get_file_offset(pos)
    # Save-unpublished staging (safetensors): real offset >=0.
    # Real DCP load: metadata does not expose raw byte offsets -> explicit Err,
    # not a fake offset, not NotFound for an existing pos.
    if fmt == "distcp":
        assert type(off) is gb.Err, off
    else:
        assert type(off) is gb.Ok and off.value is not None and off.value >= 0, off
    assert type(session.init_data_accessor()) is gb.Ok
    assert type(session.finish()) is gb.Ok
    for call in (
        lambda: session.get_file_offset(pos),
        session.init_data_accessor,
        session.finish,
    ):
        result = call()
        assert (
            type(result) is gb.Err and type(result.value) is gb.InvalidArgumentError
        ), result
