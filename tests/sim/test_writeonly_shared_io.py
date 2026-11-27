"""Regression tests for multi-host NFS-safe O_WRONLY shared-file writes."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from gdtensor.builder import (
    ArrangedMesh,
    DTensorTable,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    Ok,
    Shard,
)
from gdtensor.tag import WorldSize
from gdtensor_conn._layout import build_payload_index
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.formats import SafeTensorsSaveSession
from gdtensor_conn.iobackend import FileIO
from gdtensor_conn.models import build_layout


def test_file_io_opens_write_descriptors_wronly_and_fsyncs_on_close(
    tmp_path, monkeypatch
):
    opened: list[tuple[str, int]] = []
    fsynced: list[int] = []
    ftruncated: list[int] = []
    real_open = os.open
    real_fsync = os.fsync
    real_ftruncate = os.ftruncate

    def spy_open(path, flags, *args, **kwargs):
        opened.append((os.fspath(path), flags))
        return real_open(path, flags, *args, **kwargs)

    def spy_fsync(fd):
        fsynced.append(fd)
        return real_fsync(fd)

    def spy_ftruncate(fd, length):
        ftruncated.append(length)
        return real_ftruncate(fd, length)

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "ftruncate", spy_ftruncate)

    io = FileIO()
    target = str(tmp_path / "shared.safetensors.tmp")
    io.create_staging(target, 16)
    io.create_staging(target, 16)  # second call sees size already matches
    io.pwrite(0, target, 0, b"01234567")
    io.pwrite(1, target, 8, b"89abcdef")

    write_flags = [flags for path, flags in opened if path == target]
    assert len(write_flags) == 4  # 2x create_staging + rank 0 + rank 1 pwrite
    for flags in write_flags:
        assert (flags & os.O_ACCMODE) == os.O_WRONLY
        assert (flags & os.O_ACCMODE) != os.O_RDWR
    assert ftruncated == [16]

    fsynced_before_close = len(fsynced)
    io.close_rank(0)
    io.close_rank(1)
    assert len(fsynced) == fsynced_before_close + 2
    assert io.pread(0, target, 0, 16) == b"0123456789abcdef"


def test_shared_save_session_uses_wronly_across_ranks(tmp_path, monkeypatch):
    opened_flags: list[int] = []
    real_open = os.open

    def spy_open(path, flags, *args, **kwargs):
        if os.fspath(path).endswith(".tmp"):
            opened_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", spy_open)

    world = 4
    mesh = ArrangedMesh(WorldSize(world))
    dt = NaiveDTensor(GlobalTensorMeta([world, 4], "float32", 4), [Shard(0)], mesh)
    table = DTensorTable([NamedDTensor("w", dt)])
    layout = build_layout(
        table, world, "safetensors", lambda _t, _w, _nd: "shared.safetensors"
    )
    index = build_payload_index(table, layout)
    coord = ThreadCoordinator(world)
    io = FileIO()

    def run(rank: int) -> None:
        session = SafeTensorsSaveSession(
            rank, str(tmp_path), table, layout, index, io, coord=coord
        )
        pos = index.files["shared.safetensors"].slots[rank].pos
        offset_res = session.get_file_offset(pos)
        assert type(offset_res) is Ok, offset_res
        offset = offset_res.value
        assert isinstance(offset, int)
        io.pwrite(
            rank,
            session.abspath("shared.safetensors"),
            offset,
            bytes([rank]) * 16,
        )
        res = session.commit()
        assert type(res) is Ok, res

    with ThreadPoolExecutor(max_workers=world) as pool:
        futs = [pool.submit(run, r) for r in range(world)]
        for fut in futs:
            fut.result()

    assert opened_flags
    assert any((flags & os.O_ACCMODE) == os.O_WRONLY for flags in opened_flags)
    assert all((flags & os.O_ACCMODE) != os.O_RDWR for flags in opened_flags)
    raw = (tmp_path / "shared.safetensors").read_bytes()
    header_len = int.from_bytes(raw[:8], "little")
    payload = raw[8 + header_len :]
    assert payload == b"".join(bytes([r]) * 16 for r in range(world))
