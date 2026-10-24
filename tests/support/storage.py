"""Filesystem fixture built only from the documented public interfaces."""

import functools
import itertools
import json
import os
import struct
from pathlib import Path

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.session as gs
import gdtensor.tag as gt


def _result(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        try:
            value = fn(*args, **kwargs)
            return value if type(value) in (gb.Ok, gb.Err) else gb.Ok(value)
        except Exception as exc:
            return gb.Err(gb.GDTensorIOError(operation=fn.__name__, msg=str(exc)))

    return call


def _file_name(kind, order, rank, stages):
    if kind == "single":
        return "model"
    if kind == "hf":
        return f"model-{order % 3}"
    if kind == "per_stage":
        return f"stage{stages[order]}"
    return f"__{rank}_{order % 3 if kind == 'rank_multi' else 0}"


def _group_cells(storage, points, kind, stages):
    groups = {}
    for order, tid in enumerate(storage.ids):
        ranks = [rank for rank in range(storage.world) if len(points[tid, rank])]
        for rank in ranks:
            path = _file_name(kind, order, rank, stages) + "." + storage.fmt
            groups.setdefault(path, []).append((order, rank))
    return groups


def _group_meshes(base, cells):
    meshes = []
    for order, rows in itertools.groupby(sorted(cells), key=lambda c: c[0]):
        ranks = [rank for _, rank in rows]
        for _, run in itertools.groupby(
            enumerate(ranks), key=lambda pair: pair[1] - pair[0]
        ):
            run = [rank for _, rank in run]
            meshes.append(
                gb.SlicedMesh(
                    base, [gb.Slice(order, order + 1), gb.Slice(run[0], run[-1] + 1)]
                )
            )
    return meshes


def _file_slots(storage, cells, points, path, group, entries):
    slots, offset = [], 0
    for order, rank in sorted(cells):
        nd = storage.table.named_d_tensors[order]
        indices = points[nd.global_tensor_id, rank]
        size = len(indices) * gb.global_tensor_meta_of(nd).itemsize
        pos = gb.TensorComponentPos(nd.global_tensor_id, rank)
        slots.append((order, rank, offset, size, indices))
        storage.offsets[pos] = (path, offset)
        entries[pos] = group
        offset += size
    return slots, offset


def _chunk_header(nd, row):
    _, rank, start, size, indices = row
    naive = (
        nd.d_tensor.underlying_d_tensor
        if type(nd.d_tensor) is gb.ReshapedDTensor
        else nd.d_tensor
    )
    coords = np.unravel_index(indices, naive.global_tensor_meta.global_shape)
    shape = [int(axis.max() - axis.min() + 1) for axis in coords]
    return f"{nd.global_tensor_id}/slot{rank}", {
        "dtype": gb.global_tensor_meta_of(nd).dtype,
        "shape": shape,
        "data_offsets": [start, start + size],
    }


def _tensor_header(nd, rows):
    meta = gb.global_tensor_meta_of(nd)
    indices = np.concatenate([slot[4] for slot in rows])
    if np.array_equal(indices, np.arange(int(np.prod(meta.global_shape)))):
        return {
            nd.global_tensor_id: {
                "dtype": meta.dtype,
                "shape": list(meta.global_shape),
                "data_offsets": [rows[0][2], rows[-1][2] + rows[-1][3]],
            }
        }
    return dict(_chunk_header(nd, row) for row in rows)


def _file_header(table, slots):
    header: dict[str, object] = {"__metadata__": {"writer": "test-fixture"}}
    for order, rows in itertools.groupby(slots, key=lambda slot: slot[0]):
        header.update(_tensor_header(table.named_d_tensors[order], list(rows)))
    return header


def _file_layout(storage, points, kind, stages):
    groups = _group_cells(storage, points, kind, stages)
    base = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.TensorCount(len(storage.ids))),
        gb.ArrangedMesh(gt.WorldSize(storage.world)),
    )
    entries, paths = {}, {}
    for path, cells in groups.items():
        # tensor_ids is the FULL DTensorTable row order (N labels), indexed by
        # absolute row; not truncated to rows this group covers.
        row_labels = [nd.global_tensor_id for nd in storage.table.named_d_tensors]
        group = gb.DisjointStorageMeshList(_group_meshes(base, cells), row_labels)
        leader = gb.leader_tensor_component_pos_of(group)  # canonical leader rank
        t0 = min(order for order, _ in cells)  # leader row in table order
        paths[gb.TensorComponentPos(storage.ids[t0], leader.rank)] = path
        slots, size = _file_slots(storage, cells, points, path, group, entries)
        header = _file_header(storage.table, slots)
        storage.headers[path] = header
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        prefix = (
            struct.pack("<Q", len(raw)) + raw if storage.fmt == "safetensors" else b""
        )
        storage.files[path] = (leader.rank, size, prefix, slots)
    return gb.PhysicalLayoutTable(storage.fmt, entries, paths)


def _prepare_file(storage, path, info, rank):
    leader, size, header, slots = info
    if leader != rank and not any(slot[1] == rank for slot in slots):
        return
    fd = os.open(storage.root / (path + ".tmp"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, len(header) + size)
        if leader == rank and header:
            os.pwrite(fd, header, 0)
    finally:
        os.close(fd)


def _validate_headers(storage):
    for path in storage.files:
        with (storage.root / path).open("rb") as file:
            size = struct.unpack("<Q", file.read(8))[0]
            assert json.loads(file.read(size)) == storage.headers[path]


def _metadata_chunks(storage):
    chunks = []
    for path, (_, _, _, slots) in storage.files.items():
        for order, rank, start, size, _ in slots:
            chunks.append(
                {
                    "tensor_id": storage.ids[order],
                    "slot": rank,
                    "file": path,
                    "offset": start,
                    "size": size,
                }
            )
    return chunks


def _leader_paths(storage, rank):
    return [path for path, info in storage.files.items() if info[0] == rank]


def _publish_metadata(storage, rank):
    if storage.fmt != "distcp" or rank != min(
        info[0] for info in storage.files.values()
    ):
        return
    tmp = storage.root / ".metadata.tmp"
    tmp.write_text(json.dumps(storage.metadata()))
    os.replace(tmp, storage.root / ".metadata")


class FileStorage:
    def __init__(
        self, root, table, points, world, fmt, kind="single", stages=None, coord=None
    ):
        self.root, self.table = Path(root), table
        self.world, self.fmt, self.coord = world, fmt, coord
        self.ids = [nd.global_tensor_id for nd in table.named_d_tensors]
        self.files, self.offsets, self.headers = {}, {}, {}
        layout = _file_layout(self, points, kind, stages)
        self.layout = layout
        self.view = gb.StorageDTensorsView(
            self.ids,
            table,
            _FileAccessor(self, False),
            layout,
            self.begin_save,
            self.begin_load,
        )

    def _real_dcp_session(self, load: bool, rank: int):
        """Route distcp through the genuine torch DCP backend (no JSON metadata)."""
        from gdtensor_sim._layout import build_payload_index
        from gdtensor_sim.dcp_backend import DcpLoadSession, DcpSaveSession

        index = build_payload_index(self.table, self.layout)
        if load:
            return DcpLoadSession(rank, self.root, self.table, self.layout, index)
        return DcpSaveSession(
            rank,
            str(self.root),
            self.table,
            self.layout,
            index,
            coord=self.coord,
            mode="process",
        )

    @_result
    def begin_save(self, view, rank):
        if self.fmt == "distcp":
            return self._real_dcp_session(False, rank)
        self.root.mkdir(parents=True, exist_ok=True)
        for path, info in self.files.items():
            _prepare_file(self, path, info, rank)
        return _Save(self, rank)

    @_result
    def begin_load(self, view, rank):
        if self.fmt == "distcp":
            return self._real_dcp_session(True, rank)
        _validate_headers(self)
        return _Load(self)

    def metadata(self):
        tensors = []
        for nd in self.table.named_d_tensors:
            meta = gb.global_tensor_meta_of(nd)
            tensors.append(
                {
                    "id": nd.global_tensor_id,
                    "shape": list(meta.global_shape),
                    "dtype": meta.dtype,
                    "itemsize": meta.itemsize,
                }
            )
        return {
            "format": "gdtensor-distcp",
            "tensors": tensors,
            "chunks": _metadata_chunks(self),
        }


class _FileAccessor(ga.DTensorDataAccessor):
    def __init__(self, storage, writing):
        self.storage, self.writing = storage, writing

    def _address(self, rng):
        _, size, header, _ = self.storage.files[rng.file_path]
        assert 0 <= rng.contiguous_payload_offset <= size - rng.size
        path = self.storage.root / (rng.file_path + (".tmp" if self.writing else ""))
        return path, len(header) + rng.contiguous_payload_offset

    @_result
    def read_to_buffer(self, rng, callback):
        path, offset = self._address(rng)
        with path.open("rb") as f:
            f.seek(offset)
            data = np.frombuffer(f.read(rng.size), dtype=np.uint8).copy()
        assert data.nbytes == rng.size
        return callback(data.ctypes.data, data.nbytes)

    @_result
    def write_from_buffer(self, rng, callback):
        path, offset = self._address(rng)
        data = np.empty(rng.size, dtype=np.uint8)
        result = callback(data.ctypes.data, data.nbytes)
        if type(result) is gb.Err:
            return result
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            assert os.pwrite(fd, data.tobytes(), offset) == data.nbytes
        finally:
            os.close(fd)


class _Save(gs.StorageSaveSession):
    def __init__(self, storage, rank):
        self.storage, self.rank = storage, rank

    def get_file_offset(self, pos):
        path, offset = self.storage.offsets[pos]
        return gb.Ok(len(self.storage.files[path][2]) + offset)

    def init_data_accessor(self):
        return gb.Ok(_FileAccessor(self.storage, True))

    @_result
    def commit(self):
        storage = self.storage
        if storage.coord is not None and not storage.coord.vote(True):
            return gb.Err(
                gb.GDTensorIOError(operation="commit", msg="another rank aborted")
            )
        for path in _leader_paths(storage, self.rank):
            os.replace(storage.root / (path + ".tmp"), storage.root / path)
        if storage.coord is not None:
            storage.coord.wait()
        _publish_metadata(storage, self.rank)
        if storage.coord is not None:
            storage.coord.wait()

    @_result
    def abort(self):
        if self.storage.coord is not None:
            self.storage.coord.vote(False)
        for path in _leader_paths(self.storage, self.rank):
            (self.storage.root / (path + ".tmp")).unlink(missing_ok=True)


class _Load(gs.StorageLoadSession):
    def __init__(self, storage):
        self.storage = storage

    def get_file_offset(self, pos):
        path, offset = self.storage.offsets[pos]
        return gb.Ok(len(self.storage.files[path][2]) + offset)

    def init_data_accessor(self):
        return gb.Ok(_FileAccessor(self.storage, False))

    def finish(self):
        return gb.Ok(None)
