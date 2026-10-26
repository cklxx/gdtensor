"""File-backed cache oracle used only by public IO tests."""

import ctypes
import os
import threading
from pathlib import Path

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.tag as gt


class FileCache(ga.DTensorDataAccessor):
    """Persist each slot to disk; track reads against declared per-rank bytes."""

    def __init__(self, root, expected):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.expected = expected
        self.local = threading.local()
        self.lock = threading.Lock()
        self.slots = {}
        self.acks = {}
        self.writes = []
        self.reads = []
        self.peak = 0
        self.entered = threading.Event()
        self.second_write = threading.Event()
        self.release = None
        self.block_rank = None

    @property
    def rank(self):
        return self.local.rank

    @rank.setter
    def rank(self, value):
        self.local.rank = value

    def write_from_buffer(self, global_range, write_from_buffer):
        rng, callback = global_range, write_from_buffer
        data = bytearray(rng.size)
        ptr = ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data))
        result = callback(ptr, len(data))
        if type(result) is not gb.Ok:
            return result
        key = (rng.global_tensor_id, rng.global_tensor_offset, rng.size)
        with self.lock:
            self._persist(key, data)
        return result

    def _persist(self, key, data):
        path = self.root / f"slot-{len(self.writes)}.bin"
        with path.open("wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        self.slots[key] = path
        self.writes.append(key)
        if len(self.writes) > 1:
            self.second_write.set()
        self.peak = max(self.peak, len(self.slots))

    def read_to_buffer(self, global_range, read_to_buffer):
        rng, callback = global_range, read_to_buffer
        if self.rank == self.block_rank:
            self.entered.set()
            assert self.release is not None
            assert self.release.wait(5), "file-cache read barrier timed out"
        key, path = self._locate(rng)
        fd = os.open(path, os.O_RDONLY)
        try:
            data = bytearray(os.pread(fd, rng.size, rng.global_tensor_offset - key[1]))
        finally:
            os.close(fd)
        assert len(data) == rng.size, "file-cache short read"
        ptr = ctypes.addressof((ctypes.c_char * len(data)).from_buffer(data))
        result = callback(ptr, len(data))
        if type(result) is gb.Ok:
            self._ack(key, rng)
        return result

    def _locate(self, rng):
        with self.lock:
            return self._find(rng)

    def _find(self, rng):
        hit = next(
            (
                (key, path)
                for key, path in self.slots.items()
                if self._contains(key, rng)
            ),
            None,
        )
        assert hit is not None, "cache slot released before the required read"
        return hit

    @staticmethod
    def _contains(key, rng):
        tid, start, size = key
        end = rng.global_tensor_offset + rng.size
        return (
            tid == rng.global_tensor_id
            and start <= rng.global_tensor_offset
            and end <= start + size
        )

    def _ack(self, key, rng):
        with self.lock:
            self._record_ack(key, rng)

    def _record_ack(self, key, rng):
        cov = self.acks.setdefault(key, {})
        cov.setdefault(self.rank, set()).update(
            range(rng.global_tensor_offset, rng.global_tensor_offset + rng.size)
        )
        self.reads.append((self.rank, key, rng.global_tensor_offset, rng.size))
        required = self.expected[key[0]]
        if all(set(iv).issubset(cov.get(rank, set())) for rank, iv in required.items()):
            self.slots.pop(key).unlink()


def compute_table(names, size=4, exclusive=False):
    dts = []
    for rank, name in enumerate(names):
        mesh = gb.ArrangedMesh(gt.WorldSize(2))
        if exclusive:
            mesh = gb.SlicedMesh(mesh, [gb.Slice(rank, rank + 1)])
        placement = gb.Replicate() if exclusive else gb.Shard(0)
        dt = gb.NaiveDTensor(gb.GlobalTensorMeta([size], "U8", 1), [placement], mesh)
        dts.append(gb.NamedDTensor(name, dt))
    return gb.DTensorTable(dts)


def cache_view(names, accessor, size=4):
    mesh = gb.StCacheOuterProductMesh(
        gb.ArrangedMesh(gt.NumSteps(1)),
        gb.ArrangedMesh(gt.NumNodes(1)),
        gb.ArrangedMesh(gt.ConcurrencyWidth(1)),
    )
    dts = [
        gb.NamedDTensor(
            name,
            gb.NaiveDTensor(
                gb.GlobalTensorMeta([size], "U8", 1),
                [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
                mesh,
            ),
        )
        for name in names
    ]
    return gb.CacheDTensorsView(names, gb.DTensorTable(dts), accessor)


def storage_schema(names, size=4):
    dts = []
    mesh = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
    )
    for name in names:
        dt = gb.NaiveDTensor(
            gb.GlobalTensorMeta([size], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh
        )
        dts.append(gb.NamedDTensor(name, dt))
    group = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(len(names))),
                gb.ArrangedMesh(gt.WorldSize(1)),
            )
        ],
        names,
    )
    entries = {gb.TensorComponentPos(name, 0): group for name in names}
    layout = gb.PhysicalLayoutTable(
        "safetensors",
        entries,
        {gb.TensorComponentPos(names[0], 0): "model.safetensors"},
    )
    return gb.DTensorTable(dts), layout
