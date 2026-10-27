"""File-backed cache oracle used only by public IO tests."""

import ctypes
import os
import threading
from pathlib import Path

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.tag as gt
from gdtensor import Result


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


def _find_slot(
    data: dict[str, dict[int, np.ndarray]], tid: str, offset: int, size: int
) -> tuple[int, np.ndarray] | None:
    for o, b in data[tid].items():
        if o <= offset and offset + size <= o + len(b):
            return o, b
    return None


class ResidentCache(ga.DTensorDataAccessor):
    """Per-node cache allocator: regions live from write-done to matching-tid ack.

    The test runner sets ``current_rank`` before each ``chain_load``; the
    owner node is derived as ``rank // num_ranks_per_node``. Live regions are
    keyed (node, tid, offset) — per-node, per-tensor, never shared.
    ``peak_by_node[node]`` is that node's max simultaneously-resident bytes.
    """

    def __init__(self, events: list, num_ranks_per_node: int, name: str = "cache"):
        self.data: dict[str, dict[int, np.ndarray]] = {}
        self.events = events
        self.name = name
        self.num_ranks_per_node = num_ranks_per_node
        self._tl = threading.local()
        self._tl.rank = 0
        self.live: list[tuple[int, str, int, int]] = []  # (node, tid, off, size)
        self.peak_by_node: dict[int, int] = {}
        self._lock = threading.Lock()
        self._acked: dict[
            tuple, dict[int, set]
        ] = {}  # (node,tid,base) -> {rank: covered offsets}
        self.expected_reads: dict[
            tuple, dict[int, range]
        ] = {}  # caller-declared per-rank required coverage
        self._read_calls = 0
        self.barriers: dict = {}  # rank -> Event; empty = no read block

    @property
    def current_rank(self) -> int:
        return getattr(self._tl, "rank", 0)

    @current_rank.setter
    def current_rank(self, rank: int) -> None:
        self._tl.rank = rank

    @property
    def node(self) -> int:
        return self.current_rank // self.num_ranks_per_node

    def _snapshot(self):
        for node in {n for n, _t, _o, _s in self.live}:
            total = sum(sz for n, _t, _o, sz in self.live if n == node)
            self.peak_by_node[node] = max(self.peak_by_node.get(node, 0), total)

    def write_from_buffer(self, global_range, write_from_buffer) -> Result[None]:
        rng = global_range
        callback = write_from_buffer
        tid = getattr(rng, "global_tensor_id", "") or "?"
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        buf = np.zeros(size, dtype=np.uint8)
        self.events.append((self.name, "write", tid, offset, size))
        res = callback(buf.ctypes.data, buf.nbytes)
        with self._lock:
            self.data.setdefault(tid, {})[offset] = buf
            self.live.append((self.node, tid, offset, size))
            self._snapshot()
        return res

    def _release(self, node, tid, offset, size):
        entry = (node, tid, offset, size)
        with self._lock:
            self.live = [e for e in self.live if e != entry]

    def _no_slot_err(self, tid, offset, size):
        return gb.Err(
            gb.GDTensorIOError(
                operation="internal",
                msg=f"no slot covers {tid}@{offset}+{size}",
            )
        )

    def _locked_read(
        self, tid: str, offset: int, size: int
    ) -> tuple[int, np.ndarray, np.ndarray] | tuple[None, None, None]:
        with self._lock:
            found = _find_slot(self.data, tid, offset, size)
        if found is None:
            return None, None, None
        base_off, buf = found
        view = buf[offset - base_off : offset - base_off + size]
        return base_off, buf, view

    def read_to_buffer(self, global_range, read_to_buffer) -> Result[None]:
        rng = global_range
        callback = read_to_buffer
        tid = getattr(rng, "global_tensor_id", "") or "?"
        offset = getattr(rng, "global_tensor_offset", None)
        if offset is None:
            offset = rng.contiguous_payload_offset
        size = rng.size
        # Per-rank read barrier (cohort delay).
        if self.current_rank in self.barriers:
            self.barriers[self.current_rank].wait()
        self._read_calls += 1
        base_off, buf, view = self._locked_read(tid, offset, size)
        if view is None:
            return self._no_slot_err(tid, offset, size)
        assert buf is not None and base_off is not None
        self.events.append((self.name, "read", tid, offset, size))
        res = callback(view.ctypes.data, view.nbytes)
        # ACK per-rank required coverage; release only when ALL declared ranks full.
        key = (self.node, tid, base_off)
        cov = self._acked.setdefault(key, {})
        rank_cov = cov.setdefault(self.current_rank, set())
        rank_cov.update(range(offset, offset + size))
        expected = self.expected_reads.get(
            key, {self.current_rank: range(base_off, base_off + len(buf))}
        )
        all_done = all(set(expected[r]).issubset(cov.get(r, set())) for r in expected)
        if all_done:
            self._release(self.node, tid, base_off, len(buf))
        return res
