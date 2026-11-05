"""File-backed accessors for the fixed golden test; no environment switches."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from pathlib import Path

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb


def file_identity(path):
    path = os.path.realpath(path)
    s = os.stat(path)
    return path, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def pwrite_all(fd, data, off):
    view = memoryview(data)
    while view:
        try:
            n = os.pwrite(fd, view, off)
        except InterruptedError:
            continue
        if n <= 0 or n > len(view):
            raise OSError("cache short write made no progress")
        off += n
        view = view[n:]


class Device:
    """One serial device per node; record service and queue time separately."""

    def __init__(self, num_nodes, bandwidth, latency):
        self.bandwidth, self.latency = bandwidth, latency
        self.locks = {n: threading.Lock() for n in range(num_nodes)}
        self.service = dict.fromkeys(self.locks, 0.0)
        self.queue = dict.fromkeys(self.locks, 0.0)
        self.lock = threading.Lock()

    @property
    def charged(self):
        return sum(self.service.values())

    def charge(self, node, nbytes):
        delay = self.latency + (nbytes / self.bandwidth if self.bandwidth else 0)
        start = time.monotonic()
        with self.locks[node]:
            queued = time.monotonic() - start
            time.sleep(delay)
        with self.lock:
            self.queue[node] += queued
            self.service[node] += delay


def _write_raw_source(path, data):
    with open(path, "wb") as file:
        for value in data.values():
            file.write(np.asarray(value, np.uint8).tobytes())


def _write_source(path, data, checkpoint):
    if not checkpoint:
        _write_raw_source(path, data)
        return 0
    import safetensors.numpy
    import struct

    values = {t: np.asarray(v, np.uint8).view(np.float16) for t, v in data.items()}
    safetensors.numpy.save_file(values, path)
    with open(path, "rb") as file:
        return 8 + struct.unpack("<Q", file.read(8))[0]


class DiskStorage(ga.DTensorDataAccessor):
    def __init__(self, cluster, data, root, device, checkpoint=False):
        self.cluster, self.device = cluster, device
        self.path = os.path.join(
            root, "model.safetensors" if checkpoint else "storage.bin"
        )
        self.payload_base = _write_source(self.path, data, checkpoint)
        self.fd = os.open(self.path, os.O_RDONLY)
        self.spans, start = [], 0
        for tid, value in data.items():
            self.spans.append((start, start + len(value), tid))
            start += len(value)
        self.reads, self.io = [], 0.0

    def cache_identity(self):
        identity = file_identity(self.path)
        if identity != self._fd_identity():
            raise OSError("source path was replaced; open a new load session")
        return identity

    def _fd_identity(self):
        s = os.fstat(self.fd)
        return (
            os.path.realpath(self.path),
            s.st_dev,
            s.st_ino,
            s.st_size,
            s.st_mtime_ns,
            s.st_ctime_ns,
        )

    def _tensor_of(self, off, size):
        hits = [(lo, hi, tid) for lo, hi, tid in self.spans if lo <= off < hi]
        assert len(hits) == 1, "storage range outside payload"
        lo, hi, tid = hits[0]
        assert off + size <= hi, "storage range crosses tensor boundary"
        return tid, off - lo

    def read_to_buffer(self, global_range, read_to_buffer):
        rng, callback = global_range, read_to_buffer
        node = self.cluster.node_of_rank(self.cluster.rank)
        off, size = rng.contiguous_payload_offset, rng.size
        tid, toff = self._tensor_of(off, size)
        start = time.monotonic()
        raw = os.pread(self.fd, size, self.payload_base + off)
        assert len(raw) == size, "short source read"
        with self.cluster.lock:
            self.io += time.monotonic() - start
            self.reads.append((node, tid, toff, size))
            self.cluster.storage_in[node] += size
        self.device.charge(node, size)
        buf = np.frombuffer(raw, np.uint8)
        return callback(buf.ctypes.data, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        return gb.Err(gb.GDTensorIOError(operation="write", msg="read-only storage"))

    def close(self):
        os.close(self.fd)


def _merge_range(merged, a, b):
    if merged and a <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
    else:
        merged.append((a, b))
    return merged


class _Ranges:
    def __init__(self):
        self.spans = []

    def add(self, off, size):
        edges = sorted(self.spans + [(off, off + size)])
        merged = []
        for a, b in edges:
            merged = _merge_range(merged, a, b)
        self.spans = merged

    def covers(self, off, size):
        return any(a <= off and off + size <= b for a, b in self.spans)

    def clear(self):
        self.spans.clear()


class SparseRankBuffers(ga.DTensorDataAccessor):
    def __init__(self, cluster):
        self.cluster, self.parts = cluster, {}
        self.lock = threading.Lock()

    def write_from_buffer(self, global_range, write_from_buffer):
        rng, callback = global_range, write_from_buffer
        buf = np.empty(rng.size, np.uint8)
        result = callback(buf.ctypes.data, rng.size)
        if type(result) is not gb.Ok:
            return result
        with self.lock:
            self._publish(rng, buf)
        return result

    def _publish(self, rng, buf):
        parts = self.parts.setdefault(self.cluster.rank, {})
        key = rng.global_tensor_id, rng.global_tensor_offset
        assert key not in parts, "duplicate compute write"
        parts[key] = buf

    def read_to_buffer(self, global_range, read_to_buffer):
        return gb.Err(
            gb.GDTensorIOError(operation="read", msg="write-only destination")
        )


def _truncate_to(fd, size):
    if os.fstat(fd).st_size != size:
        os.ftruncate(fd, size)


class DiskNodeCache(ga.DTensorDataAccessor):
    """Path-scoped cache; publish metadata only after complete physical writes.

    Metadata checks detect external file mutation. SHA256 checks detect payload
    corruption before passing bytes to a consumer. Persistence across process
    restart is deliberately not claimed: unindexed files are cold.
    """

    def __init__(self, cluster, sizes, target_of_rank, root, device):
        self.cluster, self.sizes, self.target_of_rank = cluster, sizes, target_of_rank
        self.root, self.device = root, device
        self.mem, self.fds, self.integrity, self.stamps = {}, {}, {}, {}
        self.integrity_keys = {}
        self.lock = threading.RLock()
        self.node_locks = {}
        self.physical_read_bytes = 0
        self.physical_reads = {}
        self.wrong_node, self.io = 0, 0.0
        self.writes, self.fills, self.reads = [], [], []
        self.resident, self.peak = {}, {}
        self.bases, start = {}, 0
        for tid, size in sizes.items():
            self.bases[tid] = start
            start += size
        self.total = start

    def _ends(self, rng):
        actor = self.cluster.node_of_rank(self.cluster.rank)
        target = getattr(rng, "cache_node", self.target_of_rank(self.cluster.rank))
        identity = getattr(rng, "source_identity", None)
        namespace = hashlib.sha256(repr(identity).encode()).hexdigest()[:20]
        return actor, target, namespace

    def _node_lock(self, namespace, node):
        with self.lock:
            return self.node_locks.setdefault((namespace, node), threading.RLock())

    def file_path(self, namespace, node):
        return os.path.join(self.root, namespace, f"cache{node}.bin")

    def _fd(self, namespace, node):
        key = namespace, node
        if key not in self.fds:
            path = self.file_path(namespace, node)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            _truncate_to(fd, self.total)
            self.fds[key] = fd
        return self.fds[key]

    def _stamp(self, namespace, node):
        return file_identity(self.file_path(namespace, node))

    def _check_stamp(self, namespace, node):
        key = namespace, node
        if key in self.stamps and self.stamps[key] != self._stamp(namespace, node):
            raise OSError("cache file changed or was truncated")

    def _validate(self, rng):
        tid, off, size = rng.global_tensor_id, rng.global_tensor_offset, rng.size
        if tid not in self.sizes or off < 0 or size < 0 or off + size > self.sizes[tid]:
            raise ValueError("invalid cache range")
        return tid, off, size

    def has_bytes(self, global_range) -> bool:
        rng = global_range
        _, node, namespace = self._ends(rng)
        tid, off, size = self._validate(rng)
        with self._node_lock(namespace, node):
            self._check_stamp(namespace, node)
            mask = self.mem.get((namespace, node, tid))
            return mask is not None and mask.covers(off, size)

    def write_from_buffer(self, global_range, write_from_buffer):
        rng, callback = global_range, write_from_buffer
        actor, node, namespace = self._ends(rng)
        tid, off, size = self._validate(rng)
        tmp = np.empty(size, np.uint8)
        result = callback(tmp.ctypes.data, size)
        if type(result) is not gb.Ok:
            return result
        raw = tmp.tobytes()
        self.cluster.transfer(actor, node, size)
        with self._node_lock(namespace, node):
            self._check_stamp(namespace, node)
            fd = self._fd(namespace, node)
            start = time.monotonic()
            pwrite_all(fd, raw, self.bases[tid] + off)
            elapsed = time.monotonic() - start
            self.stamps[namespace, node] = self._stamp(namespace, node)
            mask = self.mem.setdefault((namespace, node, tid), _Ranges())
            previous = sum(b - a for a, b in mask.spans)
            mask.add(off, size)
            added = sum(b - a for a, b in mask.spans) - previous
            self.resident[node] = self.resident.get(node, 0) + added
            self.peak[node] = max(self.peak.get(node, 0), self.resident[node])
            self.integrity[namespace, node, tid, off, size] = hashlib.sha256(
                raw
            ).digest()
            self.integrity_keys.setdefault((namespace, node, tid), set()).add(
                (off, size)
            )
            self._record_fill(actor, node, tid, off, size, elapsed)
        self.device.charge(node, size)
        return result

    def _record_fill(self, actor, node, tid, off, size, elapsed):
        with self.lock:
            self.io += elapsed
            self.writes.append((actor, node, off, size))
            self.fills.append((actor, node, tid, off, size))

    def _read_checked(self, namespace, node, tid, off, size):
        fd = self._fd(namespace, node)
        exact = self.integrity.get((namespace, node, tid, off, size))
        if exact is not None:
            return self._verify_read(fd, node, tid, off, size, exact)
        hits = [
            (a, n, self.integrity[namespace, node, tid, a, n])
            for a, n in self.integrity_keys.get((namespace, node, tid), ())
            if a <= off and off + size <= a + n
        ]
        if len(hits) != 1:
            raise OSError("cache range has no unique integrity record")
        a, n, digest = hits[0]
        raw = self._verify_read(fd, node, tid, a, n, digest)
        return raw[off - a : off - a + size]

    def _verify_read(self, fd, node, tid, off, size, digest):
        raw = os.pread(fd, size, self.bases[tid] + off)
        with self.lock:
            self.physical_read_bytes += len(raw)
            self.physical_reads[node] = self.physical_reads.get(node, 0) + len(raw)
        self.device.charge(node, len(raw))
        if len(raw) != size or hashlib.sha256(raw).digest() != digest:
            raise OSError("cache checksum mismatch")
        return raw

    def _require_present(self, rng):
        if not self.has_bytes(rng):
            self.wrong_node += 1
            raise OSError("cache range was never published")

    def read_to_buffer(self, global_range, read_to_buffer):
        rng, callback = global_range, read_to_buffer
        actor, node, namespace = self._ends(rng)
        tid, off, size = self._validate(rng)
        with self._node_lock(namespace, node):
            self._check_stamp(namespace, node)
            self._require_present(rng)
            start = time.monotonic()
            raw = self._read_checked(namespace, node, tid, off, size)
            self._check_stamp(namespace, node)
            elapsed = time.monotonic() - start
            self._record_read(node, size, elapsed)
        self.cluster.transfer(node, actor, size)
        buf = np.frombuffer(raw, np.uint8)
        return callback(buf.ctypes.data, size)

    def _record_read(self, node, size, elapsed):
        with self.lock:
            self.io += elapsed
            self.reads.append((self.cluster.rank, node, size))

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

    def clear(self):
        with self.lock:
            self.mem.clear()
            self.integrity.clear()
            self.integrity_keys.clear()
            self.resident.clear()


PROFILES = {
    "slow-storage": (256.0, 200.0, 4096.0, 50.0, 4096.0),
    "slow-network": (4096.0, 20.0, 4096.0, 20.0, 256.0),
}
