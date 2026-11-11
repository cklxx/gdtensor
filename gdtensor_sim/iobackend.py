"""File IO backends for the safetensors / distcp sessions.

The sessions own format and offset logic. Backends own exact byte IO, file
identity, staging, publication, and handle cleanup. This module is not public
core API.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, TypeVar

MiB = 1 << 20
T = TypeVar("T")


@dataclass(frozen=True)
class IoRecord:
    rank: int
    op: str  # "pwrite" | "pread"
    path: str
    offset: int
    size: int
    t: float


class IoTracer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: List[IoRecord] = []
        self.counters: Dict[str, int] = defaultdict(int)

    def add(self, rec: IoRecord) -> None:
        with self._lock:
            self.records.append(rec)
            self.counters[f"{rec.op}_bytes"] += rec.size


class FdPool:
    """Per-rank file descriptor cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fds: Dict[Tuple[int, str, str], int] = {}

    def get(self, rank: int, path: str, mode: str) -> int:
        key = (rank, path, mode)
        with self._lock:
            return self._get_unlocked(key)

    def _get_unlocked(self, key):
        fd = self._fds.get(key)
        if fd is None:
            _, path, mode = key
            fd = _open_fd(path, mode)
            self._fds[key] = fd
        return fd

    def _close(self, keys):
        for key in keys:
            os.close(self._fds.pop(key))

    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None:
        with self._lock:
            keys = [
                k
                for k in self._fds
                if k[0] == rank and (prefix is None or k[1].startswith(prefix))
            ]
            self._close(keys)

    def close_path(self, path: str) -> None:
        with self._lock:
            self._close([k for k in self._fds if k[1] == path])


def _open_fd(path, mode):
    if mode == "w":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    return os.open(path, os.O_RDONLY)


def _pread_all(fd, offset, size):
    out = bytearray()
    while len(out) < size:
        chunk = os.pread(fd, size - len(out), offset + len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)


def _file_identity(path, fd=None):
    path = os.path.realpath(path)
    stat = os.stat(path)
    identity = (
        path,
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    if fd is not None:
        opened = os.fstat(fd)
        actual = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity[1:] != actual:
            raise OSError("source path was replaced; open a new load session")
    return identity


class FileSystemClient(Protocol):
    """Synchronous range-IO client used by SdkFileIO."""

    def metadata(self, key: str) -> Tuple[int, str]: ...

    def read_range(self, key: str, offset: int, size: int, version: str) -> bytes: ...

    def create_staging(self, key: str, size: int) -> None: ...

    def write_range(self, key: str, offset: int, data: memoryview) -> Optional[int]: ...

    def sync(self, key: str) -> None: ...

    def publish(self, temporary_key: str, final_key: str) -> None: ...

    def discard(self, key: str) -> None: ...

    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None: ...


class FileIO:
    """Traced pwrite / pread against a local directory acting as the store."""

    def __init__(self, tracer: Optional[IoTracer] = None) -> None:
        self.tracer = tracer or IoTracer()
        self.fds = FdPool()

    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None:
        self.fds.close_rank(rank, prefix)

    def create_staging(self, path: str, size: int) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size)
        finally:
            os.close(fd)

    def sync(self, path: str) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def publish(self, temporary_path: str, final_path: str) -> None:
        os.replace(temporary_path, final_path)

    def discard(self, path: str) -> None:
        Path(path).unlink(missing_ok=True)

    def read_all(self, rank: int, path: str) -> bytes:
        return Path(path).read_bytes()

    def write_all(self, rank: int, path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    def identity(self, rank: int, path: str, check_open: bool) -> tuple:
        return _file_identity(
            path, self.fds.get(rank, path, "r") if check_open else None
        )

    def pwrite(self, rank: int, path: str, offset: int, data) -> None:
        fd = self.fds.get(rank, path, "w")
        view = memoryview(data)
        from .file_cache import pwrite_all

        pwrite_all(fd, view, offset)
        self.tracer.add(
            IoRecord(rank, "pwrite", path, offset, len(view), time.perf_counter())
        )

    def pread(self, rank: int, path: str, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        fd = self.fds.get(rank, path, "r")
        out = _pread_all(fd, offset, size)
        self.tracer.add(
            IoRecord(rank, "pread", path, offset, len(out), time.perf_counter())
        )
        if len(out) != size:
            raise IOError(f"short read {path}@{offset}: {len(out)} of {size} bytes")
        return out


class SdkFileIO(FileIO):
    """FileIO backend for a synchronous, versioned range-IO SDK client."""

    def __init__(self, client: FileSystemClient, tracer: Optional[IoTracer] = None):
        super().__init__(tracer)
        self.client = client
        self._versions: Dict[Tuple[int, str], Tuple[int, str]] = {}
        self._version_lock = threading.Lock()

    def _call(self, call: Callable[[], T]) -> T:
        try:
            return call()
        except OSError:
            raise
        except Exception as exc:
            raise OSError(str(exc)) from exc

    def _metadata(self, path: str) -> Tuple[int, str]:
        size, version = self._call(lambda: self.client.metadata(path))
        if (
            type(size) is not int
            or size < 0
            or not isinstance(version, str)
            or not version
        ):
            raise OSError(f"invalid SDK metadata for {path}: {(size, version)!r}")
        return size, version

    def _check_current(self, rank: int, path: str, current: Tuple[int, str]) -> None:
        key = (rank, path)
        with self._version_lock:
            pinned = self._versions.setdefault(key, current)
        if pinned != current:
            raise OSError("source path was replaced; open a new load session")

    def _pinned(self, rank: int, path: str) -> Tuple[int, str]:
        key = (rank, path)
        with self._version_lock:
            current = self._versions.get(key)
        if current is not None:
            return current
        current = self._metadata(path)
        self._check_current(rank, path, current)
        return current

    def close_rank(self, rank: int, prefix: Optional[str] = None) -> None:
        with self._version_lock:
            self._versions = {
                key: value
                for key, value in self._versions.items()
                if key[0] != rank
                or (prefix is not None and not key[1].startswith(prefix))
            }
        self._call(lambda: self.client.close_rank(rank, prefix))

    def create_staging(self, path: str, size: int) -> None:
        self._call(lambda: self.client.create_staging(path, size))

    def sync(self, path: str) -> None:
        self._call(lambda: self.client.sync(path))

    def publish(self, temporary_path: str, final_path: str) -> None:
        self._call(lambda: self.client.publish(temporary_path, final_path))

    def discard(self, path: str) -> None:
        self._call(lambda: self.client.discard(path))

    def _read(self, rank: int, path: str, offset: int, size: int, version: str):
        out = self._call(lambda: self.client.read_range(path, offset, size, version))
        self.tracer.add(
            IoRecord(rank, "pread", path, offset, len(out), time.perf_counter())
        )
        if len(out) != size:
            raise IOError(f"short read {path}@{offset}: {len(out)} of {size} bytes")
        return out

    def read_all(self, rank: int, path: str) -> bytes:
        size, version = self._metadata(path)
        return self._read(rank, path, 0, size, version) if size else b""

    def _write_count(self, path: str, written, size: int) -> int:
        if written is None:
            return size
        if type(written) is not int or not 0 < written <= size:
            raise IOError(f"invalid SDK write count for {path}: {written!r}")
        return written

    def _write(self, path: str, offset: int, data) -> None:
        view = memoryview(data)
        while view:
            written = self._call(lambda: self.client.write_range(path, offset, view))
            written = self._write_count(path, written, len(view))
            offset += written
            view = view[written:]

    def write_all(self, rank: int, path: str, data: bytes) -> None:
        self.create_staging(path, len(data))
        self._write(path, 0, data)
        self.sync(path)

    def identity(self, rank: int, path: str, check_open: bool) -> tuple:
        current = self._metadata(path)
        if check_open:
            self._check_current(rank, path, current)
        return (path, *current)

    def pwrite(self, rank: int, path: str, offset: int, data) -> None:
        size = len(memoryview(data))
        self._write(path, offset, data)
        self.tracer.add(
            IoRecord(rank, "pwrite", path, offset, size, time.perf_counter())
        )

    def pread(self, rank: int, path: str, offset: int, size: int) -> bytes:
        if size == 0:
            return b""
        _, version = self._pinned(rank, path)
        return self._read(rank, path, offset, size, version)
