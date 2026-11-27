"""Minimal in-memory FileSystemClient fixture for SdkFileIO integration tests."""

from __future__ import annotations

import threading


def _synchronized(method):
    def run(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return run


class MemorySdkClient:
    """Thread-safe in-memory range-IO client matching SdkFileIO's protocol."""

    def __init__(self) -> None:
        self.objects: dict[str, bytearray] = {}
        self.versions: dict[str, str] = {}
        self.history: dict[tuple[str, str], bytes] = {}
        self.synced: set[str] = set()
        self.events: list[tuple] = []
        self.counter = 0
        self.max_write: int | None = None
        self.zero_write = False
        self.short_read = False
        self.fail_publish: str | None = None
        self._lock = threading.RLock()

    @_synchronized
    def _set(self, key: str, data) -> None:
        self.counter += 1
        version = f"v{self.counter}"
        raw = bytes(data)
        self.objects[key] = bytearray(raw)
        self.versions[key] = version
        self.history[(key, version)] = raw

    @_synchronized
    def metadata(self, key: str) -> tuple[int, str]:
        if key not in self.objects:
            raise OSError(f"missing {key}")
        return len(self.objects[key]), self.versions[key]

    @_synchronized
    def read_range(self, key: str, offset: int, size: int, version: str) -> bytes:
        try:
            raw = self.history[(key, version)]
        except KeyError:
            raise OSError(f"missing version {key}@{version}") from None
        if self.short_read and size:
            size -= 1
        return raw[offset : offset + size]

    @_synchronized
    def create_staging(self, key: str, size: int) -> None:
        current = self.objects.get(key)
        if current is not None and len(current) == size:
            return
        raw = bytearray(current or b"")
        del raw[size:]
        raw.extend(b"\0" * (size - len(raw)))
        self._set(key, raw)

    @_synchronized
    def write_range(self, key: str, offset: int, data: memoryview) -> int | None:
        if self.zero_write:
            return 0
        raw = bytes(data)
        size = len(raw) if self.max_write is None else min(len(raw), self.max_write)
        self.objects[key][offset : offset + size] = raw[:size]
        self._set(key, self.objects[key])
        return size

    @_synchronized
    def sync(self, key: str) -> None:
        self.events.append(("sync", key))
        self.synced.add(key)

    @_synchronized
    def publish(self, temporary_key: str, final_key: str) -> None:
        self.events.append(("publish", temporary_key, final_key))
        if final_key == self.fail_publish:
            raise OSError(f"intentional publish failure: {final_key}")
        if temporary_key not in self.synced:
            raise OSError(f"unsynced publication: {temporary_key}")
        self._set(final_key, self.objects[temporary_key])
        self.discard(temporary_key)

    @_synchronized
    def discard(self, key: str) -> None:
        self.objects.pop(key, None)
        self.versions.pop(key, None)
        self.synced.discard(key)

    @_synchronized
    def close_rank(self, rank: int, prefix: str | None = None) -> None:
        self.events.append(("close_rank", rank, prefix))
