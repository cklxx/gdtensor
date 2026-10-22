"""Cross-rank coordination for shared-file save (barrier + vote).

A shared file can be published only after every writer is done. Each rank
votes once at commit / abort time; if any rank votes "fail", every rank
discards. This is the "barrier" item that the spec leaves to the session
implementation layer.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from typing import Optional


class ThreadCoordinator:
    def __init__(self, n: int, timeout: float = 600.0) -> None:
        self.n = n
        self.timeout = timeout
        self._barrier = threading.Barrier(n)
        self._lock = threading.Lock()
        self._fail = 0
        self._gen = 0

    def vote(self, ok: bool) -> bool:
        with self._lock:
            self._fail += int(not ok)
        idx = self._barrier.wait(self.timeout)
        all_ok = self._fail == 0
        self._barrier.wait(self.timeout)  # everyone read the result
        if idx == 0:
            self._reset_failures()
        self._barrier.wait(self.timeout)
        return all_ok

    def _reset_failures(self):
        with self._lock:
            self._fail = 0

    def wait(self) -> None:
        self._barrier.wait(self.timeout)


class ProcessCoordinator:
    """Same contract across forked processes."""

    def __init__(self, n: int, ctx=None, timeout: float = 600.0) -> None:
        ctx = ctx or mp.get_context("fork")
        self.n = n
        self.timeout = timeout
        self._barrier = ctx.Barrier(n)
        self._fail = ctx.Value("i", 0)

    def vote(self, ok: bool) -> bool:
        with self._fail.get_lock():
            self._fail.value += int(not ok)
        idx = self._barrier.wait(self.timeout)
        all_ok = self._fail.value == 0
        self._barrier.wait(self.timeout)
        if idx == 0:
            self._reset_failures()
        self._barrier.wait(self.timeout)
        return all_ok

    def _reset_failures(self):
        with self._fail.get_lock():
            self._fail.value = 0

    def wait(self) -> None:
        self._barrier.wait(self.timeout)


Coordinator = Optional[
    object
]  # ThreadCoordinator | ProcessCoordinator | None (single rank)
