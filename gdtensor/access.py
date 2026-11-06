"""Public range accessor interface (design B.2.3.2 and C.2)."""

from __future__ import annotations

import abc as _abc
from typing import Callable as _Callable, Generic as _Generic, TypeVar as _TypeVar

from . import Result as _Result

_RangeT = _TypeVar("_RangeT")
_BufferCallback = _Callable[[int, int], _Result[None]]

__all__ = ["DTensorDataAccessor"]


class DTensorDataAccessor(_abc.ABC, _Generic[_RangeT]):
    @_abc.abstractmethod
    def read_to_buffer(
        self, global_range: _RangeT, read_to_buffer: _BufferCallback
    ) -> _Result[None]: ...

    def has_bytes(self, global_range: _RangeT) -> bool:
        """Whether every byte of the range is present, without reading it.

        Caches that track validity answer from metadata; the default answer
        is no, so reuse stays off for accessors that cannot speak. Cache ranges
        carry an internal source_identity. An implementation that returns True
        must bind both published bytes and queries to that identity. Unknown
        identities do not permit reuse across calls. Reading published bytes
        must reject detected corruption before invoking the consumer callback.
        """
        return False

    def cache_identity(self) -> object | None:
        """Stable source identity for cache reuse, or None if unavailable.

        File backends use resolved paths and file metadata. This is not a
        caller-supplied version number. A changed identity selects new bytes.
        """
        return None

    @_abc.abstractmethod
    def write_from_buffer(
        self, global_range: _RangeT, write_from_buffer: _BufferCallback
    ) -> _Result[None]: ...
