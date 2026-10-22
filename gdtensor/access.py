"""gdtensor.access: DTensorDataAccessor (design B.2.3.2 / C.2)."""
from __future__ import annotations

import abc
from typing import Callable, Generic, TypeVar

RangeT = TypeVar("RangeT")

# (buffer_ptr, buffer_len) -> Result[void]
BufferCallback = Callable[[int, int], object]

__all__ = ["DTensorDataAccessor"]


class DTensorDataAccessor(abc.ABC, Generic[RangeT]):
    @abc.abstractmethod
    def read_to_buffer(self, global_range: RangeT, read_to_buffer: BufferCallback):
        ...

    @abc.abstractmethod
    def write_from_buffer(self, global_range: RangeT, write_from_buffer: BufferCallback):
        ...
