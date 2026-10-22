"""Session and accessor abstract base classes (spec B.2.2.5/B.2.2.6/C.2).

The lite runtime keeps concrete sessions and the thread-local accessor
binding. These ABCs name the spec surface: get_file_offset,
init_data_accessor, commit/abort (save) and finish (load).
"""
from __future__ import annotations

import abc
from typing import Callable, Generic, TypeVar

from .layout import TensorComponentPos
from .ranges import RangeFromStorageView
from typing import Any

from .tag import Ok

RangeT = TypeVar("RangeT")

# Spec C.2: (buffer_ptr, buffer_len) -> Result[void]. The lite runtime hands a
# memoryview to callbacks; BufferCallback documents the contract.
BufferCallback = Callable[..., Any]


class StorageSaveSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos: TensorComponentPos) -> Any: ...

    @abc.abstractmethod
    def init_data_accessor(self):
        """Return a data accessor bound to this session."""

    @abc.abstractmethod
    def commit(self) -> Any: ...

    @abc.abstractmethod
    def abort(self) -> Any: ...


class StorageLoadSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos: TensorComponentPos) -> Any: ...

    @abc.abstractmethod
    def init_data_accessor(self):
        """Return a data accessor bound to this session."""

    @abc.abstractmethod
    def finish(self) -> Any: ...


class DTensorDataAccessor(abc.ABC, Generic[RangeT]):
    @abc.abstractmethod
    def read_to_buffer(self, global_range: RangeT, read_to_buffer: BufferCallback) -> Any: ...

    @abc.abstractmethod
    def write_from_buffer(self, global_range: RangeT, write_from_buffer: BufferCallback) -> Any: ...


# Register the concrete lite sessions/accessor as virtual subclasses so
# isinstance checks match the spec types without changing their hierarchy.
def _register_virtuals() -> None:
    from .formats import DcpLoadSession, DcpSaveSession, SafeTensorsLoadSession, SafeTensorsSaveSession
    from .views import StorageDataAccessor

    for cls in (SafeTensorsSaveSession, DcpSaveSession):
        StorageSaveSession.register(cls)
    for cls in (SafeTensorsLoadSession, DcpLoadSession):
        StorageLoadSession.register(cls)
    DTensorDataAccessor.register(StorageDataAccessor)


_register_virtuals()

# Spec-defined init_data_accessor for the concrete session mixin: the lite
# storage accessor is session-bound via thread-local; return it directly.
