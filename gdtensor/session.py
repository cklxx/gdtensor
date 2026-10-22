"""Public storage session interfaces (design B.2.2.5/B.2.2.6 and C.2)."""

from __future__ import annotations

import abc as _abc

from . import Result as _Result
from .access import DTensorDataAccessor as _Accessor
from .builder import TensorComponentPos as _TensorComponentPos
from .internal.ranges import RangeFromStorageView as _StorageRange

__all__ = ["StorageSaveSession", "StorageLoadSession"]


class StorageSaveSession(_abc.ABC):
    @_abc.abstractmethod
    def get_file_offset(self, pos: _TensorComponentPos) -> _Result[int]: ...

    @_abc.abstractmethod
    def init_data_accessor(self) -> _Result[_Accessor[_StorageRange]]: ...

    @_abc.abstractmethod
    def commit(self) -> _Result[None]: ...

    @_abc.abstractmethod
    def abort(self) -> _Result[None]: ...


class StorageLoadSession(_abc.ABC):
    @_abc.abstractmethod
    def get_file_offset(self, pos: _TensorComponentPos) -> _Result[int]: ...

    @_abc.abstractmethod
    def init_data_accessor(self) -> _Result[_Accessor[_StorageRange]]: ...

    @_abc.abstractmethod
    def finish(self) -> _Result[None]: ...
