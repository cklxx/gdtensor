"""gdtensor.session: StorageSaveSession / StorageLoadSession (design B.2.2.5 / B.2.2.6 / C.2)."""
from __future__ import annotations

import abc

__all__ = ["StorageSaveSession", "StorageLoadSession"]


class StorageSaveSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos):
        ...

    @abc.abstractmethod
    def init_data_accessor(self):
        ...

    @abc.abstractmethod
    def commit(self):
        ...

    @abc.abstractmethod
    def abort(self):
        ...


class StorageLoadSession(abc.ABC):
    @abc.abstractmethod
    def get_file_offset(self, pos):
        ...

    @abc.abstractmethod
    def init_data_accessor(self):
        ...

    @abc.abstractmethod
    def finish(self):
        ...
