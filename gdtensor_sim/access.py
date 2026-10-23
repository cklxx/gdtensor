"""Concrete file accessor for the simulation storage sessions."""

import ctypes
import numpy as np
from gdtensor.access import DTensorDataAccessor
from ._layout import PayloadIndex
from ._result import Fail, catch, invalid, io_error
from gdtensor.builder import Err, Ok
from .iobackend import FileIO


def buffer_address(buf) -> int:
    """Address of a writable numpy array or bytearray."""
    if isinstance(buf, np.ndarray):
        return buf.ctypes.data
    return ctypes.addressof((ctypes.c_char * len(buf)).from_buffer(buf))


def call_buffer_callback(callback, buf) -> None:
    out = callback(buffer_address(buf) if len(buf) else 0, len(buf))
    if type(out) is Err:
        raise Fail(out.value)
    if type(out) is not Ok:
        raise invalid("callback", f"buffer callback must return a Result, got {out!r}")


class StorageDataAccessor(DTensorDataAccessor):
    """DTensorDataAccessor[RangeFromStorageView].

    Sessions translate contiguous payload offsets to file offsets.
    Each accessor returned by init_data_accessor belongs to one session.
    """

    def __init__(self, index: PayloadIndex, io: FileIO, session=None) -> None:
        self.index = index
        self.io = io
        self.session = session

    def _file_offset(self, rng):
        s = self.session
        if s is None:
            raise invalid("session", "storage accessor used outside a session")
        fp = self.index.files.get(rng.file_path)
        if fp is None:
            raise invalid("global_range", f"unknown file {rng.file_path}")
        starts = [x.contiguous_payload_offset for x in fp.slots]
        k = (
            int(np.searchsorted(starts, rng.contiguous_payload_offset, side="right"))
            - 1
        )
        if k < 0:
            raise invalid(
                "global_range",
                f"offset {rng.contiguous_payload_offset} before first slot",
            )
        slot = fp.slots[k]
        if (
            rng.contiguous_payload_offset + rng.size
            > slot.contiguous_payload_offset + slot.nbytes
        ):
            raise invalid(
                "global_range", f"storage range {rng} crosses a slot boundary"
            )
        base = s.get_file_offset(slot.pos)
        if type(base) is Err:
            raise Fail(base.value)
        return (
            s,
            s.abspath(rng.file_path),
            base.value + rng.contiguous_payload_offset - slot.contiguous_payload_offset,
        )

    @catch
    def read_to_buffer(self, global_range, read_to_buffer) -> None:
        s, path, off = self._file_offset(global_range)
        try:
            data = self.io.pread(s.rank, path, off, global_range.size)
        except OSError as exc:
            raise io_error("read", str(exc), path) from None
        call_buffer_callback(read_to_buffer, bytearray(data))

    @catch
    def write_from_buffer(self, global_range, write_from_buffer) -> None:
        s, path, off = self._file_offset(global_range)
        buf = bytearray(global_range.size)
        call_buffer_callback(write_from_buffer, buf)
        try:
            self.io.pwrite(s.rank, path, off, buf)
        except OSError as exc:
            raise io_error("write", str(exc), path) from None
