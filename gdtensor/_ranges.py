"""Ranges and CopyPlan (DTensorsView B.3.1).

The oneof tag is the Python class: ``$compute`` = RangeFromComputeView,
``$storage`` = RangeFromStorageView. Units are bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union


@dataclass(frozen=True)
class RangeFromComputeView:
    global_tensor_id: str
    global_tensor_offset: int  # byte offset in the flat row-major global tensor
    size: int


@dataclass(frozen=True)
class RangeFromStorageView:
    file_path: str
    # Offset in the contiguous payload region of this file (no header, no
    # alignment, empty slots use 0 bytes). NOT the file offset.
    contiguous_payload_offset: int
    size: int


DTensorRange = Union[RangeFromComputeView, RangeFromStorageView]

# list[($src DTensorRange, $dst DTensorRange)]
CopyPlan = List[Tuple[DTensorRange, DTensorRange]]


def is_compute(r: DTensorRange) -> bool:
    return isinstance(r, RangeFromComputeView)


def is_storage(r: DTensorRange) -> bool:
    return isinstance(r, RangeFromStorageView)


from ._builder_mixin import attach_builders as _ab
_ab(RangeFromComputeView)
_ab(RangeFromStorageView)
