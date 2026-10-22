"""Ranges and CopyPlan (design B.3.1). Internal construction types (section 8).

The oneof branch is the Python class. Units are bytes.
"""
from __future__ import annotations

from dataclasses import dataclass

from .builder_mixin import field_hash, make_builder_mixin

RangeFromComputeViewBuilderMixin = make_builder_mixin(
    "RangeFromComputeView", [("global_tensor_id", "scalar"), ("global_tensor_offset", "scalar"), ("size", "scalar")])
RangeFromStorageViewBuilderMixin = make_builder_mixin(
    "RangeFromStorageView", [("file_path", "scalar"), ("contiguous_payload_offset", "scalar"), ("size", "scalar")])


@dataclass(eq=True)
class RangeFromComputeView(RangeFromComputeViewBuilderMixin):
    global_tensor_id: str = ""
    global_tensor_offset: int = 0  # byte offset in the flat row-major global tensor
    size: int = 0
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints
        return check_constraints(self)


@dataclass(eq=True)
class RangeFromStorageView(RangeFromStorageViewBuilderMixin):
    file_path: str = ""
    # Offset in the contiguous payload region of this file (no header, no
    # alignment, empty slots use 0 bytes). Not the file offset.
    contiguous_payload_offset: int = 0
    size: int = 0
    __hash__ = field_hash

    def check_constraints(self):
        from .constraints import check_constraints
        return check_constraints(self)


DTensorRange = RangeFromComputeView | RangeFromStorageView

CopyPlan = list[tuple[DTensorRange, DTensorRange]]
