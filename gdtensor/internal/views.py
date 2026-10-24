"""Compute and storage view products (design B.2.1.1 / B.2.2.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


from .builder_mixin import field_hash, make_builder_mixin
from .dtensor import (
    DTensorTable,
)
from .layout import PhysicalLayoutTable


# ===================================================================== compute


ComputeDTensorsViewBuilderMixin = make_builder_mixin(
    "ComputeDTensorsView",
    [("tensor_ids", "list"), ("d_tensor_table", "object"), ("data_accessor", "scalar")],
)


@dataclass(eq=True)
class ComputeDTensorsView(ComputeDTensorsViewBuilderMixin):
    tensor_ids: List[str] = field(default_factory=list)
    d_tensor_table: DTensorTable = field(default_factory=DTensorTable)
    data_accessor: Any = None
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


# ===================================================================== storage


StorageDTensorsViewBuilderMixin = make_builder_mixin(
    "StorageDTensorsView",
    [
        ("tensor_ids", "list"),
        ("d_tensor_table", "object"),
        ("data_accessor", "scalar"),
        ("physical_layout", "object"),
        ("begin_save", "scalar"),
        ("begin_load", "scalar"),
    ],
)


@dataclass(eq=True)
class StorageDTensorsView(StorageDTensorsViewBuilderMixin):
    tensor_ids: List[str] = field(default_factory=list)
    d_tensor_table: DTensorTable = field(default_factory=DTensorTable)
    data_accessor: Any = None
    __hash__ = field_hash
    physical_layout: PhysicalLayoutTable = field(default_factory=PhysicalLayoutTable)
    # Function fields, called as view.begin_save(view, rank) / view.begin_load(view, rank).
    begin_save: Optional[Callable] = field(default=None, repr=False)
    begin_load: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        from .constraints import check_constraints
        from .result import invalid

        table_ids = [n.global_tensor_id for n in self.d_tensor_table.named_d_tensors]
        bad = [
            l
            for l in self.physical_layout.tensor_entries.values()
            if list(l.tensor_ids) != table_ids
        ]
        if bad:
            raise invalid("tensor_ids", "labels order != DTensorTable header order")
        return check_constraints(self)


# ===================================================================== cache


CacheDTensorsViewBuilderMixin = make_builder_mixin(
    "CacheDTensorsView",
    [("tensor_ids", "list"), ("d_tensor_table", "object"), ("data_accessor", "scalar")],
)


@dataclass(eq=True)
class CacheDTensorsView(CacheDTensorsViewBuilderMixin):
    tensor_ids: List[str] = field(default_factory=list)
    d_tensor_table: DTensorTable = field(default_factory=DTensorTable)
    data_accessor: Any = None
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        from .constraints import check_constraints

        return check_constraints(self)


DTensorsView = ComputeDTensorsView | StorageDTensorsView | CacheDTensorsView
