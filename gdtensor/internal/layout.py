"""Physical model (design B.2.2.2 / B.2.2.3) and payload maps (internal).

Terms:
* cell (t, w): slot of tensor row ``t`` (DTensorTable order) and content
  column ``w`` (global rank). It is non-empty when ``w`` is on the tensor's
  SingleTensorStorageMesh and the column owns > 0 elements.
* A file payload is the concatenation of its non-empty cells, ordered by
  tensor row (header order), then by column. Empty cells use 0 bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from .builder_mixin import field_hash, make_builder_mixin
from .dtensor import (
    DTensorTable,
    NamedDTensor,
    global_tensor_meta_of,
    iv_total,
    owned_intervals,
    participants_of,
    tensor_ids_of,
)
from .mesh import is_multi_tensors_storage_mesh, storage_mesh_cells
from .result import catch, invalid, not_found


def _cc(self):
    from .constraints import check_constraints

    return check_constraints(self)


ContainerFormat = str  # "safetensors" / "distcp"

TensorComponentPosBuilderMixin = make_builder_mixin(
    "TensorComponentPos", [("tensor_id", "scalar"), ("rank", "scalar")]
)
DisjointStorageMeshListBuilderMixin = make_builder_mixin(
    "DisjointStorageMeshList",
    [("disjoint_storage_mesh_list", "list"), ("tensor_ids", "list")],
)
PhysicalLayoutTableBuilderMixin = make_builder_mixin(
    "PhysicalLayoutTable",
    [
        ("container_format", "scalar"),
        ("tensor_entries", "dict"),
        ("file_paths", "dict"),
    ],
)


@dataclass(eq=True)
class TensorComponentPos(TensorComponentPosBuilderMixin):
    tensor_id: str = ""
    rank: int = 0
    __hash__ = field_hash

    def check_constraints(self):
        return _cc(self)


def pos_key(pos: TensorComponentPos) -> Tuple[str, int]:
    return (pos.tensor_id, pos.rank)


@dataclass(eq=True)
class DisjointStorageMeshList(DisjointStorageMeshListBuilderMixin):
    disjoint_storage_mesh_list: List[Any] = field(default_factory=list)
    # Row labels supplied by the business constructor in table order.
    tensor_ids: List[str] = field(default_factory=list)
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.disjoint_storage_mesh_list = list(self.disjoint_storage_mesh_list)
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        return _cc(self)


def list_cells(lst: DisjointStorageMeshList) -> List[Tuple[int, int]]:
    """Sorted (row, column) cells of a list; meshes must be multi-tensor and disjoint."""
    out: set = set()
    for m in lst.disjoint_storage_mesh_list:
        if not is_multi_tensors_storage_mesh(m):
            raise invalid(
                "disjoint_storage_mesh_list", "list holds MultiTensorsStorageMesh only"
            )
        c = storage_mesh_cells(m)
        if out & c:
            raise invalid(
                "disjoint_storage_mesh_list", "meshes inside one list overlap"
            )
        out |= c
    return sorted(out)


def leader_tensor_component_pos_of(
    disjoint_storage_meshes: DisjointStorageMeshList,
) -> TensorComponentPos:
    """Canonical key: the smallest (row, column) cell."""
    cells = list_cells(disjoint_storage_meshes)
    if not cells:
        raise invalid(
            "disjoint_storage_meshes", "empty DisjointStorageMeshList has no leader"
        )
    t, w = cells[0]
    return TensorComponentPos(disjoint_storage_meshes.tensor_ids[t], w)


@dataclass(eq=True)
class PhysicalLayoutTable(PhysicalLayoutTableBuilderMixin):
    container_format: ContainerFormat = ""
    tensor_entries: Dict[TensorComponentPos, DisjointStorageMeshList] = field(
        default_factory=dict
    )
    file_paths: Dict[TensorComponentPos, str] = field(default_factory=dict)
    __hash__ = field_hash

    def check_constraints(self):
        return _cc(self)


@catch
def file_path_of(physical_layout: PhysicalLayoutTable, pos: TensorComponentPos) -> str:
    """tensor_entries[pos] -> list -> leader -> file_paths."""
    lst = physical_layout.tensor_entries.get(pos)
    if lst is None:
        raise not_found(
            "tensor_entries", pos_key(pos), f"{pos_key(pos)} has no physical file"
        )
    leader = leader_tensor_component_pos_of(lst)
    if leader not in physical_layout.file_paths:
        raise not_found(
            "file_paths", pos_key(leader), f"leader {pos_key(leader)} has no file path"
        )
    return physical_layout.file_paths[leader]


# ------------------------------------------------------------- payload maps (internal)


@dataclass(frozen=True)
class SlotPayload:
    pos: TensorComponentPos
    tensor_order: int
    intervals: np.ndarray  # owned element intervals, ascending
    itemsize: int
    contiguous_payload_offset: int  # bytes, inside its file

    @property
    def nbytes(self) -> int:
        return iv_total(self.intervals) * self.itemsize


@dataclass
class FilePayload:
    path: str
    leader: TensorComponentPos
    slots: List[SlotPayload] = field(default_factory=list)

    @property
    def payload_size(self) -> int:
        return sum(s.nbytes for s in self.slots)


@dataclass
class PayloadIndex:
    """Derived, deterministic view of a layout: file -> ordered slots."""

    files: Dict[str, FilePayload]
    slots: Dict[TensorComponentPos, Tuple[str, SlotPayload]]


def nonempty_columns(nd: NamedDTensor) -> Dict[int, np.ndarray]:
    out = {}
    for w in participants_of(nd, "storage"):
        iv = owned_intervals(nd, "storage", w)
        if iv_total(iv) > 0:
            out[w] = iv
    return out


def unique_lists(
    layout: PhysicalLayoutTable,
) -> Dict[TensorComponentPos, DisjointStorageMeshList]:
    """leader -> list, in a deterministic order."""
    lists: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    seen = set()
    for lst in layout.tensor_entries.values():
        if id(lst) in seen:
            continue
        seen.add(id(lst))
        leader = leader_tensor_component_pos_of(lst)
        if leader in lists and lists[leader] != lst:
            raise invalid("tensor_entries", "different physical lists share a leader")
        lists.setdefault(leader, lst)
    return dict(sorted(lists.items(), key=lambda kv: pos_key(kv[0])))


def _nonempty_cells(lst, columns):
    for tensor, rank in list_cells(lst):
        intervals = columns[tensor].get(rank)
        if intervals is not None:
            yield tensor, rank, intervals


def build_payload_index(
    table: DTensorTable, layout: PhysicalLayoutTable
) -> PayloadIndex:
    files: Dict[str, FilePayload] = {}
    slots: Dict[TensorComponentPos, Tuple[str, SlotPayload]] = {}
    cols_by_row = [nonempty_columns(nd) for nd in table.named_d_tensors]
    expected_ids = tensor_ids_of(table)
    for meshes in layout.tensor_entries.values():
        if meshes.tensor_ids != expected_ids:
            raise invalid(
                "tensor_ids", "physical list row labels differ from table order"
            )
    lists = unique_lists(layout)
    for leader, lst in sorted(
        lists.items(), key=lambda kv: layout.file_paths.get(kv[0], "")
    ):
        if leader not in layout.file_paths:
            raise not_found(
                "file_paths",
                pos_key(leader),
                f"leader {pos_key(leader)} has no file path",
            )
        path = layout.file_paths[leader]
        fp = FilePayload(path=path, leader=leader)
        off = 0
        for t, w, iv in _nonempty_cells(lst, cols_by_row):
            nd = table.named_d_tensors[t]
            pos = TensorComponentPos(nd.global_tensor_id, w)
            _check_slot_entry(layout, pos, lst, slots)
            sp = SlotPayload(pos, t, iv, global_tensor_meta_of(nd).itemsize, off)
            off += sp.nbytes
            fp.slots.append(sp)
            slots[pos] = (path, sp)
        if path in files:
            raise invalid("file_paths", f"two lists map to one path {path}")
        files[path] = fp
    return PayloadIndex(files, slots)


def _check_slot_entry(layout, pos, meshes, slots):
    if layout.tensor_entries.get(pos) != meshes:
        raise invalid(
            "tensor_entries", f"nonempty slot {pos_key(pos)} has no matching entry"
        )
    if pos in slots:
        raise invalid(
            "tensor_entries", f"nonempty slot {pos_key(pos)} is covered twice"
        )
