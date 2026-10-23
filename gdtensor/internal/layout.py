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
    [("disjoint_storage_mesh_list", "list")],
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
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.disjoint_storage_mesh_list = list(self.disjoint_storage_mesh_list)

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
    tensor_ids: List[str] | None = None,
) -> TensorComponentPos:
    """Canonical key: the smallest (row, column) cell.

    ``tensor_ids`` maps tensor-slot rows to tensor ids in header order.
    Without it positional row labels are used; internal plan layers always
    supply the table ids so layouts are validated lazily per file.
    """
    cells = list_cells(disjoint_storage_meshes)
    if not cells:
        raise invalid(
            "disjoint_storage_meshes", "empty DisjointStorageMeshList has no leader"
        )
    t, w = cells[0]
    label = tensor_ids[t] if tensor_ids is not None else f"tensor_{t}"
    return TensorComponentPos(label, w)


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
def file_path_of(
    physical_layout: PhysicalLayoutTable,
    pos: TensorComponentPos,
    tensor_ids: List[str] | None = None,
) -> str:
    """tensor_entries[pos] -> list -> that list's unique file path.

    Each physical list has exactly one ``file_paths`` key sitting on one of
    its own component positions. Resolving by list identity avoids remapping
    positional tensor rows to ids, which needs DTensorTable header order that
    the layout alone does not carry.
    """
    lst = physical_layout.tensor_entries.get(pos)
    if lst is None:
        raise not_found(
            "tensor_entries", pos_key(pos), f"{pos_key(pos)} has no physical file"
        )
    for key, path in physical_layout.file_paths.items():
        if physical_layout.tensor_entries.get(key) is lst:
            return path
    raise not_found(
        "file_paths", pos_key(pos), f"{pos_key(pos)} list has no file path"
    )


def _layout_tensor_ids(layout: PhysicalLayoutTable) -> List[str]:
    """Header-order tensor ids: first-appearance order in tensor_entries.

    Business code populates ``tensor_entries`` in DTensorTable (header) order,
    so the layout alone recovers the positional row labels without sorting
    alphabetically or taking a tensor_ids argument.
    """
    ids: List[str] = []
    for pos in layout.tensor_entries:
        if pos.tensor_id not in ids:
            ids.append(pos.tensor_id)
    return ids


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
    tensor_ids: List[str] | None = None,
) -> Dict[TensorComponentPos, DisjointStorageMeshList]:
    """leader -> list, in a deterministic order."""
    lists: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    seen = set()
    labels = tensor_ids if tensor_ids is not None else _layout_tensor_ids(layout)
    for lst in layout.tensor_entries.values():
        if id(lst) in seen:
            continue
        seen.add(id(lst))
        leader = leader_tensor_component_pos_of(lst, labels)
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
    table_ids = tensor_ids_of(table)
    lists = unique_lists(layout, table_ids)
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
