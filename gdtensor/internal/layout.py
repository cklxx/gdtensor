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
    order_of,
    owned_intervals,
    participants_of,
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
    tensor_ids: List[str] = field(default_factory=list)
    __hash__ = field_hash

    def __post_init__(self) -> None:
        self.disjoint_storage_mesh_list = list(self.disjoint_storage_mesh_list)
        self.tensor_ids = list(self.tensor_ids)

    def check_constraints(self):
        return _cc(self)


def _check_tensor_ids(lst: DisjointStorageMeshList) -> None:
    rows: set = set()
    for m in lst.disjoint_storage_mesh_list:
        if not is_multi_tensors_storage_mesh(m):
            raise invalid(
                "disjoint_storage_mesh_list", "list holds MultiTensorsStorageMesh only"
            )
        for t, _ in storage_mesh_cells(m):
            rows.add(t)
    for t in rows:
        if t >= len(lst.tensor_ids):
            raise invalid("tensor_ids", f"row {t} has no label")
    if any(not isinstance(t, str) or not t for t in lst.tensor_ids):
        raise invalid("tensor_ids", "labels must be non-empty strings")
    if len(set(lst.tensor_ids)) != len(lst.tensor_ids):
        raise invalid("tensor_ids", "labels must be unique")


def list_cells(lst: DisjointStorageMeshList) -> List[Tuple[int, int]]:
    """Sorted (row, column) cells of a list; meshes must be multi-tensor and disjoint."""
    _check_tensor_ids(lst)
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
    """Canonical key: the smallest (row, column) cell, row -> real tensor id.

    ``tensor_ids[row]`` labels each row in DTensorTable header order (design
    delta approved by owner). This helper is self-contained: no extra table or
    registry argument.
    """
    cells = list_cells(disjoint_storage_meshes)
    if not cells:
        raise invalid(
            "disjoint_storage_meshes", "empty DisjointStorageMeshList has no leader"
        )
    _t, w = cells[0]
    if _t >= len(disjoint_storage_meshes.tensor_ids):
        raise invalid(
            "tensor_ids",
            f"row {_t} has no tensor_ids label (got {len(disjoint_storage_meshes.tensor_ids)})",
        )
    return TensorComponentPos(disjoint_storage_meshes.tensor_ids[_t], w)


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
) -> str:
    """tensor_entries[pos] -> list -> that list's unique file path.

    Resolving by list identity (design B.2.2.2): tensor_entries[pos] gives the
    list, and the list's single file_paths key is whichever entry points at the
    same list object. A missing key at either step returns Err(NotFoundError).
    """
    lst = physical_layout.tensor_entries.get(pos)
    if lst is None:
        raise not_found(
            "tensor_entries", pos_key(pos), f"{pos_key(pos)} has no physical file"
        )
    for key, path in physical_layout.file_paths.items():
        if physical_layout.tensor_entries.get(key) == lst:
            return path
    raise not_found("file_paths", pos_key(pos), f"{pos_key(pos)} list has no file path")


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
    table: DTensorTable,
) -> Dict[TensorComponentPos, DisjointStorageMeshList]:
    """leader -> list, ordered by (DTensorTable header row, rank).

    Entries pointing at value-equal lists (deepcopy) are the same physical file.
    The leader of a list is its smallest actual ``tensor_entries`` key once
    keys are ordered by DTensorTable header row of their tensor_id, then by
    rank. This is independent of dict insertion order or file row order.
    """
    by_list: Dict[Any, list] = {}
    for pos, lst in layout.tensor_entries.items():
        by_list.setdefault(lst, [lst, []])[1].append(pos)
    leaders: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    for lst, positions in by_list.values():
        leader = min(positions, key=lambda p: (order_of(table, p.tensor_id), p.rank))
        if leader in leaders and leaders[leader] != lst:
            raise invalid("tensor_entries", "different physical lists share a leader")
        leaders[leader] = lst
    return dict(sorted(leaders.items(), key=lambda kv: pos_key(kv[0])))


def _nonempty_cells(lst, columns):
    for tensor, rank in list_cells(lst):
        intervals = columns[tensor].get(rank)
        if intervals is not None:
            yield tensor, rank, intervals


def _path_for_list(layout: PhysicalLayoutTable, lst) -> str:
    for key, path in layout.file_paths.items():
        if layout.tensor_entries.get(key) == lst:
            return path
    raise not_found(
        "file_paths",
        "leader",
        "physical list has no file path",
    )


def build_payload_index(
    table: DTensorTable, layout: PhysicalLayoutTable
) -> PayloadIndex:
    files: Dict[str, FilePayload] = {}
    slots: Dict[TensorComponentPos, Tuple[str, SlotPayload]] = {}
    cols_by_row = [nonempty_columns(nd) for nd in table.named_d_tensors]
    table_ids = [n.global_tensor_id for n in table.named_d_tensors]
    for lst in layout.tensor_entries.values():
        if list(lst.tensor_ids) != table_ids:
            raise invalid("tensor_ids", "labels order != DTensorTable header order")
    lists = unique_lists(layout, table)
    for leader, lst in sorted(
        lists.items(), key=lambda kv: _path_for_list(layout, kv[1])
    ):
        path = _path_for_list(layout, lst)
        # The business must key file_paths by the real header-order leader;
        # any other component pos of the same list is rejected (B.2.2.2).
        fp_key = next(
            pos
            for pos, p in layout.file_paths.items()
            if p == path and layout.tensor_entries.get(pos) == lst
        )
        if fp_key != leader:
            raise invalid(
                "file_paths",
                f"file_paths key {pos_key(fp_key)} is not the leader {pos_key(leader)}",
            )
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
