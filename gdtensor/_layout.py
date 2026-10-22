"""Physical model (DTensorsView B.2.2.2 / B.2.2.3) plus lite payload maps.

Terms used here:
* cell (t, w): slot of tensor row ``t`` (DTensorTable order) and content
  column ``w`` (global rank). It is *non-empty* when ``w`` is on the tensor's
  SingleTensorStorageMesh and the column owns > 0 elements.
* A file payload is the concatenation of its non-empty cells, ordered by
  tensor row (header order), then by column. Empty cells use 0 bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np

from ._dtensor import (
    DTensorTable,
    NamedDTensor,
    global_tensor_meta_of,
    iv_total,
    owned_intervals,
    participants_of,
)
from ._mesh import (
    ArrangedMesh,
    OuterProductMesh,
    Slice,
    SlicedMesh,
    StorageMesh,
    is_multi_tensors_storage_mesh,
    storage_mesh_cells,
)
from ._result import Result, ResultError, catch
from ._constraints import check_constraints as _check

ContainerFormat = Literal["safetensors", "distcp"]


class TensorComponentPos(NamedTuple):
    tensor_id: str
    rank: int


@dataclass(frozen=True)
class DisjointStorageMeshList:
    def check_constraints(self):
        return _check(self)

    """One logical list per physical file.

    ``tensor_ids`` is a lite addition: the row labels of the "tensor_count"
    axis. It lets ``leader_tensor_component_pos_of`` stay a pure function of
    the list, as the spec signature requires.
    """

    disjoint_storage_mesh_list: Tuple[StorageMesh, ...]
    tensor_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        for m in self.disjoint_storage_mesh_list:
            if not is_multi_tensors_storage_mesh(m):
                raise ResultError("DisjointStorageMeshList holds MultiTensorsStorageMesh only")

    @cached_property
    def cells(self) -> Tuple[Tuple[int, int], ...]:
        out: set = set()
        for m in self.disjoint_storage_mesh_list:
            c = storage_mesh_cells(m)
            if out & c:
                raise ResultError("meshes inside one DisjointStorageMeshList overlap")
            out |= c
        return tuple(sorted(out))


def leader_tensor_component_pos_of(disjoint_storage_meshes: DisjointStorageMeshList) -> TensorComponentPos:
    """Canonical key: the lexicographically smallest (row, column) cell."""
    cells = disjoint_storage_meshes.cells
    if not cells:
        raise ResultError("empty DisjointStorageMeshList has no leader")
    t, w = cells[0]
    return TensorComponentPos(disjoint_storage_meshes.tensor_ids[t], w)


@dataclass
class PhysicalLayoutTable:
    def check_constraints(self):
        return _check(self)

    """Pure configuration data. No function fields; JSON-serializable."""

    container_format: ContainerFormat
    tensor_entries: Dict[TensorComponentPos, DisjointStorageMeshList]
    file_paths: Dict[TensorComponentPos, str]

    def to_json(self) -> dict:
        lists = {}
        for pos, lst in self.tensor_entries.items():
            lists.setdefault(leader_tensor_component_pos_of(lst), lst)
        return {
            "container_format": self.container_format,
            "files": [
                {
                    "leader": list(leader),
                    "path": self.file_paths[leader],
                    "cells": [list(c) for c in lst.cells],
                }
                for leader, lst in sorted(lists.items())
            ],
            "tensor_entries": sorted(
                [list(pos), list(leader_tensor_component_pos_of(lst))] for pos, lst in self.tensor_entries.items()
            ),
        }


@catch
def file_path_of(layout: PhysicalLayoutTable, pos: TensorComponentPos) -> str:
    """tensor_entries[pos] -> list -> leader -> file_paths."""
    lst = layout.tensor_entries.get(TensorComponentPos(*pos))
    if lst is None:
        raise ResultError(f"{tuple(pos)} has no physical file")
    leader = leader_tensor_component_pos_of(lst)
    if leader not in layout.file_paths:
        raise ResultError(f"leader {tuple(leader)} has no file path")
    return layout.file_paths[leader]


# ------------------------------------------------------------- payload maps


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


def build_payload_index(table: DTensorTable, layout: PhysicalLayoutTable) -> PayloadIndex:
    files: Dict[str, FilePayload] = {}
    slots: Dict[TensorComponentPos, Tuple[str, SlotPayload]] = {}
    cols_by_row = [nonempty_columns(nd) for nd in table.named_d_tensors]
    lists: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    for lst in layout.tensor_entries.values():
        lists.setdefault(leader_tensor_component_pos_of(lst), lst)
    for leader, lst in sorted(lists.items(), key=lambda kv: layout.file_paths.get(kv[0], "")):
        if leader not in layout.file_paths:
            raise ResultError(f"leader {tuple(leader)} has no file path")
        path = layout.file_paths[leader]
        fp = FilePayload(path=path, leader=leader)
        off = 0
        for t, w in lst.cells:  # row-major: header order, then column order
            iv = cols_by_row[t].get(w)
            if iv is None:
                continue  # empty slot: 0 bytes, no offset accumulation
            nd = table.named_d_tensors[t]
            pos = TensorComponentPos(nd.global_tensor_id, w)
            sp = SlotPayload(pos, t, iv, global_tensor_meta_of(nd).itemsize, off)
            off += sp.nbytes
            fp.slots.append(sp)
            slots[pos] = (path, sp)
        if path in files:
            raise ResultError(f"two lists map to one path {path}")
        files[path] = fp
    return PayloadIndex(files, slots)


# ------------------------------------------------------------- business builder


def _cells_to_meshes(cells: List[Tuple[int, int]], n_tensors: int, world: int) -> Tuple[StorageMesh, ...]:
    """Cover a cell set with disjoint rectangles: per row, contiguous column
    runs; identical runs on consecutive rows merge into one rectangle."""
    base = OuterProductMesh(ArrangedMesh(n_tensors, "tensor_count"), ArrangedMesh(world, "world_size"))
    by_row: Dict[int, List[int]] = {}
    for t, w in cells:
        by_row.setdefault(t, []).append(w)
    runs_by_row: Dict[int, List[Tuple[int, int]]] = {}
    for t, ws in by_row.items():
        ws = sorted(ws)
        runs, s = [], ws[0]
        for a, b in zip(ws, ws[1:]):
            if b != a + 1:
                runs.append((s, a + 1))
                s = b
        runs.append((s, ws[-1] + 1))
        runs_by_row[t] = runs
    rects: List[Tuple[int, int, int, int]] = []
    open_: Dict[Tuple[int, int], int] = {}  # run -> start row
    rows = sorted(runs_by_row)
    prev_row = None
    for t in rows:
        cur = set(runs_by_row[t])
        contiguous = prev_row is not None and t == prev_row + 1
        for run in list(open_):
            if not contiguous or run not in cur:
                rects.append((open_.pop(run), prev_row + 1, *run))
        for run in cur:
            if run not in open_:
                open_[run] = t
        prev_row = t
    for run, t0 in open_.items():
        rects.append((t0, prev_row + 1, *run))
    rects.sort()
    return tuple(SlicedMesh(base, (Slice(t0, t1), Slice(w0, w1))) for t0, t1, w0, w1 in rects)


def build_layout(
    table: DTensorTable,
    world_size: int,
    container_format: ContainerFormat,
    file_of_cell,
) -> PhysicalLayoutTable:
    """Generic layout builder (business layer, not core).

    ``file_of_cell(t, w, named_d_tensor) -> path`` assigns each non-empty cell
    to one file. Empty cells get no entry.
    """
    ids = tuple(table.tensor_ids)
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for t, nd in enumerate(table.named_d_tensors):
        for w in nonempty_columns(nd):
            if not 0 <= w < world_size:
                raise ResultError(f"{nd.global_tensor_id}: column {w} outside world {world_size}")
            groups.setdefault(file_of_cell(t, w, nd), []).append((t, w))
    tensor_entries: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    file_paths: Dict[TensorComponentPos, str] = {}
    for path in sorted(groups):
        cells = groups[path]
        lst = DisjointStorageMeshList(_cells_to_meshes(cells, len(ids), world_size), ids)
        file_paths[leader_tensor_component_pos_of(lst)] = path
        for t, w in cells:
            tensor_entries[TensorComponentPos(ids[t], w)] = lst
    return PhysicalLayoutTable(container_format, tensor_entries, file_paths)


from ._builder_mixin import attach_builders as _ab
_ab(DisjointStorageMeshList)
_ab(PhysicalLayoutTable)
