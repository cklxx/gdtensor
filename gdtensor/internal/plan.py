"""Plan APIs (B.1.2) and check APIs (B.1.3). Pure functions, no session.

Range selection lives here only. The executors in api.py never re-select.

Who writes what:
* plan_save   : compute -> storage. The rank writes what it owns on the
                compute mesh. Replicate dims are deduplicated: one
                coordinate writes, rotated by tensor order.
* plan_load   : storage -> compute. The rank fills every element it owns.
* plan_reshard: any branch pair. The rank fills its destination share
                (compute: owned elements; storage: content column == rank).
                Sources: storage slots, or compute owners (own rank first,
                then the lowest other rank; that read stands for comm).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .dtensor import (
    EMPTY,
    DTensorTable,
    global_tensor_meta_of,
    iv_intersect,
    iv_normalize,
    iv_prefix,
    iv_subtract,
    iv_total,
    named_of,
    nbytes_of,
    tensor_ids_of,
    is_designated_writer,
    order_of,
    owned_intervals,
    participants_of,
)
from .layout import (
    PhysicalLayoutTable, TensorComponentPos, build_payload_index, leader_tensor_component_pos_of, list_cells, pos_key,
)
from .mesh import storage_mesh_cells
from .ranges import CopyPlan, DTensorRange, RangeFromComputeView, RangeFromStorageView
from .result import catch, check_failed, invalid
from .views import ComputeDTensorsView, StorageDTensorsView


def is_compute(r) -> bool:
    return type(r) is RangeFromComputeView


def is_storage(r) -> bool:
    return type(r) is RangeFromStorageView

# ------------------------------------------------------------------ helpers


class _Side:
    """Uniform 'who owns which elements, and how to address them' for one view."""

    def __init__(self, view) -> None:
        self.view = view
        self.table: DTensorTable = view.d_tensor_table
        self.is_storage = type(view) is StorageDTensorsView
        if self.is_storage:
            self.index = view.data_accessor.index

    def storage_pieces(self, tid: str) -> List[Tuple[np.ndarray, str, int, int]]:
        """Non-empty slots with a physical file: (intervals, path, cpo, itemsize)."""
        out = []
        for w in participants_of(named_of(self.table, tid), "storage"):
            hit = self.index.slots.get(TensorComponentPos(tid, w))
            if hit is not None:
                path, s = hit
                out.append((s.intervals, path, s.contiguous_payload_offset, s.itemsize))
        return out

    def dst_intervals(self, tid: str, rank: int):
        """Destination share of a rank: list of (intervals, address)."""
        nd = named_of(self.table, tid)
        if self.is_storage:
            hit = self.index.slots.get(TensorComponentPos(tid, rank))
            if hit is None:
                return []
            path, s = hit
            return [(s.intervals, ("storage", path, s.contiguous_payload_offset, s.itemsize))]
        iv = owned_intervals(nd, "compute", rank)
        return [(iv, ("compute", tid, global_tensor_meta_of(nd).itemsize))] if iv.size else []

    def src_candidates(self, tid: str, rank: int):
        """Ordered source owners: list of (intervals, address)."""
        nd = named_of(self.table, tid)
        if self.is_storage:
            return [(iv, ("storage", p, cpo, isz)) for iv, p, cpo, isz in self.storage_pieces(tid)]
        isz = global_tensor_meta_of(nd).itemsize
        ranks = participants_of(nd, "compute")
        ranks = ([rank] if rank in ranks else []) + [r for r in ranks if r != rank]
        return [(owned_intervals(nd, "compute", r), ("compute", tid, isz)) for r in ranks]


def _make_range(addr, owner_iv: np.ndarray, piece: np.ndarray) -> List[DTensorRange]:
    """Ranges for element pieces inside one owner's intervals."""
    if addr[0] == "compute":
        _, tid, isz = addr
        return [RangeFromComputeView(tid, int(a) * isz, int(b - a) * isz) for a, b in piece]
    _, path, cpo, isz = addr
    local = iv_prefix(owner_iv, piece[:, 0])
    return [RangeFromStorageView(path, cpo + int(l) * isz, int(b - a) * isz) for l, (a, b) in zip(local, piece)]


def _pair(dst_list, src_list) -> CopyPlan:
    plan: CopyPlan = []
    for dst_iv, dst_addr in dst_list:
        todo = iv_normalize(dst_iv.copy())
        for src_iv, src_addr in src_list:
            if todo.size == 0:
                break
            piece = iv_intersect(todo, src_iv)
            if piece.size == 0:
                continue
            srcs = _make_range(src_addr, src_iv, piece)
            dsts = _make_range(dst_addr, dst_iv, piece)
            plan.extend(zip(srcs, dsts))
            todo = iv_subtract(todo, piece)
        if todo.size:
            raise invalid("src_view", f"no source covers {iv_total(todo)} destination elements")
    return plan


def _check_same_tensors(a, b) -> None:
    if list(a.tensor_ids) != list(b.tensor_ids):
        raise invalid("tensor_ids", f"tensor_ids differ: {a.tensor_ids} vs {b.tensor_ids}")
    for tid in a.tensor_ids:
        ma = global_tensor_meta_of(named_of(a.d_tensor_table, tid))
        mb = global_tensor_meta_of(named_of(b.d_tensor_table, tid))
        if ma != mb:
            raise invalid("global_tensor_meta", f"global_tensor_meta differs for {tid}: {ma} vs {mb}")


# ------------------------------------------------------------------ B.1.2


@catch
def d_tensors_view_rank_plan_save(rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView) -> CopyPlan:
    _check_same_tensors(compute_view, storage_view)
    c, s = _Side(compute_view), _Side(storage_view)
    plan: CopyPlan = []
    for tid in compute_view.tensor_ids:
        nd = named_of(c.table, tid)
        if not is_designated_writer(nd, rank, order_of(c.table, tid)):
            continue
        iv = owned_intervals(nd, "compute", rank)
        if iv.size == 0:
            continue
        isz = global_tensor_meta_of(nd).itemsize
        for s_iv, path, cpo, s_isz in s.storage_pieces(tid):
            piece = iv_intersect(iv, s_iv)
            if piece.size:
                srcs = _make_range(("compute", tid, isz), iv, piece)
                dsts = _make_range(("storage", path, cpo, s_isz), s_iv, piece)
                plan.extend(zip(srcs, dsts))
    return plan


@catch
def d_tensors_view_rank_plan_load(rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView) -> CopyPlan:
    _check_same_tensors(compute_view, storage_view)
    c, s = _Side(compute_view), _Side(storage_view)
    plan: CopyPlan = []
    for tid in compute_view.tensor_ids:
        plan.extend(_pair(c.dst_intervals(tid, rank), s.src_candidates(tid, rank)))
    return plan


@catch
def d_tensors_view_rank_plan_reshard(rank: int, src_view, dst_view) -> CopyPlan:
    # Assert[tensor_ids equal and global_tensor_meta equal]
    _check_same_tensors(src_view, dst_view)
    src, dst = _Side(src_view), _Side(dst_view)
    plan: CopyPlan = []
    for tid in dst_view.tensor_ids:
        plan.extend(_pair(dst.dst_intervals(tid, rank), src.src_candidates(tid, rank)))
    return plan


# ------------------------------------------------------------------ B.1.3
# Internal checks. Disjoint checks run on one rank's plan or on the union of
# all ranks' plans; storage disjointness over the union is the invariant
# that lets ranks pwrite one shared file without locks. Covering checks run
# on the union.


def _ranges(plan: CopyPlan, pred) -> List[DTensorRange]:
    return [r for pair in plan for r in pair if pred(r)]


def _byte_iv(pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    arr = np.array(list(pairs), dtype=np.int64).reshape(-1, 2)
    arr[:, 1] += arr[:, 0]
    return arr


def _assert_disjoint(iv: np.ndarray, what: str, check: str) -> None:
    if iv.size == 0:
        return
    iv = iv[iv[:, 1] > iv[:, 0]]
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    bad = np.flatnonzero(iv[1:, 0] < iv[:-1, 1])
    if bad.size:
        k = int(bad[0])
        raise check_failed(check, f"{what}: overlap {iv[k].tolist()} / {iv[k + 1].tolist()}")


def _compute_by_tensor(plan: CopyPlan) -> Dict[str, np.ndarray]:
    out: Dict[str, list] = {}
    for r in _ranges(plan, is_compute):
        out.setdefault(r.global_tensor_id, []).append((r.global_tensor_offset, r.size))
    return {k: _byte_iv(v) for k, v in out.items()}


def _storage_by_file(plan: CopyPlan) -> Dict[str, np.ndarray]:
    out: Dict[str, list] = {}
    for r in _ranges(plan, is_storage):
        out.setdefault(r.file_path, []).append((r.contiguous_payload_offset, r.size))
    return {k: _byte_iv(v) for k, v in out.items()}


@catch
def check_compute_d_tensors_access_disjoint(d_tensor_table: DTensorTable, copy_plan: CopyPlan) -> None:
    for tid, iv in _compute_by_tensor(copy_plan).items():
        order_of(d_tensor_table, tid)
        _assert_disjoint(iv, f"compute {tid}", "check_compute_d_tensors_access_disjoint")


@catch
def check_compute_d_tensors_access_full_covering(d_tensor_table: DTensorTable, copy_plan: CopyPlan) -> None:
    by_t = _compute_by_tensor(copy_plan)
    for nd in d_tensor_table.named_d_tensors:
        total = nbytes_of(global_tensor_meta_of(nd))
        iv = iv_normalize(by_t.get(nd.global_tensor_id, EMPTY))
        if not (iv.shape[0] == 1 and iv[0, 0] == 0 and iv[0, 1] == total) and total:
            raise check_failed("check_compute_d_tensors_access_full_covering",
                               f"compute {nd.global_tensor_id}: covered {iv_total(iv)} of {total} bytes")


def _storage_common(table: DTensorTable, layout: PhysicalLayoutTable, file_paths: List[str], plan: CopyPlan,
                    check: str):
    index = build_payload_index(table, layout)
    known = set(file_paths)
    by_f = _storage_by_file(plan)
    for path in by_f:
        if path not in known:
            raise check_failed(check, f"storage range on unknown file {path}")
        # every range must lie inside one slot
        starts = np.array([s.contiguous_payload_offset for s in index.files[path].slots])
        ends = starts + np.array([s.nbytes for s in index.files[path].slots])
        iv = by_f[path]
        k = np.searchsorted(starts, iv[:, 0], side="right") - 1
        if np.any(k < 0) or np.any(iv[:, 1] > ends[np.maximum(k, 0)]):
            raise check_failed(check, f"{path}: storage range crosses a slot boundary")
    return index, by_f


@catch
def check_storage_d_tensors_access_disjoint(
    d_tensor_table: DTensorTable, physical_layout: PhysicalLayoutTable, file_paths: List[str], copy_plan: CopyPlan
) -> None:
    _, by_f = _storage_common(d_tensor_table, physical_layout, file_paths, copy_plan,
                              "check_storage_d_tensors_access_disjoint")
    for path, iv in by_f.items():
        _assert_disjoint(iv, f"storage {path}", "check_storage_d_tensors_access_disjoint")


@catch
def check_storage_d_tensors_access_full_covering(
    d_tensor_table: DTensorTable, physical_layout: PhysicalLayoutTable, file_paths: List[str], copy_plan: CopyPlan
) -> None:
    index, by_f = _storage_common(d_tensor_table, physical_layout, file_paths, copy_plan,
                                  "check_storage_d_tensors_access_full_covering")
    for path in file_paths:
        total = index.files[path].payload_size
        iv = iv_normalize(by_f.get(path, EMPTY))
        if total and not (iv.shape[0] == 1 and iv[0, 0] == 0 and iv[0, 1] == total):
            raise check_failed("check_storage_d_tensors_access_full_covering",
                               f"storage {path}: covered {iv_total(iv)} of {total} bytes")


@catch
def check_storage_mesh_disjoint(lhs, rhs) -> None:
    both = storage_mesh_cells(lhs) & storage_mesh_cells(rhs)
    if both:
        raise check_failed("check_storage_mesh_disjoint", f"storage meshes overlap at {sorted(both)[:4]}")


@catch
def check_physical_layout(table: DTensorTable, layout: PhysicalLayoutTable) -> None:
    """Section 6 invariants of PhysicalLayoutTable (internal)."""
    name = "check_physical_layout"
    ids = tensor_ids_of(table)
    lists = {}
    for pos, lst in layout.tensor_entries.items():
        if lst.tensor_ids != ids:
            raise check_failed(name, "DisjointStorageMeshList rows differ from DTensorTable order")
        leader = leader_tensor_component_pos_of(lst)
        if leader not in layout.file_paths:
            raise check_failed(name, f"leader {pos_key(leader)} has no file path")
        lists.setdefault(leader, lst)
        t = order_of(table, pos.tensor_id)
        if (t, pos.rank) not in list_cells(lst):
            raise check_failed(name, f"{pos_key(pos)} is not covered by its own list")
    if set(layout.file_paths) != set(lists):
        raise check_failed(name, "file_paths keys must be exactly the list leaders")
    if len(set(layout.file_paths.values())) != len(layout.file_paths):
        raise check_failed(name, "two lists share one file path")
    seen: Dict[Tuple[int, int], TensorComponentPos] = {}
    for leader, lst in lists.items():
        for cell in list_cells(lst):
            if cell in seen:
                raise check_failed(name, f"slot {cell} is covered by two files")
            seen[cell] = leader
    index = build_payload_index(table, layout)
    for t, nd in enumerate(table.named_d_tensors):
        # single-tensor mesh == row slice of the multi-tensor meshes
        row = {w for (tt, w) in seen if tt == t}
        nonempty = {s.pos.rank for _, s in index.slots.values() if s.tensor_order == t}
        if not nonempty <= row:
            raise check_failed(name, f"{nd.global_tensor_id}: non-empty slot without physical file")
        cols = set(participants_of(nd, "storage"))
        for w in row:
            if w not in cols:
                raise check_failed(name, f"{nd.global_tensor_id}: list covers column {w} outside its storage mesh")
