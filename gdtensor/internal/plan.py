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
    is_designated_writer,
    order_of,
    owned_intervals,
    participants_of,
)
from .layout import (
    PhysicalLayoutTable,
    TensorComponentPos,
    build_payload_index,
)
from .mesh import storage_mesh_cells
from .ranges import CopyPlan, DTensorRange, RangeFromComputeView, RangeFromStorageView
from .result import catch, check_failed, invalid
from .views import ComputeDTensorsView, DTensorsView, StorageDTensorsView
from .. import Result
from ..tag import Ok


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
            self.index = build_payload_index(self.table, view.physical_layout)

    def storage_pieces(self, tid: str) -> List[Tuple[np.ndarray, str, int, int]]:
        """Non-empty slots with a physical file: (intervals, path, cpo, itemsize)."""
        hits = (
            self.index.slots.get(TensorComponentPos(tid, rank))
            for rank in participants_of(named_of(self.table, tid), "storage")
        )
        return [_storage_piece(hit) for hit in hits if hit is not None]

    def dst_intervals(self, tid: str, rank: int):
        """Destination share of a rank: list of (intervals, address)."""
        nd = named_of(self.table, tid)
        if not self.is_storage:
            iv = owned_intervals(nd, "compute", rank)
            return (
                [(iv, ("compute", tid, global_tensor_meta_of(nd).itemsize))]
                if iv.size
                else []
            )
        hit = self.index.slots.get(TensorComponentPos(tid, rank))
        if hit is None:
            return []
        path, slot = hit
        return [
            (
                slot.intervals,
                ("storage", path, slot.contiguous_payload_offset, slot.itemsize),
            )
        ]

    def src_candidates(self, tid: str, rank: int):
        """Ordered source owners: list of (intervals, address)."""
        nd = named_of(self.table, tid)
        if self.is_storage:
            return [
                (iv, ("storage", p, cpo, isz))
                for iv, p, cpo, isz in self.storage_pieces(tid)
            ]
        isz = global_tensor_meta_of(nd).itemsize
        ranks = participants_of(nd, "compute")
        ranks = ([rank] if rank in ranks else []) + [r for r in ranks if r != rank]
        return [
            (owned_intervals(nd, "compute", r), ("compute", tid, isz)) for r in ranks
        ]


def _storage_piece(hit):
    path, slot = hit
    return slot.intervals, path, slot.contiguous_payload_offset, slot.itemsize


def _make_range(addr, owner_iv: np.ndarray, piece: np.ndarray) -> List[DTensorRange]:
    """Ranges for element pieces inside one owner's intervals."""
    if addr[0] == "compute":
        _, tid, isz = addr
        return [
            RangeFromComputeView(tid, int(a) * isz, int(b - a) * isz) for a, b in piece
        ]
    _, path, cpo, isz = addr
    local = iv_prefix(owner_iv, piece[:, 0])
    return [
        RangeFromStorageView(path, cpo + int(l) * isz, int(b - a) * isz)
        for l, (a, b) in zip(local, piece)
    ]


def _pair_destination(dst_iv, dst_addr, src_list):
    plan = []
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
        raise invalid(
            "src_view", f"no source covers {iv_total(todo)} destination elements"
        )
    return plan


def _pair(dst_list, src_list) -> CopyPlan:
    plan = []
    for dst_iv, dst_addr in dst_list:
        plan.extend(_pair_destination(dst_iv, dst_addr, src_list))
    return plan


def _save_pieces(tid, intervals, itemsize, storage):
    plan = []
    remaining = intervals.copy()
    for slot_intervals, path, offset, slot_itemsize in storage.storage_pieces(tid):
        piece = iv_intersect(intervals, slot_intervals)
        if not piece.size:
            continue
        srcs = _make_range(("compute", tid, itemsize), intervals, piece)
        dsts = _make_range(
            ("storage", path, offset, slot_itemsize), slot_intervals, piece
        )
        plan.extend(zip(srcs, dsts))
        remaining = iv_subtract(remaining, piece)
    if remaining.size:
        raise invalid(
            "physical_layout",
            f"{tid}: {iv_total(remaining)} compute elements have no physical storage",
        )
    return plan


def _check_same_tensors(a, b) -> None:
    if list(a.tensor_ids) != list(b.tensor_ids):
        raise invalid(
            "tensor_ids", f"tensor_ids differ: {a.tensor_ids} vs {b.tensor_ids}"
        )
    for tid in a.tensor_ids:
        ma = global_tensor_meta_of(named_of(a.d_tensor_table, tid))
        mb = global_tensor_meta_of(named_of(b.d_tensor_table, tid))
        if ma != mb:
            raise invalid(
                "global_tensor_meta",
                f"global_tensor_meta differs for {tid}: {ma} vs {mb}",
            )


# ------------------------------------------------------------------ B.1.2


@catch
def d_tensors_view_rank_plan_save(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[CopyPlan]:
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
        plan.extend(_save_pieces(tid, iv, isz, s))
    return Ok(plan)


@catch
def d_tensors_view_rank_plan_load(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[CopyPlan]:
    _check_same_tensors(compute_view, storage_view)
    c, s = _Side(compute_view), _Side(storage_view)
    plan: CopyPlan = []
    for tid in compute_view.tensor_ids:
        plan.extend(_pair(c.dst_intervals(tid, rank), s.src_candidates(tid, rank)))
    return Ok(plan)


@catch
def d_tensors_view_rank_plan_reshard(
    rank: int, src_view: DTensorsView, dst_view: DTensorsView
) -> Result[CopyPlan]:
    # Assert[tensor_ids equal and global_tensor_meta equal]
    _check_same_tensors(src_view, dst_view)
    src, dst = _Side(src_view), _Side(dst_view)
    plan: CopyPlan = []
    for tid in dst_view.tensor_ids:
        plan.extend(_pair(dst.dst_intervals(tid, rank), src.src_candidates(tid, rank)))
    return Ok(plan)


# ------------------------------------------------------------------ B.1.3
# Internal checks. Disjoint checks run on one rank's plan or on the union of
# all ranks' plans; storage disjointness over the union is the invariant
# that lets ranks pwrite one shared file without locks. Covering checks run
# on the union.


def _ranges(plan: CopyPlan, pred) -> List[DTensorRange]:
    return [r for pair in plan for r in pair if pred(r)]


# ------------------------------------------------- runtime validation
#
# B.1.3 check_* are callable externally, but a correct rank executor must
# also enforce the per-rank safety properties itself: aggregate/union checks
# cannot see a rank that silently copied half of its shard (another rank's
# bytes would still make the union look full). The guards below run on one
# rank's plan in rank_save / rank_load / rank_reshard, before any session is
# opened, so a malformed plan never reaches the accessors.

CHECK = "runtime_plan_check"


def _dst_compute_pairs(plan: CopyPlan, tid: str):
    return [
        (dst.global_tensor_offset, dst.size)
        for _, dst in plan
        if is_compute(dst) and dst.global_tensor_id == tid
    ]


def _compute_sources(plan: CopyPlan) -> Dict[str, np.ndarray]:
    out: Dict[str, list] = {}
    for src, _ in plan:
        if is_compute(src):
            out.setdefault(src.global_tensor_id, []).append(
                (src.global_tensor_offset, src.size)
            )
    return {tid: _byte_iv(pairs) for tid, pairs in out.items()}


def _storage_side_within(view, plan: CopyPlan, which: str):
    """Bounds-check storage ranges on one copy side (src or dst) and return
    payload index plus file -> byte intervals. Each side is inspected alone
    so an ss reshard reading and writing the same file is not counted twice."""
    ranges = [src for src, _ in plan] if which == "src" else [dst for _, dst in plan]
    by_raw: Dict[str, list] = {}
    for r in ranges:
        if is_storage(r):
            by_raw.setdefault(r.file_path, []).append(
                (r.contiguous_payload_offset, r.size)
            )
    index = _Side(view).index
    by_f = {
        path: _check_bounds(
            pairs, index.files[path].payload_size, f"storage {path}", CHECK
        )
        for path, pairs in by_raw.items()
    }
    return index, by_f


def validate_rank_save(rank: int, compute_view, storage_view, plan: CopyPlan) -> None:
    """One rank reads exactly the compute elements it is designated to write,
    no more and no less, and writes disjoint bytes into storage.

    Coverage is per rank. The union of all ranks' plans looking complete
    cannot hide one rank that persisted only half of its shard.
    """
    for nd in compute_view.d_tensor_table.named_d_tensors:
        tid = nd.global_tensor_id
        pairs = [
            (src.global_tensor_offset, src.size)
            for src, _ in plan
            if is_compute(src) and src.global_tensor_id == tid
        ]
        designated = is_designated_writer(
            nd, rank, order_of(compute_view.d_tensor_table, tid)
        )
        owns = iv_total(owned_intervals(nd, "compute", rank)) > 0
        if designated and owns:
            _covers_owned(nd, rank, pairs, CHECK)
        elif pairs:
            raise check_failed(
                CHECK, f"rank {rank}: plan reads tensor {tid} it does not write"
            )
    _, by_f = _storage_side_within(storage_view, plan, "dst")
    for path, iv in by_f.items():
        _assert_disjoint(iv, f"storage {path}", CHECK)


def _aligned_byte_iv(nd, pairs, check: str) -> np.ndarray:
    itemsize = global_tensor_meta_of(nd).itemsize
    for offset, size in pairs:
        if offset < 0 or size < 0:
            raise check_failed(check, f"{nd.global_tensor_id}: negative range")
        if offset % itemsize or size % itemsize:
            raise check_failed(
                check,
                f"{nd.global_tensor_id}: range offset={offset} size={size} "
                f"is not aligned to itemsize {itemsize}",
            )
    return _byte_iv(pairs)


def _covers_owned(nd, rank: int, pairs, check: str) -> None:
    """Plan byte ranges on a compute tensor must equal the rank's owned
    elements exactly: no overlap, no gap, no extra byte. Owned elements are
    compared in element space because a shard on a non-leading axis yields
    several disjoint byte blocks, not one [0, nbytes) interval."""
    itemsize = global_tensor_meta_of(nd).itemsize
    byte_iv = _aligned_byte_iv(nd, pairs, check) if pairs else EMPTY
    _assert_disjoint(byte_iv, f"compute {nd.global_tensor_id}", check)
    got = iv_normalize((byte_iv // itemsize).astype(np.int64)) if pairs else EMPTY
    want = iv_normalize(owned_intervals(nd, "compute", rank))
    if not (got.shape == want.shape and bool(np.array_equal(got, want))):
        raise check_failed(
            check,
            f"rank {rank} compute {nd.global_tensor_id}: plan covers "
            f"{iv_total(got)} of {iv_total(want)} elements",
        )


def _check_rank_writes_only_owned(rank: int, tid: str, pairs, owns: bool) -> None:
    if not owns and pairs:
        raise check_failed(
            CHECK, f"rank {rank}: plan writes tensor {tid} it does not own"
        )


def validate_rank_load(rank: int, compute_view, storage_view, plan: CopyPlan) -> None:
    """One rank must fill every element it owns on the compute side, and
    its reads of one storage file must stay within that file's payload.

    Coverage is checked per rank, not over the union of all ranks: a
    replicated tensor must arrive in full at every replica, so another
    replica reading the missing bytes cannot hide this rank's gap.
    """
    nd_by_id = {
        nd.global_tensor_id: nd for nd in compute_view.d_tensor_table.named_d_tensors
    }
    got_ids = {dst.global_tensor_id for _, dst in plan if is_compute(dst)}
    extra = got_ids - set(nd_by_id)
    if extra:
        raise check_failed(CHECK, f"compute range on unknown tensor {sorted(extra)}")
    for tid, nd in nd_by_id.items():
        pairs = _dst_compute_pairs(plan, tid)
        owns = iv_total(owned_intervals(nd, "compute", rank)) > 0
        _check_rank_writes_only_owned(rank, tid, pairs, owns)
        if owns:
            _covers_owned(nd, rank, pairs, CHECK)
    index, by_f = _storage_side_within(storage_view, plan, "src")
    for path, iv in by_f.items():
        total = index.files[path].payload_size
        if iv.size and (int(iv[:, 0].min()) < 0 or int(iv[:, 1].max()) > total):
            raise check_failed(
                CHECK, f"storage {path}: read range outside [0, {total})"
            )


def validate_rank_reshard(rank: int, src_view, dst_view, plan: CopyPlan) -> None:
    # Destination coverage when it is compute; destination disjointness
    # when it is storage. Source ranges are bounds-checked.
    if type(dst_view) is ComputeDTensorsView:
        if type(src_view) is StorageDTensorsView:
            validate_rank_load(rank, dst_view, src_view, plan)
        else:
            _validate_compute_to_compute(rank, dst_view, plan)
    else:
        _validate_storage_destination(dst_view, plan)
        if type(src_view) is StorageDTensorsView:
            _validate_storage_source(src_view, plan)


def _validate_compute_to_compute(rank: int, compute_view, plan: CopyPlan) -> None:
    for nd in compute_view.d_tensor_table.named_d_tensors:
        tid = nd.global_tensor_id
        pairs = _dst_compute_pairs(plan, tid)
        owns = iv_total(owned_intervals(nd, "compute", rank)) > 0
        _check_rank_writes_only_owned(rank, tid, pairs, owns)
        if owns:
            _covers_owned(nd, rank, pairs, CHECK)
    for tid, iv in _compute_sources(plan).items():
        _assert_disjoint(iv, f"compute {tid}", CHECK)


def _validate_storage_destination(storage_view, plan: CopyPlan) -> None:
    _, by_f = _storage_side_within(storage_view, plan, "dst")
    for path, iv in by_f.items():
        _assert_disjoint(iv, f"storage {path}", CHECK)


def _validate_storage_source(storage_view, plan: CopyPlan) -> None:
    _, by_f = _storage_side_within(storage_view, plan, "src")
    for path, iv in by_f.items():
        total = _Side(storage_view).index.files[path].payload_size
        if iv.size and int(iv[:, 1].max()) > total:
            raise check_failed(
                CHECK, f"storage {path}: read range outside [0, {total})"
            )


# ------------------------------------------------------------------ B.1.3 public checks


def _byte_iv(pairs: Iterable[Tuple[int, int]]) -> np.ndarray:
    arr = np.array(list(pairs), dtype=np.int64).reshape(-1, 2)
    arr[:, 1] += arr[:, 0]
    return arr


def _check_bounds(pairs, total: int, what: str, check: str) -> np.ndarray:
    pairs = list(pairs)
    for offset, size in pairs:
        if offset < 0 or size < 0 or offset > total or size > total - offset:
            raise check_failed(
                check, f"{what}: offset={offset}, size={size} outside [0, {total})"
            )
    return _byte_iv(pairs)


def _assert_disjoint(iv: np.ndarray, what: str, check: str) -> None:
    if iv.size == 0:
        return
    iv = iv[iv[:, 1] > iv[:, 0]]
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    bad = np.flatnonzero(iv[1:, 0] < iv[:-1, 1])
    if bad.size:
        k = int(bad[0])
        raise check_failed(
            check, f"{what}: overlap {iv[k].tolist()} / {iv[k + 1].tolist()}"
        )


def _compute_by_tensor(table: DTensorTable, plan: CopyPlan, check: str):
    out: Dict[str, list] = {}
    for r in _ranges(plan, is_compute):
        out.setdefault(r.global_tensor_id, []).append((r.global_tensor_offset, r.size))
    totals = {
        nd.global_tensor_id: nbytes_of(global_tensor_meta_of(nd))
        for nd in table.named_d_tensors
    }
    for tid in out:
        if tid not in totals:
            raise check_failed(check, f"compute range on unknown tensor {tid}")
    return {
        tid: _check_bounds(pairs, totals[tid], f"compute {tid}", check)
        for tid, pairs in out.items()
    }


def _storage_by_file(plan: CopyPlan):
    out: Dict[str, list] = {}
    for r in _ranges(plan, is_storage):
        out.setdefault(r.file_path, []).append((r.contiguous_payload_offset, r.size))
    return out


@catch
def check_compute_d_tensors_access_disjoint(
    d_tensor_table: DTensorTable, copy_plan: CopyPlan
) -> Result[None]:
    check = "check_compute_d_tensors_access_disjoint"
    for tid, iv in _compute_by_tensor(d_tensor_table, copy_plan, check).items():
        _assert_disjoint(iv, f"compute {tid}", check)
    return Ok(None)


@catch
def check_compute_d_tensors_access_full_covering(
    d_tensor_table: DTensorTable, copy_plan: CopyPlan
) -> Result[None]:
    by_t = _compute_by_tensor(
        d_tensor_table, copy_plan, "check_compute_d_tensors_access_full_covering"
    )
    for nd in d_tensor_table.named_d_tensors:
        total = nbytes_of(global_tensor_meta_of(nd))
        iv = iv_normalize(by_t.get(nd.global_tensor_id, EMPTY))
        if not (iv.shape[0] == 1 and iv[0, 0] == 0 and iv[0, 1] == total) and total:
            raise check_failed(
                "check_compute_d_tensors_access_full_covering",
                f"compute {nd.global_tensor_id}: covered {iv_total(iv)} of {total} bytes",
            )
    return Ok(None)


def _storage_common(
    table: DTensorTable,
    layout: PhysicalLayoutTable,
    file_paths: List[str],
    plan: CopyPlan,
    check: str,
):
    index = build_payload_index(table, layout)
    known = set(file_paths)
    unknown = known - index.files.keys()
    if unknown:
        raise check_failed(check, f"unknown file_paths: {sorted(unknown)}")
    by_f = _storage_by_file(plan)
    for path in by_f:
        if path not in known:
            raise check_failed(check, f"storage range on unknown file {path}")
    return index, {
        path: _check_bounds(
            pairs, index.files[path].payload_size, f"storage {path}", check
        )
        for path, pairs in by_f.items()
    }


@catch
def check_storage_d_tensors_access_disjoint(
    d_tensor_table: DTensorTable,
    physical_layout: PhysicalLayoutTable,
    file_paths: List[str],
    copy_plan: CopyPlan,
) -> Result[None]:
    _, by_f = _storage_common(
        d_tensor_table,
        physical_layout,
        file_paths,
        copy_plan,
        "check_storage_d_tensors_access_disjoint",
    )
    for path, iv in by_f.items():
        _assert_disjoint(
            iv, f"storage {path}", "check_storage_d_tensors_access_disjoint"
        )
    return Ok(None)


@catch
def check_storage_d_tensors_access_full_covering(
    d_tensor_table: DTensorTable,
    physical_layout: PhysicalLayoutTable,
    file_paths: List[str],
    copy_plan: CopyPlan,
) -> Result[None]:
    index, by_f = _storage_common(
        d_tensor_table,
        physical_layout,
        file_paths,
        copy_plan,
        "check_storage_d_tensors_access_full_covering",
    )
    for path in file_paths:
        total = index.files[path].payload_size
        iv = iv_normalize(by_f.get(path, EMPTY))
        if total and not (iv.shape[0] == 1 and iv[0, 0] == 0 and iv[0, 1] == total):
            raise check_failed(
                "check_storage_d_tensors_access_full_covering",
                f"storage {path}: covered {iv_total(iv)} of {total} bytes",
            )
    return Ok(None)


@catch
def check_storage_mesh_disjoint(lhs, rhs) -> Result[None]:
    both = storage_mesh_cells(lhs) & storage_mesh_cells(rhs)
    if both:
        raise check_failed(
            "check_storage_mesh_disjoint",
            f"storage meshes overlap at {sorted(both)[:4]}",
        )
    return Ok(None)
