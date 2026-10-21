"""Plan oracle: an independent model of what every CopyPlan pair must mean.

A pair (src, dst) is correct iff both sides name the same tensor and the same
global elements, in the same order. The oracle maps a range to its global
element ids without the plan code path:

* compute range  -> arange(offset / itemsize, (offset + size) / itemsize)
* storage range  -> slot intervals expanded to element ids, then the local
                    slice [cpo - slot.cpo, + size) in units of itemsize

On top of the per-pair identity, ``check_rank_plans`` asserts the global
contracts of save / load / reshard.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from gdtensor.dtensor import global_tensor_meta_of, iv_normalize, iv_total, named_of, owned_intervals
from gdtensor.layout import PayloadIndex
from gdtensor.models import interval_elems
from gdtensor.ranges import RangeFromComputeView, RangeFromStorageView


def _slot_of(index: PayloadIndex, rng: RangeFromStorageView):
    fp = index.files[rng.file_path]
    starts = [s.contiguous_payload_offset for s in fp.slots]
    k = int(np.searchsorted(starts, rng.contiguous_payload_offset, side="right")) - 1
    assert k >= 0, f"{rng}: before first slot"
    s = fp.slots[k]
    assert rng.contiguous_payload_offset + rng.size <= s.contiguous_payload_offset + s.nbytes, f"{rng}: crosses slot"
    return s


def range_elems(rng, table, index: PayloadIndex) -> Tuple[str, np.ndarray]:
    """(tensor_id, global element ids) addressed by one range."""
    if isinstance(rng, RangeFromComputeView):
        isz = global_tensor_meta_of(named_of(table, rng.global_tensor_id)).itemsize
        assert rng.global_tensor_offset % isz == 0 and rng.size % isz == 0, f"{rng}: not element aligned"
        e0 = rng.global_tensor_offset // isz
        return rng.global_tensor_id, np.arange(e0, e0 + rng.size // isz, dtype=np.int64)
    s = _slot_of(index, rng)
    lb = rng.contiguous_payload_offset - s.contiguous_payload_offset
    assert lb % s.itemsize == 0 and rng.size % s.itemsize == 0, f"{rng}: not element aligned"
    elems = interval_elems(s.intervals)[lb // s.itemsize:(lb + rng.size) // s.itemsize]
    return s.pos.tensor_id, elems


def check_pair_identity(plan, src_table, src_index, dst_table, dst_index) -> None:
    for src, dst in plan:
        assert src.size == dst.size and src.size > 0, (src, dst)
        ts, es = range_elems(src, src_table, src_index)
        td, ed = range_elems(dst, dst_table, dst_index)
        assert ts == td, (src, dst)
        assert np.array_equal(es, ed), f"element mismatch {src} -> {dst}"


def _byte_iv(ranges, key, off) -> Dict[str, np.ndarray]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for r in ranges:
        o = getattr(r, off)
        out.setdefault(getattr(r, key), []).append((o, o + r.size))
    return {k: np.array(sorted(v), dtype=np.int64) for k, v in out.items()}


def assert_disjoint(iv: np.ndarray, what: str) -> None:
    if len(iv) > 1:
        bad = np.flatnonzero(iv[1:, 0] < iv[:-1, 1])
        assert bad.size == 0, f"{what}: overlap at {iv[bad[0]]} / {iv[bad[0] + 1]}"


def compute_side(ranges):
    return _byte_iv([r for r in ranges if isinstance(r, RangeFromComputeView)], "global_tensor_id", "global_tensor_offset")


def storage_side(ranges):
    return _byte_iv([r for r in ranges if isinstance(r, RangeFromStorageView)], "file_path", "contiguous_payload_offset")


def owned_bytes(nd, side: str, rank: int) -> np.ndarray:
    isz = global_tensor_meta_of(nd).itemsize
    return owned_intervals(nd, side, rank) * isz


def check_save(plans: Dict[int, list], ct, st, index: PayloadIndex) -> None:
    """Every rank reads only local memory; every payload byte is written once."""
    for r, plan in plans.items():
        check_pair_identity(plan, ct, None, st, index)
        for tid, iv in compute_side([s for s, _ in plan]).items():
            own = owned_bytes(named_of(ct, tid), "compute", r)
            assert iv_total(iv_normalize(np.concatenate([iv, own]))) == iv_total(own), f"rank {r} reads remote {tid}"
    allw = storage_side([d for p in plans.values() for _, d in p])
    for path, fp in index.files.items():
        iv = allw.get(path, np.zeros((0, 2), dtype=np.int64))
        assert_disjoint(iv, f"save {path}")
        assert iv_total(iv) == fp.payload_size, f"save {path}: {iv_total(iv)} of {fp.payload_size} bytes"


def check_load(plans: Dict[int, list], ct, st, index: PayloadIndex) -> None:
    """Every rank fills exactly its owned bytes, each byte once."""
    for r, plan in plans.items():
        check_pair_identity(plan, st, index, ct, None)
        got = compute_side([d for _, d in plan])
        for nd in ct.named_d_tensors:
            iv = got.get(nd.global_tensor_id, np.zeros((0, 2), dtype=np.int64))
            assert_disjoint(iv, f"load rank {r} {nd.global_tensor_id}")
            own = owned_bytes(nd, "compute", r)
            assert np.array_equal(iv_normalize(iv), iv_normalize(own)), f"load rank {r} {nd.global_tensor_id}"
