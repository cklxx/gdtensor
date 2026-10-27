"""Public IO executors: plan, open sessions, copy, and close sessions."""

from __future__ import annotations

from dataclasses import replace
import ctypes
import logging
import threading
import weakref

from .plan import (
    d_tensors_view_rank_plan_load,
    d_tensors_view_rank_plan_save,
    validate_rank_load,
    validate_rank_save,
)
from .ranges import CopyPlan, RangeFromComputeView
from .result import catch, invalid, must_ok
from .views import ComputeDTensorsView, StorageDTensorsView
from .. import Result
from .products import Ok


_LOG = logging.getLogger(__name__)


_view_slots: "weakref.WeakKeyDictionary[object, _SlotTable]" = (
    weakref.WeakKeyDictionary()
)
_table_lock = threading.Lock()


def _rng_key(rng) -> tuple:
    tid = getattr(rng, "global_tensor_id", "")
    off = getattr(rng, "global_tensor_offset", None)
    if off is None:
        off = rng.contiguous_payload_offset
    return (tid, off, rng.size)


def _seg(cache_rng, eff_src, src_rng) -> dict:
    tid = getattr(cache_rng, "global_tensor_id", "")
    off = getattr(cache_rng, "global_tensor_offset", None)
    if off is None:
        off = cache_rng.contiguous_payload_offset
    return {
        "key": (tid, off, cache_rng.size),
        "eff_src": eff_src,
        "src_rng": src_rng,
        "tid": tid,
        "off": off,
        "size": cache_rng.size,
    }


def _overlap(rng, seg) -> int:
    off = getattr(rng, "global_tensor_offset", None)
    if off is None:
        off = rng.contiguous_payload_offset
    end = off + rng.size
    s_e = seg["off"] + seg["size"]
    return max(0, min(end, s_e) - max(off, seg["off"]))


def _overlap_iv(rng, seg):
    """Global-offset overlap interval (off, size) between rng and a slot seg."""
    off = getattr(rng, "global_tensor_offset", None)
    if off is None:
        off = rng.contiguous_payload_offset
    end = off + rng.size
    s_e = seg["off"] + seg["size"]
    a, b = max(off, seg["off"]), min(end, s_e)
    if b <= a:
        return (off, 0)
    return (a, b - a)


def _rng_to_view_range(view, key):
    tid, off, size = key
    return RangeFromComputeView(tid, off, size)


class _NodeState:
    """Per-node, per-cache-view residency for one atomic fill segment.

    A segment is keyed by the global interval (tid, offset, size) it occupies
    on the cache node. Live segments are bounded by the cache mesh concurrency
    width C. A fill failure records one node-wide primary error bound to the
    current cohort generation: every rank sharing that generation raises it; a
    later generation (next explicit chain_load) starts clean. A segment releases
    only after every expected consumer rank on this node has ACKed its drain —
    including replicated ranks that read the same bytes.
    """

    def __init__(self, cap: int, consumers: int) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.slots: dict[tuple, dict] = {}
        self.live = 0
        self.cap = max(1, cap)
        self.consumers = max(1, consumers)
        self.error: BaseException | None = None
        self.error_gen = -1
        self.gen = 0
        self.active = 0
        self.cohort_joined: set = set()

    def generation_start(self, rank: int) -> int:
        """Join the node's current cohort, or open a fresh one only after the
        previous cohort's expected ranks have all observed it (so a slightly late
        same-cohort rank still sees the node-wide error). Per-rank join set, so a
        same-rank re-entry cannot prematurely advance the epoch."""
        with self.lock:
            self._cohort_join(rank)
        return self.gen

    def _cohort_join(self, rank: int) -> None:
        if self.active != 0 or len(self.cohort_joined) < self.consumers:
            self.active += 1
            self.cohort_joined.add(rank)
            return
        self.gen += 1
        self.error = None
        self.error_gen = -1
        self.cohort_joined = set()
        self.active += 1
        self.cohort_joined.add(rank)

    def generation_end(self) -> None:
        with self.lock:
            self.active = max(0, self.active - 1)
            self.cond.notify_all()

    @staticmethod
    def _new_slot():
        return {
            "filled": False,
            "error": None,
            "error_gen": -1,
            "required": {},  # rank -> set[(a,b)] it must drain
            "cov": {},  # rank -> set[(a,b)] acked so far
        }

    def set_expected(self, key: tuple, required: dict) -> None:
        with self.lock:
            self._set_expected_locked(key, required)

    def _set_expected_locked(self, key, required):
        slot = self.slots.get(key)
        if slot is not None and not slot["required"]:
            slot["required"] = {r: set(iv) for r, iv in required.items()}
        self.cond.notify_all()

    @staticmethod
    def _covers(ivs, required):
        merged = set(ivs)
        return all(any(x <= a and y >= b for x, y in merged) for a, b in required)

    def ack_drain(self, key: tuple, rank: int, off: int, size: int) -> None:
        with self.lock:
            self._ack_drain_locked(key, rank, off, size)

    def _ack_drain_locked(self, key, rank, off, size):
        slot = self.slots.get(key)
        if slot is None:
            return
        slot["cov"].setdefault(rank, set()).add((off, off + size))
        required = slot["required"] or {rank: {(off, off + size)}}
        done = all(
            self._covers(slot["cov"].get(r, set()), iv) for r, iv in required.items()
        )
        if done:
            del self.slots[key]
            self.live -= 1
        self.cond.notify_all()

    def begin_fill(self, key: tuple, gen: int, block: bool = True) -> bool:
        """Return True if the caller must write this segment; False to reuse a
        resident, already-filled segment. Blocks while C is saturated unless the
        caller is a downstream-cache pull in the same recursion (block=False)."""
        with self.lock:
            return self._begin_fill_locked(key, gen, block)

    def _begin_fill_locked(self, key, gen, block):
        decision = self._begin_decision(key, gen, block)
        while decision is None:
            self.cond.wait()
            decision = self._begin_decision(key, gen, block)
        return decision

    def _begin_decision(self, key, gen, block):
        """Return a value if the loop should exit, None to keep waiting."""
        if self.error is not None and self.error_gen == gen:
            raise self.error
        slot = self.slots.get(key)
        if slot is not None:
            return self._begin_existing(slot, key, gen, block)
        if block and self.live >= self.cap:
            return None
        self.slots[key] = self._new_slot()
        self.live += 1
        return True

    def _begin_existing(self, slot, key, gen, block):
        if slot["filled"] and slot["error"] is None:
            return False
        if slot["filled"] and slot["error"] is not None:
            return self._begin_filled_error(slot, key, gen, block)
        return None

    def _begin_filled_error(self, slot, key, gen, block):
        if slot["error_gen"] == gen:
            raise slot["error"]
        del self.slots[key]
        if block and self.live >= self.cap:
            return None
        self.slots[key] = self._new_slot()
        self.live += 1
        return True

    def finish_fill(
        self, key: tuple, ok: bool, gen: int, exc: BaseException | None = None
    ) -> None:
        with self.lock:
            self._finish_locked(key, ok, gen, exc)
            self.cond.notify_all()

    def _finish_locked(self, key, ok, gen, exc):
        slot = self.slots.get(key)
        if slot is not None:
            self._apply_finish(slot, ok, gen, exc)

    def _apply_finish(self, slot, ok, gen, exc):
        if ok:
            slot["filled"] = True
            return
        self.error = exc
        self.error_gen = gen
        slot["error"] = exc
        slot["error_gen"] = gen
        slot["filled"] = True
        self.live -= 1

    def wait_filled(self, key: tuple, gen: int) -> None:
        with self.lock:
            decision = self._wait_loop(key, gen)
        if decision is not False:
            raise decision

    def _wait_loop(self, key, gen):
        decision = self._wait_decision(key, gen)
        while decision is None:
            self.cond.wait()
            decision = self._wait_decision(key, gen)
        return decision

    def _wait_decision(self, key, gen):
        """None = keep waiting; an error instance = raise; False = proceed."""
        if self.error is not None and self.error_gen == gen:
            return self.error
        slot = self.slots.get(key)
        if slot is None:
            return False
        if not slot["filled"]:
            return None
        if slot["error"] is not None and slot["error_gen"] == gen:
            return slot["error"]
        return False


class _SlotTable:
    def __init__(self, cap: int, consumers: int) -> None:
        self.nodes: dict[int, _NodeState] = {}
        self.cap = cap
        self.consumers = consumers
        self._lock = threading.Lock()

    def node(self, node: int) -> _NodeState:
        state = self.nodes.get(node)
        if state is None:
            state = self._locked_get(node)
        return state

    def _locked_get(self, node):
        with self._lock:
            return self.nodes.setdefault(node, _NodeState(self.cap, self.consumers))


def _slots(cache_view, consumers: int, live_limit: int) -> _SlotTable:
    with _table_lock:
        table = _view_slots.get(cache_view)
        if table is None:
            table = _SlotTable(live_limit, consumers)
            _view_slots[cache_view] = table
        return table


def _read(src_acc, src) -> bytes:
    chunks = []
    read_size = 0

    def take(buffer_ptr: int, buffer_len: int):
        nonlocal read_size
        if buffer_len < 0 or read_size + buffer_len > src.size:
            raise invalid("buffer_len", "read chunks exceed range size")
        if buffer_len:
            chunks.append(ctypes.string_at(buffer_ptr, buffer_len))
        read_size += buffer_len
        return Ok(None)

    must_ok(src_acc.read_to_buffer(src, take))
    if read_size != src.size:
        raise invalid(
            "buffer_len", f"read chunks have {read_size} bytes, expected {src.size}"
        )
    return b"".join(chunks)


def _write(dst_acc, dst, data: bytes) -> None:
    written = 0

    def give(buffer_ptr: int, buffer_len: int):
        nonlocal written
        if buffer_len < 0 or written + buffer_len > dst.size:
            raise invalid("buffer_len", "write chunks exceed range size")
        if buffer_len:
            ctypes.memmove(buffer_ptr, data[written : written + buffer_len], buffer_len)
        written += buffer_len
        return Ok(None)

    must_ok(dst_acc.write_from_buffer(dst, give))
    if written != dst.size:
        raise invalid(
            "buffer_len", f"write chunks have {written} bytes, expected {dst.size}"
        )


def _copy(plan: CopyPlan, src_view, dst_view) -> None:
    for src, dst in plan:
        if src.size != dst.size:
            raise invalid("copy_plan", f"size mismatch in plan pair {src} -> {dst}")
        _write(dst_view.data_accessor, dst, _read(src_view.data_accessor, src))


def _log_plan(operation, rank, plan):
    _LOG.info(
        "%s rank=%d phase=plan pairs=%d bytes=%d",
        operation,
        rank,
        len(plan),
        sum(src.size for src, _ in plan),
    )


def _session_view(view, session):
    return replace(view, data_accessor=must_ok(session.init_data_accessor()))


def _cleanup(session, method):
    if session is None:
        return
    try:
        getattr(session, method)()
    except Exception as exc:
        _LOG.warning("session %s failed during error cleanup: %s", method, exc)


@catch
def d_tensors_view_rank_save(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[None]:
    if (
        type(compute_view) is not ComputeDTensorsView
        or type(storage_view) is not StorageDTensorsView
    ):
        raise invalid("views", "rank_save requires a compute view and a storage view")
    plan = must_ok(d_tensors_view_rank_plan_save(rank, compute_view, storage_view))
    assert isinstance(plan, list)
    validate_rank_save(rank, compute_view, storage_view, plan)
    _log_plan("save", rank, plan)
    assert storage_view.begin_save is not None
    session = must_ok(storage_view.begin_save(storage_view, rank))
    assert session is not None
    try:
        _copy(plan, compute_view, _session_view(storage_view, session))
    except BaseException:
        _cleanup(session, "abort")
        raise
    must_ok(session.commit())
    _LOG.info("save rank=%d phase=committed", rank)


@catch
def d_tensors_view_rank_load(
    rank: int, compute_view: ComputeDTensorsView, storage_view: StorageDTensorsView
) -> Result[None]:
    if (
        type(compute_view) is not ComputeDTensorsView
        or type(storage_view) is not StorageDTensorsView
    ):
        raise invalid("views", "rank_load requires a compute view and a storage view")
    plan = must_ok(d_tensors_view_rank_plan_load(rank, compute_view, storage_view))
    assert isinstance(plan, list)
    validate_rank_load(rank, compute_view, storage_view, plan)
    _log_plan("load", rank, plan)
    assert storage_view.begin_load is not None
    session = must_ok(storage_view.begin_load(storage_view, rank))
    assert session is not None
    try:
        _copy(plan, _session_view(storage_view, session), compute_view)
    except BaseException:
        _cleanup(session, "finish")
        raise
    must_ok(session.finish())
    _LOG.info("load rank=%d phase=finished", rank)


# ---------------------------------------------------------------- B.1.1 chain load


def _finish_chain_sessions(opened):
    """finish load sessions; keep and return the first error."""
    primary = None
    for session in reversed(opened):
        primary = _finish_one_session(session, primary)
    return primary


def _finish_one_session(session, primary):
    try:
        must_ok(session.finish())
    except BaseException as exc:
        if primary is None:
            return exc
    return primary


def _collect_concrete_chains(read_chain, plan_chain, legs_for):
    """(read chain, plan chain, legs) tuples in data-flow order."""
    from .chain import (
        DirectDTensorsReadChain,
        MixedDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    )

    t = type(read_chain)
    if t in (
        DirectDTensorsReadChain,
        OneLayerCachedDTensorsReadChain,
        TwoLayerCachedDTensorsReadChain,
    ):
        return [(read_chain, plan_chain, legs_for(read_chain))]
    if t is MixedDTensorsReadChain:
        return [
            (seg.read_chain, seg_pc, legs_for(seg.read_chain))
            for seg, seg_pc in zip(
                read_chain.disjoint_tensor_ids_and_read_chain_list,
                plan_chain.disjoint_tensor_ids_and_copy_plan_chain_list,
            )
        ]
    raise invalid("read_chain", f"unknown read chain type {t.__name__}")


def _run_one_concrete(rank, node, chain, pc, legs, consumers) -> None:
    """Open (and finish) one load session for a single concrete chain.

    Per design line 386, each Mixed segment has its own source session; we do
    NOT group segments by storage-view identity and share one session across
    disjoint tensor-id segments."""
    storage = chain.src
    assert type(storage) is StorageDTensorsView
    assert storage.begin_load is not None
    _LOG.info("chain_load rank=%d phase=begin concrete", rank)
    session = must_ok(storage.begin_load(storage, rank))
    assert session is not None
    storage_session_view = _session_view(storage, session)
    try:
        _execute_chain_legs(
            pc, legs, storage, storage_session_view, node, consumers, rank
        )
    except BaseException:
        primary = _finish_chain_sessions([session])
        if primary is not None:
            raise primary from None
        raise
    primary = _finish_chain_sessions([session])
    if primary is not None:
        raise primary
    _LOG.info("chain_load rank=%d phase=finished concrete", rank)


def _cache_live_limit(cache_view) -> int:
    """Bounded in-flight slot count C for a cache view.

    The concurrency width is read via the same mesh helper used by the stats
    path (``cache_concurrency_width``); it unwraps PermutedMesh and validates
    the ConcurrencyWidth tag. A malformed cache mesh is a programming error and
    must surface, not silently degrade to a single buffer (which would make a
    declared double buffer behave as serial fills).
    """
    from .mesh import cache_concurrency_width

    mesh = cache_view.d_tensor_table.named_d_tensors[0].d_tensor.mesh
    return cache_concurrency_width(mesh)


def _split_step(off: int, size: int, step: int):
    """Yield (off, size) chunks covering [off, off+size) aligned to the step grid."""
    if step <= 0:
        yield off, size
        return
    end = off + size
    c = (off // step) * step
    o = off
    while o < end:
        chunk_end = min(c + step, end)
        yield o, chunk_end - o
        c += step
        o = chunk_end


def _rng_slice(rng, offset_delta: int, size: int):
    """Clone a range shifted by offset_delta with a new size (cache vs storage)."""
    if hasattr(rng, "global_tensor_offset"):
        return replace(
            rng, global_tensor_offset=rng.global_tensor_offset + offset_delta, size=size
        )
    return replace(
        rng,
        contiguous_payload_offset=rng.contiguous_payload_offset + offset_delta,
        size=size,
    )


def _slot_cells(view, cache_rng):
    """Yield (off, size) atomic mesh cells covering cache_rng.

    The atomic slot is the real box_of cell for (temporal step s, concurrency c)
    on this node: the two Shard axes (temporal dim0 and concurrency dim0) split
    dim0; spatial is Replicate and does NOT split. Cells are clipped to the
    requested range. Falls back to the whole range when the mesh is unknown."""
    co0 = cache_rng.global_tensor_offset
    csz = cache_rng.size
    try:
        tid = cache_rng.global_tensor_id
        found = _find_slot(view, tid, co0, csz)
        if found is not None:
            return found
    except Exception:
        pass
    return [(co0, csz)]


def _find_slot(view, tid, co0, csz):
    for nd in view.d_tensor_table.named_d_tensors:
        out = _slot_for_nd(nd, tid, co0, csz)
        if out is not None:
            return out
    return None


def _slot_for_nd(nd, tid, co0, csz):
    if nd.global_tensor_id != tid:
        return None
    out = _cells_for_nd(nd, co0, co0 + csz)
    if out:
        return sorted(out)
    return None


def _cells_for_nd(nd, lo_r, hi_r):
    from .dtensor import box_of, naive_of
    from .mesh import cache_concurrency_width, cache_num_nodes, cache_num_steps

    naive = naive_of(nd)
    S = cache_num_steps(naive.mesh)
    C = cache_concurrency_width(naive.mesh)
    N = cache_num_nodes(naive.mesh)
    shape = naive.global_tensor_meta.global_shape
    out = []
    for s in range(S):
        for c in range(C):
            lo, hi = box_of(naive, (S, N, C), (s, 0, c))
            out.extend(_clip_cell(shape, lo, hi, lo_r, hi_r))
    return out


def _clip_cell(shape, lo, hi, lo_r, hi_r):
    from .dtensor import box_to_intervals

    iv = box_to_intervals(shape, lo, hi)
    out = []
    for a, b in iv:
        ca, cb = max(int(a), lo_r), min(int(b), hi_r)
        if cb > ca:
            out.append((ca, cb - ca))
    return out


def _register_fill_runs(reg, view, pairs, eff_src):
    runs = []
    for src_rng, cache_rng in pairs:
        so = src_rng.contiguous_payload_offset
        path = src_rng.file_path
        if runs and runs[-1][1] == so and runs[-1][3] == path:
            runs[-1][1] = so + src_rng.size
            runs[-1][2].append((src_rng, cache_rng))
        else:
            runs.append([so, so + src_rng.size, [(src_rng, cache_rng)], path])
    for src_start, _src_end, sub, _path in runs:
        # Retain each validated physical range. Adjacent payload offsets may
        # belong to distinct slots, so a file accessor cannot read their union.
        job = {"eff_src": eff_src, "ranges": [rng for rng, _ in sub]}
        for src_rng, cache_rng in sub:
            _append_fill_cells(
                reg, view, job, src_start, src_rng.contiguous_payload_offset, cache_rng
            )


def _append_fill_cells(reg, view, job, src_start, src_off, cache_rng):
    for co, cs in _slot_cells(view, cache_rng):
        reg.append(
            {
                "key": (cache_rng.global_tensor_id, co, cs),
                "job": job,
                "buf_off": src_off + (co - cache_rng.global_tensor_offset) - src_start,
                "size": cs,
                "tid": cache_rng.global_tensor_id,
                "off": co,
                "size2": cs,
            }
        )


def _covering_segments(fill_source, view, rng):
    tid = getattr(rng, "global_tensor_id", "")
    off = getattr(rng, "global_tensor_offset", None)
    if off is None:
        off = rng.contiguous_payload_offset
    end = off + rng.size
    out = []
    for seg in fill_source.get(id(view), ()):
        if seg["tid"] == tid and _seg_overlap(seg, off, end):
            out.append(seg)
    return out


def _seg_overlap(seg, off, end):
    s_e = seg["off"] + seg["size"]
    return seg["off"] < end and off < s_e


def _slot_expected(node, consumers, rank, view, tid, off, size):
    # Ranks on this node whose compute dst want overlaps [off,off+size); each
    # rank's required intersection intervals (global coords) must all ACK.
    try:
        from .dtensor import named_of

        nd = named_of(view.d_tensor_table, tid)
        return _required_for_ranks(nd, node, consumers, rank, off, size)
    except Exception:
        return {rank: [(off, off + size)]}


def _required_for_ranks(nd, node, consumers, rank, off, size):
    required = {}
    lo, hi = off, off + size
    for r in range(node * consumers, (node + 1) * consumers):
        ivs = _intersect_owned(nd, r, lo, hi)
        if ivs:
            required[r] = ivs
    return required or {rank: [(lo, hi)]}


def _intersect_owned(nd, r, lo, hi):
    from .dtensor import owned_intervals

    ivs = []
    for a, b in owned_intervals(nd, "compute", r):
        ca, cb = max(int(a), lo), min(int(b), hi)
        if cb > ca:
            ivs.append((ca, cb))
    return ivs


def _seg_job_data(seg):
    job = seg["job"]
    if "buffer" not in job:
        accessor = job["eff_src"].data_accessor
        job["buffer"] = b"".join(_read(accessor, rng) for rng in job["ranges"])
    return job["buffer"][seg["buf_off"] : seg["buf_off"] + seg["size"]]


def _execute_chain_legs(
    pc, legs, storage, storage_session_view, node, consumers, rank
) -> None:
    leg_plans = _chain_plans(pc)
    cache_names = "CacheDTensorsView"
    fill_source: dict[int, list] = {}
    gens: dict[int, int] = {}
    used_states: list = []

    def cap_of(view):
        return _cache_live_limit(view)

    def table_of(view):
        return _slots(view, consumers, cap_of(view))

    def gen_for(state) -> int:
        g = gens.get(id(state))
        if g is None:
            g = state.generation_start(rank)
            gens[id(state)] = g
            used_states.append(state)
        return g

    def covering(view, rng):
        return _covering_segments(fill_source, view, rng)

    def expected(view, tid, off, size):
        return _slot_expected(node, consumers, rank, view, tid, off, size)

    def ensure_segment(view, seg, block=True) -> None:
        state = table_of(view).node(node)
        gen = gen_for(state)
        key = seg["key"]
        if not state.begin_fill(key, gen, block=block):
            return
        try:
            data = _seg_job_data(seg) if "job" in seg else _recurse(view, seg)
            _write(view.data_accessor, _rng_to_view_range(view, key), data)
            state.finish_fill(key, ok=True, gen=gen)
        except BaseException as exc:
            state.finish_fill(key, ok=False, gen=gen, exc=exc)
            raise

    def _recurse(view, seg):
        eff_src, src_rng = seg["eff_src"], seg["src_rng"]
        for up_seg in covering(eff_src, src_rng):
            ensure_segment(eff_src, up_seg, block=False)
        up = table_of(eff_src).node(node)
        up.wait_filled(_rng_key(src_rng), gen_for(up))
        data = _read(eff_src.data_accessor, src_rng)
        for up_seg in covering(eff_src, src_rng):
            off2, sz2 = _overlap_iv(src_rng, up_seg)
            up.ack_drain(up_seg["key"], rank, off2, sz2)
        return data

    def handle_fill(dst, plan, eff_src):
        reg = fill_source.setdefault(id(dst), [])
        pairs = sorted(plan, key=lambda p: p[0].contiguous_payload_offset)
        _register_fill_runs(reg, dst, pairs, eff_src)

    def drain_cell(src, dst, plan, state, gen):
        for cache_rng, dst_rng in plan:
            _drain_plan(
                src,
                dst,
                cache_rng,
                dst_rng,
                state,
                gen,
                covering,
                expected,
                rank,
                ensure_segment,
            )

    try:
        for (src, dst), plan in zip(legs, leg_plans):
            eff_src = storage_session_view if src is storage else src
            run_leg(
                src,
                dst,
                plan,
                eff_src,
                cache_names,
                handle_fill,
                fill_source,
                table_of,
                node,
                gen_for,
                drain_cell,
            )
    finally:
        for state in used_states:
            state.generation_end()


def run_leg(
    src,
    dst,
    plan,
    eff_src,
    cache_names,
    handle_fill,
    fill_source,
    table_of,
    node,
    gen_for,
    drain_cell,
):
    src_cache = type(src).__name__ == cache_names
    dst_cache = type(dst).__name__ == cache_names
    if dst_cache and not src_cache:
        handle_fill(dst, plan, eff_src)
        return
    if src_cache and not dst_cache:
        state = table_of(src).node(node)
        drain_cell(src, dst, plan, state, gen_for(state))
        return
    if src_cache and dst_cache:
        reg = fill_source.setdefault(id(dst), [])
        for cache_rng, out_rng in plan:
            reg.append(_seg(out_rng, src, cache_rng))
        return
    _copy(plan, eff_src, dst)


def _drain_plan(
    src, dst, cache_rng, dst_rng, state, gen, covering, expected, rank, ensure_segment
):
    for co, cs in _slot_cells(src, cache_rng):
        _drain_cell(
            src,
            dst,
            cache_rng,
            dst_rng,
            state,
            gen,
            co,
            cs,
            covering,
            expected,
            rank,
            ensure_segment,
        )


def _drain_cell(
    src,
    dst,
    cache_rng,
    dst_rng,
    state,
    gen,
    co,
    cs,
    covering,
    expected,
    rank,
    ensure_segment,
):
    rel = co - cache_rng.global_tensor_offset
    c_chunk = _rng_slice(cache_rng, rel, cs)
    for seg in covering(src, c_chunk):
        ensure_segment(src, seg)
        state.set_expected(
            seg["key"], expected(dst, seg["tid"], seg["off"], seg["size"])
        )
    state.wait_filled(_rng_key(c_chunk), gen)
    _write(
        dst.data_accessor,
        _rng_slice(dst_rng, rel, cs),
        _read(src.data_accessor, c_chunk),
    )
    for seg in covering(src, c_chunk):
        off2, sz2 = _overlap_iv(c_chunk, seg)
        state.ack_drain(seg["key"], rank, off2, sz2)


@catch
def d_tensors_view_rank_chain_load(rank: int, read_chain) -> Result[None]:
    from .chain import d_tensors_view_rank_plan_chain_load, legs_of

    plan_chain = must_ok(d_tensors_view_rank_plan_chain_load(rank, read_chain))
    chains = _collect_concrete_chains(read_chain, plan_chain, legs_of)
    consumers = getattr(read_chain, "num_ranks_per_node", 1)
    node = rank // consumers
    for chain, pc, legs in chains:
        _run_one_concrete(rank, node, chain, pc, legs, consumers)


def _chain_plans(plan_chain):
    from .chain import (
        DirectCopyPlanChain,
        DisjointTensorIdsAndCopyPlanChain,
        OneLayerCachedCopyPlanChain,
        TwoLayerCachedCopyPlanChain,
    )

    if type(plan_chain) is DisjointTensorIdsAndCopyPlanChain:
        plan_chain = plan_chain.copy_plan_chain
    t = type(plan_chain)
    if t is DirectCopyPlanChain:
        return [plan_chain.src_to_dst]
    if t is OneLayerCachedCopyPlanChain:
        return [plan_chain.src_to_cache, plan_chain.cache_to_dst]
    if t is TwoLayerCachedCopyPlanChain:
        return [
            plan_chain.src_to_small_cache,
            plan_chain.small_cache_to_big_cache,
            plan_chain.big_cache_to_dst,
        ]
    raise invalid("copy_plan_chain", f"unknown plan chain type {t.__name__}")
