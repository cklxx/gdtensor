"""Setup results observed through the public API, as comparable digests.

``collect()`` runs every case below and returns one small record per case:
the plans that ``rank_save`` / ``rank_load`` execute, the file each layout
entry resolves to, and the leader cell of each storage mesh list. Setup
optimizations must leave every record unchanged; ``setup_plan_golden.json``
holds the records produced by the implementation before they were made.

Regenerate the golden file only for an intended behavior change:
``python -m tests.support.setup_plans > tests/support/setup_plan_golden.json``
"""

import hashlib
import json
import sys

import numpy as np

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.session as gs
import gdtensor.tag as gt
from gdtensor_conn.models import build_layout
from tests.support import axes

LAYOUTS = ("single", "per_rank", "rank_multi", "per_stage")


class Trace(ga.DTensorDataAccessor):
    """Record every range of a plan; serve zero bytes of the right size."""

    def __init__(self):
        self.ranges = []

    def _call(self, rng, callback):
        if hasattr(rng, "global_tensor_offset"):
            address = ("compute", rng.global_tensor_id, rng.global_tensor_offset)
        else:
            address = ("storage", rng.file_path, rng.contiguous_payload_offset)
        self.ranges.append([*address, rng.size])
        data = np.zeros(rng.size, dtype=np.uint8)
        return callback(data.ctypes.data if rng.size else 0, rng.size)

    def read_to_buffer(self, global_range, read_to_buffer):
        return self._call(global_range, read_to_buffer)

    def write_from_buffer(self, global_range, write_from_buffer):
        return self._call(global_range, write_from_buffer)


class _Save(gs.StorageSaveSession):
    def __init__(self, trace):
        self.trace = trace

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.trace)

    def commit(self):
        return gb.Ok(None)

    def abort(self):
        return gb.Ok(None)


class _Load(gs.StorageLoadSession):
    def __init__(self, trace):
        self.trace = trace

    def get_file_offset(self, pos):
        return gb.Ok(0)

    def init_data_accessor(self):
        return gb.Ok(self.trace)

    def finish(self):
        return gb.Ok(None)


def file_of_cell(kind, stages):
    names = {
        "single": lambda t, w, nd: "model.bin",
        "per_rank": lambda t, w, nd: f"rank{w}.bin",
        "rank_multi": lambda t, w, nd: f"rank{w}_{t % 3}.bin",
        "per_stage": lambda t, w, nd: f"stage{stages[t]}.bin",
    }
    return names[kind]


def compute_view(table, trace):
    ids = [nd.global_tensor_id for nd in table.named_d_tensors]
    return gb.ComputeDTensorsView(ids, table, trace)


def storage_view(table, layout, trace):
    ids = [nd.global_tensor_id for nd in table.named_d_tensors]
    return gb.StorageDTensorsView(
        ids,
        table,
        trace,
        layout,
        lambda view, rank: gb.Ok(_Save(trace)),
        lambda view, rank: gb.Ok(_Load(trace)),
    )


def source_storage(topology, kind):
    table, _, _, stages = axes.schema(topology, storage=True)
    world = int(np.prod(topology))
    layout = build_layout(table, world, "safetensors", file_of_cell(kind, stages))
    return table, layout


def _outcome(result, trace):
    if type(result) is gb.Err:
        return ["Err", type(result.value).__name__]
    return ["Ok", trace.ranges]


def save_plan(rank, topology, table, layout):
    trace = Trace()
    compute, _, _, _ = axes.schema(topology)
    view = storage_view(table, layout, trace)
    return _outcome(gio.rank_save(rank, compute_view(compute, trace), view), trace)


def load_plan(rank, topology, table, layout):
    trace = Trace()
    compute, _, _, _ = axes.schema(topology)
    view = storage_view(table, layout, trace)
    return _outcome(gio.rank_load(rank, compute_view(compute, trace), view), trace)


def _digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def _files(layout):
    out = []
    for pos in layout.tensor_entries:
        path = gb.file_path_of(layout, pos)
        out.append([pos.tensor_id, pos.rank, type(path).__name__, str(path.value)])
    return sorted(out)


def _pairs(plans):
    return sum(len(plan[1]) // 2 for plan in plans if plan[0] == "Ok")


def layout_case(label, source, target, kind):
    """One layout: save on ``source``, load on ``source`` and on ``target``."""
    table, layout = source_storage(source, kind)
    ranks = {
        name: range(int(np.prod(grid))) for name, grid in (("s", source), ("t", target))
    }
    saves = [save_plan(rank, source, table, layout) for rank in ranks["s"]]
    loads = [load_plan(rank, source, table, layout) for rank in ranks["s"]]
    reshard = [load_plan(rank, target, table, layout) for rank in ranks["t"]]
    return {
        "check": type(layout.check_constraints()).__name__,
        "files": _digest(_files(layout)),
        "save": [_pairs(saves), _digest(saves)],
        "load": [_pairs(loads), _digest(loads)],
        "reshard": [_pairs(reshard), _digest(reshard)],
    }


def layout_cases():
    out = {}
    for label, source, target in axes.CASES:
        for kind in LAYOUTS:
            out[f"layout/{label}/{kind}"] = layout_case(label, source, target, kind)
    return out


# (rows, world size, content ranks or None for all ranks)
MESH_BASES = [
    (1, 1, None),
    (1, 4, None),
    (3, 4, None),
    (5, 6, None),
    (3, 4, [2, 0, 3]),
    (4, 6, [5, 1]),
    (2, 3, [1]),
    (3, 5, [4, 4, 0]),
    (2, 3, [7]),
    (0, 3, None),
]

# Slices of the (rows, world) grid, as (start, stop, step) or None per dim.
MESH_SLICES = [
    None,
    [None, None],
    [(0, 1, None), None],
    [None, (1, 3, None)],
    [(1, None, None), (0, None, 2)],
    [(0, None, 2), (1, 2, None)],
    [(-1, None, None), (-2, None, None)],
    [(2, 2, None), None],
]


def _base_mesh(rows, world, ranks):
    content = gb.ArrangedMesh(gt.WorldSize(world))
    if ranks is not None:
        content = gb.LiteralMesh(list(ranks), gt.WorldSize(world))
    return gb.SlotOuterProductMesh(gb.ArrangedMesh(gt.TensorCount(rows)), content)


def _sliced(base, slices):
    if slices is None:
        return base
    parts = [None if part is None else gb.Slice(*part) for part in slices]
    return gb.SlicedMesh(base, parts)


def _leader(group):
    try:
        leader = gb.leader_tensor_component_pos_of(group)
    except Exception as exc:  # noqa: BLE001
        return ["raise", type(exc).__name__]
    return [leader.tensor_id, leader.rank]


def mesh_case(base, slices):
    rows = base[0]
    group = gb.DisjointStorageMeshList(
        [_sliced(_base_mesh(*base), slices)], [f"t{row}" for row in range(rows)]
    )
    try:
        check = group.check_constraints()
    except Exception as exc:  # noqa: BLE001
        return {"check": ["raise", type(exc).__name__], "leader": _leader(group)}
    name = type(check.value).__name__ if type(check) is gb.Err else "Ok"
    return {"check": name, "leader": _leader(group)}


def mesh_cases():
    out = {}
    for b, base in enumerate(MESH_BASES):
        for s, slices in enumerate(MESH_SLICES):
            out[f"mesh/base{b}/slice{s}"] = mesh_case(base, slices)
    return out


def collect():
    return {**mesh_cases(), **layout_cases()}


if __name__ == "__main__":
    json.dump(collect(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
