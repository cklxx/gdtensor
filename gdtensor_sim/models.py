"""Business layer: toy transformer -> DTensorTables + PhysicalLayoutTables.

Not part of the core. It plays the role of the training framework (compute
table) and of the layout builder (storage table + PhysicalLayoutTable).

World layout follows Megatron order: rank = (pp * DP + dp) * TP + tp
(TP innermost, then DP, PP outermost). Every mesh is a slice of the same
reshaped world ``(PP, DP, TP)``; a SlicedMesh keeps all three dims, so every
NaiveDTensor has three placements.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from gdtensor.builder import (
    DTensorTable,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    ReshapedDTensor,
    global_tensor_meta_of,
    ContainerFormat,
    PhysicalLayoutTable,
    DisjointStorageMeshList,
    leader_tensor_component_pos_of,
    TensorComponentPos,
    ArrangedMesh,
    SlotOuterProductMesh,
    ReshapedMesh,
    Slice,
    SlicedMesh,
)
from gdtensor.tag import ConstantOne, WorldSize
from gdtensor.builder import Replicate, Shard
from ._geometry import naive_of, tensor_ids_of, nbytes_of, numel_of, nonempty_columns
from ._result import invalid

R = Replicate()


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    hidden: int = 256
    ffn: int = 512
    vocab: int = 1024
    experts: int = 0  # 0 = dense
    dtype: str = "BF16"
    itemsize: int = 2
    optimizer: bool = False  # add Adam exp_avg / exp_avg_sq (F32, ZeRO sharded)


@dataclass(frozen=True)
class ParallelConfig:
    tp: int = 2
    dp: int = 1
    pp: int = 1

    @property
    def world(self) -> int:
        return self.tp * self.dp * self.pp

    def stage_of_layer(self, layer: int, layers: int) -> int:
        return min(layer * self.pp // layers, self.pp - 1)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    kind: str  # col | row | rep | expert | zero
    stage: int
    expert: int = -1
    dtype: str = "BF16"
    itemsize: int = 2


def model_tensors(m: ModelConfig, p: ParallelConfig) -> List[TensorSpec]:
    H, F, V, L = m.hidden, m.ffn, m.vocab, m.layers
    out: List[TensorSpec] = []
    last = p.pp - 1
    out.append(TensorSpec("model.embed_tokens.weight", (V, H), "col", 0))
    for i in range(L):
        s = p.stage_of_layer(i, L)
        pre = f"model.layers.{i}"
        out += [
            TensorSpec(f"{pre}.input_layernorm.weight", (H,), "rep", s),
            TensorSpec(f"{pre}.self_attn.q_proj.weight", (H, H), "col", s),
            TensorSpec(f"{pre}.self_attn.k_proj.weight", (H, H), "col", s),
            TensorSpec(f"{pre}.self_attn.v_proj.weight", (H, H), "col", s),
            TensorSpec(f"{pre}.self_attn.o_proj.weight", (H, H), "row", s),
            TensorSpec(f"{pre}.post_attention_layernorm.weight", (H,), "rep", s),
        ]
        if m.experts:
            out.extend(_expert_tensors(m, s, pre))
        else:
            out += [
                TensorSpec(f"{pre}.mlp.gate_proj.weight", (F, H), "col", s),
                TensorSpec(f"{pre}.mlp.up_proj.weight", (F, H), "col", s),
                TensorSpec(f"{pre}.mlp.down_proj.weight", (H, F), "row", s),
            ]
    out.append(TensorSpec("model.norm.weight", (H,), "rep", last))
    out.append(TensorSpec("lm_head.weight", (V, H), "col", last))
    out = [
        TensorSpec(t.name, t.shape, t.kind, t.stage, t.expert, m.dtype, m.itemsize)
        for t in out
    ]
    if m.optimizer:
        out.extend(_optimizer_tensors(out))
    return out


def _expert_tensors(model, stage, prefix):
    hidden, ffn = model.hidden, model.ffn
    tensors = [
        TensorSpec(f"{prefix}.mlp.gate.weight", (model.experts, hidden), "rep", stage)
    ]
    for expert in range(model.experts):
        name = f"{prefix}.mlp.experts.{expert}"
        tensors.extend(
            [
                TensorSpec(
                    f"{name}.gate_proj.weight", (ffn, hidden), "expert", stage, expert
                ),
                TensorSpec(
                    f"{name}.up_proj.weight", (ffn, hidden), "expert", stage, expert
                ),
                TensorSpec(
                    f"{name}.down_proj.weight", (hidden, ffn), "expert", stage, expert
                ),
            ]
        )
    return tensors


def _optimizer_tensors(tensors):
    states = []
    for tensor in tensors:
        for name in ("exp_avg", "exp_avg_sq"):
            states.append(
                TensorSpec(
                    f"optimizer.state.{tensor.name}.{name}",
                    tensor.shape,
                    "zero",
                    tensor.stage,
                    tensor.expert,
                    "F32",
                    4,
                )
            )
    return states


# ------------------------------------------------------------------ meshes


def _world_shape(p: ParallelConfig) -> Tuple[int, int, int]:
    return (p.pp, p.dp, p.tp)


def _slices(stage: int, dp: Optional[int] = None, tp: Optional[int] = None):
    one = lambda i: Slice(i, i + 1)  # noqa: E731
    return (
        one(stage),
        None if dp is None else one(dp),
        None if tp is None else one(tp),
    )


def compute_mesh(
    p: ParallelConfig, stage: int, dp: Optional[int] = None, tp: Optional[int] = None
):
    return SlicedMesh(
        ReshapedMesh(ArrangedMesh(WorldSize(p.world)), list(_world_shape(p))),
        list(_slices(stage, dp, tp)),
    )


def storage_mesh(p_world: int, shape, slices):
    base = SlotOuterProductMesh(
        ArrangedMesh(ConstantOne(1)), ArrangedMesh(WorldSize(p_world))
    )
    return SlicedMesh(ReshapedMesh(base, shape), slices)


def _expert_owner(e: int, p: ParallelConfig) -> Tuple[int, int]:
    """EP folded over (DP, TP): expert e lives on one rank of its stage."""
    k = e % (p.dp * p.tp)
    return k // p.tp, k % p.tp


def compute_table(m: ModelConfig, p: ParallelConfig) -> DTensorTable:
    nds = []
    for t in model_tensors(m, p):
        meta = GlobalTensorMeta(list(t.shape), t.dtype, t.itemsize)
        if t.kind == "col":
            dt = NaiveDTensor(meta, [R, R, Shard(0)], compute_mesh(p, t.stage))
        elif t.kind == "row":
            dt = NaiveDTensor(meta, [R, R, Shard(1)], compute_mesh(p, t.stage))
        elif t.kind == "rep":
            dt = NaiveDTensor(meta, [R, R, R], compute_mesh(p, t.stage))
        elif t.kind == "expert":
            d, k = _expert_owner(t.expert, p)
            dt = NaiveDTensor(meta, [R, R, R], compute_mesh(p, t.stage, d, k))
        elif t.kind == "zero":
            # Flatten, then split over DP x TP (distributed optimizer). The
            # ReshapedDTensor keeps the HF shape as the global meta.
            flat = GlobalTensorMeta([int(np.prod(t.shape))], t.dtype, t.itemsize)
            mesh = compute_mesh(
                p,
                t.stage,
                *(_expert_owner(t.expert, p) if t.expert >= 0 else (None, None)),
            )
            dt = ReshapedDTensor(
                NaiveDTensor(flat, [R, Shard(0), Shard(0)], mesh), list(t.shape)
            )
        else:
            raise ValueError(t.kind)
        nds.append(NamedDTensor(t.name, dt))
    return DTensorTable(nds)


def storage_table(
    compute: DTensorTable,
    p: ParallelConfig,
    mode: str = "hf",
    aggregate_tp: bool = False,
) -> DTensorTable:
    """Storage DTensorTable derived from a compute table.

    * ``hf``    : HF file layout. Columns = the dedup writer group of the
                  tensor; Shard(0) along the flat or row axis, so the slots
                  concatenate to the full tensor in row-major order. A TP
                  dim1 shard then writes strided pieces.
    * ``mirror``: DCP-like layout. Storage boxes = writer compute boxes; the
                  flat ``zero`` tensors keep their flat shard.

    ``aggregate_tp`` (hf only): TP-group gather before write. For row
    (dim1) tensors the TP dim becomes Replicate, so one rank of the TP group
    owns the whole tensor and writes it in one piece; plan_reshard semantics
    (cross-rank read = comm) supply the other TP shards. The design doc says
    the business side does NOT aggregate; this switch shows the difference.
    """
    return DTensorTable(
        [
            _storage_tensor(nd, order, p, mode, aggregate_tp)
            for order, nd in enumerate(compute.named_d_tensors)
        ]
    )


def _storage_tensor(nd, order, p, mode, aggregate_tp):
    shape = _world_shape(p)
    naive = naive_of(nd)
    cmesh = naive.mesh
    slices = list(cmesh.slices)
    # Fix each Replicate dim of size > 1 to the designated writer coordinate.
    for d, pl in enumerate(naive.placements):
        size = cmesh.underlying_mesh.shape[d] if slices[d] is None else 1
        if type(pl) is Replicate and size > 1:
            k = order % size
            slices[d] = Slice(k, k + 1)
    smesh = storage_mesh(p.world, shape, tuple(slices))
    if mode == "mirror":
        placements = naive.placements
    elif mode == "hf":
        # Any Shard(axis) becomes Shard(0) of the naive shape (flat for
        # ZeRO tensors): slots then concatenate in row-major order.
        placements = tuple(
            Shard(0) if type(pl) is Shard else R for pl in naive.placements
        )
        tp_dim = 2
        if (
            aggregate_tp
            and isinstance(naive.placements[tp_dim], Shard)
            and naive.placements[tp_dim].axis == 1
            and slices[tp_dim] is None
        ):
            placements = placements[:tp_dim] + (R,) + placements[tp_dim + 1 :]
            k = order % p.tp
            slices[tp_dim] = Slice(k, k + 1)
            smesh = storage_mesh(p.world, shape, tuple(slices))
    else:
        raise ValueError(mode)
    sd = NaiveDTensor(naive.global_tensor_meta, list(placements), smesh)
    dt = (
        ReshapedDTensor(sd, nd.d_tensor.shape)
        if isinstance(nd.d_tensor, ReshapedDTensor)
        else sd
    )
    return NamedDTensor(nd.global_tensor_id, dt, nd.alias)


# ------------------------------------------------------------------ layouts


def _stage_of(nd: NamedDTensor, p: ParallelConfig) -> int:
    return int(naive_of(nd).mesh.slices[0].start)


def layout_builder(
    kind: str,
    table: DTensorTable,
    p: ParallelConfig,
    fmt: ContainerFormat,
    hf_shards: int = 2,
) -> PhysicalLayoutTable:
    """Business layout builders. ``kind``:

    hf        : tensors grouped into ``hf_shards`` files by size; every file is
                shared by many ranks.
    single    : one file for everything.
    per_rank  : one file per content column (rank).
    rank_multi: up to three files per rank (tensor row mod 3).
    per_stage : one file per PP stage.
    """
    ext = "safetensors" if fmt == "safetensors" else "distcp"
    if kind == "hf":
        sizes = [nbytes_of(global_tensor_meta_of(nd)) for nd in table.named_d_tensors]
        shard_of = _hf_shard_ids(sizes, hf_shards)
        n = max(shard_of) + 1
        fn = lambda t, w, nd: f"model-{shard_of[t] + 1:05d}-of-{n:05d}.{ext}"  # noqa: E731
    elif kind == "single":
        fn = lambda t, w, nd: f"model.{ext}"  # noqa: E731
    elif kind == "per_rank":
        fn = lambda t, w, nd: f"__{w}_0.{ext}"  # noqa: E731
    elif kind == "rank_multi":
        fn = lambda t, w, nd: f"__{w}_{t % 3}.{ext}"  # noqa: E731
    elif kind == "per_stage":
        fn = lambda t, w, nd: f"stage{_stage_of(nd, p)}.{ext}"  # noqa: E731
    else:
        raise ValueError(kind)
    return build_layout(table, p.world, fmt, fn)


def _hf_shard_ids(sizes, count):
    target = -(-sum(sizes) // count)
    shards, accumulated, current = [], 0, 0
    for size in sizes:
        if accumulated >= target and current < count - 1:
            current, accumulated = current + 1, 0
        shards.append(current)
        accumulated += size
    return shards


# ------------------------------------------------------------------ data


def tensor_seed(tensor_id: str) -> int:
    return zlib.crc32(tensor_id.encode()) & 0xFF


def pattern_bytes(tensor_id: str, itemsize: int, elems: np.ndarray) -> np.ndarray:
    """Deterministic bytes for global elements ``elems`` (verifiable anywhere)."""
    e = elems.astype(np.int64)[:, None]
    b = np.arange(itemsize, dtype=np.int64)[None, :]
    return ((e * 131 + b * 29 + tensor_seed(tensor_id)) & 0xFF).astype(np.uint8).ravel()


def interval_elems(iv: np.ndarray) -> np.ndarray:
    if iv.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in iv])


def fill_pattern(nd: NamedDTensor, iv: np.ndarray) -> np.ndarray:
    return pattern_bytes(
        nd.global_tensor_id, global_tensor_meta_of(nd).itemsize, interval_elems(iv)
    )


def full_tensor_bytes(nd: NamedDTensor) -> np.ndarray:
    meta = global_tensor_meta_of(nd)
    return pattern_bytes(
        nd.global_tensor_id, meta.itemsize, np.arange(numel_of(meta), dtype=np.int64)
    )


# ------------------------------------------------------------- generic layout builder (business)


def _cells_to_meshes(
    cells: List[Tuple[int, int]], n_tensors: int, world: int
) -> List[Any]:
    """Cover a cell set with disjoint rectangles: per row, contiguous column
    runs; identical runs on consecutive rows merge into one rectangle."""
    from gdtensor.builder import ArrangedMesh, SlotOuterProductMesh, Slice, SlicedMesh
    from gdtensor.tag import TensorCount, WorldSize

    base = SlotOuterProductMesh(
        ArrangedMesh(TensorCount(n_tensors)), ArrangedMesh(WorldSize(world))
    )
    by_row: Dict[int, List[int]] = {}
    for t, w in cells:
        by_row.setdefault(t, []).append(w)
    runs_by_row: Dict[int, List[Tuple[int, int]]] = {}
    for t, ws in by_row.items():
        runs_by_row[t] = _column_runs(ws)
    rects: List[Tuple[int, int, int, int]] = []
    open_: Dict[Tuple[int, int], int] = {}  # run -> start row
    prev_row = -1
    for t in sorted(runs_by_row):
        cur = set(runs_by_row[t])
        contiguous = t == prev_row + 1
        closing = [run for run in open_ if not contiguous or run not in cur]
        for run in closing:
            rects.append((open_.pop(run), prev_row + 1, run[0], run[1]))
        for run in cur:
            open_.setdefault(run, t)
        prev_row = t
    for run, t0 in open_.items():
        rects.append((t0, prev_row + 1, *run))
    rects.sort()
    return [
        SlicedMesh(base, [Slice(t0, t1), Slice(w0, w1)]) for t0, t1, w0, w1 in rects
    ]


def _column_runs(columns):
    columns = sorted(columns)
    runs, start = [], columns[0]
    for left, right in zip(columns, columns[1:]):
        if right != left + 1:
            runs.append((start, left + 1))
            start = right
    runs.append((start, columns[-1] + 1))
    return runs


def build_layout(
    table: DTensorTable,
    world_size: int,
    container_format: ContainerFormat,
    file_of_cell,
) -> PhysicalLayoutTable:
    """``file_of_cell(t, w, named_d_tensor) -> path`` assigns each non-empty
    cell to one file. Empty cells get no entry."""
    ids = tensor_ids_of(table)
    groups: Dict[str, List[Tuple[int, int]]] = {}
    for t, nd in enumerate(table.named_d_tensors):
        columns = nonempty_columns(nd)
        outside = [rank for rank in columns if not 0 <= rank < world_size]
        if outside:
            raise invalid(
                "world_size",
                f"{nd.global_tensor_id}: column {outside[0]} outside world {world_size}",
            )
        for rank in columns:
            groups.setdefault(file_of_cell(t, rank, nd), []).append((t, rank))
    tensor_entries: Dict[TensorComponentPos, DisjointStorageMeshList] = {}
    file_paths: Dict[TensorComponentPos, str] = {}
    for path in sorted(groups):
        cells = groups[path]
        lst = DisjointStorageMeshList(_cells_to_meshes(cells, len(ids), world_size))
        leader = leader_tensor_component_pos_of(lst)  # canonical leader rank, tid=""
        t0 = min(t for t, _ in cells)  # leader row in table order
        file_paths[TensorComponentPos(ids[t0], leader.rank)] = path
        for t, w in cells:
            tensor_entries[TensorComponentPos(ids[t], w)] = lst
    return PhysicalLayoutTable(container_format, tensor_entries, file_paths)
