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
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._dtensor import (
    DTensorTable,
    GlobalTensorMeta,
    NaiveDTensor,
    NamedDTensor,
    Replicate,
    ReshapedDTensor,
    Shard,
    global_tensor_meta_of,
    naive_of,
)
from ._dtensor import nbytes_of
from ._layout import ContainerFormat, PhysicalLayoutTable, build_layout
from ._mesh import ArrangedMesh, OuterProductMesh, ReshapedMesh, Slice, SlicedMesh
from .tag import ConstantOne, TensorCount, WorldSize

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
            out.append(TensorSpec(f"{pre}.mlp.gate.weight", (m.experts, H), "rep", s))
            for e in range(m.experts):
                ep = f"{pre}.mlp.experts.{e}"
                out += [
                    TensorSpec(f"{ep}.gate_proj.weight", (F, H), "expert", s, e),
                    TensorSpec(f"{ep}.up_proj.weight", (F, H), "expert", s, e),
                    TensorSpec(f"{ep}.down_proj.weight", (H, F), "expert", s, e),
                ]
        else:
            out += [
                TensorSpec(f"{pre}.mlp.gate_proj.weight", (F, H), "col", s),
                TensorSpec(f"{pre}.mlp.up_proj.weight", (F, H), "col", s),
                TensorSpec(f"{pre}.mlp.down_proj.weight", (H, F), "row", s),
            ]
    out.append(TensorSpec("model.norm.weight", (H,), "rep", last))
    out.append(TensorSpec("lm_head.weight", (V, H), "col", last))
    out = [TensorSpec(t.name, t.shape, t.kind, t.stage, t.expert, m.dtype, m.itemsize) for t in out]
    if m.optimizer:
        opt = []
        for t in out:
            for st in ("exp_avg", "exp_avg_sq"):
                opt.append(TensorSpec(f"optimizer.state.{t.name}.{st}", t.shape, "zero", t.stage, t.expert, "F32", 4))
        out += opt
    return out


# ------------------------------------------------------------------ meshes


def _world_shape(p: ParallelConfig) -> Tuple[int, int, int]:
    return (p.pp, p.dp, p.tp)


def _slices(stage: int, dp: Optional[int] = None, tp: Optional[int] = None):
    one = lambda i: Slice(i, i + 1)  # noqa: E731
    return (one(stage), None if dp is None else one(dp), None if tp is None else one(tp))


def compute_mesh(p: ParallelConfig, stage: int, dp: Optional[int] = None, tp: Optional[int] = None):
    return SlicedMesh(ReshapedMesh(ArrangedMesh(WorldSize(p.world)), _world_shape(p)), _slices(stage, dp, tp))


def storage_mesh(p_world: int, shape, slices):
    base = OuterProductMesh(ArrangedMesh(ConstantOne(1)), ArrangedMesh(WorldSize(p_world)))
    return SlicedMesh(ReshapedMesh(base, shape), slices)


def _expert_owner(e: int, p: ParallelConfig) -> Tuple[int, int]:
    """EP folded over (DP, TP): expert e lives on one rank of its stage."""
    k = e % (p.dp * p.tp)
    return k // p.tp, k % p.tp


def compute_table(m: ModelConfig, p: ParallelConfig) -> DTensorTable:
    nds = []
    for t in model_tensors(m, p):
        meta = GlobalTensorMeta(t.shape, t.dtype, t.itemsize)
        if t.kind == "col":
            dt = NaiveDTensor(meta, (R, R, Shard(0)), compute_mesh(p, t.stage))
        elif t.kind == "row":
            dt = NaiveDTensor(meta, (R, R, Shard(1)), compute_mesh(p, t.stage))
        elif t.kind == "rep":
            dt = NaiveDTensor(meta, (R, R, R), compute_mesh(p, t.stage))
        elif t.kind == "expert":
            d, k = _expert_owner(t.expert, p)
            dt = NaiveDTensor(meta, (R, R, R), compute_mesh(p, t.stage, d, k))
        elif t.kind == "zero":
            # Flatten, then split over DP x TP (distributed optimizer). The
            # ReshapedDTensor keeps the HF shape as the global meta.
            flat = GlobalTensorMeta((int(np.prod(t.shape)),), t.dtype, t.itemsize)
            mesh = compute_mesh(p, t.stage, *(_expert_owner(t.expert, p) if t.expert >= 0 else (None, None)))
            dt = ReshapedDTensor(NaiveDTensor(flat, (R, Shard(0), Shard(0)), mesh), t.shape)
        else:
            raise ValueError(t.kind)
        nds.append(NamedDTensor(t.name, dt))
    return DTensorTable(nds)


def storage_table(compute: DTensorTable, p: ParallelConfig, mode: str = "hf",
                  aggregate_tp: bool = False) -> DTensorTable:
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
    shape = _world_shape(p)
    nds = []
    for order, nd in enumerate(compute.named_d_tensors):
        naive = naive_of(nd)
        cmesh = naive.mesh
        slices = list(cmesh.slices)
        # Fix each Replicate dim of size > 1 to the designated writer coordinate.
        for d, pl in enumerate(naive.placements):
            size = (cmesh.underlying_mesh.shape[d] if slices[d] is None else 1)
            if type(pl) is Replicate and size > 1:
                k = order % size
                slices[d] = Slice(k, k + 1)
        smesh = storage_mesh(p.world, shape, tuple(slices))
        if mode == "mirror":
            placements = naive.placements
        elif mode == "hf":
            # Any Shard(axis) becomes Shard(0) of the naive shape (flat for
            # ZeRO tensors): slots then concatenate in row-major order.
            placements = tuple(Shard(0) if type(pl) is Shard else R for pl in naive.placements)
            tp_dim = 2
            if (aggregate_tp and isinstance(naive.placements[tp_dim], Shard) and naive.placements[tp_dim].value == 1
                    and slices[tp_dim] is None):
                placements = placements[:tp_dim] + (R,) + placements[tp_dim + 1:]
                k = order % p.tp
                slices[tp_dim] = Slice(k, k + 1)
                smesh = storage_mesh(p.world, shape, tuple(slices))
        else:
            raise ValueError(mode)
        sd = NaiveDTensor(naive.global_tensor_meta, placements, smesh)
        dt = ReshapedDTensor(sd, nd.d_tensor.shape) if isinstance(nd.d_tensor, ReshapedDTensor) else sd
        nds.append(NamedDTensor(nd.global_tensor_id, dt, nd.alias))
    return DTensorTable(nds)


# ------------------------------------------------------------------ layouts


def _stage_of(nd: NamedDTensor, p: ParallelConfig) -> int:
    return int(naive_of(nd).mesh.slices[0].start)


def layout_builder(kind: str, table: DTensorTable, p: ParallelConfig, fmt: ContainerFormat,
                   hf_shards: int = 2) -> PhysicalLayoutTable:
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
        total = sum(sizes)
        target = -(-total // hf_shards)
        shard_of, acc, cur = [], 0, 0
        for sz in sizes:
            if acc >= target and cur < hf_shards - 1:
                cur, acc = cur + 1, 0
            shard_of.append(cur)
            acc += sz
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
    return pattern_bytes(nd.global_tensor_id, global_tensor_meta_of(nd).itemsize, interval_elems(iv))


def full_tensor_bytes(nd: NamedDTensor) -> np.ndarray:
    meta = global_tensor_meta_of(nd)
    return pattern_bytes(nd.global_tensor_id, meta.itemsize, np.arange(meta.numel, dtype=np.int64))
