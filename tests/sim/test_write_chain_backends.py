"""Real backend integration tests for cached write chains (`gio.chain_save`).

Covers:
- Multi-file safetensors (`single`, `per_rank`, `per_stage`, `rank_multi`) over
  both `FileIO` and `SdkFileIO`, including delayed leader staging via
  `SaveSessionBase.ensure_staging`;
- Real PyTorch DCP (`distcp`) across both `thread` and `process` staging modes
  with 1-layer and 2-layer write chains over multi-tensor mixed-dtype schemas
  (including `F64`, `I64`, `F32`, `F16`, `BF16`, `ReshapedDTensor`, and empty tensors);
- `chain_save -> chain_load` roundtrips verified against independent
  `safetensors` and `torch.distributed.checkpoint.load` decodes + CRC-32;
- Real-file `FileIO.tracer` `pwrite` syscall coalescing comparison (`rank_save` vs
  `chain_save`).
"""

from __future__ import annotations

import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import safetensors.numpy

import gdtensor.access as ga
import gdtensor.builder as gb
import gdtensor.io as gio
import gdtensor.tag as gt
from gdtensor_conn.coord import ThreadCoordinator
from gdtensor_conn.file_cache import DiskNodeCache
from gdtensor_conn.formats import COMMIT_MARKER_NAME, make_storage_view
from gdtensor_conn.iobackend import FileIO, SdkFileIO
from gdtensor_conn.models import (
    ParallelConfig,
    layout_builder,
    storage_table,
)
from tests.support.sdk_client import MemorySdkClient as MemoryClient
from tests.support.cluster_sim import Cluster, parent_of
from tests.support.oracle import ArrayFabric
from tests.support.schemas import (
    bf16_schema,
    run_ranks,
    tensor_schema,
    verify_crc,
)


class _DummyDevice:
    def charge(self, _node: int, _nbytes: int) -> None:
        pass


def _cache_view_for(ct: gb.DTensorTable, acc, steps: int, nodes: list[int], width: int):
    mesh = gb.StCacheOuterProductMesh(
        temporal=gb.ArrangedMesh(gt.NumSteps(steps)),
        spatial=gb.LiteralMesh(list(nodes), gt.NumNodes(max(nodes) + 1)),
        concurrency=gb.ArrangedMesh(gt.ConcurrencyWidth(width)),
    )
    named = [
        gb.NamedDTensor(
            nd.global_tensor_id,
            gb.NaiveDTensor(
                gb.global_tensor_meta_of(nd),
                [gb.Shard(0), gb.Replicate(), gb.Shard(0)],
                mesh,
            ),
        )
        for nd in ct.named_d_tensors
    ]
    ids = [nd.global_tensor_id for nd in ct.named_d_tensors]
    return gb.CacheDTensorsView(ids, gb.DTensorTable(named), acc)


def _tensor_nbytes(ct: gb.DTensorTable) -> dict[str, int]:
    out = {}
    for nd in ct.named_d_tensors:
        m = gb.global_tensor_meta_of(nd)
        out[nd.global_tensor_id] = int(np.prod(m.global_shape)) * m.itemsize
    return out


def _sdk_decode_safetensors(client: MemoryClient, root: str, ct: gb.DTensorTable):
    files = sorted(
        k
        for k in client.objects
        if k.startswith(root + "/") and k.endswith(".safetensors")
    )
    assert files
    out: dict[str, bytes] = {}
    for key in files:
        decoded = safetensors.numpy.load(bytes(client.objects[key]))
        out.update({k: v.tobytes() for k, v in decoded.items()})
    return out


def _model_values(ct: gb.DTensorTable) -> dict[str, np.ndarray]:
    values = {}
    for idx, nd in enumerate(ct.named_d_tensors):
        m = gb.global_tensor_meta_of(nd)
        shape = tuple(m.global_shape)
        dtype = {
            "I64": np.int64,
            "F64": np.float64,
            "F32": np.float32,
            "U16": np.uint16,
            "BF16": np.uint16,
            "F16": np.float16,
        }[m.dtype]
        n = int(np.prod(shape))
        values[nd.global_tensor_id] = (
            np.arange(n, dtype=np.int64).reshape(shape) + idx * 17
        ).astype(dtype)
    return values


def _build_backend_views(
    root: str,
    ct: gb.DTensorTable,
    p: tuple[int, int, int],
    kind: str,
    mode: str,
    fmt: str,
    io,
    *,
    coord=None,
    dcp_mode: str = "thread",
):
    par = ParallelConfig(pp=p[0], dp=p[1], tp=p[2])
    st = storage_table(ct, par, mode)
    layout = layout_builder(kind, st, par, fmt, hf_shards=2)
    ids = [nd.global_tensor_id for nd in st.named_d_tensors]
    sv = make_storage_view(ids, st, layout, root, io=io, coord=coord, dcp_mode=dcp_mode)
    return st, layout, sv


@pytest.mark.parametrize("kind", ["single", "per_rank", "per_stage", "rank_multi"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("backend", ["file", "sdk"])
def test_multifile_safetensors_write_and_read_chain_roundtrip(
    tmp_path: Path, monkeypatch, kind: str, layers: int, backend: str
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    p = (2, 1, 2)
    k = 2
    world = int(np.prod(p))
    ct, points, values = tensor_schema(p)
    sizes = _tensor_nbytes(ct)
    cluster = Cluster(num_nodes=2, k=k)
    io = SdkFileIO(MemoryClient()) if backend == "sdk" else FileIO()
    root = "sdk://ckpt" if backend == "sdk" else str(tmp_path / "ckpt")
    mode = "hf" if kind in ("single", "per_stage") else "mirror"
    st, layout, sv = _build_backend_views(
        root, ct, p, kind, mode, "safetensors", io, coord=ThreadCoordinator(world)
    )
    source = ArrayFabric(ct, points, values)
    l0_dir = tmp_path / "wcache0"
    l1_dir = tmp_path / "wcache1"
    w0 = DiskNodeCache(cluster, sizes, lambda r: r // k, str(l0_dir), _DummyDevice())
    w1 = DiskNodeCache(
        cluster, sizes, parent_of([0, 1], [0]).get, str(l1_dir), _DummyDevice()
    )
    try:
        cv = gb.ComputeDTensorsView(
            [nd.global_tensor_id for nd in ct.named_d_tensors],
            ct,
            _ClusterRankFabric(cluster, source),
        )
        chain = _make_write_chain(ct, cv, w0, w1, sv, layers=layers, k=k)
        assert type(chain.check_constraints()) is gb.Ok
        _run_with_cluster(cluster, world, lambda r: gio.chain_save(r, chain))
        if isinstance(io, SdkFileIO):
            assert isinstance(io.client, MemoryClient)
            assert f"{root}/{COMMIT_MARKER_NAME}" in io.client.objects
            decoded = _sdk_decode_safetensors(io.client, root, ct)
            assert decoded and not [
                key for key in io.client.objects if key.endswith(".tmp")
            ]
        else:
            assert (Path(root) / COMMIT_MARKER_NAME).exists()
            verify_crc(Path(root), "safetensors", ct, p, mode, values)
        _verify_chain_load_roundtrip(
            tmp_path, root, st, layout, ct, points, values, p, k, io, layers=layers
        )
    finally:
        w0.close()
        w1.close()


@pytest.mark.parametrize("dcp_mode", ["thread", "process"])
@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.parametrize("schema_kind", ["mixed", "bf16"])
def test_real_dcp_write_and_read_chain_roundtrip(
    tmp_path: Path, monkeypatch, dcp_mode: str, layers: int, schema_kind: str
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    p = (1, 2, 2)
    k = 2
    world = int(np.prod(p))
    ct, points, values = bf16_schema(p) if schema_kind == "bf16" else tensor_schema(p)
    sizes = _tensor_nbytes(ct)
    cluster = Cluster(num_nodes=2, k=k)
    io = FileIO()
    root = str(tmp_path / "dcp-ckpt")
    st, layout, sv = _build_backend_views(
        root,
        ct,
        p,
        "per_rank",
        "hf",
        "distcp",
        io,
        coord=ThreadCoordinator(world),
        dcp_mode=dcp_mode,
    )
    source = ArrayFabric(ct, points, values)
    w0 = DiskNodeCache(
        cluster, sizes, lambda r: r // k, str(tmp_path / "d0"), _DummyDevice()
    )
    w1 = DiskNodeCache(
        cluster,
        sizes,
        parent_of([0, 1], [0]).get,
        str(tmp_path / "d1"),
        _DummyDevice(),
    )
    try:
        cv = gb.ComputeDTensorsView(
            [nd.global_tensor_id for nd in ct.named_d_tensors],
            ct,
            _ClusterRankFabric(cluster, source),
        )
        chain = _make_write_chain(ct, cv, w0, w1, sv, layers=layers, k=k)
        assert type(chain.check_constraints()) is gb.Ok
        _run_with_cluster(cluster, world, lambda r: gio.chain_save(r, chain))
        verify_crc(Path(root), "distcp", ct, p, "hf", values)
        _verify_chain_load_roundtrip(
            tmp_path,
            root,
            st,
            layout,
            ct,
            points,
            values,
            p,
            k,
            FileIO(),
            layers=layers,
        )
    finally:
        w0.close()
        w1.close()


def test_2d_col_sharded_chain_save_coalesces_pwrite_syscalls(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    p = (1, 1, 4)
    k = 2
    world = int(np.prod(p))
    rows, cols = 16, 64
    mesh = gb.SlicedMesh(
        gb.ReshapedMesh(gb.ArrangedMesh(gt.WorldSize(world)), list(p)),
        [gb.Slice(0, 1), None, None],
    )
    ct = gb.DTensorTable(
        [
            gb.NamedDTensor(
                f"w{i}",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([rows, cols], "F32", 4),
                    [gb.Replicate(), gb.Replicate(), gb.Shard(1)],
                    mesh,
                ),
            )
            for i in range(4)
        ]
    )
    points = _col_shard_points(ct, p)
    values = _model_values(ct)
    source = ArrayFabric(ct, points, values)

    direct_io = FileIO()
    direct_root = str(tmp_path / "direct")
    _, _, direct_sv = _build_backend_views(
        direct_root,
        ct,
        p,
        "single",
        "hf",
        "safetensors",
        direct_io,
        coord=ThreadCoordinator(world),
    )
    run_ranks(world, lambda r: gio.rank_save(r, source.view(r), direct_sv))
    direct_pwrites = [rec for rec in direct_io.tracer.records if rec.op == "pwrite"]

    cached_io = FileIO()
    cached_root = str(tmp_path / "cached")
    _, _, cached_sv = _build_backend_views(
        cached_root,
        ct,
        p,
        "single",
        "hf",
        "safetensors",
        cached_io,
        coord=ThreadCoordinator(world),
    )
    cluster = Cluster(num_nodes=2, k=k)
    cache = DiskNodeCache(
        cluster,
        _tensor_nbytes(ct),
        lambda r: r // k,
        str(tmp_path / "c"),
        _DummyDevice(),
    )
    try:
        cv = gb.ComputeDTensorsView(
            [nd.global_tensor_id for nd in ct.named_d_tensors],
            ct,
            _ClusterRankFabric(cluster, source),
        )
        chain = gb.OneLayerCachedDTensorsWriteChain(
            src=cv,
            cache=_cache_view_for(ct, cache, steps=1, nodes=[0, 1], width=1),
            dst=cached_sv,
            num_ranks_per_node=k,
        )
        _run_with_cluster(cluster, world, lambda r: gio.chain_save(r, chain))
    finally:
        cache.close()

    cached_pwrites = [rec for rec in cached_io.tracer.records if rec.op == "pwrite"]
    # direct writes 4 tensors * 16 rows * 4 ranks = 256 small strided pieces (+1 header).
    # 1-layer cache (2 nodes) halves pwrite syscalls to 129 (1 header + 128 coalesced node rows).
    assert (
        len(cached_pwrites) <= 130
        and len(cached_pwrites) * 2 <= len(direct_pwrites) + 2
    ), (
        len(cached_pwrites),
        len(direct_pwrites),
    )

    two_io = FileIO()
    two_root = str(tmp_path / "cached2")
    _, _, two_sv = _build_backend_views(
        two_root,
        ct,
        p,
        "single",
        "hf",
        "safetensors",
        two_io,
        coord=ThreadCoordinator(world),
    )
    cluster2 = Cluster(num_nodes=2, k=k)
    big_c = DiskNodeCache(
        cluster2,
        _tensor_nbytes(ct),
        lambda r: r // k,
        str(tmp_path / "c2b"),
        _DummyDevice(),
    )
    small_c = DiskNodeCache(
        cluster2,
        _tensor_nbytes(ct),
        lambda _r: 0,
        str(tmp_path / "c2s"),
        _DummyDevice(),
    )
    try:
        cv2 = gb.ComputeDTensorsView(
            [nd.global_tensor_id for nd in ct.named_d_tensors],
            ct,
            _ClusterRankFabric(cluster2, source),
        )
        chain2 = gb.TwoLayerCachedDTensorsWriteChain(
            src=cv2,
            big_cache=_cache_view_for(ct, big_c, steps=1, nodes=[0, 1], width=1),
            small_cache=_cache_view_for(ct, small_c, steps=1, nodes=[0], width=1),
            dst=two_sv,
            num_ranks_per_node=k,
        )
        _run_with_cluster(cluster2, world, lambda r: gio.chain_save(r, chain2))
    finally:
        big_c.close()
        small_c.close()

    two_pwrites = [rec for rec in two_io.tracer.records if rec.op == "pwrite"]
    # 2-layer convergence on node 0 coalesces all 256 strided 64B writes into
    # 16 slot-aligned 1024B writes (4 tensors x 4 storage slots) plus 1 header.
    assert len(two_pwrites) <= 17 and len(two_pwrites) * 15 <= len(direct_pwrites), (
        len(two_pwrites),
        len(direct_pwrites),
    )
    direct_payload = [
        rec.size for rec in direct_pwrites if rec.size != two_pwrites[0].size
    ]
    two_payload = [rec.size for rec in two_pwrites if rec.size == 1024]
    assert len(direct_payload) == 256 and all(s == 64 for s in direct_payload)
    assert len(two_payload) == 16 and all(s == 1024 for s in two_payload)
    assert _decode_st_all(Path(cached_root)) == _decode_st_all(Path(direct_root))
    assert _decode_st_all(Path(two_root)) == _decode_st_all(Path(direct_root))
    for tid, arr in values.items():
        got = safetensors.numpy.load_file(Path(cached_root) / "model.safetensors")[tid]
        assert zlib.crc32(got.tobytes()) == zlib.crc32(arr.tobytes())
        np.testing.assert_array_equal(got, arr)


def test_sdk_delayed_leader_cross_rank_staging_succeeds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GDTENSOR_CACHE_COORDINATOR", "thread")
    _run_delayed_leader_sdk_check(tmp_path)


def _run_delayed_leader_sdk_check(tmp_path: Path) -> None:
    world, k = 2, 2
    mesh_c = gb.ArrangedMesh(gt.WorldSize(world))
    # "a" is Replicate across ranks (0, 1) -> designated writer is ONLY rank 0.
    # Storage puts "a" in a file led by rank 1 (LiteralMesh([1])), so rank 0's
    # my_files() does NOT include remote.safetensors. Delayed rank 1 opens its
    # session 50ms later, so rank 0 flushes BEFORE leader rank 1 runs begin_save!
    ct = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "a",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([8], "U8", 1), [gb.Replicate()], mesh_c
                ),
            )
        ]
    )
    sm = gb.SlotOuterProductMesh(
        gb.ArrangedMesh(gt.ConstantOne(1)), gb.LiteralMesh([1], gt.WorldSize(world))
    )
    st = gb.DTensorTable(
        [
            gb.NamedDTensor(
                "a",
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([8], "U8", 1),
                    [gb.Replicate(), gb.Replicate()],
                    sm,
                ),
            )
        ]
    )
    grp = gb.DisjointStorageMeshList(
        [
            gb.SlotOuterProductMesh(
                gb.ArrangedMesh(gt.TensorCount(1)),
                gb.LiteralMesh([1], gt.WorldSize(world)),
            )
        ],
        ["a"],
    )
    leader = gb.TensorComponentPos("a", 1)
    layout = gb.PhysicalLayoutTable(
        "safetensors", {leader: grp}, {leader: "remote.safetensors"}
    )
    client = MemoryClient()
    io = SdkFileIO(client)
    sv = make_storage_view(
        ["a"], st, layout, "sdk://delayed", io=io, coord=ThreadCoordinator(world)
    )
    orig_begin = sv.begin_save
    assert orig_begin is not None

    def slow_leader_begin(view, rank: int):
        if rank == 1:
            time.sleep(0.05)
        return orig_begin(view, rank)

    sv_delayed = gb.StorageDTensorsView(
        sv.tensor_ids,
        sv.d_tensor_table,
        sv.data_accessor,
        sv.physical_layout,
        slow_leader_begin,
        sv.begin_load,
    )
    values = {"a": np.arange(8, dtype=np.uint8)}
    points = {0: {"a": np.arange(8)}, 1: {"a": np.arange(8)}}
    source = ArrayFabric(ct, points, values)
    cluster = Cluster(num_nodes=1, k=k)
    cache = DiskNodeCache(
        cluster, {"a": 8}, lambda _r: 0, str(tmp_path / "dc"), _DummyDevice()
    )
    try:
        cv = gb.ComputeDTensorsView(["a"], ct, _ClusterRankFabric(cluster, source))
        chain = gb.OneLayerCachedDTensorsWriteChain(
            src=cv,
            cache=_cache_view_for(ct, cache, steps=1, nodes=[0], width=1),
            dst=sv_delayed,
            num_ranks_per_node=k,
        )
        _run_with_cluster(cluster, world, lambda r: gio.chain_save(r, chain))
    finally:
        cache.close()

    decoded = safetensors.numpy.load(
        bytes(client.objects["sdk://delayed/remote.safetensors"])
    )
    np.testing.assert_array_equal(decoded["a"], values["a"])


class _ClusterRankFabric(ga.DTensorDataAccessor):
    def __init__(self, cluster: Cluster, fabric: ArrayFabric) -> None:
        self.cluster = cluster
        self.fabric = fabric

    def read_to_buffer(self, global_range, read_to_buffer):
        return self.fabric.view(self.cluster.rank).data_accessor.read_to_buffer(
            global_range, read_to_buffer
        )

    def write_from_buffer(self, global_range, write_from_buffer):
        return self.fabric.view(self.cluster.rank).data_accessor.write_from_buffer(
            global_range, write_from_buffer
        )


def _make_write_chain(ct, cv, w0, w1, sv, *, layers: int, k: int):
    if layers == 1:
        return gb.OneLayerCachedDTensorsWriteChain(
            src=cv,
            cache=_cache_view_for(ct, w0, steps=2, nodes=[0, 1], width=1),
            dst=sv,
            num_ranks_per_node=k,
        )
    return gb.TwoLayerCachedDTensorsWriteChain(
        src=cv,
        big_cache=_cache_view_for(ct, w0, steps=2, nodes=[0, 1], width=1),
        small_cache=_cache_view_for(ct, w1, steps=2, nodes=[0], width=1),
        dst=sv,
        num_ranks_per_node=k,
    )


def _verify_chain_load_roundtrip(
    tmp_path: Path,
    root: str,
    st: gb.DTensorTable,
    layout: gb.PhysicalLayoutTable,
    ct: gb.DTensorTable,
    points: dict,
    values: dict,
    p: tuple[int, int, int],
    k: int,
    io,
    *,
    layers: int,
) -> None:
    world = int(np.prod(p))
    sizes = _tensor_nbytes(ct)
    cluster = Cluster(num_nodes=2, k=k)
    loaded = ArrayFabric(ct, points)
    ids = [nd.global_tensor_id for nd in st.named_d_tensors]
    reader = make_storage_view(ids, st, layout, root, io=io)
    dst = gb.ComputeDTensorsView(ids, ct, _ClusterRankFabric(cluster, loaded))
    r0 = DiskNodeCache(
        cluster, sizes, lambda r: r // k, str(tmp_path / "rc0"), _DummyDevice()
    )
    r1 = DiskNodeCache(
        cluster,
        sizes,
        parent_of([0, 1], [0]).get,
        str(tmp_path / "rc1"),
        _DummyDevice(),
    )
    try:
        if layers == 1:
            rchain = gb.OneLayerCachedDTensorsReadChain(
                src=reader,
                cache=_cache_view_for(ct, r0, steps=2, nodes=[0, 1], width=1),
                dst=dst,
                num_ranks_per_node=k,
            )
        else:
            rchain = gb.TwoLayerCachedDTensorsReadChain(
                src=reader,
                small_cache=_cache_view_for(ct, r1, steps=2, nodes=[0], width=1),
                big_cache=_cache_view_for(ct, r0, steps=2, nodes=[0, 1], width=1),
                dst=dst,
                num_ranks_per_node=k,
            )
        _run_with_cluster(cluster, world, lambda r: gio.chain_load(r, rchain))
    finally:
        r0.close()
        r1.close()
    loaded.assert_values(values)


def _run_with_cluster(cluster: Cluster, world: int, fn) -> None:
    results = {}

    def one(rank: int):
        cluster.rank = rank
        results[rank] = fn(rank)

    threads = [
        threading.Thread(target=one, args=(r,), daemon=True) for r in range(world)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive()
    bad = {r: v.value for r, v in results.items() if type(v) is not gb.Ok}
    assert len(results) == world and not bad, bad


def _col_shard_points(ct: gb.DTensorTable, p: tuple[int, int, int]) -> dict:
    from tests.support.oracle import slice_points

    world = int(np.prod(p))
    pts: dict = {r: {} for r in range(world)}
    for nd in ct.named_d_tensors:
        m = gb.global_tensor_meta_of(nd)
        shape = tuple(m.global_shape)
        for r in range(world):
            pts[r][nd.global_tensor_id] = slice_points(
                shape, p, [gb.Replicate(), gb.Replicate(), gb.Shard(1)], 0, r
            )
    return pts


def _decode_st_all(root: Path) -> dict[str, bytes]:
    out = {}
    for path in sorted(root.glob("*.safetensors")):
        out.update(
            {k: v.tobytes() for k, v in safetensors.numpy.load_file(path).items()}
        )
    return out
