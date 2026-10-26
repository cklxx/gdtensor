"""F3: LAWA over a multi-shard HuggingFace safetensors directory.

Real HF layout: tensors are split across several ``model-XXXXX-of-YYYYY
.safetensors`` files described by ``model.safetensors.index.json``. Single
file safetensors LAWA is already covered everywhere else, so this test pins
the sharded-directory contract:

* build k real multi-shard HF directories with a valid index via the
  safetensors library (the index is hand-built exactly as HF specifies);
* average tensor-by-tensor, routing each read through the index weight_map;
* write the mean as another sharded HF directory and rebuild its index;
* assert the output is a legal HF directory: index weight_map covers every
  tensor, each referenced shard exists and actually holds that tensor, and
  ``total_size`` equals the summed payload bytes;
* decoded mean matches the dense numpy mean elementwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import numpy as st_numpy

SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
INDEX = "model.safetensors.index.json"
K = 3


def _tensors(version: int):
    base = {
        "a.weight": (np.arange(6 * 4, dtype="<f4").reshape(6, 4) + version * version),
        "b.bias": (np.arange(8, dtype="<f4") * (version + 1)).reshape(8),
        "c.weight": (np.arange(3 * 5, dtype="<f4").reshape(3, 5) - version),
        "d.bias": (np.arange(2, dtype="<f4") + version * 0.5).reshape(2),
    }
    return {k: np.ascontiguousarray(v) for k, v in base.items()}


def _shard_for(tid: str) -> int:
    # First two tensors in shard 0, the rest in shard 1.
    return 0 if tid in ("a.weight", "b.bias") else 1


def save_hf_dir(root: Path, tensors: dict[str, np.ndarray]):
    root.mkdir(parents=True, exist_ok=True)
    groups = [
        {k: v for k, v in tensors.items() if _shard_for(k) == i} for i in range(2)
    ]
    weight_map = {}
    total_size = 0
    for i, group in enumerate(groups):
        st_numpy.save_file(group, str(root / SHARDS[i]))
        for tid, arr in group.items():
            weight_map[tid] = SHARDS[i]
            total_size += int(arr.nbytes)
    (root / INDEX).write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map})
    )
    return total_size


def load_via_index(root: Path, tid: str) -> np.ndarray:
    index = json.loads((root / INDEX).read_text())
    shard = index["weight_map"][tid]
    return st_numpy.load_file(str(root / shard))[tid]


def assert_valid_hf_dir(root: Path, tids: set[str]):
    index = json.loads((root / INDEX).read_text())
    weight_map = index["weight_map"]
    assert set(weight_map) == tids
    referenced = set(weight_map.values())
    assert referenced <= {p.name for p in root.glob("*.safetensors")}
    total = 0
    for shard_name in referenced:
        decoded = st_numpy.load_file(str(root / shard_name))
        for tid, mapped in weight_map.items():
            if mapped == shard_name:
                assert tid in decoded, (shard_name, tid)
                total += int(decoded[tid].nbytes)
    assert index["metadata"]["total_size"] == total


def test_lawa_over_sharded_hf_directory(tmp_path):
    versions = _tensors(0).keys()
    tids = set(versions)
    all_versions = [{tid: _tensors(i)[tid] for tid in tids} for i in range(K)]
    for i, tensors in enumerate(all_versions):
        save_hf_dir(tmp_path / f"in{i}", tensors)

    # Caller-side LAWA: every read is routed through the shard index.
    mean = {}
    for tid in tids:
        stacked = np.stack([load_via_index(tmp_path / f"in{i}", tid) for i in range(K)])
        mean[tid] = stacked.mean(axis=0).astype("<f4")

    out = tmp_path / "mean"
    save_hf_dir(out, mean)

    # Output is a legal, self-consistent HF sharded directory.
    assert_valid_hf_dir(out, tids)

    # Independent dense oracle, read back only through the published index.
    for tid in tids:
        expected = np.stack([v[tid] for v in all_versions]).mean(axis=0)
        np.testing.assert_allclose(load_via_index(out, tid), expected, rtol=1e-6)

    # A single version must differ from the mean: all k shards were consumed.
    for i in range(K):
        assert not np.allclose(
            load_via_index(out, "a.weight"), all_versions[i]["a.weight"]
        )
