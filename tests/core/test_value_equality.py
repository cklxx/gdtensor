"""Value equality and hashing must be consistent across independent builds.

Products that land in dicts/sets (design section 6 invariants): equal objects
must compare equal and hash the same, regardless of construction path or the
order fields are set.
"""

import os
import subprocess
import sys

import gdtensor.builder as gb


def test_tensor_component_pos_value_equality_and_hash():
    a = gb.TensorComponentPos("w", 3)
    b = gb.TensorComponentPos()
    b.tensor_id = "w"
    b.rank = 3
    assert a == b
    assert hash(a) == hash(b)
    assert {a, b} == {a}


def test_shard_axis_value_equality_and_hash():
    a = gb.Shard(1)
    b = gb.Shard(axis=1)
    assert a == b
    assert hash(a) == hash(b)
    assert hash(gb.Shard(0)) != hash(gb.Shard(1))


def test_global_tensor_meta_value_equality_and_hash():
    a = gb.GlobalTensorMeta([2, 3], "F32", 4)
    b = gb.GlobalTensorMeta()
    b.global_shape = [2, 3]
    b.dtype = "F32"
    b.itemsize = 4
    assert a == b
    assert hash(a) == hash(b)


def test_replicate_partial_fieldless_equality_and_hash():
    assert gb.Replicate() == gb.Replicate()
    assert hash(gb.Replicate()) == hash(gb.Replicate())
    assert gb.Partial() == gb.Partial()
    assert hash(gb.Partial()) == hash(gb.Partial())
    assert gb.Replicate() != gb.Partial()


_PROBE = """
import gdtensor.builder as gb
import gdtensor.tag as gt
mesh = gb.SlotOuterProductMesh(
    gb.ArrangedMesh(gt.ConstantOne(1)), gb.ArrangedMesh(gt.WorldSize(1))
)
a = gb.NaiveDTensor(gb.GlobalTensorMeta([8], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh)
b = gb.NaiveDTensor(gb.GlobalTensorMeta([8], "U8", 1), [gb.Replicate(), gb.Shard(0)], mesh)
pos = gb.TensorComponentPos("w", 3)
# Both equality and dict routing must hold inside this seed.
assert a == b and hash(a) == hash(b)
d = {pos: 1}
assert gb.TensorComponentPos("w", 3) in d
print("OK")
"""


def test_product_equality_and_dict_routing_across_hash_seeds():
    for seed in (0, 1, 42, 12345):
        out = subprocess.run(
            [sys.executable, "-c", _PROBE],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "OK"
