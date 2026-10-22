"""Builder Mixin API tests (design 5.7 / C.1).

Every product class Foo inherits a generated FooBuilderMixin with
get_/set_/add_/mutable_/clear_ methods.
"""
from __future__ import annotations

from gdtensor.internal.dtensor import DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor
from gdtensor.internal.layout import PhysicalLayoutTable
from gdtensor.internal.mesh import ArrangedMesh
from gdtensor.internal.ranges import RangeFromComputeView
from gdtensor.tag import Replicate, WorldSize


def test_global_tensor_meta_builder():
    m = GlobalTensorMeta([2, 3], "F32", 4)
    assert m.get_itemsize() == 4
    m.set_itemsize(8)
    assert m.get_itemsize() == 8
    assert m.get_dtype() == "F32"


def test_named_d_tensor_list_add_and_mutable():
    mesh = ArrangedMesh(WorldSize(1))
    a = NamedDTensor("a", NaiveDTensor(GlobalTensorMeta([1], "U8", 1), [Replicate()], mesh))
    t = DTensorTable()
    t.add_named_d_tensor(a)
    assert t.mutable_named_d_tensors()[0] is a
    assert t.get_named_d_tensors()[0].global_tensor_id == "a"
    t.clear_named_d_tensors()
    assert t.get_named_d_tensors() == []


def test_set_scalar_replaces_field():
    r = RangeFromComputeView("t", 0, 8)
    r.set_size(16)
    assert r.get_size() == 16


def test_dict_mutable_handles():
    lay = PhysicalLayoutTable("safetensors", {}, {})
    assert lay.mutable_tensor_entries() == {}
    lay.set_container_format("distcp")
    assert lay.get_container_format() == "distcp"


def test_nullable_clear():
    n = NaiveDTensor(GlobalTensorMeta([1], "U8", 1), [Replicate()], ArrangedMesh(WorldSize(1)),
                     torch_d_tensor=object())
    n.clear_torch_d_tensor()
    assert n.get_torch_d_tensor() is None


def test_mixin_is_separate_class_with_suffix():
    # C.1: build interface lives on FooBuilderMixin; the data class inherits it.
    assert "DTensorTableBuilderMixin" in {c.__name__ for c in DTensorTable.__mro__}
    assert "GlobalTensorMetaBuilderMixin" in {c.__name__ for c in GlobalTensorMeta.__mro__}
