"""Builder Mixin / new_* API tests (spec 5.7 / C.1)."""
from __future__ import annotations

import pytest

from gdtensor.builder_mixin import new_
from gdtensor.dtensor import (
    DTensorTable, GlobalTensorMeta, NaiveDTensor, NamedDTensor,
)
from gdtensor.layout import DisjointStorageMeshList, PhysicalLayoutTable
from gdtensor.mesh import ArrangedMesh, Slice
from gdtensor.ranges import RangeFromComputeView
from gdtensor.tag import Replicate


def test_global_tensor_meta_builder():
    m = GlobalTensorMeta((2, 3), "F32", 4)
    assert m.get_itemsize() == 4
    m.set_itemsize(8)
    assert m.get_itemsize() == 8
    assert m.get_dtype() == "F32"


def test_named_d_tensor_list_add_and_mutable():
    mesh = ArrangedMesh(1)
    a = NamedDTensor("a", NaiveDTensor(GlobalTensorMeta((1,), "U8", 1), (Replicate(),), mesh))
    t = DTensorTable()
    t.add_named_d_tensors(a)
    assert t.mutable_named_d_tensors()[0] is a
    assert t.get_named_d_tensors()[0].global_tensor_id == "a"
    t.clear_named_d_tensors()
    assert t.get_named_d_tensors() == []


def test_new_factory():
    m = new_("GlobalTensorMeta", (4,), "U8", 1)
    assert isinstance(m, GlobalTensorMeta) and m.itemsize == 1
    r = new_("RangeFromComputeView", "t", 0, 8)
    assert isinstance(r, RangeFromComputeView) and r.size == 8


def test_set_scalar_replaces_frozen_field():
    r = RangeFromComputeView("t", 0, 8)
    r.set_size(16)
    assert r.get_size() == 16


def test_unknown_factory_raises():
    with pytest.raises(KeyError):
        new_("DoesNotExist")


def test_dict_mutable_handles():
    lay = PhysicalLayoutTable("safetensors", {}, {})
    assert lay.mutable_tensor_entries() == {}
    lay.set_container_format("distcp")
    assert lay.get_container_format() == "distcp"
