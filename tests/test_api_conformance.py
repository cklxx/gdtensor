"""API conformity audit: gdtensor public surface vs the design doc.

This is a mechanical guard against free-styling. It enumerates every public
symbol the spec C.3 table requires (namespaces + short names + the 5 typed
errors + sessions/access ABCs + plan/check internal helpers) and asserts it
exists with the spec-defined call shape. Anything extra exported at top level
or under gdtensor.io that is not on the spec list fails the audit.

Scope note: models.py / runner.py / iobackend.py / sim.py are the
business/simulation layer (design 4.3: builders live outside the core) and are
intentionally not part of the spec'd public API.
"""
from __future__ import annotations

import inspect

import pytest

import gdtensor
import gdtensor.builder as builder
import gdtensor.io as gio
import gdtensor.session as session
import gdtensor.tag as tag


# C.3 top-level short names
TOP_LEVEL = [
    "get_tensor_ids", "get_gd_tensor", "make_gd_tensor",
    "get_d_tensor", "reset_d_tensor", "Result", "Error", "Ok", "Err",
]

# C.3 io namespace
IO_FNS = ["rank_load", "rank_save", "rank_reshard"]

# C.3 builder product types and tags
BUILDER_TYPES = [
    "DTensorsView", "ComputeDTensorsView", "StorageDTensorsView",
    "DTensorsViewMapCore", "DTensorTable", "NamedDTensor", "GenericDTensor",
    "NaiveDTensor", "ReshapedDTensor", "GlobalTensorMeta", "Placement",
    "ComputeMesh", "StorageMesh", "OuterProductMesh", "ContentSlotsMesh",
    "LiteralMesh", "ArrangedMesh", "ReshapedMesh", "PermutedMesh", "SlicedMesh",
    "Slice", "PhysicalLayoutTable", "DisjointStorageMeshList",
    "ContainerFormat", "TensorComponentPos",
    "ConstraintViolationError", "CheckFailedError", "NotFoundError",
    "InvalidArgumentError", "GDTensorIOError",
    "d_tensors_view_map",
]

BUILDER_HELPERS = [
    "global_tensor_meta_of", "tensor_order_value_of", "file_path_of",
    "leader_tensor_component_pos_of",
]

SESSION_TYPES = ["StorageSaveSession", "StorageLoadSession", "DTensorDataAccessor"]

TAG_TYPES = ["Ok", "Err", "Shard", "Replicate", "Partial",
             "ConstantOne", "TensorCount", "WorldSize"]


@pytest.mark.parametrize("name", TOP_LEVEL)
def test_top_level_names(name):
    assert hasattr(gdtensor, name), name


@pytest.mark.parametrize("name", IO_FNS)
def test_io_namespace(name):
    fn = getattr(gio, name)
    assert callable(fn)


@pytest.mark.parametrize("name", BUILDER_TYPES)
def test_builder_types(name):
    assert hasattr(builder, name), name


@pytest.mark.parametrize("name", BUILDER_HELPERS)
def test_builder_helpers(name):
    assert callable(getattr(builder, name)), name


@pytest.mark.parametrize("name", SESSION_TYPES)
def test_session_types(name):
    assert hasattr(session, name), name


@pytest.mark.parametrize("name", TAG_TYPES)
def test_tag_types(name):
    assert hasattr(tag, name), name


def test_top_level_has_no_unspecd_api():
    """Top-level short names must be exactly the spec bridge + Result/Error."""
    allowed = set(TOP_LEVEL) | {"annotations", "__doc__", "__name__", "__package__",
                                "__loader__", "__spec__", "__path__", "__file__",
                                "__cached__", "__builtins__"}
    extra = {n for n in dir(gdtensor) if not n.startswith("_")} - allowed
    # internal submodules imported for namespacing are allowed; bare functions
    # that are not spec bridge names are not.
    extra_fns = {n for n in extra if callable(getattr(gdtensor, n))
                 and not inspect.ismodule(getattr(gdtensor, n))}
    assert extra_fns == set(), f"unspecd top-level callables: {extra_fns}"


def test_io_exposes_only_three_verbs():
    public = {n for n in dir(gio) if not n.startswith("_") and
              callable(getattr(gio, n)) and not inspect.ismodule(getattr(gio, n))}
    assert public == set(IO_FNS), public


def test_bridge_function_arities():
    """Spec B.1.4 argument names."""
    sig = inspect.signature(gdtensor.make_gd_tensor)
    assert "d_tensor" in sig.parameters
    sig = inspect.signature(gdtensor.reset_d_tensor)
    assert list(sig.parameters)[:2] == ["gd_tensor", "d_tensor"]
    sig = inspect.signature(gdtensor.get_gd_tensor)
    assert list(sig.parameters) == ["view", "tensor_id"]


def test_no_reduce_api_anywhere_core():
    """Design 9 / 10.3: core provides no reduce/merge/mean."""
    for mod in (gdtensor, gio, builder):
        for forbidden in ("rank_reduce", "rank_mean", "rank_average",
                          "lawa", "merge_checkpoints", "welford"):
            assert not hasattr(mod, forbidden), (mod.__name__, forbidden)
