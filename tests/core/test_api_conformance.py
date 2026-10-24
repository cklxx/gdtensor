"""Check every public namespace against the checked-in design specification."""

import ast
import dataclasses
import importlib
import inspect
import re
import types
from pathlib import Path

import pytest

import gdtensor
import gdtensor.builder as builder
import gdtensor.tag as tag

ROOT = Path(__file__).resolve().parents[2]
PRODUCT_FIELDS = {
    "ComputeDTensorsView": ("tensor_ids", "d_tensor_table", "data_accessor"),
    "StorageDTensorsView": (
        "tensor_ids",
        "d_tensor_table",
        "data_accessor",
        "physical_layout",
        "begin_save",
        "begin_load",
    ),
    "DTensorsViewMapCore": (
        "convert_each_compute_view",
        "convert_each_storage_view",
        "convert_each_cache_view",
        "convert_each_storage_load_session",
        "convert_each_storage_save_session",
    ),
    "DTensorTable": ("named_d_tensors",),
    "NamedDTensor": ("global_tensor_id", "d_tensor", "alias"),
    "NaiveDTensor": ("global_tensor_meta", "placements", "mesh", "torch_d_tensor"),
    "ReshapedDTensor": ("underlying_d_tensor", "shape"),
    "GlobalTensorMeta": ("global_shape", "dtype", "itemsize"),
    "SlotOuterProductMesh": ("tensor_slots", "content_slots"),
    "LiteralMesh": ("ranks", "max"),
    "ArrangedMesh": ("size",),
    "ReshapedMesh": ("underlying_mesh", "shape"),
    "PermutedMesh": ("underlying_mesh", "permutation"),
    "SlicedMesh": ("underlying_mesh", "slices"),
    "Slice": ("start", "stop", "step"),
    "PhysicalLayoutTable": ("container_format", "tensor_entries", "file_paths"),
    "DisjointStorageMeshList": ("disjoint_storage_mesh_list", "tensor_ids"),
    "TensorComponentPos": ("tensor_id", "rank"),
    "ConstraintViolationError": ("type_name", "constraint_name", "msg"),
    "CheckFailedError": ("check_name", "msg"),
    "NotFoundError": ("key_name", "key", "msg"),
    "InvalidArgumentError": ("argument_name", "msg"),
    "GDTensorIOError": ("file_path", "operation", "msg"),
    "Ok": ("value",),
    "Err": ("value",),
    "Shard": ("axis",),
    "Replicate": (),
    "Partial": (),
}


def _snake(name):
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _declared_exports():
    design = (ROOT / "docs" / "DTensorsView-design.md").read_text()
    section = design[design.index("### C.3 用户主要 API 清单") :]
    paths = re.findall(r"^\|[^\n]*?`(gdtensor(?:\.\w+)+)`", section, re.MULTILINE)
    namespaces = {
        n: set()
        for n in (
            "gdtensor",
            "gdtensor.io",
            "gdtensor.builder",
            "gdtensor.session",
            "gdtensor.access",
            "gdtensor.tag",
        )
    }
    for path in paths:
        module, name = path.rsplit(".", 1)
        namespaces[module].add(name)
    for name in PRODUCT_FIELDS:
        namespaces["gdtensor.builder"].update(
            (name + "BuilderMixin", "new_" + _snake(name))
        )
    # C.2 explicitly defines these two tag construction primitives.
    namespaces["gdtensor.tag"].update(("Box", "make_box_subclass"))
    return namespaces


@pytest.mark.parametrize("namespace", list(_declared_exports()))
def test_exports_are_exactly_the_design(namespace):
    module = importlib.import_module(namespace)
    expected = _declared_exports()[namespace]
    assert len(module.__all__) == len(set(module.__all__))
    assert set(module.__all__) == expected
    visible = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_")
        and name != "annotations"
        and not inspect.ismodule(value)
    }
    assert visible == expected


@pytest.mark.parametrize("name,fields", list(PRODUCT_FIELDS.items()))
def test_products_mixins_and_factories(name, fields):
    design = (ROOT / "docs" / "DTensorsView-design.md").read_text()
    if name == "TensorComponentPos":
        match = re.search(r"^TensorComponentPos = (.*)$", design, re.MULTILINE)
        assert match is not None
        declaration = match.group(1)
        declared_fields = tuple(re.findall(r"\$(\w+)", declaration))
    elif name in ("Replicate", "Partial"):
        # Field-less products: `Name = Object` on one line, no `* $field`.
        assert re.search(rf"^{name} = Object\s*$", design, re.MULTILINE)
        declared_fields = ()
    else:
        match = re.search(
            rf"^{name}(?:\[[^\n]*\])? =\n    Object\n(.*?)(?=^\S)",
            design,
            re.MULTILINE | re.DOTALL,
        )
        assert match is not None
        declaration = match.group(1)
        declared_fields = tuple(re.findall(r"^  \* \$(\w+)", declaration, re.MULTILINE))
        if name == "DisjointStorageMeshList":
            # Approved delta (owner 2026-10-24): tensor_ids row labels added so
            # leader helper can resolve real tensor_id self-containedly.
            declared_fields = declared_fields + ("tensor_ids",)
    assert fields == declared_fields
    cls = getattr(builder, name)
    assert dataclasses.is_dataclass(cls)
    assert tuple(f.name for f in dataclasses.fields(cls)) == fields
    mixin = getattr(builder, name + "BuilderMixin")
    assert mixin in cls.__bases__
    assert (
        "check_constraints" in cls.__dict__
        and "check_constraints" not in mixin.__dict__
    )
    factory = getattr(builder, "new_" + _snake(name))
    assert list(inspect.signature(factory).parameters) == []
    assert type(factory()) is cls
    for field in fields:
        assert hasattr(mixin, "get_" + field) and hasattr(mixin, "set_" + field)
    for method, fn in vars(mixin).items():
        if method.startswith(("get_", "set_", "add_", "mutable_", "clear_")):
            expected = (
                ["self", "value"] if method.startswith(("set_", "add_")) else ["self"]
            )
            assert list(inspect.signature(fn).parameters) == expected
    assert not any(n.startswith(("has_", "which_")) for n in dir(mixin))


@pytest.mark.parametrize(
    "path,parameters",
    [
        ("gdtensor.get_tensor_ids", ["view"]),
        ("gdtensor.get_gd_tensor", ["view", "tensor_id"]),
        ("gdtensor.make_gd_tensor", ["d_tensor"]),
        ("gdtensor.get_d_tensor", ["gd_tensor"]),
        ("gdtensor.reset_d_tensor", ["gd_tensor", "d_tensor"]),
        ("gdtensor.io.rank_save", ["rank", "compute_view", "storage_view"]),
        ("gdtensor.io.rank_load", ["rank", "compute_view", "storage_view"]),
        ("gdtensor.io.chain_load", ["rank", "read_chain"]),
        (
            "gdtensor.io.chain_stat_mem_overhead",
            ["read_chain"],
        ),
        (
            "gdtensor.io.chain_stat_outbound_traffic",
            ["read_chain"],
        ),
        (
            "gdtensor.io.chain_stat_inbound_traffic",
            ["read_chain"],
        ),
        ("gdtensor.get_chain_tensor_ids", ["read_chain"]),
        ("gdtensor.builder.d_tensors_view_map", ["origin", "map_core"]),
        ("gdtensor.builder.global_tensor_meta_of", ["named_d_tensor"]),
        ("gdtensor.builder.tensor_order_value_of", ["d_tensor_table", "tensor_id"]),
        (
            "gdtensor.builder.file_path_of",
            ["layout", "pos"],
        ),
        (
            "gdtensor.builder.leader_tensor_component_pos_of",
            ["disjoint_storage_meshes"],
        ),
    ],
)
def test_function_signatures(path, parameters):
    namespace, name = path.rsplit(".", 1)
    fn = getattr(importlib.import_module(namespace), name)
    assert list(inspect.signature(fn).parameters) == parameters


def test_internal_plan_and_check_declarations_match_design_without_importing_them():
    design = (ROOT / "docs" / "DTensorsView-design.md").read_text()
    section = design[design.index("#### B.1.2 Plan APIs") : design.index("#### B.1.4")]
    declarations = re.findall(
        r"^(\w+)(?:\[[^\n]*\])? =\n    (Result\[\w+\])\n(.*?)(?=^\w|^```)",
        section,
        re.MULTILINE | re.DOTALL,
    )
    expected = {
        name: (re.findall(r"^  <- \$(\w+) ", args, re.MULTILINE), result)
        for name, result, args in declarations
    }
    functions = {}
    for rel in ("gdtensor/internal/plan.py", "gdtensor/internal/chain.py"):
        tree = ast.parse((ROOT / rel).read_text())
        functions.update(
            {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
                and node.name.startswith(("d_tensors_view_rank_plan_", "check_"))
            }
        )
    assert len(expected) == 10 and functions.keys() == expected.keys()
    for name, node in functions.items():
        parameters, result = expected[name]
        assert [arg.arg for arg in node.args.args] == parameters
        assert node.returns is not None
        assert ast.unparse(node.returns) == result.replace("void", "None")


def test_type_aliases_and_namespace_separation():
    for name in (
        "DTensorsView",
        "GenericDTensor",
        "Placement",
        "ComputeMesh",
        "StorageMesh",
        "ContentSlotsMesh",
        "CacheMesh",
        "ConcreteDTensorsReadChain",
    ):
        assert isinstance(getattr(builder, name), types.UnionType)
        assert not hasattr(builder, name + "BuilderMixin")
    assert isinstance(gdtensor.Error, types.UnionType)
    assert not hasattr(builder, "Error")
    # B.2.4.7 / B.3.4: Ok / Err / Shard / Replicate / Partial are concrete
    # products in gdtensor.builder; tag only keeps the six scalar markers.
    for name in ("Ok", "Err", "Shard", "Replicate", "Partial"):
        assert hasattr(builder, name) and not hasattr(gdtensor, name)
        assert not hasattr(tag, name)
    for name in (
        "ConstantOne",
        "TensorCount",
        "WorldSize",
        "NumSteps",
        "NumNodes",
        "ConcurrencyWidth",
    ):
        assert hasattr(tag, name) and not hasattr(builder, name)
    for name in (
        "get_tensor_ids",
        "get_gd_tensor",
        "make_gd_tensor",
        "get_d_tensor",
        "reset_d_tensor",
    ):
        assert hasattr(gdtensor, name) and not hasattr(builder, name)
    for name in ("StorageSaveSession", "StorageLoadSession", "DTensorDataAccessor"):
        assert not hasattr(builder, name)


@pytest.mark.parametrize(
    "file",
    sorted((ROOT / "tests").rglob("*.py"))
    + sorted((ROOT / "gdtensor_sim").rglob("*.py")),
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_tests_and_sim_use_no_core_internal_api(file):
    tree = ast.parse(file.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        else:
            names = [node.module or ""]
        assert all(not name.startswith("gdtensor.internal") for name in names), (
            file.name,
            node.lineno,
        )
        assert all(not re.search(r"(^|\.)test_", name) for name in names), (
            "move shared fixtures to tests.support instead of importing test modules",
            file,
            node.lineno,
        )
    # Dynamic imports must not bypass the namespace checks.
    assert not re.search(
        r"(?:import_module|__import__)\([\"']gdtensor\.internal", file.read_text()
    )


def test_core_contains_no_simulation_or_business_runtime():
    forbidden = {
        "ComputeFabric",
        "ComputeDataAccessor",
        "ModelConfig",
        "ParallelConfig",
        "gd_for_layout",
        "fabric_from_torch",
        "torch_from_fabric",
        "make_storage_view",
        "FileIO",
        "build_layout",
    }
    for file in sorted((ROOT / "gdtensor").rglob("*.py")):
        tree = ast.parse(file.read_text())
        for node in ast.walk(tree):
            _check_core_node(node, file, forbidden)


def _check_core_node(node, file, forbidden):
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        assert node.name not in forbidden, (file, node.name)
    if isinstance(node, ast.ImportFrom):
        assert not (node.module or "").startswith(("gdtensor_sim", "tests")), file


def test_no_reduction_api():
    for namespace in _declared_exports():
        module = importlib.import_module(namespace)
        for name in (
            "rank_reduce",
            "rank_mean",
            "average",
            "lawa",
            "merge_checkpoints",
            "welford",
        ):
            assert not hasattr(module, name)


@pytest.mark.parametrize(
    "file",
    sorted(
        path
        for directory in ("gdtensor", "gdtensor_sim", "tests")
        for path in (ROOT / directory).rglob("*.py")
    ),
    ids=lambda path: str(path.relative_to(ROOT)),
)
def test_code_blocks_use_at_most_three_indent_levels(file):
    source = file.read_text()
    indentation = {
        number: len(line) - len(line.lstrip())
        for number, line in enumerate(source.splitlines(), 1)
    }
    nodes = ast.walk(ast.parse(source))
    violations = [
        (node.lineno, indentation[node.lineno] // 4)
        for node in nodes
        if isinstance(node, ast.stmt) and indentation[node.lineno] > 12
    ]
    assert not violations, (
        f"{file.relative_to(ROOT)}: (line, indent level)={violations}"
    )
