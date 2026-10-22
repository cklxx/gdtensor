"""BuilderMixin generation (design C.1).

``make_builder_mixin(name, fields)`` creates ``<name>BuilderMixin`` with
protobuf-style methods for the given fields:

* ``get_<f>`` / ``set_<f>`` for every field;
* ``add_<singular>`` for list fields (``named_d_tensors`` -> ``add_named_d_tensor``);
* ``mutable_<f>`` for list / dict / nested object fields;
* ``clear_<f>`` for list / dict / nullable fields.

The mixin holds no data; it reads and writes ``self.<field>``. No has_* /
which_* methods. ``check_constraints`` stays on the data class.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple
from dataclasses import fields

# kind: "scalar" | "list" | "dict" | "object" | "oneof" | "nullable"
FieldSpec = Tuple[str, str]


def _singular(name: str) -> str:
    if name == "alias":
        return name
    if name.endswith("ies"):
        return name[:-3] + "y"
    if name.endswith("s"):
        return name[:-1]
    return name


def field_hash(self) -> int:
    """Hash a product data class by its fields (C.2: products are dict keys)."""

    def freeze(v):
        if isinstance(v, list) or isinstance(v, tuple):
            return tuple(freeze(x) for x in v)
        if isinstance(v, dict):
            return frozenset((freeze(k), freeze(x)) for k, x in v.items())
        return v

    return hash(
        (type(self),)
        + tuple(freeze(getattr(self, f.name)) for f in fields(self) if f.compare)
    )


def _field_methods(name: str, kind: str) -> dict:
    def get(self):
        return getattr(self, name)

    def set(self, value) -> None:
        setattr(self, name, value)

    methods = {f"get_{name}": get, f"set_{name}": set}
    if kind in ("list", "dict", "object", "oneof"):

        def mutable(self):
            return getattr(self, name)

        methods[f"mutable_{name}"] = mutable
    if kind == "list":

        def add(self, value) -> None:
            getattr(self, name).append(value)

        def clear(self) -> None:
            setattr(self, name, [])

        methods[f"add_{_singular(name)}"] = add
        methods[f"clear_{name}"] = clear
    elif kind == "dict":

        def clear(self) -> None:
            setattr(self, name, {})

        methods[f"clear_{name}"] = clear
    elif kind in ("oneof", "nullable"):

        def clear(self) -> None:
            setattr(self, name, None)

        methods[f"clear_{name}"] = clear
    return methods


def make_builder_mixin(class_name: str, fields: Iterable[FieldSpec]) -> Any:
    ns: Dict[str, object] = {"__slots__": (), "__module__": "gdtensor.builder"}
    for name, kind in fields:
        for method, fn in _field_methods(name, kind).items():
            fn.__name__ = method
            ns[method] = fn
    return type(f"{class_name}BuilderMixin", (), ns)
