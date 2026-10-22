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

from typing import Dict, Iterable, Tuple

# kind: "scalar" | "list" | "dict" | "object" | "nullable"
FieldSpec = Tuple[str, str]


def _singular(name: str) -> str:
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
            # order-insensitive without comparing keys directly
            return tuple((type(k).__name__, getattr(k, "tensor_id", None), getattr(k, "rank", None),
                          freeze(x)) if hasattr(k, "tensor_id") else (k, freeze(x))
                         for k, x in v.items())
        return v
    return hash((type(self),) + tuple(freeze(v) for v in self.__dict__.values()))


def make_builder_mixin(class_name: str, fields: Iterable[FieldSpec]) -> type:
    ns: Dict[str, object] = {"__slots__": (), "__module__": "gdtensor.builder"}
    for fname, kind in fields:
        def getter(self, _f=fname):
            return getattr(self, _f)

        def setter(self, value, _f=fname) -> None:
            setattr(self, _f, value)

        ns[f"get_{fname}"] = getter
        ns[f"set_{fname}"] = setter
        if kind == "list":
            def adder(self, value, _f=fname) -> None:
                getattr(self, _f).append(value)

            def clear_list(self, _f=fname) -> None:
                setattr(self, _f, [])

            ns[f"add_{_singular(fname)}"] = adder
            ns[f"mutable_{fname}"] = getter
            ns[f"clear_{fname}"] = clear_list
        elif kind == "dict":
            def clear_dict(self, _f=fname) -> None:
                setattr(self, _f, {})

            ns[f"mutable_{fname}"] = getter
            ns[f"clear_{fname}"] = clear_dict
        elif kind == "object":
            ns[f"mutable_{fname}"] = getter
        elif kind == "nullable":
            def clear_none(self, _f=fname) -> None:
                setattr(self, _f, None)

            ns[f"clear_{fname}"] = clear_none
    for name, fn in ns.items():
        if callable(fn):
            fn.__name__ = name
    return type(f"{class_name}BuilderMixin", (), ns)
