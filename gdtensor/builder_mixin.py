"""Builder methods and new_* factories (spec 5.7 / C.1).

Every product data class gets protobuf-style build methods:

    get_<field>() / set_<field>(v)
    add_<field>(v)       list fields
    mutable_<field>()    nested object / dict / list handle
    clear_<field>()

plus a module-level ``new_<lower_name>()`` factory registered in
``gdtensor.builder``.

No has_*/which_*: null checks use get_x() is None; oneof branches use type().

Methods are installed directly on the dataclass (not via a changed MRO), which
keeps frozen dataclasses working: scalar set_* uses object.__setattr__, while
list/dict containers are mutated in place.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Tuple

_FACTORIES: Dict[str, type] = {}


def _kind(f: dataclasses.Field) -> str:
    t = str(f.type)
    if "Dict" in t or "dict" in t:
        return "dict"
    if "List" in t or "list" in t or "Tuple" in t or "tuple" in t:
        return "list"
    return "scalar"


def attach_builders(cls: type) -> type:
    """Install get/set/add/mutable/clear methods and register new_* factory."""
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        fname, kind = f.name, _kind(f)

        def make_getter(name=fname):
            def getter(self):
                return getattr(self, name)
            getter.__name__ = f"get_{name}"
            return getter

        def make_setter(name=fname):
            def setter(self, value) -> None:
                object.__setattr__(self, name, value)
            setter.__name__ = f"set_{name}"
            return setter

        setattr(cls, f"get_{fname}", make_getter())
        setattr(cls, f"set_{fname}", make_setter())

        if kind == "list":
            def make_adder(name=fname):
                def adder(self, value) -> None:
                    getattr(self, name).append(value)
                adder.__name__ = f"add_{name}"
                return adder

            def make_mutable(name=fname):
                def mutable(self):
                    return getattr(self, name)
                mutable.__name__ = f"mutable_{name}"
                return mutable

            def make_clear(name=fname):
                def clearer(self) -> None:
                    object.__setattr__(self, name, [])
                clearer.__name__ = f"clear_{name}"
                return clearer

            setattr(cls, f"add_{fname}", make_adder())
            setattr(cls, f"mutable_{fname}", make_mutable())
            setattr(cls, f"clear_{fname}", make_clear())
        elif kind == "dict":
            def make_mutable(name=fname):
                def mutable(self):
                    return getattr(self, name)
                mutable.__name__ = f"mutable_{name}"
                return mutable

            def make_clear(name=fname):
                def clearer(self) -> None:
                    object.__setattr__(self, name, {})
                clearer.__name__ = f"clear_{name}"
                return clearer

            setattr(cls, f"mutable_{fname}", make_mutable())
            setattr(cls, f"clear_{fname}", make_clear())

    _FACTORIES[cls.__name__] = cls
    return cls


def new_(type_name: str, *args, **kwargs):
    """Factory: new_<type> -> call the constructor with no/empty args."""
    cls = _FACTORIES.get(type_name)
    if cls is None:
        raise KeyError(f"no builder-registered type {type_name!r}")
    return cls(*args, **kwargs)
