"""Concrete product classes for Result and Placement (design B.2.4.7 / B.3.4).

These are named product types, not field tags:

* ``Shard(axis)`` carries the tensor dimension.
* ``Replicate`` / ``Partial`` are field-less; the type carries the meaning.
* ``Ok(value)`` / ``Err(error)`` are the two ``Result`` branches.
"""

from __future__ import annotations

import dataclasses
import types as _types
from typing import Any as _Any, Generic as _Generic, TypeVar as _TypeVar

_T = _TypeVar("_T")


@dataclasses.dataclass(eq=True)
class Ok(_Generic[_T]):
    """Success branch of Result[T]; ``value`` is the payload (None for void)."""

    value: _T = None
    __hash__ = object.__hash__

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def __repr__(self):
        return f"Ok({self.value!r})"


@dataclasses.dataclass(eq=True)
class Err:
    """Failure branch of Result; ``value`` is one of the Error branches."""

    value: _Any = None
    __hash__ = object.__hash__

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def __repr__(self):
        return f"Err({self.value!r})"


@dataclasses.dataclass(eq=True)
class Shard:
    """One contiguous logical slice along tensor dimension ``axis``."""

    axis: int
    __hash__ = object.__hash__

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def __repr__(self):
        return f"Shard({self.axis!r})"


class Replicate:
    """Every participant holds the full tensor. Field-less type."""

    __slots__ = ()
    __match_args__ = ()

    def __init__(self) -> None:
        pass

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return "Replicate()"


class Partial:
    """Every participant holds a reduction contribution. Field-less type."""

    __slots__ = ()
    __match_args__ = ()

    def __init__(self) -> None:
        pass

    def __eq__(self, other):
        return type(self) is type(other)

    def __hash__(self):
        return hash(type(self))

    def __repr__(self):
        return "Partial()"
