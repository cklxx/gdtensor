"""Concrete product classes for Result and Placement (design B.2.4.7 / B.3.4).

These are named product types, not field tags:

* ``Ok(value)`` carries the success payload; ``Err(value)`` carries one Error branch.
* ``Shard(axis)`` carries the tensor dimension.
* ``Replicate`` / ``Partial`` are field-less; the type carries the meaning.

Every product ships a ``<Name>BuilderMixin`` (C.1) and ``check_constraints`` (C.3).
"""

from __future__ import annotations

import dataclasses
import types as _types
from typing import (
    Any as _Any,
    Generic as _Generic,
    Optional as _Optional,
    TypeVar as _TypeVar,
)

from .builder_mixin import field_hash, make_builder_mixin

_T = _TypeVar("_T")


# ------------------------------------------------------------------ Result

OkBuilderMixin = make_builder_mixin("Ok", [("value", "scalar")])
ErrBuilderMixin = make_builder_mixin("Err", [("value", "scalar")])


@dataclasses.dataclass(eq=True)
class Ok(OkBuilderMixin, _Generic[_T]):
    """Success branch of Result[T]; ``value`` is the payload (None for void)."""

    value: _Optional[_T] = None

    __hash__ = field_hash

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def check_constraints(self):
        return Ok(None)

    def __repr__(self):
        return f"Ok({self.value!r})"


@dataclasses.dataclass(eq=True)
class Err(ErrBuilderMixin):
    """Failure branch of Result; ``value`` is one of the Error branches."""

    value: _Any = None

    __hash__ = field_hash

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def check_constraints(self):
        from .result import ERROR_TYPES, ConstraintViolationError

        if self.value is None or type(self.value) not in ERROR_TYPES:
            return Err(
                ConstraintViolationError(
                    type_name="Err",
                    constraint_name="value_non_empty",
                    msg="Err value must be one of the Error branches",
                )
            )
        return self.value.check_constraints()

    def __repr__(self):
        return f"Err({self.value!r})"


# ------------------------------------------------------------------ Placement

ShardBuilderMixin = make_builder_mixin("Shard", [("axis", "scalar")])
ReplicateBuilderMixin = make_builder_mixin("Replicate", [])
PartialBuilderMixin = make_builder_mixin("Partial", [])


@dataclasses.dataclass(eq=True)
class Shard(ShardBuilderMixin):
    """One contiguous logical slice along tensor dimension ``axis``."""

    axis: int = 0

    __hash__ = field_hash

    def __class_getitem__(cls, item):
        return _types.GenericAlias(cls, item)

    def check_constraints(self):
        return Ok(None)

    def __repr__(self):
        return f"Shard({self.axis!r})"


@dataclasses.dataclass(eq=True)
class Replicate(ReplicateBuilderMixin):
    """Every participant holds the full tensor. Field-less type."""

    __hash__ = field_hash

    def check_constraints(self):
        return Ok(None)

    def __repr__(self):
        return "Replicate()"


@dataclasses.dataclass(eq=True)
class Partial(PartialBuilderMixin):
    """Every participant holds a reduction contribution. Field-less type."""

    __hash__ = field_hash

    def check_constraints(self):
        return Ok(None)

    def __repr__(self):
        return "Partial()"
