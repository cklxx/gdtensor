"""gdtensor.tag: box tags, Result, and the typed Error hierarchy.

Spec B.3.3 / C.2:

* Oneof branches with an int/void payload are Box tag classes:
  Ok / Err / Shard / Replicate / Partial / ConstantOne / TensorCount /
  WorldSize.
* Result = Ok[T] | Err[Error].
* Error is a oneof of five named object branches. Every branch carries msg.
* Discrimination uses type(x) is Cls, never isinstance with a parameterized
  generic.

Lite compatibility: Shard(0) == Shard(axis=0); Ok(v)/Err(e); every error is
also a GDTensorError (old name ResultError), so existing raises keep working.
"""
from __future__ import annotations

import types
from typing import Union


class Box:
    """A tag is the class itself; the payload hangs on ``value``."""

    __slots__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        args = item if isinstance(item, tuple) else (item,)
        return types.GenericAlias(cls, args)

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self).__name__, self.value))

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


def make_box_subclass(name, init=None):
    ns = {"__module__": __name__, "__slots__": ()}
    if init is not None:
        ns["__init__"] = init
    return type(name, (Box,), ns)


# ------------------------------------------------------------------ Errors (defined early: Err.unwrap raises them)


class GDTensorError(Exception):
    """Base of the typed Error oneof. Old lite name: ResultError."""

    branch = "Error"

    def __init__(self, msg: str = ""):
        super().__init__(msg)
        self.msg = msg

    def __str__(self):
        return self.msg


class ConstraintViolationError(GDTensorError):
    branch = "ConstraintViolationError"

    def __init__(self, msg: str = "", type_name: str = "", constraint_name: str = ""):
        super().__init__(msg)
        self.type_name = type_name
        self.constraint_name = constraint_name


class CheckFailedError(GDTensorError):
    branch = "CheckFailedError"

    def __init__(self, msg: str = "", check_name: str = ""):
        super().__init__(msg)
        self.check_name = check_name


class NotFoundError(GDTensorError):
    branch = "NotFoundError"

    def __init__(self, msg: str = "", key_name: str = "", key: str = ""):
        super().__init__(msg)
        self.key_name = key_name
        self.key = key


class InvalidArgumentError(GDTensorError):
    branch = "InvalidArgumentError"

    def __init__(self, msg: str = "", argument_name: str = ""):
        super().__init__(msg)
        self.argument_name = argument_name


class GDTensorIOError(GDTensorError):
    branch = "GDTensorIOError"

    def __init__(self, msg: str = "", file_path=None, operation: str = ""):
        super().__init__(msg)
        self.file_path = file_path
        self.operation = operation


Error = Union[
    ConstraintViolationError, CheckFailedError, NotFoundError,
    InvalidArgumentError, GDTensorIOError,
]

ResultError = GDTensorError  # back-compat alias


# ------------------------------------------------------------------ generic tags


class Ok(Box):
    ok = True

    def unwrap(self):
        return self.value


class Err(Box):
    ok = False

    @property
    def error(self) -> str:
        return str(self.value)

    def unwrap(self):
        if isinstance(self.value, BaseException):
            raise self.value
        raise GDTensorError(str(self.value))


def _void_init(self, value=None):
    Box.__init__(self, None)


Replicate = make_box_subclass("Replicate", _void_init)
Partial = make_box_subclass("Partial", _void_init)


def _shard_init(self, axis: int = 0, value=None):
    if value is not None:
        axis = value
    if not isinstance(axis, int) or isinstance(axis, bool) or axis < 0:
        raise TypeError("Shard axis must be a non-negative int")
    Box.__init__(self, int(axis))


Shard = make_box_subclass("Shard", _shard_init)
Shard.axis = property(lambda self: self.value)


def _tagged_int(name):
    def init(self, value=0):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} payload must be int")
        Box.__init__(self, int(value))
    return make_box_subclass(name, init)


ConstantOne = _tagged_int("ConstantOne")
TensorCount = _tagged_int("TensorCount")
WorldSize = _tagged_int("WorldSize")

Result = Union[Ok, Err]
