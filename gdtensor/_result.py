"""Error branches and Result (design B.3.3 / C.2).

Each Error branch is a product data class with a ``msg`` field. Results are
``tag.Ok(value)`` or ``tag.Err(error)``.

Internal plumbing (not public API): implementation code raises ``Fail(err)``
and ``@catch`` turns it into ``tag.Err(err)``. ``must_ok`` returns the Ok
payload or raises ``Fail``. Callers of the public API only see Results.
"""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .tag import Err, Ok


@dataclass(eq=True, frozen=False)
class ConstraintViolationError:
    type_name: str = ""
    constraint_name: str = ""
    msg: str = ""


@dataclass(eq=True, frozen=False)
class CheckFailedError:
    check_name: str = ""
    msg: str = ""


@dataclass(eq=True, frozen=False)
class NotFoundError:
    key_name: str = ""
    key: str = ""
    msg: str = ""


@dataclass(eq=True, frozen=False)
class InvalidArgumentError:
    argument_name: str = ""
    msg: str = ""


@dataclass(eq=True, frozen=False)
class GDTensorIOError:
    file_path: Optional[str] = None
    operation: str = ""
    msg: str = ""


ERROR_TYPES = (ConstraintViolationError, CheckFailedError, NotFoundError, InvalidArgumentError, GDTensorIOError)

for _cls in ERROR_TYPES:
    _cls.__hash__ = lambda self: hash((type(self),) + dataclasses.astuple(self))


class Fail(Exception):
    """Internal carrier of one Error branch value through Python control flow."""

    def __init__(self, error) -> None:
        super().__init__(error.msg)
        self.error = error


# ------------------------------------------------------------------ raise helpers


def not_found(key_name: str, key, msg: str) -> Fail:
    return Fail(NotFoundError(key_name=key_name, key=str(key), msg=msg))


def invalid(argument_name: str, msg: str) -> Fail:
    return Fail(InvalidArgumentError(argument_name=argument_name, msg=msg))


def check_failed(check_name: str, msg: str) -> Fail:
    return Fail(CheckFailedError(check_name=check_name, msg=msg))


def io_error(operation: str, msg: str, file_path: Optional[str] = None) -> Fail:
    return Fail(GDTensorIOError(file_path=file_path, operation=operation, msg=msg))


def violation(type_name: str, constraint_name: str, msg: str) -> Fail:
    return Fail(ConstraintViolationError(type_name=type_name, constraint_name=constraint_name, msg=msg))


# ------------------------------------------------------------------ Result plumbing


def is_ok(result) -> bool:
    return type(result) is Ok


def must_ok(result):
    """Ok payload, or raise Fail with the Err payload (internal)."""
    if type(result) is Ok:
        return result.value
    if type(result) is Err:
        raise Fail(result.value)
    raise Fail(InvalidArgumentError(argument_name="result", msg=f"not a Result: {result!r}"))


def catch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return tag.Ok(value) or tag.Err(error). Unknown exceptions are I/O failures."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            out = fn(*args, **kwargs)
        except Fail as exc:
            return Err(exc.error)
        except OSError as exc:
            return Err(GDTensorIOError(file_path=getattr(exc, "filename", None), operation="io",
                                       msg=f"{type(exc).__name__}: {exc}"))
        except Exception as exc:  # noqa: BLE001
            return Err(GDTensorIOError(file_path=None, operation="internal", msg=f"{type(exc).__name__}: {exc}"))
        if type(out) in (Ok, Err):
            return out
        return Ok(out)

    return wrapper
