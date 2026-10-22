"""Result wrapper and @catch decorator (spec B.3.3).

The Ok / Err tags and the typed Error hierarchy live in gdtensor.tag. This
module keeps the lite call style (``@catch``, ``ResultError``) as thin
compatibility wrappers over them.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Union

from .tag import Err, GDTensorError, Ok  # re-exported

T = type("T", (), {})()  # placeholder not used; real signatures use Any below

ResultError = GDTensorError
Result = Union[Ok, Err]


def catch(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a function so it returns Ok(value) or Err(typed error)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            out = fn(*args, **kwargs)
        except GDTensorError as exc:
            return Err(exc)
        except Exception as exc:  # noqa: BLE001 - unknown failure maps to IO-ish Err
            from .tag import GDTensorIOError
            return Err(GDTensorIOError(f"{type(exc).__name__}: {exc}"))
        if isinstance(out, (Ok, Err)):
            return out
        return Ok(out)

    return wrapper
