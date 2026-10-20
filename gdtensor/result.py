"""Result[T] = $ok T | $err str  (DTensorsView appendix B.3.3)."""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, Union

T = TypeVar("T")


class ResultError(RuntimeError):
    """Raised by Err.unwrap()."""


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T
    ok = True

    def unwrap(self) -> T:
        return self.value


@dataclass(frozen=True)
class Err:
    error: str
    ok = False

    def unwrap(self) -> Any:
        raise ResultError(self.error)


Result = Union[Ok[T], Err]


def catch(fn: Callable[..., Any]) -> Callable[..., Result]:
    """Wrap a function so that it returns Ok(value) or Err(message)."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Result:
        try:
            out = fn(*args, **kwargs)
        except ResultError as exc:
            return Err(str(exc))
        except Exception as exc:  # noqa: BLE001 - lite core maps every failure to Err
            return Err(f"{type(exc).__name__}: {exc}")
        if isinstance(out, (Ok, Err)):
            return out
        return Ok(out)

    return wrapper
