"""Annotation-only ``Result`` alias for internal modules.

Internal modules cannot import the package ``Result`` alias without an import
cycle. All internal functions use ``from __future__ import annotations``, so
annotations are never evaluated; the real ``Result[T] = Ok[T] | Err[Error]``
alias is defined in the package ``__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar, Union

    from .internal.result import Error
    from .internal.products import Err, Ok

    T = TypeVar("T")
    Result = Union["Ok[T]", "Err[Error]"]
else:  # pragma: no cover - never evaluated at runtime
    Result = object

__all__ = ["Result"]
