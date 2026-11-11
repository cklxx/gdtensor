"""Public range accessor interface (design B.2.3.2 and C.2)."""

from __future__ import annotations

import abc as _abc
import hashlib as _hashlib
import pickle as _pickle
import types as _types
from typing import Callable as _Callable, Generic as _Generic, TypeVar as _TypeVar

from . import Result as _Result
from .tag import Box as _Box

_RangeT = _TypeVar("_RangeT")
_BufferCallback = _Callable[[int, int], _Result[None]]

__all__ = ["DTensorDataAccessor"]


def _type_name(value):
    cls = type(value)
    return cls.__module__, cls.__qualname__


def _unordered(values, memo):
    return tuple(sorted((_canonical(v, memo) for v in values), key=repr))


def _reduction(value, memo):
    reduced = value.__reduce_ex__(5)
    if isinstance(reduced, str):
        _pickle.dumps(value, protocol=5)
        return _canonical(reduced, memo)
    if not isinstance(reduced, tuple) or not 2 <= len(reduced) <= 6:
        raise TypeError("cache identity reduction must return a string or tuple")
    if not callable(reduced[0]) or not isinstance(reduced[1], tuple):
        raise TypeError("cache identity reduction has invalid callable or arguments")
    reduced = list(reduced)
    for index in (3, 4):
        if len(reduced) > index and reduced[index] is not None:
            reduced[index] = tuple(reduced[index])
    if (
        isinstance(value, dict)
        and type(value) is not dict
        and len(reduced) > 4
        and reduced[4] is not None
    ):
        reduced[4] = _unordered(reduced[4], memo)
    args = reduced[1]
    if (
        isinstance(value, (set, frozenset))
        and type(value) not in (set, frozenset)
        and len(args) == 1
        and isinstance(args[0], (list, tuple, set, frozenset))
    ):
        reduced[1] = (_unordered(args[0], memo),)
    return _canonical(tuple(reduced), memo)


def _primitive_subclass(value, base, memo):
    state = _reduction(value, memo)
    primitive = _hashlib.sha256(_pickle.dumps(base(value), protocol=5)).digest()
    return _type_name(value), primitive, state


def _canonical(value, memo=None):
    if memo is None:
        memo = {}
    key = id(value)
    cached = memo.get(key)
    if cached is not None:
        if cached[1] is None:
            raise RecursionError
        return cached[1]
    # Keep each value alive so Python cannot reuse its id during this digest.
    memo[key] = value, None
    if value is None or type(value) in (bool, int, float, str, bytes):
        primitive = _hashlib.sha256(_pickle.dumps(value, protocol=5)).digest()
        result = _type_name(value), primitive
    elif isinstance(value, int):
        result = _primitive_subclass(value, int, memo)
    elif isinstance(value, float):
        result = _primitive_subclass(value, float, memo)
    elif isinstance(value, str):
        result = _primitive_subclass(value, str, memo)
    elif isinstance(value, bytes):
        result = _primitive_subclass(value, bytes, memo)
    elif type(value) is _Box:
        result = _type_name(value), _canonical(value.value, memo)
    elif isinstance(value, (type, _types.FunctionType, _types.BuiltinFunctionType)):
        result = (
            _type_name(value),
            value.__module__,
            value.__qualname__,
            _pickle.dumps(value, protocol=5),
        )
    elif isinstance(value, dict) and type(value) is not dict:
        result = _type_name(value), _reduction(value, memo)
    elif isinstance(value, dict):
        result = _type_name(value), _unordered(value.items(), memo)
    elif isinstance(value, (list, tuple)) and type(value) not in (list, tuple):
        result = _type_name(value), _reduction(value, memo)
    elif isinstance(value, (list, tuple)):
        result = _type_name(value), tuple(_canonical(item, memo) for item in value)
    elif isinstance(value, (set, frozenset)) and type(value) not in (set, frozenset):
        result = _type_name(value), _reduction(value, memo)
    elif isinstance(value, (set, frozenset)):
        result = _type_name(value), _unordered(value, memo)
    else:
        result = _type_name(value), _reduction(value, memo)
    # Fixed-size child digests keep shared DAGs linear without encoding aliases.
    result = _hashlib.sha256(repr(result).encode("utf-8")).digest()
    memo[key] = value, result
    return result


def _identity_payload(value):
    try:
        return _canonical(value)
    except RecursionError:
        return repr((_type_name(value), _pickle.dumps(value, protocol=5))).encode()


def _content_digest(value) -> str:
    """Hash content; ordinary acyclic aliases do not affect the key.

    Cyclic and recursion-limited values use pickle and can retain aliases.
    """
    try:
        payload = _identity_payload(value)
    except Exception as exc:
        raise ValueError(f"cache identity is not serializable: {exc}") from exc
    return _hashlib.sha256(payload).hexdigest()


class DTensorDataAccessor(_abc.ABC, _Generic[_RangeT]):
    @_abc.abstractmethod
    def read_to_buffer(
        self, global_range: _RangeT, read_to_buffer: _BufferCallback
    ) -> _Result[None]: ...

    def has_bytes(self, global_range: _RangeT) -> bool:
        """Whether every byte of the range is present, without reading it.

        Caches that track validity answer from metadata; the default answer
        is no, so reuse stays off for accessors that cannot speak. Cache ranges
        carry an internal source_identity. An implementation that returns True
        must bind both published bytes and queries to that identity. Unknown
        identities do not permit reuse across calls. Reading published bytes
        must reject detected corruption before invoking the consumer callback.
        """
        return False

    def cache_identity(self) -> object | None:
        """Stable source identity for cache reuse, or None if unavailable.

        File backends use resolved paths and file metadata. This is not a
        caller-supplied version number. A changed identity selects new bytes.
        Coordinated caches compare serialized identity content, not object id.
        Identity-only tokens are unsupported. Custom identities must support
        serialization and encode every field that affects equality.
        """
        return None

    def cache_namespace(self) -> object | None:
        """Stable physical-cache namespace shared by equivalent accessors.

        Return None for object-local memory. Coordinated caches then isolate
        the object instead of sharing completion state with another cache.
        """
        return None

    @_abc.abstractmethod
    def write_from_buffer(
        self, global_range: _RangeT, write_from_buffer: _BufferCallback
    ) -> _Result[None]: ...
