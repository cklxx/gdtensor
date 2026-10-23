"""gdtensor.tag: named scalar tags (design C.2).

Only ``tagged[...]`` scalar markers live here:
``ConstantOne``, ``TensorCount``, ``WorldSize``, ``NumSteps``, ``NumNodes``,
``ConcurrencyWidth``. Mesh oneof branches carry no tag, so the branch product
types (``Shard`` / ``Replicate`` / ``Partial``) and the Result branches
(``Ok`` / ``Err``) are products in ``gdtensor.builder``, not tags.
"""

import types as _types
from typing import Any as _Any

__all__ = [
    "Box",
    "make_box_subclass",
    "ConstantOne",
    "TensorCount",
    "WorldSize",
    "NumSteps",
    "NumNodes",
    "ConcurrencyWidth",
]


class Box:
    # A named scalar tag: the tag is the class itself, the payload is value.
    __slots__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        args = item if isinstance(item, tuple) else (item,)
        return _types.GenericAlias(cls, args)

    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.value))

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


def make_box_subclass(name) -> _Any:
    return type(name, (Box,), {"__module__": __name__, "__slots__": ()})


ConstantOne = make_box_subclass("ConstantOne")
TensorCount = make_box_subclass("TensorCount")
WorldSize = make_box_subclass("WorldSize")

NumSteps = make_box_subclass("NumSteps")
NumNodes = make_box_subclass("NumNodes")
ConcurrencyWidth = make_box_subclass("ConcurrencyWidth")
