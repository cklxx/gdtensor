"""gdtensor.tag: tag classes (design C.2 / C.3).

Each tag is a Box subclass made by make_box_subclass, one line per tag. The
payload is ``value``. Discriminate with ``type(x) is tag.Shard``.
"""
import types

__all__ = [
    "Box", "make_box_subclass",
    "Ok", "Err", "Shard", "Replicate", "Partial",
    "ConstantOne", "TensorCount", "WorldSize",
]


class Box:
    # A tag: the tag is the class itself, the payload hangs on value.
    __slots__ = ("value",)
    __match_args__ = ("value",)

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        args = item if isinstance(item, tuple) else (item,)
        return types.GenericAlias(cls, args)

    # Product types compare and hash by field (C.2), and a field may hold a
    # tag, so a tag compares and hashes by (class, value).
    def __eq__(self, other):
        return type(self) is type(other) and self.value == other.value

    def __hash__(self):
        return hash((type(self), self.value))

    def __repr__(self):
        return f"{type(self).__name__}({self.value!r})"


def make_box_subclass(name):
    return type(name, (Box,), {"__module__": __name__, "__slots__": ()})


Ok = make_box_subclass("Ok")
Err = make_box_subclass("Err")
Shard = make_box_subclass("Shard")
Replicate = make_box_subclass("Replicate")
Partial = make_box_subclass("Partial")
ConstantOne = make_box_subclass("ConstantOne")
TensorCount = make_box_subclass("TensorCount")
WorldSize = make_box_subclass("WorldSize")
