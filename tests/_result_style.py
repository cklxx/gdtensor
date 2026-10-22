"""Test compatibility shim (not part of the package).

The design Result is ``tag.Ok | tag.Err`` with no methods; tests discriminate
with ``type(x) is Ok`` and read ``x.value``. These helpers adapt the old
unwrap/.ok/.error call sites in the test suite.
"""
from __future__ import annotations

from gdtensor.tag import Err, Ok


def is_ok(r) -> bool:
    return type(r) is Ok


def value_or(r):
    return r.value


def err_msg(r) -> str:
    v = r.value
    return getattr(v, "msg", str(v))


def install():
    """Install unwrap()/.ok/.error on Ok/Err for the test suite only."""
    if getattr(Ok, "_test_style", False):
        return
    def unwrap_ok(self):
        return self.value
    def unwrap_err(self):
        import gdtensor.internal.result as R
        raise R.Fail(self.value)
    Ok.unwrap = unwrap_ok
    Err.unwrap = unwrap_err
    Ok.ok = property(lambda self: True)
    Err.ok = property(lambda self: False)
    Err.error = property(lambda self: getattr(self.value, "msg", str(self.value)))
    Ok._test_style = True
    Err._test_style = True
