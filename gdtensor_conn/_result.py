"""Result handling for the simulation's file and memory adapters."""

from functools import wraps

from gdtensor.builder import GDTensorIOError, InvalidArgumentError, NotFoundError
from gdtensor.builder import Err, Ok


class Fail(Exception):
    def __init__(self, error):
        super().__init__(error.msg)
        self.error = error


def invalid(argument_name, msg):
    return Fail(InvalidArgumentError(argument_name, msg))


def io_error(operation, msg, file_path=None):
    return Fail(GDTensorIOError(file_path, operation, msg))


def not_found(key_name, key, msg):
    return Fail(NotFoundError(key_name, str(key), msg))


def catch(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            value = fn(*args, **kwargs)
        except Fail as exc:
            return Err(exc.error)
        except Exception as exc:
            return Err(GDTensorIOError(operation=fn.__name__, msg=str(exc)))
        return value if type(value) in (Ok, Err) else Ok(value)

    return wrapped
