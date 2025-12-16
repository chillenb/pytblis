from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal, Union

_default_order = ContextVar("default_order", default=None)


@contextmanager
def use_default_array_order(order: str):
    """Create a temporary context in which pytblis creates new arrays
    with the specified memory order (either 'C' or 'F').
    """
    assert order in ("C", "F"), "order must be either 'C' or 'F'"
    prev_order = _default_order.get()
    _default_order.set(order)
    try:
        yield
    finally:
        _default_order.set(prev_order)


def get_default_array_order() -> Union[Literal["C"], Literal["F"]]:
    """Get the current default array order for new arrays created by
    pytblis.
    """
    default_order = _default_order.get()
    if default_order is None:
        return "C"
    return default_order
