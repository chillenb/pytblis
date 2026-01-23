from contextlib import contextmanager
from contextvars import ContextVar

# You really shouldn't be mixing pytblis with Python threading,
# but we'll use contextvars just in case.
from typing import Literal, Union

_default_order: ContextVar[Literal["C", "F"]] = ContextVar("default_order", default="C")


@contextmanager
def use_default_array_order(order: Literal["C", "F"]):
    """Create a temporary context in which pytblis creates new arrays
    with the specified memory order (either 'C' or 'F').

    Example:
        >>> A = np.random.random((100, 100))
        >>> B = np.random.random((100, 100))
        >>> with pytblis.use_default_array_order('F'):
        >>>   C = pytblis.contract('ij,jk->ik', A, B)
        >>>   assert C.flags.f_contiguous

    """
    if order not in ("C", "F"):
        raise ValueError("order must be one of 'C' or 'F'")
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
    return _default_order.get()
