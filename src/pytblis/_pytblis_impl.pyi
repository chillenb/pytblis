import enum
from typing import Union

import numpy.typing as npt

scalar = Union[float, complex]

__all__ = ["add", "dot", "mult", "reduce", "shift"]

def add(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    idx_A: str,
    idx_B: str,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    conja: bool = False,
    conjb: bool = False,
) -> None: ...
def dot(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    idx_A: str,
    idx_B: str,
    alpha: scalar = 1.0,
    beta: scalar = 1.0,
    conja: bool = False,
    conjb: bool = False,
) -> scalar: ...
def mult(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    c: npt.ArrayLike,
    idx_A: str,
    idx_B: str,
    idx_C: str,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    conja: bool = False,
    conjb: bool = False,
) -> None: ...
def shift(
    a: npt.ArrayLike,
    idx_A: str,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
) -> None: ...
def tensor_set(
    a: npt.ArrayLike,
    idx_A: str,
    value: scalar,
) -> None: ...
def reduce(a: npt.ArrayLike, idx_A: str, op: enum.Enum, conja: bool = False) -> Union[scalar, tuple[scalar, int]]: ...
def get_num_threads() -> int: ...
def set_num_threads(num_threads: int) -> None: ...
