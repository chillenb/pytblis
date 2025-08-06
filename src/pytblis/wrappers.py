import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ._pytblis_impl import add, mult
from .typecheck import _accepted_types, _check_strides, _check_tblis_types, _valid_labels

scalar = Union[float, complex]


def transpose_add(
    subscripts: str,
    a: npt.ArrayLike,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    out: Optional[npt.ArrayLike] = None,
    conja: bool = False,
    conjout: bool = False,
) -> npt.ArrayLike:
    """
    Perform tensor transpose and addition based on the provided subscripts.
    High-level wrapper for tblis_tensor_add.
    B (stored in `out` if provided) is computed as:
    B = alpha * transpose(a, subscripts) + beta * B.
    Optionally, conjugation can be applied to `a` and `out`.

    Parameters
    ----------
    subscripts : str
        Subscripts defining the contraction.
    a : array_like
        Tensor operand
    alpha : scalar, optional
        Scaling factor for `a`.
    beta : scalar, optional
        Scaling factor for the output tensor `b`. Must be 0.0 if `out` is None.
    conja: bool, optional
        If True, conjugate the first tensor `a`. Alpha is not conjugated.
    conjout: bool, optional
        If True, conjugate the output tensor `b`. Beta is not conjugated.
    out : array_like, optional
        Output tensor containing `b`.

    Returns
    -------
    ndarray
        Result of the tensor contraction.

    Examples
    --------
    >>> import numpy as np
    >>> from pytblis import transpose_add
    >>> a = np.random.rand(3, 4, 5)
    >>> b = transpose_add("ijk->ikj", a, alpha=1.0)
    """
    a = np.asarray(a)
    scalar_type = _check_tblis_types(a, out) if out is not None else _check_tblis_types(a)
    strides_ok = _check_strides(a, out) if out is not None else _check_strides(a)

    if scalar_type is None:
        raise TypeError(
            "TBLIS only supports float32, float64, complex64, and complex128. "
            "Types do not match or unsupported type detected. "
        )
    if not strides_ok:
        msg = "Input tensor has non-positive strides."
        raise ValueError(msg)

    subscripts = subscripts.replace(" ", "")
    a_idx, b_idx = subscripts.split("->")

    # a_idx, b_idx, c_idx = re.split(",|->", subscripts)

    if not set(a_idx) >= set(b_idx):
        msg = f"Invalid subscripts '{subscripts}'"
        raise ValueError(msg)
    a_shape_dic = dict(zip(a_idx, a.shape))

    b_shape = tuple(a_shape_dic[x] for x in b_idx)

    if out is None:
        out = np.empty(b_shape, dtype=scalar_type)
        assert beta == 0.0, "beta must be 0.0 if out is None"
    else:
        out = np.asarray(out)
        if out.shape != b_shape:
            msg = f"Output shape {out.shape} does not match expected shape {b_shape} for subscripts '{subscripts}'"
            raise ValueError(msg)

    add(a, out, a_idx, b_idx, alpha=alpha, beta=beta, conja=conja, conjb=conjout)
    return out


def contract(
    subscripts: str,
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    out: Optional[npt.ArrayLike] = None,
    conja: bool = False,
    conjb: bool = False,
) -> npt.ArrayLike:
    """
    Perform tensor contraction based on the provided subscripts.
    C (stored in `out` if provided) is computed as:
    C = alpha * einsum(subscripts, a, b) + beta * C if `out` is provided.

    Parameters
    ----------
    subscripts : str
        Subscripts defining the contraction.
    a : array_like
        First tensor operand.
    b : array_like
        Second tensor operand.
    alpha : scalar, optional
        Scaling factor for the product of `a` and `b`.
    beta : scalar, optional
        Scaling factor for the output tensor. Must be 0.0 if `out` is None.
    conja: bool, optional
        If True, conjugate the first tensor `a` before contraction. Alpha is not conjugated.
    conjb: bool, optional
        If True, conjugate the second tensor `b` before contraction. Beta is not conjugated.
    out : array_like, optional
        Output tensor to store the result.

    Returns
    -------
    ndarray
        Result of the tensor contraction.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    scalar_type = _check_tblis_types(a, b, out) if out is not None else _check_tblis_types(a, b)
    strides_ok = _check_strides(a, b, out) if out is not None else _check_strides(a, b)

    if scalar_type is None or not strides_ok:
        if scalar_type is None:
            warnings.warn(
                "TBLIS only supports float32, float64, complex64, and complex128. "
                "Types do not match or unsupported type detected. "
                "Will attempt to fall back to numpy tensordot.",
                stacklevel=2,
            )
        if not strides_ok:
            warnings.warn(
                "Input tensor has non-positive strides. Will attempt to fall back to numpy tensordot.", stacklevel=2
            )
        if alpha != 1.0 or beta != 0.0:
            msg = "Cannot fall back to numpy tensordot unless alpha = 1.0 and beta = 0.0"
            raise ValueError(msg)
        return np.einsum(subscripts, a, b)

    subscripts = subscripts.replace(" ", "")
    input_str, c_idx = subscripts.split("->")
    a_idx, b_idx = input_str.split(",")

    # a_idx, b_idx, c_idx = re.split(",|->", subscripts)

    if not (set(a_idx) | set(b_idx)) >= set(c_idx):
        msg = f"Invalid subscripts '{subscripts}'"
        raise ValueError(msg)
    a_shape_dic = dict(zip(a_idx, a.shape))
    b_shape_dic = dict(zip(b_idx, b.shape))
    if any(a_shape_dic[x] != b_shape_dic[x] for x in set(a_idx) & set(b_idx)):
        msg = f"Shape mismatch for subscripts '{subscripts}': {a.shape} {b.shape}"
        raise ValueError(msg)

    ab_shape_dic = {**a_shape_dic, **b_shape_dic}
    c_shape = tuple(ab_shape_dic[x] for x in c_idx)

    if out is None:
        out = np.empty(c_shape, dtype=scalar_type)
        assert beta == 0.0, "beta must be 0.0 if out is None"
    else:
        out = np.asarray(out)
        if out.shape != c_shape:
            msg = f"Output shape {out.shape} does not match expected shape {c_shape} for subscripts '{subscripts}'"
            raise ValueError(msg)

    mult(a, b, out, a_idx, b_idx, c_idx, alpha=alpha, beta=beta, conja=conja, conjb=conjb)
    return out


def ascontiguousarray(a):
    """Parallel transpose the input to C-contiguous layout.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Contiguous array.
    """
    a = np.asarray(a)

    if not _check_strides(a):
        warnings.warn("Input tensor has non-positive strides. Falling back to numpy ascontiguousarray.", stacklevel=2)
        return np.ascontiguousarray(a)

    if a.flags.c_contiguous:
        return a
    if a.dtype.type not in _accepted_types:
        warnings.warn(
            "TBLIS only supports float32, float64, complex64, and complex128. Falling back to numpy ascontiguousarray.",
            stacklevel=2,
        )
        return np.ascontiguousarray(a)
    out = np.empty(a.shape, dtype=a.dtype, order="C")
    assert len(a.shape) < len(_valid_labels), (
        f"a.ndim is {len(a.shape)}, but only {len(_valid_labels)} labels are valid."
    )
    a_inds = _valid_labels[: len(a.shape)]
    a_inds = "".join(a_inds)
    add(a, out, a_inds, a_inds, alpha=1.0, beta=0.0)
    return out
