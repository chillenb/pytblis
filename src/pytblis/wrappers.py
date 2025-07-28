import warnings

import numpy as np

from ._pytblis_impl import _mult


def _check_tblis_types(*tensors):
    """
    Returns the scalar type if all tensors have the same datatype, and this datatype is
    one of the supported types (float, double, complex float, complex double).
    Otherwise return None.
    """
    ok_types = (np.float32, np.float64, np.complex64, np.complex128)
    if len(tensors) == 0:
        return None
    first_type = tensors[0].dtype.type
    for tensor in tensors:
        if tensor.dtype.type != first_type or tensor.dtype.type not in ok_types:
            return None
    return first_type


def contract(subscripts, a, b, alpha=1.0, beta=0.0, out=None, conja=False, conjb=False):
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
    alpha : float, optional
        Scaling factor for the product of `a` and `b`.
    beta : float, optional
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
    if scalar_type is None:
        warnings.warn(
            "TBLIS only supports float32, float64, complex64, and complex128. "
            "Types do not match or unsupported type detected. "
            "Falling back to numpy tensordot.", stacklevel=2
        )
        if alpha != 1.0 or beta != 0.0:
            msg = "Cannot fall back to numpy tensordot with non-default alpha or beta."
            raise ValueError(msg)
        return np.einsum(subscripts, a, b, alpha=alpha, beta=beta, out=out)

    input_str, c_idx = subscripts.split('->')
    a_idx, b_idx = input_str.split(',')

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
    elif out.shape != c_shape:
        msg = f"Output shape {out.shape} does not match expected shape {c_shape} for subscripts '{subscripts}'"
        raise ValueError(msg)

    _mult(a, b, out, a_idx, b_idx, c_idx, alpha=alpha, beta=beta, conja=conja, conjb=conjb)
    return out
