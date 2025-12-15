import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from ._pytblis_impl import add, mult, shift
from .typecheck import _accepted_types, _check_strides, _check_tblis_types, _valid_labels, contraction_result_shape

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
    input_strides_ok, output_strides_ok = _check_strides(a, out=out)

    if scalar_type is None:
        raise TypeError(
            "TBLIS only supports float32, float64, complex64, and complex128. "
            "Types do not match or unsupported type detected. "
        )
    # It's okay if A has bad strides if its size is 0, because nothing will be read.
    if not input_strides_ok and a.size != 0:
        msg = f"Input tensor of shape {a.shape} has non-positive strides: {a.strides}"
        raise ValueError(msg)
    if not output_strides_ok and out.size != 0:
        msg = f"Output tensor of shape {out.shape} has non-positive strides: {out.strides}"
        raise ValueError(msg)

    subscripts = subscripts.replace(" ", "")
    a_idx, b_idx = subscripts.split("->")

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

    # handle zero-sized input
    # if a is size zero, tensor_add quits early and does not scale B.
    if a.size == 0:
        shift(out, b_idx, alpha=0.0, beta=beta)
    else:
        add(a, out, a_idx, b_idx, alpha=alpha, beta=beta, conja=conja, conjb=conjout)
    return out


def contract_same_type(
    subscripts: str,
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    out: Optional[npt.ArrayLike] = None,
    conja: bool = False,
    conjb: bool = False,
    allow_partial_trace: bool = False,
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
    allow_partial_trace : bool, optional
        If True, handle redundant indices in subscripts for `a` and `b` by doing partial trace before contraction.
    out : array_like, optional
        Output tensor to store the result.

    Returns
    -------
    ndarray
        Result of the tensor contraction.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    scalar_type = _check_tblis_types(a, b, out=out)
    input_strides_ok, output_strides_ok = _check_strides(a, b, out=out)
    is_trivial = a.size == 0 or b.size == 0

    fallback = False

    if scalar_type is None:
        warnings.warn(
            "TBLIS only supports float32, float64, complex64, and complex128. "
            "Types do not match or unsupported type detected. "
            "Will attempt to fall back to numpy tensordot.",
            stacklevel=2,
        )
        fallback = True
    if not output_strides_ok and not is_trivial:
        warnings.warn(
            f"Output tensor of shape {out.shape} has non-positive strides: {out.strides}. "
            "Will attempt to fall back to numpy tensordot.",
            stacklevel=2,
        )
        fallback = True

    if not input_strides_ok and not is_trivial:
        warnings.warn(
            f"Input tensor of shape {a.shape} has non-positive strides: {a.strides}. "
            "Will attempt to fall back to numpy tensordot.",
            stacklevel=2,
        )
        fallback = True

    if fallback:
        if alpha != 1.0 or beta != 0.0:
            msg = "Cannot fall back to numpy tensordot unless alpha = 1.0 and beta = 0.0"
            raise ValueError(msg)
        return np.einsum(subscripts, a, b)

    subscripts = subscripts.replace(" ", "")
    input_str, subscript_c = subscripts.split("->")
    subscript_a, subscript_b = input_str.split(",")

    idx_a = frozenset(subscript_a)
    idx_b = frozenset(subscript_b)
    idx_c = frozenset(subscript_c)
    idx_redundant_a = idx_a - idx_b - idx_c
    idx_redundant_b = idx_b - idx_a - idx_c
    idx_redundant_c = idx_c - idx_a - idx_b

    if idx_redundant_c:
        raise RuntimeError("Should never have redundant indices in the output. This is probably a bug.")

    if not allow_partial_trace and (idx_redundant_a or idx_redundant_b):
        msg = (
            f"Subscripts '{subscripts}' require partial trace on "
            f"{'a ' if idx_redundant_a else ''}{'b' if idx_redundant_b else ''}. "
            "Pass allow_partial_trace=True to enable."
        )
        raise ValueError(msg)

    c_shape = contraction_result_shape(subscripts, a.shape, b.shape)

    if idx_redundant_a:
        # partial trace on a
        subscript_a_traced = "".join([i for i in subscript_a if i not in idx_redundant_a])
        einsum_str_traced = f"{subscript_a}->{subscript_a_traced}"
        a = transpose_add(einsum_str_traced, a)
        subscript_a = subscript_a_traced
    if idx_redundant_b:
        # partial trace on b
        subscript_b_traced = "".join([i for i in subscript_b if i not in idx_redundant_b])
        einsum_str_traced = f"{subscript_b}->{subscript_b_traced}"
        b = transpose_add(einsum_str_traced, b)
        subscript_b = subscript_b_traced

    if out is None:
        out = np.empty(c_shape, dtype=scalar_type)
        assert beta == 0.0, "beta must be 0.0 if out is None"
    else:
        out = np.asarray(out)
        if out.shape != c_shape:
            msg = f"Output shape {out.shape} does not match expected shape {c_shape} for subscripts '{subscripts}'"
            raise ValueError(msg)

    # handle zero-sized input
    # if A or B is size zero, mult quits early and does not scale C.
    if is_trivial:
        shift(out, subscript_c, alpha=0.0, beta=beta)
    else:
        mult(a, b, out, subscript_a, subscript_b, subscript_c, alpha=alpha, beta=beta, conja=conja, conjb=conjb)
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
    allow_partial_trace: bool = False,
    complex_real_contractions: bool = False,
) -> npt.ArrayLike:
    """
    Similar to `pytblis.contract`, but with support for contractions between complex and real tensors.
    If one of the input tensors is complex and the other is real, we perform separate contractions for the
    real and imaginary parts of the complex tensor.
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
    allow_partial_trace : bool, optional
        If True, handle redundant indices in subscripts for `a` and `b` by doing partial trace before contraction.
    complex_real_contractions : bool, default False
        If True, handle contractions between complex and real tensors by performing
        separate contractions for the real and imaginary parts of the complex tensor.
        alpha and beta must be real in this case.
    out : array_like, optional
        Output tensor to store the result.

    Returns
    -------
    ndarray
        Result of the tensor contraction.

    Examples
    --------
    >>> import numpy as np
    >>> from pytblis import contract_complex_real
    >>> A = np.random.rand(3, 4, 5)
    >>> B = np.random.rand(4, 5, 6).astype(np.complex128)
    >>> result = contract_complex_real('ijk,jkl->il', A, B, alpha=2.0)
    """

    if not complex_real_contractions:
        return contract_same_type(
            subscripts,
            a,
            b,
            alpha=alpha,
            beta=beta,
            out=out,
            conja=conja,
            conjb=conjb,
            allow_partial_trace=allow_partial_trace,
        )
    a = np.asarray(a)
    b = np.asarray(b)

    if out is not None:
        real_scalar_type = _check_tblis_types(a.real, b.real, out=out.real)
    else:
        real_scalar_type = _check_tblis_types(a.real, b.real)

    if real_scalar_type is None:
        raise TypeError("Input arrays and output array (if provided) must use the same IEEE floating point type.")

    if np.iscomplexobj(a) and np.iscomplexobj(b):
        return contract(subscripts, a, b, alpha, beta, out, conja, conjb, allow_partial_trace)
    if (not np.iscomplexobj(a)) and (not np.iscomplexobj(b)):
        return contract(subscripts, a, b, alpha, beta, out, conja, conjb, allow_partial_trace)

    # exactly one of a or b is complex, so the result must be complex
    if out is not None and not np.iscomplexobj(out):
        msg = "Output array must have complex dtype when contracting a complex tensor with a real tensor."
        raise ValueError(msg)

    result_type = np.result_type(real_scalar_type, 1j).type
    # allocate complex output if not provided

    c_shape = contraction_result_shape(subscripts, a.shape, b.shape)
    if out is None:
        out = np.empty(c_shape, dtype=result_type)
        assert beta == 0.0, "beta must be 0.0 if out is None"

    assert alpha.imag == 0.0, "alpha must be real when contracting a complex tensor with a real tensor."
    assert beta.imag == 0.0, "beta must be real when contracting a complex tensor with a real tensor."

    if c_shape:
        if np.iscomplexobj(a) and not np.iscomplexobj(b):
            imag_fac = -1.0 if conja else 1.0
            a_real = a.real
            contract(
                subscripts,
                a_real,
                b,
                alpha=alpha,
                beta=beta,
                out=out.real,
                conja=conja,
                conjb=conjb,
                allow_partial_trace=allow_partial_trace,
            )
            a_imag = a.imag
            contract(
                subscripts,
                a_imag,
                b,
                alpha=alpha * imag_fac,
                beta=beta,
                out=out.imag,
                conja=conja,
                conjb=conjb,
                allow_partial_trace=allow_partial_trace,
            )
        else:
            imag_fac = -1.0 if conjb else 1.0
            b_real = b.real
            contract(
                subscripts,
                a,
                b_real,
                alpha=alpha,
                beta=beta,
                out=out.real,
                conja=conja,
                conjb=conjb,
                allow_partial_trace=allow_partial_trace,
            )
            b_imag = b.imag
            contract(
                subscripts,
                a,
                b_imag,
                alpha=alpha * imag_fac,
                beta=beta,
                out=out.imag,
                conja=conja,
                conjb=conjb,
                allow_partial_trace=allow_partial_trace,
            )
    # handle scalar output
    # c_shape == ()
    elif np.iscomplexobj(a) and not np.iscomplexobj(b):
        imag_fac = -1.0 if conja else 1.0
        out = (
            beta * out
            + alpha * contract(subscripts, a.real, b, conja=conja, conjb=conjb, allow_partial_trace=allow_partial_trace)
            + 1j
            * alpha
            * imag_fac
            * contract(subscripts, a.imag, b, conja=conja, conjb=conjb, allow_partial_trace=allow_partial_trace)
        )
    else:
        imag_fac = -1.0 if conjb else 1.0
        out = (
            beta * out
            + alpha * contract(subscripts, a, b.real, conja=conja, conjb=conjb, allow_partial_trace=allow_partial_trace)
            + 1j
            * alpha
            * imag_fac
            * contract(subscripts, a, b.imag, conja=conja, conjb=conjb, allow_partial_trace=allow_partial_trace)
        )
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
    if a.flags.c_contiguous:
        return a

    if not _check_strides(a):
        warnings.warn(
            f"Input tensor of shape {a.shape} has non-positive strides: {a.strides}. Falling back to numpy ascontiguousarray.",
            stacklevel=2,
        )
        return np.ascontiguousarray(a)

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


def asfortranarray(a):
    """Parallel transpose the input to F-contiguous layout.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    ndarray
        Fortran-order array.
    """
    a = np.asarray(a)
    if a.flags.f_contiguous:
        return a

    if not _check_strides(a):
        warnings.warn(
            f"Input tensor of shape {a.shape} has non-positive strides: {a.strides}. Falling back to numpy asfortranarray.",
            stacklevel=2,
        )
        return np.asfortranarray(a)

    if a.dtype.type not in _accepted_types:
        warnings.warn(
            "TBLIS only supports float32, float64, complex64, and complex128. Falling back to numpy asfortranarray.",
            stacklevel=2,
        )
        return np.asfortranarray(a)

    out = np.empty(a.shape, dtype=a.dtype, order="F")
    assert len(a.shape) < len(_valid_labels), (
        f"a.ndim is {len(a.shape)}, but only {len(_valid_labels)} labels are valid."
    )
    a_inds = _valid_labels[: len(a.shape)]
    a_inds = "".join(a_inds)
    add(a, out, a_inds, a_inds, alpha=1.0, beta=0.0)
    return out
