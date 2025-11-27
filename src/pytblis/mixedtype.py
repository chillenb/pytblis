from typing import Union

import numpy as np

from .typecheck import _check_strides, _check_tblis_types
from .wrappers import transpose_add

scalar = Union[float, complex]


def complexify(realpart, imagpart, conj=False, scale=1.0, out=None):
    """Returns a complex tensor formed from the given real and imaginary parts.

    Parameters
    ----------
    realpart : array_like
        Must be a real-valued array.
    imagpart : array_like
        Must be a real-valued array with the same shape as `realpart`.
    conj : bool, optional
        If True, return (realpart - 1j * imagpart), by default False
    scale : float, optional
        Multiply the result by a real scalar, by default 1.0
    out : array_like, optional
        Output array to store the result, by default None.
        If given, must be a complex array with the same shape as `realpart` and `imagpart`,
        and use the same precision as the inputs.
    Returns
    -------
    output : ndarray
        The complex tensor formed from the given real and imaginary parts.

    Examples
    --------
    >>> import numpy as np
    >>> from pytblis import complexify
    >>> re = np.random.rand(3, 4, 5)
    >>> im = np.random.rand(3, 4, 5)
    >>> arr = complexify(re, im)
    """
    _check_tblis_types(realpart, imagpart, out=out)
    assert not np.iscomplexobj(realpart), "Inputs must be real arrays."
    assert not np.iscomplexobj(imagpart), "Inputs must be real arrays."
    input_strides_ok, output_strides_ok = _check_strides(realpart, imagpart, out=out)
    is_trivial = realpart.size == 0 or imagpart.size == 0
    shape = realpart.shape
    ndim = realpart.ndim
    assert shape == imagpart.shape, "Input arrays must have the same shape."
    inds = "abcdefghijklmnopqrstuvwxyz"[:ndim]
    result_type = np.result_type(realpart.dtype, 1j).type
    if out is None:
        out = np.empty(shape, dtype=result_type)

    if is_trivial:
        return np.asarray(realpart) + np.asarray(imagpart) * 1j

    assert out.shape == shape, "Output array must have the same shape as input arrays."
    assert input_strides_ok, "Input arrays must not have negative strides."
    assert output_strides_ok, "Output array must not have negative strides."

    conj_fac = -1.0 if conj else 1.0
    transpose_add(inds + "->" + inds, realpart, out=out.real, alpha=scale)
    transpose_add(inds + "->" + inds, imagpart, out=out.imag, alpha=scale * conj_fac)
    return out
