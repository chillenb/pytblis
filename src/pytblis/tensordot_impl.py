# Contains code from NumPy, which is licensed under the BSD 3-Clause License.
# Copyright (c) 2005-2025, NumPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.

#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

import numpy as np

from ._pytblis_impl import mult
from .typecheck import _accepted_types, _valid_labels


def tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes using TBLIS.

    Given two tensors, `a` and `b`, and an array_like object containing
    two array_like objects, ``(a_axes, b_axes)``, sum the products of
    `a`'s and `b`'s elements (components) over the axes specified by
    ``a_axes`` and ``b_axes``. The third argument can be a single non-negative
    integer_like scalar, ``N``; if it is such, then the last ``N`` dimensions
    of `a` and the first ``N`` dimensions of `b` are summed over.

    Parameters
    ----------
    a, b : array_like
        Tensors to "dot".

    axes : int or (2,) array_like
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.
        * (2,) array_like
          Or, a list of axes to be summed over, first sequence applying to `a`,
          second to `b`. Both elements array_like must be of the same length.

    Returns
    -------
    output : ndarray
        The tensor dot product of the input.
    """
    try:
        iter(axes)
    except Exception:
        axes_a = list(range(-axes, 0))
        axes_b = list(range(axes))
    else:
        axes_a, axes_b = axes
    try:
        na = len(axes_a)
        axes_a = list(axes_a)
    except TypeError:
        axes_a = [axes_a]
        na = 1
    try:
        nb = len(axes_b)
        axes_b = list(axes_b)
    except TypeError:
        axes_b = [axes_b]
        nb = 1

    a, b = np.asarray(a), np.asarray(b)
    as_ = a.shape
    nda = a.ndim
    bs = b.shape
    ndb = b.ndim
    equal = True
    if na != nb:
        equal = False
    else:
        for k in range(na):
            if as_[axes_a[k]] != bs[axes_b[k]]:
                equal = False
                break
            if axes_a[k] < 0:
                axes_a[k] += nda
            if axes_b[k] < 0:
                axes_b[k] += ndb
    if not equal:
        msg = "shape-mismatch for sum"
        raise ValueError(msg)

    if a.dtype.type != b.dtype.type:
        warnings.warn("The types of the input tensors do not match. Falling back to numpy tensordot.", stacklevel=2)
        return np.tensordot(a, b, axes=axes)

    if a.dtype.type not in _accepted_types:
        warnings.warn(
            "TBLIS only supports float32, float64, complex64, and complex128. Falling back to numpy tensordot.",
            stacklevel=2,
        )
        return np.tensordot(a, b, axes=axes)

    # Move the axes to sum over to the end of "a"
    # and to the front of "b"
    notin_a = [k for k in range(nda) if k not in axes_a]
    olda = [as_[axis] for axis in notin_a]

    notin_b = [k for k in range(ndb) if k not in axes_b]
    oldb = [bs[axis] for axis in notin_b]

    outshape = olda + oldb

    if len(outshape) + len(axes_a) > len(_valid_labels):
        msg = f"Too many axes: maximum is {len(_valid_labels)}"
        raise ValueError(msg)

    labels_common = _valid_labels[: len(axes_a)]
    remaining_labels = _valid_labels[len(axes_a) :].copy()
    inds_A = []
    inds_B = []
    inds_C = []
    for i in range(nda):
        if i not in axes_a:
            newind = remaining_labels.pop(0)
            inds_A.append(newind)
            inds_C.append(newind)
        else:
            inds_A.append(labels_common[axes_a.index(i)])

    for i in range(ndb):
        if i not in axes_b:
            newind = remaining_labels.pop(0)
            inds_B.append(newind)
            inds_C.append(newind)
        else:
            inds_B.append(labels_common[axes_b.index(i)])
    inds_A = "".join(inds_A)
    inds_B = "".join(inds_B)
    inds_C = "".join(inds_C)

    restype = np.result_type(a, b)
    c = np.empty(outshape, dtype=restype)

    mult(a, b, c, inds_A, inds_B, inds_C)
    return c
