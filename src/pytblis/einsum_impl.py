# Contains code from opt_einsum, which is licensed under the MIT License.
# The MIT License (MIT)

# Copyright (c) 2014 Daniel Smith

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from contextlib import nullcontext

import numpy as np

from .defaultorder import _default_order, get_default_array_order, use_default_array_order
from .numpy_einsumpath import einsum_path
from .wrappers import contract, transpose_add


def einsum(*operands, out=None, optimize=True, complex_real_contractions=True, **kwargs):
    """
    einsum(subscripts, *operands, out=None, order='K',
           optimize='greedy')
    Evaluates the Einstein summation convention on the operands.

    Drop-in replacement for numpy.einsum, using TBLIS for tensor contractions.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation as comma separated list of
        subscript labels. An implicit (classical Einstein summation)
        calculation is performed unless the explicit indicator '->' is
        included as well as subscript labels of the precise output form.
    operands : list of array_like
        These are the arrays for the operation.
    out : ndarray, optional
        If provided, the calculation is done into this array.
    order : {'C', 'F', 'A', 'K'}, optional
        Controls the memory layout of the output. 'C' means it should
        be C contiguous. 'F' means it should be Fortran contiguous,
        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
        'K' is ignored, for now.
        Default is 'C'.
    optimize : {bool, list, tuple, 'greedy', 'optimal'}, default True
        Controls the optimization strategy used to compute the contraction.
        If a tuple is provided, the second argument is assumed to be 
        the maximum intermediate size created.
        Also accepts an explicit contraction list from the ``np.einsum_path``
        function. See ``np.einsum_path`` for more details. 
    complex_real_contractions : bool, default True
        If True, handle contractions between complex and real tensors by performing
        separate contractions for the real and imaginary parts of the complex tensor.
        This avoids NumPy type promotion if the complex and real tensors
        have the same precision (e.g., complex128 and float64).

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.
    """
    specified_out = out is not None

    # Check the kwargs to avoid a more cryptic error later, without having to
    # repeat default values here
    valid_einsum_kwargs = ["order"]
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if unknown_kwargs:
        msg = f"Did not understand the following kwargs: {unknown_kwargs}"
        raise TypeError(msg)

    # calculate contraction path
    operands, contraction_list = einsum_path(*operands, optimize=optimize, einsum_call=True)

    # Handle order kwarg for output array, c_einsum allows mixed case
    order_given = "order" in kwargs
    output_order = kwargs.get("order", _default_order.get())
    if output_order not in ("C", "F", "A", "K"):
        raise ValueError("order must be one of 'C', 'F', 'A', or 'K'")
    if output_order == "A":
        output_order = "F" if all(arr.flags.f_contiguous for arr in operands) else "C"
    elif output_order == "K":
        # ignore K.
        output_order = get_default_array_order()

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, einsum_str, _ = contraction
        tmp_operands = [operands.pop(x) for x in inds]

        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))

        if handle_out:
            out_kwarg = out
        else:
            out_kwarg = None

        if ((num + 1) == len(contraction_list)) and order_given:
            # Set the requested output order on the final contraction.
            order_context = use_default_array_order(output_order)
        else:
            order_context = nullcontext()

        if len(tmp_operands) == 2:
            # two operands: use contract
            with order_context:
                new_view = contract(
                    einsum_str,
                    *tmp_operands,
                    out=out_kwarg,
                    allow_partial_trace=True,
                    complex_real_contractions=complex_real_contractions,
                )

        elif len(tmp_operands) == 1:
            # check if only a transpose
            subscript_a, subscript_b = einsum_str.split("->")
            if sorted(subscript_a) == sorted(subscript_b):
                # only a transpose, use numpy for this (should return view)
                new_view = np.einsum(einsum_str, tmp_operands[0], out=out_kwarg, **kwargs)
            # may involve a trace or replication, use tblis transpose_add for this
            else:
                with order_context:
                    new_view = transpose_add(einsum_str, tmp_operands[0], out=out_kwarg)
        else:
            # 3 or more operands, fall back to numpy einsum
            new_view = np.einsum(einsum_str, *tmp_operands, out=out_kwarg, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    return operands[0]
