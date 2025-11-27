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

import numpy as np

from .wrappers import contract, transpose_add


def einsum(*operands, out=None, optimize="greedy", complex_real_contractions=False, **kwargs):
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
        'K' means it should be as close to the layout as the inputs as
        is possible, including arbitrarily permuted axes.
        Default is 'K'.
    optimize : {'greedy', 'optimal'}, default 'greedy'
        Controls the optimization strategy used to compute the contraction.
    complex_real_contractions : bool, default False
        If True, handle contractions between complex and real tensors by performing
        separate contractions for the real and imaginary parts of the complex tensor.

    Returns
    -------
    output : ndarray
        The calculation based on the Einstein summation convention.
    """
    specified_out = out is not None
    assert optimize in ("greedy", "optimal"), "optimize must be 'greedy' or 'optimal'"

    # Check the kwargs to avoid a more cryptic error later, without having to
    # repeat default values here
    valid_einsum_kwargs = ["order"]
    unknown_kwargs = [k for (k, v) in kwargs.items() if k not in valid_einsum_kwargs]
    if unknown_kwargs:
        msg = f"Did not understand the following kwargs: {unknown_kwargs}"
        raise TypeError(msg)

    # Build the contraction list and operand
    operands, contraction_list = np.einsum_path(*operands, optimize=optimize, einsum_call=True)

    # Handle order kwarg for output array, c_einsum allows mixed case
    output_order = kwargs.pop("order", "K")
    if output_order.upper() == "A":
        output_order = "F" if all(arr.flags.f_contiguous for arr in operands) else "C"

    # Start contraction loop
    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, blas = contraction
        tmp_operands = [operands.pop(x) for x in inds]
        # Do we need to deal with the output?
        handle_out = specified_out and ((num + 1) == len(contraction_list))
        out_kwarg = None
        if handle_out:
            out_kwarg = out

        if len(tmp_operands) == 2:
            new_view = contract(
                einsum_str,
                *tmp_operands,
                out=out_kwarg,
                allow_partial_trace=True,
                complex_real_contractions=complex_real_contractions,
            )

        elif len(tmp_operands) == 1:
            # check if only a transpose
            einsum_str = einsum_str.replace(" ", "")
            subscript_a, subscript_b = einsum_str.split("->")
            if sorted(subscript_a) == sorted(subscript_b):
                # only a transpose, use numpy for this (should return view)
                new_view = np.einsum(einsum_str, tmp_operands[0], out=out_kwarg, **kwargs)
            else:
                # may involve a trace or replication, use tblis transpose_add for this
                new_view = transpose_add(einsum_str, tmp_operands[0], out=out_kwarg, **kwargs)
        else:
            # fallback to numpy einsum
            # e.g. contractions of 3 tensors
            out_kwarg = None
            if handle_out:
                out_kwarg = out
            new_view = np.einsum(einsum_str, *tmp_operands, out=out_kwarg, **kwargs)

        # Append new items and dereference what we can
        operands.append(new_view)
        del tmp_operands, new_view

    if specified_out:
        return out
    return operands[0]
