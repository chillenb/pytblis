import numpy as np

_valid_labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_accepted_types = (np.float32, np.float64, np.complex64, np.complex128)


def _check_strides(*tensors, out=None):
    """
    Check for negative strides in the input tensors.
    Return False if any tensor has a negative stride, otherwise True.
    Non-contiguity is OK.
    """
    inputs_ok = all(all(s > 0 for s in tensor.strides) for tensor in tensors)
    output_ok = (out is None) or all(s > 0 for s in out.strides)
    return inputs_ok, output_ok


def _check_tblis_types(*tensors, out=None):
    """
    Returns the scalar type if all tensors have the same datatype, and this datatype is
    one of the supported types (float, double, complex float, complex double).
    Otherwise return None.
    """
    if len(tensors) == 0:
        return None
    first_type = tensors[0].dtype.type
    for tensor in tensors:
        if tensor.dtype.type != first_type or tensor.dtype.type not in _accepted_types:
            return None
    if out is not None and out.dtype.type != first_type:
        return None
    return first_type


def contraction_result_shape(subscripts, a_shape, b_shape):
    subscripts = subscripts.replace(" ", "")
    input_str, subscript_c = subscripts.split("->")
    subscript_a, subscript_b = input_str.split(",")
    if not (set(subscript_a) | set(subscript_b)) >= set(subscript_c):
        msg = f"Invalid subscripts '{subscripts}'"
        raise ValueError(msg)
    a_shape_dic = dict(zip(subscript_a, a_shape))
    b_shape_dic = dict(zip(subscript_b, b_shape))
    if any(a_shape_dic[x] != b_shape_dic[x] for x in set(subscript_a) & set(subscript_b)):
        msg = f"Shape mismatch for subscripts '{subscripts}': {a_shape} {b_shape}"
        raise ValueError(msg)

    ab_shape_dic = {**a_shape_dic, **b_shape_dic}
    return tuple(ab_shape_dic[x] for x in subscript_c)
