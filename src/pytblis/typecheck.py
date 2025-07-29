import numpy as np

_valid_labels = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_accepted_types = (np.float32, np.float64, np.complex64, np.complex128)


def _check_strides(*tensors):
    """
    Check for non-positive strides in the input tensors.
    Return False if any tensor has a non-positive stride, otherwise True.
    Non-contiguity is OK.
    """
    return all(all(s > 0 for s in tensor.strides) for tensor in tensors)


def _check_tblis_types(*tensors):
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
    return first_type
