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

import sys

import numpy as np
import pytest

import pytblis


def random_scalar(is_complex, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if is_complex:
        return rng.random() + 1j * rng.random()
    return rng.random()


def test_pytblis_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pytblis" in sys.modules


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_tensordot(scalar_type):
    rng = np.random.default_rng(0)
    A = rng.random((3, 3, 3)).astype(scalar_type)
    B = rng.random((3, 3, 3)).astype(scalar_type)

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        A = A + 1j * rng.random((3, 3, 3)).astype(scalar_type)
        B = B + 1j * rng.random((3, 3, 3)).astype(scalar_type)

    C = pytblis.tensordot(A, B, axes=([2], [0]))
    C_correct = np.tensordot(A, B, axes=([2], [0]))
    assert np.allclose(C, C_correct)

    A = rng.random((3, 5, 4)).astype(scalar_type)
    B = rng.random((5, 4, 3)).astype(scalar_type)

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        A = A + 1j * rng.random((3, 5, 4)).astype(scalar_type)
        B = B + 1j * rng.random((5, 4, 3)).astype(scalar_type)

    C = pytblis.tensordot(A, B, axes=2)
    C_correct = np.tensordot(A, B, axes=2)
    assert np.allclose(C, C_correct)


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_contract(scalar_type):
    rng = np.random.default_rng(0)
    A = rng.random((3, 3, 3)).astype(scalar_type)
    B = rng.random((3, 3, 3)).astype(scalar_type)

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        A = A + 1j * rng.random((3, 3, 3)).astype(scalar_type)
        B = B + 1j * rng.random((3, 3, 3)).astype(scalar_type)

    C = pytblis.contract("ijk, jkl->il", A, B)
    C_correct = np.einsum("ijk,jkl->il", A, B)
    assert np.allclose(C, C_correct)

    A = rng.random((3, 5, 4)).astype(scalar_type)
    B = rng.random((5, 4, 3)).astype(scalar_type)

    if np.iscomplexobj(A) or np.iscomplexobj(B):
        A = A + 1j * rng.random((3, 5, 4)).astype(scalar_type)
        B = B + 1j * rng.random((5, 4, 3)).astype(scalar_type)

    C = pytblis.contract("ijk,jkl->il", A, B)
    C_correct = np.einsum("ijk,jkl->il", A, B)
    assert np.allclose(C, C_correct)


def test_tensordot_type_mixed():
    rng = np.random.default_rng(0)
    A = rng.random((3, 3)).astype(np.float32)
    B = rng.random((3, 3)).astype(np.float64)

    with pytest.warns(
        UserWarning,
        match="The types of the input tensors do not match. Falling back to numpy tensordot.",
    ):
        C = pytblis.tensordot(A, B, axes=([0], [0]))

    C_correct = np.tensordot(A, B, axes=([0], [0]))
    assert np.allclose(C, C_correct)


def test_tensordot_type_unsupported():
    rng = np.random.default_rng(0)
    A = rng.random((3, 3)).astype(np.int32)
    B = rng.random((3, 3)).astype(np.int32)

    with pytest.warns(
        UserWarning,
        match="TBLIS only supports float32, float64, complex64, and complex128. Falling back to numpy tensordot.",
    ):
        C = pytblis.tensordot(A, B, axes=([0], [0]))

    C_correct = np.tensordot(A, B, axes=([0], [0]))
    assert np.allclose(C, C_correct)


tests = [
    # Test scalar-like operations
    "a,->a",
    "ab,->ab",
    ",ab,->ab",
    ",,->",
    # Test hadamard-like products
    "a,ab,abc->abc",
    "a,b,ab->ab",
    # Test index-transformations
    "ea,fb,gc,hd,abcd->efgh",
    "ea,fb,abcd,gc,hd->efgh",
    "abcd,ea,fb,gc,hd->efgh",
    # Test complex contractions
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "acdf,jbje,gihb,hfac,gfac,gifabc,hfac",
    "cd,bdhe,aidb,hgca,gc,hgibcd,hgac",
    "abhe,hidj,jgba,hiab,gab",
    "bde,cdh,agdb,hica,ibd,hgicd,hiac",
    "chd,bde,agbc,hiad,hgc,hgi,hiad",
    "chd,bde,agbc,hiad,bdi,cgh,agdb",
    "bdhe,acad,hiab,agac,hibd",
    # Test collapse
    "ab,ab,c->",
    "ab,ab,c->c",
    "ab,ab,cd,cd->",
    "ab,ab,cd,cd->ac",
    "ab,ab,cd,cd->cd",
    "ab,ab,cd,cd,ef,ef->",
    # Test outer products
    "ab,cd,ef->abcdef",
    "ab,cd,ef->acdf",
    "ab,cd,de->abcde",
    "ab,cd,de->be",
    "ab,bcd,cd->abcd",
    "ab,bcd,cd->abd",
    # Random test cases that have previously failed
    "eb,cb,fb->cef",
    "dd,fb,be,cdb->cef",
    "bca,cdb,dbf,afc->",
    "dcc,fce,ea,dbf->ab",
    "fdf,cdd,ccd,afe->ae",
    "abcd,ad",
    "ed,fcd,ff,bcf->be",
    "baa,dcf,af,cde->be",
    "bd,db,eac->ace",
    "fff,fae,bef,def->abd",
    "efc,dbc,acf,fd->abe",
    # Inner products
    "ab,ab",
    "ab,ba",
    "abc,abc",
    "abc,bac",
    "abc,cba",
    # GEMM test cases
    "ab,bc",
    "ab,cb",
    "ba,bc",
    "ba,cb",
    "abcd,cd",
    "abcd,ab",
    "abcd,cdef",
    "abcd,cdef->feba",
    "abcd,efdc",
    # Inner than dot
    "aab,bc ->ac",
    "ab,bcc->ac",
    "aab,bcc->ac",
    "baa,bcc->ac",
    "aab,ccb->ac",
    # Randomly build test caes
    "aab,fa,df,ecc->bde",
    "ecb,fef,bad,ed->ac",
    "bcf,bbb,fbf,fc->",
    "bb,ff,be->e",
    "bcb,bb,fc,fff->",
    "fbb,dfd,fc,fc->",
    "afd,ba,cc,dc->bf",
    "adb,bc,fa,cfc->d",
    "bbd,bda,fc,db->acf",
    "dba,ead,cad->bce",
    "aef,fbc,dca->bde",
]

_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4, 9, 10, 2, 4, 5, 3, 2, 6]
_no_collision_chars = "".join(chr(i) for i in range(7000, 7007))
_valid_chars = "abcdefghijklmnopqABC" + _no_collision_chars
_default_dim_dict = dict(zip(_valid_chars, _sizes))


def build_shapes(string, dimension_dict=None):
    if dimension_dict is None:
        dimension_dict = _default_dim_dict

    shapes = []
    string = string.replace(" ", "")
    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        shapes.append(tuple(dims))
    return tuple(shapes)


def build_views(string, dimension_dict=None, dtype=np.float64, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    views = []
    for shape in build_shapes(string, dimension_dict=dimension_dict):
        if shape:
            arr = rng.random(shape).astype(dtype)
            if np.iscomplexobj(arr):
                arr += 1j * rng.random(shape).astype(dtype)
            views.append(arr)
        else:
            views.append(rng.random())
    return tuple(views)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_einsum(string, dtype):
    views = build_views(string, dtype=dtype)
    tblis_result = pytblis.einsum(string, *views)
    numpy_result = np.einsum(string, *views)
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


single_array_tests = ["ea", "fb", "abcd", "gc", "hd", "efgh", "acdf", "gihb", "hfac", "gfac", "gifabc", "hfac"]


@pytest.mark.parametrize("string", single_array_tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_ascontiguousarray(string, dtype):
    rng = np.random.default_rng(0)
    views = build_views(string, dtype=dtype)
    arr = views[0]

    arr = np.transpose(arr, axes=rng.permutation(len(arr.shape)))
    tblis_result = pytblis.ascontiguousarray(arr)
    numpy_result = np.ascontiguousarray(arr)
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("string", single_array_tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_tensor_transpose(string, dtype):
    rng = np.random.default_rng(0)
    views = build_views(string, dtype=dtype)
    arr = views[0]

    perm = rng.permutation(len(arr.shape))

    string_perm = "".join(np.array(list(string))[perm])
    command_string = f"{string}->{string_perm}"

    numpy_result = np.transpose(arr, axes=perm)
    tblis_result = pytblis.transpose_add(command_string, arr, alpha=1.0)
    assert np.allclose(tblis_result, numpy_result), f"Failed for command: {command_string}"


@pytest.mark.parametrize("string", single_array_tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_tensor_transpose_add(string, dtype):
    rng = np.random.default_rng(0)
    a = build_views(string, dtype=dtype, rng=rng)[0]

    alpha = random_scalar(np.iscomplexobj(a), rng=rng)
    beta = random_scalar(np.iscomplexobj(a), rng=rng)
    perm = rng.permutation(len(a.shape))

    b = build_views(string, dtype=dtype, rng=rng)[0]
    b = np.ascontiguousarray(np.transpose(b, axes=perm))

    string_perm = "".join(np.array(list(string))[perm])
    command_string = f"{string}->{string_perm}"

    numpy_result = beta * b + alpha * np.transpose(a, axes=perm)
    tblis_result = pytblis.transpose_add(command_string, a, alpha=alpha, beta=beta, out=b)
    assert np.allclose(tblis_result, numpy_result), f"Failed for command: {command_string}"
