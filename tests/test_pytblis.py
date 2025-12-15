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


def random_array(shape, scalar_type, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    if np.issubdtype(scalar_type, np.complexfloating):
        arr = rng.random(shape) + 1j * rng.random(shape)
    else:
        arr = rng.random(shape)
    return arr.astype(scalar_type)


def test_pytblis_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pytblis" in sys.modules


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_transpose_add_size0(scalar_type):
    rng = np.random.default_rng(0)
    A = random_array((0, 3, 3), scalar_type, rng=rng)
    B = random_array((3, 3), scalar_type, rng=rng)
    C = pytblis.transpose_add("Zij->ij", A, alpha=1.0)
    C_correct = np.einsum("Zij->ij", A)
    assert np.allclose(C, C_correct)

    A = random_array((0, 3, 3), scalar_type, rng=rng)
    B = random_array((3, 3), scalar_type, rng=rng)
    C_correct = np.einsum("Zij->ij", A) + 0.5 * B
    C = pytblis.transpose_add("Zij->ij", A, alpha=1.0, beta=0.5, out=B)
    assert np.allclose(C, C_correct)


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
def test_tensordot(scalar_type):
    rng = np.random.default_rng(0)
    A = random_array((3, 3, 3), scalar_type, rng=rng)
    B = random_array((3, 3, 3), scalar_type, rng=rng)

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
    #    "Za,Za",
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
    "Za,ab->ab",
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
    # single array transpose
    "ea->ea",
    "fb->fb",
    "abcd->dcab",
    "gc->cg",
    "hd->dh",
    "efgh->hfge",
    "acdf->afcd",
    "gihb->ghib",
    "hfac->cfah",
    "gfac->cgaf",
    "gifabc->abifcg",
    "hfac->cfha",
    # single array partial trace
    "gfac->caf",
    "gcac->ca",
]

_sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4, 9, 0, 2, 4, 5, 3, 2, 6]
_no_collision_chars = "".join(chr(i) for i in range(7000, 7007))
_valid_chars = "abcdefghijklmnopqABCZ" + _no_collision_chars
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
            views.append(dtype(rng.random()))
    return tuple(views)


def build_views_some_complex(string, dimension_dict=None, dtype=np.float64, rng=None, complex_letter="g"):
    if rng is None:
        rng = np.random.default_rng(0)
    views = []
    if dimension_dict is None:
        dimension_dict = _default_dim_dict

    shapes = []
    string = string.replace(" ", "")
    terms = string.split("->")[0].split(",")
    for term in terms:
        dims = [dimension_dict[x] for x in term]
        shapes.append(tuple(dims))

    for shape, term in zip(shapes, terms):
        if shape:
            arr = rng.random(shape).astype(dtype)
            if complex_letter in term:
                arr = arr + 1j * rng.random(shape).astype(dtype)
            views.append(arr)
        else:
            views.append(dtype(rng.random()))
    return tuple(views)


def build_views_multi_type(string, dimension_dict=None, dtypes=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    views = []
    for shape, dtype in zip(build_shapes(string, dimension_dict=dimension_dict), dtypes):
        if shape:
            arr = rng.random(shape).astype(dtype)
            if np.iscomplexobj(arr):
                arr += 1j * rng.random(shape).astype(dtype)
            views.append(arr)
        else:
            views.append(dtype(rng.random()))
    return tuple(views)


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_einsum(string, dtype):
    views = build_views(string, dtype=dtype)
    tblis_result = pytblis.einsum(string, *views)
    numpy_result = np.einsum(string, *views)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("string", tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_einsum_complex_real(string, dtype):
    views = build_views_some_complex(string, dtype=dtype)
    tblis_result = pytblis.einsum(string, *views, complex_real_contractions=True)
    numpy_result = np.einsum(string, *views)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


single_array_tests = ["eZ", "fb", "abcd", "gc", "hd", "efgh", "acZf", "gihb", "hfac", "gfac", "gifabc", "hfac", "Z"]


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
    assert tblis_result.strides == numpy_result.strides, f"Strides mismatch for string: {string}"


@pytest.mark.parametrize("string", single_array_tests)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_asfortranarray(string, dtype):
    rng = np.random.default_rng(0)
    views = build_views(string, dtype=dtype)
    arr = views[0]

    arr = np.transpose(arr, axes=rng.permutation(len(arr.shape)))
    tblis_result = pytblis.asfortranarray(arr)
    numpy_result = np.asfortranarray(arr)
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"
    assert tblis_result.strides == numpy_result.strides, f"Strides mismatch for string: {string}"


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


contraction_tests = [
    "ijk, jkl->il",
    "abZd, Zd->ab",
    "aab, bZ->aZ",
    "ab, ba->",
    "abc, cba->",
    "abcd, cdef->abef",
    "aab, bcc->ac",
    "bbZZ, aaZ->baZ",
]


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", contraction_tests)
def test_contract(string, scalar_type):
    rng = np.random.default_rng(0)
    a, b = build_views(string, dtype=scalar_type, rng=rng)
    tblis_result = pytblis.contract(string, a, b)
    numpy_result = np.einsum(string, a, b)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", contraction_tests)
def test_contract_with_out(string, scalar_type):
    rng = np.random.default_rng(0)
    a, b = build_views(string, dtype=scalar_type, rng=rng)
    numpy_result = np.einsum(string, a, b)
    out = np.empty_like(numpy_result)
    tblis_result = pytblis.contract(string, a, b, out=out)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", contraction_tests)
def test_contract_alpha(string, scalar_type):
    rng = np.random.default_rng(0)
    alpha = random_scalar(np.iscomplexobj(scalar_type()), rng=rng)
    a, b = build_views(string, dtype=scalar_type, rng=rng)
    tblis_result = pytblis.contract(string, a, b, alpha=alpha)
    numpy_result = alpha * np.einsum(string, a, b)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", contraction_tests)
def test_contract_with_out_alphabeta(string, scalar_type):
    rng = np.random.default_rng(0)
    alpha = random_scalar(np.iscomplexobj(scalar_type()), rng=rng)
    beta = random_scalar(np.iscomplexobj(scalar_type()), rng=rng)
    a, b = build_views(string, dtype=scalar_type, rng=rng)
    numpy_result = alpha * np.einsum(string, a, b)
    C = random_array(numpy_result.shape, scalar_type, rng=rng)
    numpy_result += beta * C
    tblis_result = pytblis.contract(string, a, b, alpha=alpha, beta=beta, out=C)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize(
    "scalar_types",
    [
        (np.float32, np.complex64),
        (np.float64, np.complex128),
        (np.complex64, np.float32),
        (np.complex128, np.float64),
    ],
)
@pytest.mark.parametrize("string", contraction_tests)
@pytest.mark.parametrize("conja", [False, True])
@pytest.mark.parametrize("conjb", [False, True])
def test_contract_mixedtype(string, scalar_types, conja, conjb):
    rng = np.random.default_rng(0)

    alpha = random_scalar(False, rng=rng)
    a, b = build_views_multi_type(string, dtypes=scalar_types, rng=rng)
    aifconj = a.conj() if conja else a
    bifconj = b.conj() if conjb else b
    numpy_result = alpha * np.einsum(string, aifconj, bifconj)
    tblis_result = pytblis.contract(string, a, b, alpha=alpha, conja=conja, conjb=conjb, complex_real_contractions=True)
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("scalar_type1", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("scalar_type2", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", contraction_tests)
@pytest.mark.parametrize("conja", [False, True])
@pytest.mark.parametrize("conjb", [False, True])
def test_contract_mixedtype_alphabeta(string, scalar_type1, scalar_type2, conja, conjb):
    rng = np.random.default_rng(0)
    scalar_types = (scalar_type1, scalar_type2)
    alpha = random_scalar(False, rng=rng)
    beta = random_scalar(False, rng=rng)
    a, b = build_views_multi_type(string, dtypes=scalar_types, rng=rng)
    aifconj = a.conj() if conja else a
    bifconj = b.conj() if conjb else b
    numpy_result = alpha * np.einsum(string, aifconj, bifconj)
    C = random_array(numpy_result.shape, np.result_type(*scalar_types), rng=rng)
    numpy_result += beta * C
    if a.real.dtype.type != b.real.dtype.type:
        with pytest.raises(TypeError):
            tblis_result = pytblis.contract(
                string, a, b, alpha=alpha, beta=beta, out=C, conja=conja, conjb=conjb, complex_real_contractions=True
            )
        return
    tblis_result = pytblis.contract(
        string, a, b, alpha=alpha, beta=beta, out=C, conja=conja, conjb=conjb, complex_real_contractions=True
    )
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"


@pytest.mark.parametrize("scalar_type", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("string", single_array_tests)
@pytest.mark.parametrize("conj", [False, True])
def test_complexify(string, scalar_type, conj):
    rng = np.random.default_rng(0)
    a_real = build_views(string, dtype=scalar_type, rng=rng)[0]
    a_imag = build_views(string, dtype=scalar_type, rng=rng)[0]
    if scalar_type in [np.complex64, np.complex128]:
        with pytest.raises(AssertionError, match="Inputs must be real arrays."):
            pytblis.complexify(a_real, a_imag, conj=conj)
        return
    tblis_result = pytblis.complexify(a_real, a_imag, conj=conj)
    numpy_result = a_real + 1j * a_imag
    if conj:
        numpy_result = numpy_result.conj()
    assert tblis_result.shape == numpy_result.shape, f"Shape mismatch for string: {string}"
    assert np.allclose(tblis_result, numpy_result), f"Failed for string: {string}"
