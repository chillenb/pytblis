# pytblis: Python bindings for TBLIS

[![Actions Status][actions-badge]][actions-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- [![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link] -->

### Are your einsums too slow?

Need FP64 tensor contractions and can't buy a datacenter GPU because you already
maxed out your home equity line of credit?

Set your CPU on fire with
[TBLIS](https://github.com/MatthewsResearchGroup/tblis)!

## Installation

`pip install pytblis`

## Usage

`pytblis.einsum` and `pytblis.tensordot` are drop-in replacements for
`numpy.einsum` and `numpy.tensordot`.

In addition, low level wrappers are provided for
[`tblis_tensor_add`, `tblis_tensor_mult`, `tblis_tensor_reduce`, `tblis_tensor_shift`, and `tblis_tensor_dot`](https://github.com/MatthewsResearchGroup/tblis/wiki/C-Interface).
These are named `pytblis.add`, `pytblis.mult`, et cetera.

Finally, there are mid-level convenience wrappers for `tblis_tensor_mult` and
`tblis_tensor_add`:

```python
def contract(
    subscripts: str,
    a: ArrayLike,
    b: ArrayLike,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    out: Optional[npt.ArrayLike] = None,
    conja: bool = False,
    conjb: bool = False,
) -> ArrayLike
```

and

```python
def transpose_add(
    subscripts: str,
    a: ArrayLike,
    alpha: scalar = 1.0,
    beta: scalar = 0.0,
    out: Optional[ArrayLike] = None,
    conja: bool = False,
    conjout: bool = False,
) -> ArrayLike
```

These are used as follows:

```python
C = pytblis.contract("ij,jk->ik", A, B, alpha=1.0, beta=0.5, out=C, conja=True, conjb=False)
```

does

$$C \gets \overline{A} B + \frac{1}{2} C.$$

```python
B = pytblis.tensor_add("iklj->ijkl", A, alpha=-1.0, beta=1.0, out=B)
```

does

$$B_{ijkl} \gets B_{ijkl} - A_{iklj}.$$

Some additional documentation (work in progress) is available at
[pytblis.readthedocs.io](https://pytblis.readthedocs.io).

## Limitations

Supported datatypes: `np.float32`, `np.float64`, `np.complex64`,
`np.complex128`. Mixing arrays of different types isn't yet supported. I may add
a workaround for real-complex tensor contraction.

Arrays with negative or zero stride are not supported and will cause pytblis to
fall back to NumPy (for `einsum` and `contract`) or raise an error (all other
functions).

## Research

If you use TBLIS in your academic work, it's a good idea to cite:

- [High-Performance Tensor Contraction without Transposition](https://epubs.siam.org/doi/10.1137/16M108968X)
- [Strassen's Algorithm for Tensor Contraction](https://epubs.siam.org/doi/abs/10.1137/17M1135578)

TBLIS is not my work, and its developers are not responsible for flaws in these
Python bindings.

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/chillenb/pytblis/workflows/CI/badge.svg
[actions-link]:             https://github.com/chillenb/pytblis/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/pytblis
[conda-link]:               https://github.com/conda-forge/pytblis-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/chillenb/pytblis/discussions
[pypi-link]:                https://pypi.org/project/pytblis/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/pytblis
[pypi-version]:             https://img.shields.io/pypi/v/pytblis
[rtd-badge]:                https://readthedocs.org/projects/pytblis/badge/?version=latest
[rtd-link]:                 https://pytblis.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->
