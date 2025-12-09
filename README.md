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
`np.complex128`. Mixing arrays of different precisions isn't yet supported.

Arrays with negative or zero stride are not supported and will cause pytblis to
fall back to NumPy (for `einsum` and `contract`) or raise an error (all other
functions).

## New features

### Mixed-complex/real contractions

New in version v0.0.11: `pytblis.contract` fully supports contractions between
complex and/or real tensors of the same floating point precision, provided that
`alpha` and `beta` are both real. This just contracts the real and imaginary
parts separately with TBLIS. It's turned off by default because it's still
experimental, but you can enable it in `pytblis.contract` and `pytblis.einsum`
by passing `complex_real_contractions=True`. Otherwise, all mixed-type
contractions use NumPy.

## Installation

I will try to get this package added to conda-forge. In the meantime, conda
packages may be downloaded from my personal channel.

```
conda install pytblis -c conda-forge -c chillenb
```

The pre-built wheels on PyPI use pthreads for multithreading. To reduce the
overhead due to creating and joining threads, compile pytblis yourself and
configure it to use OpenMP, or use the conda packages.

`pip install pytblis` (not as performant)

### About OpenBLAS

[Don't use OpenBLAS configured with pthreads](https://github.com/pyscf/pyscf/discussions/3011#discussioncomment-14782315).
It causes oversubscription when used with other multithreaded libraries, in
particular anything that uses OpenMP. Instead, use MKL (`libblas=*=*mkl`) or the
[OpenMP variant](https://conda-forge.org/news/2020/07/17/conda-forge-is-building-openblas-with-both-pthreads-and-openmp-on-linux/)
of OpenBLAS (`libopenblas=*=*openmp*`).

### Installation from source

#### the easy way:

`pip install --no-binary pytblis pytblis`

The default compile options will give good performance. OpenMP is the default
thread model when building from source. You can pass additional options to CMake
via `CMAKE_ARGS`, change the thread model, compile for other
[CPU microarchitectures](https://github.com/flame/blis/blob/master/docs/ConfigurationHowTo.md#configuration-families),
etc.

#### the hard way:

1. Install TBLIS.
2. Run `CMAKE_ARGS="-DTBLIS_ROOT=wherever_tblis_is_installed" pip install .`

See [dev_install.sh](dev_install.sh) for an example. This script installs TBLIS
in `./local_tblis_prefix` and then links pytblis against it.

## Research

If you use TBLIS in your academic work, it's a good idea to cite:

- [High-Performance Tensor Contraction without Transposition](https://epubs.siam.org/doi/10.1137/16M108968X)
- [Strassen's Algorithm for Tensor Contraction](https://epubs.siam.org/doi/abs/10.1137/17M1135578)

TBLIS is not my work, and its developers are not responsible for flaws in these
Python bindings.

## Acknowledgements

The implementation of einsum and the tests are modified versions of those from
[opt_einsum](https://github.com/dgasmith/opt_einsum).

pytblis was developed in the [Zhu Group](https://www.tianyuzhu.org/), Department
of Chemistry, Yale University.

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
