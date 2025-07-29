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

## Limitations

Supported datatypes: float, double, complex float, complex double. Mixing arrays
of different types isn't yet supported. I may add a workaround for real-complex
tensor contraction.

Arrays with negative or zero stride are not supported and will cause pytblis to
fall back to NumPy.

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
