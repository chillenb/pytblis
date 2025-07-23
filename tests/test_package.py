from __future__ import annotations

import importlib.metadata

import pytblis as m


def test_version():
    assert importlib.metadata.version("pytblis") == m.__version__
