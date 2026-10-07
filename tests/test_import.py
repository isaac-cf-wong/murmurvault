"""Test importing the python-murmurvault package and its modules."""

from __future__ import annotations

import pkgutil

import pytest

import murmurvault


def get_all_submodules(package):
    """Discover all submodules in the package.

    Args:
        package: The package to inspect.

    """
    submodules = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(mod_name)
    return submodules


def test_import_main_package():
    """Test that the main murmurvault package can be imported."""
    assert hasattr(murmurvault, "__version__")
    assert murmurvault.__version__ is not None


@pytest.mark.parametrize("module_name", get_all_submodules(murmurvault))
def test_import_submodule(module_name):
    """Test that all submodules can be imported."""
    __import__(module_name)
