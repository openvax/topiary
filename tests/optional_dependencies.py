"""Test selection for optional dependency integrations."""

from importlib import import_module
import os

import pytest


def require_integration(dependency):
    """Load an integration, skipping only when the package is truly absent.

    A transitive ``ModuleNotFoundError`` or any other import failure is a
    broken installation, not an absent optional dependency, so it propagates
    unchanged. Setting ``TOPIARY_TEST_REQUIRE_<DEPENDENCY>`` to ``1`` in
    CI's dedicated integration job makes absence an error too.
    """
    try:
        return import_module(dependency)
    except ModuleNotFoundError as error:
        if error.name != dependency:
            raise
        if os.environ.get(f"TOPIARY_TEST_REQUIRE_{dependency.upper()}") == "1":
            pytest.fail(
                f"{dependency} is required in the optional-integration job but "
                f"is not installed; install with pip install 'topiary[{dependency}]'",
                pytrace=False,
            )
        pytest.skip(f"{dependency} optional integration is not installed")
