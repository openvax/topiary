"""Test selection for optional dependency integrations."""

from importlib import import_module
import os

import pytest


PIRLYGENES_REQUIRED_ENV = "TOPIARY_TEST_REQUIRE_PIRLYGENES"


def require_pirlygenes():
    """Load PirlyGenes, skipping only when the package is truly absent.

    A transitive ``ModuleNotFoundError`` or any other import failure is a
    broken installation, not an absent optional dependency, so it propagates
    unchanged. CI's dedicated integration job makes absence an error too.
    """
    try:
        return import_module("pirlygenes")
    except ModuleNotFoundError as error:
        if error.name != "pirlygenes":
            raise
        if os.environ.get(PIRLYGENES_REQUIRED_ENV) == "1":
            pytest.fail(
                "PirlyGenes is required in the optional-integration job but "
                "is not installed",
                pytrace=False,
            )
        pytest.skip("PirlyGenes optional integration is not installed")
