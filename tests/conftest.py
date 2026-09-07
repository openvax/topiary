"""Shared pytest configuration."""

from tests.optional_dependencies import require_integration


def pytest_runtest_setup(item):
    """Require a usable optional import before setting up its fixtures."""
    for dependency in ("isovar", "pirlygenes"):
        if item.get_closest_marker(dependency) is not None:
            require_integration(dependency)
