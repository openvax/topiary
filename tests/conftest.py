"""Shared pytest configuration."""

from tests.optional_dependencies import require_pirlygenes


def pytest_runtest_setup(item):
    """Require a usable PirlyGenes import for its integration tests."""
    if item.get_closest_marker("pirlygenes") is not None:
        require_pirlygenes()
