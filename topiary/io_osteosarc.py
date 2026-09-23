"""Compatibility entry point for shared Osteosarc fixture manifests."""


def osteosarc_fixture_paths(manifest, *, directory=None, cache=None):
    """Resolve and verify pinned original assets using Osteosarc's common API."""
    from osteosarc.fixture_assets import fixture_paths
    return fixture_paths(manifest, directory=directory, cache=cache)
