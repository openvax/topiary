"""Compatibility entry point for shared Osteosarc fixture manifests."""


def osteosarc_fixture_paths(manifest, *, directory=None, cache=None):
    """Resolve and verify every original asset in a shared fixture manifest.

    Parameters
    ----------
    manifest : dict
        A trusted ``osteosarc`` manifest: ``vaccine-rna-v1`` for shared vaccine
        reads or ``topiary-sid-v1`` for the full bundled test corpus. Each asset
        supplies a relative filename, immutable URL, SHA-256 and byte size.
        Original source provenance and selection policies remain in the bundle.
    directory : path-like, optional
        An existing offline export to verify. When given, read only this
        directory; do not contact the network or write to any cache.
    cache : osteosarc.Cache, optional
        Cache for explicit acquisition when ``directory`` is absent. Defaults
        to Osteosarc's shared OpenVax root. Use ``Cache(..., offline=True)`` to
        prohibit downloads. Existing objects from other OpenVax consumers are
        verified and adopted with ``Cache.import_file``; their bytes are reused.

    Returns
    -------
    dict of str to pathlib.Path
        Original filenames mapped to verified source paths. An empty asset
        list is rejected: it does not establish an offline regression dataset.

    Raises
    ------
    ValueError
        The manifest has unsafe, duplicate or incomplete asset identities.
    osteosarc.IntegrityError
        An existing asset disagrees with its pinned size or SHA-256. No repair
        is attempted. Obtain a fresh export explicitly after investigating.
    osteosarc.OfflineError
        An asset is absent from the cache while offline.

    Notes
    -----
    Osteosarc is installed with Topiary. This resolves original reads, not
    translated proteins or predictor caches. Their content
    identities and provenance must be versioned separately by the consumer.
    """
    from osteosarc.fixture_assets import fixture_paths
    return fixture_paths(manifest, directory=directory, cache=cache)
