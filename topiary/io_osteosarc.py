"""Access to the shared, immutable Osteosarc regression inputs."""

from pathlib import Path
import json
import re
from urllib.parse import unquote, urlsplit


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
    from osteosarc import Cache, IntegrityError, digest

    versions = ("vaccine-rna-v1", "topiary-sid-v1")
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != "osteosarc"
            or manifest.get("data_version") not in versions or not manifest.get("assets")):
        raise ValueError("Expected an osteosarc vaccine-rna-v1 or topiary-sid-v1 manifest")
    names = set()
    for asset in manifest["assets"]:
        name = asset["filename"]
        if (not isinstance(name, str) or not name or name == "manifest.json"
                or any(c in name for c in "\\:") or name in names
                or any(part in ("", ".", "..") for part in name.split("/"))
                or (manifest["data_version"] == "vaccine-rna-v1" and "/" in name)):
            raise ValueError(f"Unsafe or duplicate fixture filename: {name!r}")
        names.add(name)
        url = urlsplit(asset["url"])
        if url.scheme != "https" or Path(unquote(url.path)).name != Path(name).name:
            raise ValueError(f"Fixture URL must retain its original HTTPS filename: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ValueError(f"Invalid fixture SHA-256: {name}")
        if type(asset["size_bytes"]) is not int or asset["size_bytes"] < 0:
            raise ValueError(f"Invalid fixture byte size: {name}")
    for case in manifest.get("cases", []):
        if case["bam"] not in names or case["bam"] + ".bai" not in names:
            raise ValueError(f"Case lacks its BAM/index pair: {case['case_id']}")
    if directory is not None:
        saved = Path(directory) / "manifest.json"
        if not saved.is_file() or saved.is_symlink() or json.loads(saved.read_text()) != manifest:
            raise IntegrityError(f"Offline export manifest is absent or changed: {saved}")
    cache = (cache if cache is not None else Cache()) if directory is None else None
    paths = {}
    for asset in manifest["assets"]:
        name, checksum, size = asset["filename"], asset["sha256"], asset["size_bytes"]
        # This is the documented cross-consumer object convention, independent
        # of package versions or Osteosarc's per-URL acquisition receipts.
        path = (Path(directory) / name if directory is not None else
                cache.objects / (checksum + "".join(Path(name).suffixes)))
        if directory is not None or path.exists():
            linked = path.is_symlink() or (directory is not None and any(
                Path(directory).joinpath(*Path(name).parts[:i]).is_symlink()
                for i in range(1, len(Path(name).parts))))
            if (linked
                    or not path.is_file() or path.stat().st_size != size or digest(path) != checksum):
                raise IntegrityError(f"Shared fixture changed: {path}")
            if directory is None:
                receipt = cache.import_file(path, asset["url"], sha256=checksum, size=size)
                path = cache.path(receipt)
        else:
            receipt = cache.fetch(asset["url"], sha256=checksum, size=size)
            path = cache.path(receipt)
        paths[name] = path
    return paths
