"""One verified, offline Osteosarc export for every Sid fixture group."""

from functools import lru_cache
import json
from pathlib import Path

from topiary import osteosarc_fixture_paths


ROOT = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def sid_fixture_paths():
    """Verify the immutable bundled objects once per test worker; never fetch."""
    manifest = json.loads((ROOT / "manifest.json").read_text())
    return osteosarc_fixture_paths(manifest, directory=ROOT)


def sid_data_root(dataset):
    """Return one bundled fixture directory after verifying the entire export."""
    paths = sid_fixture_paths()
    if not any(name.startswith(dataset + "/") for name in paths):
        raise ValueError(f"Unknown Sid fixture group: {dataset}")
    return ROOT / dataset
