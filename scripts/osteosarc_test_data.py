"""Explicitly acquire or verify the shared original-read regression bundle.

Run ``python -m scripts.osteosarc_test_data --help`` from a checkout with
``pip install -e .`` (Python 3.10+). Ordinary tests never acquire.
"""

import argparse
import json
from pathlib import Path
import shutil

from topiary import osteosarc_fixture_paths


DEFAULT = Path(__file__).resolve().parents[1] / "tests/data/osteosarc_shared/vaccine-rna-v1/manifest.json"


def main():
    from osteosarc import Cache

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", type=Path, help="Read-only verification of an existing export")
    parser.add_argument("--output", type=Path, help="Export into a NEW directory; never overwrite")
    args = parser.parse_args()
    if args.verify and args.output:
        parser.error("Choose --verify or --output")
    manifest = json.loads(args.manifest.read_text())
    paths = osteosarc_fixture_paths(manifest, directory=args.verify,
                                    cache=Cache(args.cache_root, offline=args.offline))
    if args.output:
        # mkdir reserves this exact new output. Write the manifest last so an
        # interrupted copy cannot masquerade as a complete exported bundle.
        args.output.mkdir(parents=True, exist_ok=False)
        for name, path in paths.items():
            shutil.copyfile(path, args.output / name)
        shutil.copyfile(args.manifest, args.output / "manifest.json")
        osteosarc_fixture_paths(manifest, directory=args.output)
    print(f"Verified {len(paths)} original assets for osteosarc / vaccine-rna-v1")


if __name__ == "__main__":
    main()
