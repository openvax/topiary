"""Regenerate the reviewed Sid corpus with the shared Osteosarc implementation."""

import argparse
from pathlib import Path
from osteosarc import Cache
from osteosarc.regional_corpus import generate_regional_corpus as generate_sid_fixtures

DEFAULT = Path(__file__).resolve().parents[1] / "tests/data/sid-fixtures.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--live-alignments", action="store_true")
    parser.add_argument("--panel-recipe", type=Path, help="Shared Osteosarc v1 panel recipe")
    parser.add_argument("--panel-source", action="append", default=[], metavar="ID=LOCAL_BAM")
    args = parser.parse_args()
    if args.panel_recipe:
        from osteosarc import Cache
        from osteosarc.bundles import generate_panel
        generate_panel(args.panel_recipe, args.output, sources=args.panel_source,
                       cache=Cache(args.cache_root, offline=args.offline))
        raise SystemExit(0)

    manifest = generate_sid_fixtures(
        args.recipe, args.output, cache=Cache(args.cache_root, offline=args.offline),
        source_directory=args.source_directory, live_alignments=args.live_alignments)
    print(f"Generated {len(manifest['assets'])} assets, {manifest['total_size_bytes']:,} bytes")
    for name, counts in manifest["read_selections"].items():
        print(f"{name}: {counts['original_records']} -> {counts['selected_records']} records")
