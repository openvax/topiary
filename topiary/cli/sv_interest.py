"""Offline SV interest report: python -m topiary.cli.sv_interest --help."""

import argparse
import json
from pathlib import Path

from ..sv_interest import build_sv_interest_report, write_sv_interest_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retain and rank all SV nominations by explicit protein evidence.")
    parser.add_argument("--catalogue", required=True, type=Path)
    parser.add_argument("--orf-export", action="append", type=Path, default=[])
    parser.add_argument("--comparison", action="append", type=Path, default=[])
    parser.add_argument("--event-aliases", type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = {p.resolve() for p in [args.catalogue, *args.orf_export, *args.comparison]
                  + ([args.event_aliases] if args.event_aliases else [])}
        outputs = {Path(str(args.output_prefix.expanduser()) + suffix).resolve() for suffix in
                   (".json", ".candidates.tsv", ".proteins.tsv", ".protein.fasta", ".html")}
        if inputs & outputs:
            raise ValueError("Report outputs must not overwrite an input")
        catalogue = json.loads(args.catalogue.read_text())
        aliases = json.loads(args.event_aliases.read_text()) if args.event_aliases else None
        report = build_sv_interest_report(
            catalogue, [json.loads(p.read_text()) for p in args.orf_export],
            comparisons=[json.loads(p.read_text()) for p in args.comparison], event_aliases=aliases)
        write_sv_interest_report(report, args.output_prefix)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
