"""Fail-closed release preflight: python -m topiary.cli.release PROJECT VERSION."""

import argparse
import sys

from ..release import pypi_release_exists


def main(argv=None):
    """Return zero only when PyPI confirms that the release does not exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("version")
    args = parser.parse_args(argv)
    try:
        exists = pypi_release_exists(args.project, args.version)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if exists:
        print(f"ERROR: {args.project} {args.version} already exists on PyPI", file=sys.stderr)
        return 1
    print(f"Confirmed {args.project} {args.version} is not published on PyPI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
