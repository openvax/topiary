"""Combine every successful test-matrix artifact into one Cobertura report.

Unlike coverage combine's warning-and-skip behavior, missing, empty or corrupt
artifacts are fatal: publishing incomplete coverage would hide CI failures.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from coverage import Coverage, CoverageData


def combine_coverage(directory, expected_count, output):
    """Validate the complete matrix, merge its line/branch data, then write XML."""
    paths = sorted(directory.glob("coverage-python-*/.coverage"))
    if expected_count < 1 or len(paths) != expected_count:
        raise ValueError(
            f"Expected {expected_count} coverage artifacts, found {len(paths)} in {directory}"
        )

    with TemporaryDirectory(prefix="topiary-coverage-") as temporary:
        report = Coverage(config_file=False, data_file=str(Path(temporary) / ".coverage"))
        for path in paths:
            data = CoverageData(basename=str(path))
            data.read()
            if not data.measured_files():
                raise ValueError(f"Empty coverage artifact: {path}")
            report.get_data().update(data)
        report.xml_report(outfile=str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("expected_count", type=int)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    combine_coverage(args.directory, args.expected_count, args.output)
