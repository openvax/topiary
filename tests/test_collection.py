"""Collecting tests must not load reference data for deselected tests."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("selection", [
    ["--collect-only"],
    ["-m", "isovar", "--strict-markers"],
])
def test_collection_and_marker_selection_without_reference_data(tmp_path, selection):
    """Run the whole collection, with empty caches and forbidden I/O."""
    cache = tmp_path / "ensembl"
    cache.mkdir()
    code = textwrap.dedent("""
        import sys
        from pathlib import Path

        def forbid_network(event, args):
            if event in {"socket.connect", "socket.getaddrinfo"}:
                raise AssertionError("Network access during collection/Isovar tests")

        sys.addaudithook(forbid_network)
        import pyensembl

        def forbid_annotation(*args, **kwargs):
            raise AssertionError("Unselected Ensembl annotation was accessed")

        # Selected real-read tests may index their checked-in, checksum-pinned
        # subset in a temporary directory. No reference access at all is
        # allowed during collection, and full-genome lookup/download stays
        # forbidden in both modes.
        reference = Path("tests/data/osteosarc/protein_reference/reference.gtf.gz").resolve()
        original_db = pyensembl.Genome.db.fget
        original_index = pyensembl.Genome.index

        def require_pinned_reference(genome):
            assert "--collect-only" not in sys.argv, "Reference access during collection"
            assert genome.reference_name == "GRCh38-osteosarc-six-transcript-subset"
            assert Path(genome.to_dict()["gtf_path_or_url"]).resolve() == reference

        def pinned_db(genome):
            require_pinned_reference(genome)
            return original_db(genome)

        def pinned_index(genome, *args, **kwargs):
            require_pinned_reference(genome)
            return original_index(genome, *args, **kwargs)

        pyensembl.Genome.db = property(pinned_db)
        pyensembl.Genome.download = forbid_annotation
        pyensembl.Genome.index = pinned_index

        import pytest
        raise SystemExit(pytest.main(["tests", "-q", *sys.argv[1:]]))
    """)
    env = dict(os.environ, PYENSEMBL_CACHE_DIR=str(cache))
    # The child selects tests explicitly; do not inherit xdist/coverage options
    # or pytest-cov's subprocess instrumentation from its parent test run.
    for name in list(env):
        if name in {"PYTEST_ADDOPTS", "COVERAGE_PROCESS_START"} or name.startswith("COV_CORE_"):
            env.pop(name)
    result = subprocess.run(
        [sys.executable, "-c", code, *selection],
        cwd=Path(__file__).resolve().parents[1], env=env,
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not list(cache.rglob("*")), "Collection/Isovar selection wrote reference data"
    expected = "collected" if "--collect-only" in selection else "deselected"
    assert expected in result.stdout


def test_data_path_import_does_not_import_annotation_libraries():
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            from pathlib import Path
            from tests.data import data_path
            assert "varcode" not in sys.modules
            assert "pyensembl" not in sys.modules
            assert Path(data_path("")) == Path("tests/data").resolve()
        """)],
        cwd=Path(__file__).resolve().parents[1],
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
