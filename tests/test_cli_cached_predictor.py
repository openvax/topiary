"""CLI-level tests for --mhc-cache-* arguments.

Exercise the end-to-end CLI path through topiary's argument parser and
predict_epitopes_from_args, confirming that each supported
--mhc-cache-format wires into the right CachedPredictor loader and
that the output DataFrame carries the expected prediction values.

Uses the real NetMHC-family stdout fixtures in tests/data/netmhc_fixtures/
(captured from netmhc-bundle binaries on peptide SLLQHLIGL × a few
HLA alleles) so the tests verify both the CLI glue and the parsing
chain through mhctools.
"""
import pathlib

import pytest

from topiary.cli.args import arg_parser, predict_epitopes_from_args


_FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "netmhc_fixtures"
_HAS_FIXTURES = _FIXTURE_DIR.exists()


def _run(cli_args):
    """Parse a list of CLI tokens and run the prediction pipeline."""
    args = arg_parser.parse_args(cli_args)
    return predict_epitopes_from_args(args)


def _write_peptide_csv(path, peptide):
    """Write a one-peptide CSV at ``path`` and return its string path."""
    path.write_text(f"peptide\n{peptide}\n")
    return str(path)


# ---------------------------------------------------------------------------
# Error paths: CLI validation
# ---------------------------------------------------------------------------


class TestCachedPredictorCliErrors:
    def test_no_predictor_no_cache_raises(self, tmp_path):
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        with pytest.raises(ValueError, match="--mhc-predictor"):
            _run(["--peptide-csv", csv])

    def test_cache_file_without_format_raises(self, tmp_path):
        pytest.importorskip("mhctools")
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        if not fixture.exists():
            pytest.skip("fixture missing")
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        with pytest.raises(ValueError, match="--mhc-cache-format"):
            _run([
                "--peptide-csv", csv,
                "--mhc-cache-file", str(fixture),
            ])

    def test_file_and_directory_mutually_exclusive(self, tmp_path):
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _run([
                "--peptide-csv", csv,
                "--mhc-cache-file", "a.out",
                "--mhc-cache-directory", "b",
                "--mhc-cache-format", "netmhcpan_stdout",
            ])


# ---------------------------------------------------------------------------
# Happy-path runs through each NetMHC-family fixture
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FIXTURES, reason="NetMHC fixtures missing")
class TestCachedPredictorCliHappyPath:
    def test_netmhcpan_stdout_wires_cli_to_loader(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan_stdout",
        ])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["peptide"] == "SLLQHLIGL"
        assert row["allele"] == "HLA-A*02:01"
        assert row["affinity"] == pytest.approx(8.82, rel=1e-3)
        assert row["prediction_method_name"] == "netmhcpan"
        assert row["predictor_version"] == "4.1b"

    def test_netmhc_stdout_version_4(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhc_40_SLLQHLIGL.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhc_stdout",
            "--mhc-cache-netmhc-version", "4",
        ])
        # Multi-allele fixture → one row per (peptide, allele)
        assert set(df["allele"]) == {
            "HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02",
        }
        assert (df["prediction_method_name"] == "netmhc").all()

    def test_netmhcstabpan_stdout(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcstabpan_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcstabpan_stdout",
        ])
        assert len(df) == 1
        assert df.iloc[0]["prediction_method_name"] == "netmhcstabpan"

    def test_explicit_predictor_version_override(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan_stdout",
            "--mhc-cache-predictor-version", "my-custom-label",
        ])
        assert (df["predictor_version"] == "my-custom-label").all()


# ---------------------------------------------------------------------------
# Topiary-output round-trip via the CLI
# ---------------------------------------------------------------------------


class TestCachedPredictorCliTopiaryRoundtrip:
    def test_from_dataframe_written_by_save(self, tmp_path):
        """Save a CachedPredictor as TSV then reload via CLI."""
        import pandas as pd
        from topiary import CachedPredictor

        df = pd.DataFrame([{
            "peptide": "SLLQHLIGL",
            "allele": "HLA-A*02:01",
            "peptide_length": 9,
            "affinity": 42.0,
            "score": 0.8,
            "percentile_rank": 0.5,
            "prediction_method_name": "synthetic",
            "predictor_version": "1.0",
        }])
        cache = CachedPredictor.from_dataframe(df)
        cache_path = tmp_path / "cache.tsv"
        cache.save(cache_path)

        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        out = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(cache_path),
            "--mhc-cache-format", "topiary_output",
        ])
        assert len(out) == 1
        assert out.iloc[0]["affinity"] == 42.0
        assert out.iloc[0]["predictor_version"] == "1.0"


# ---------------------------------------------------------------------------
# Generic TSV loader with column mapping
# ---------------------------------------------------------------------------


class TestCachedPredictorCliTsv:
    def test_tsv_with_column_mapping(self, tmp_path):
        tsv_path = tmp_path / "third_party.tsv"
        tsv_path.write_text(
            "peptide\tallele\tIC50_nM\tRank\n"
            "SLLQHLIGL\tHLA-A*02:01\t125.0\t1.2\n"
        )
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        out = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(tsv_path),
            "--mhc-cache-format", "tsv",
            "--mhc-cache-predictor-name", "netchop",
            "--mhc-cache-predictor-version", "3.1",
            "--mhc-cache-tsv-column", "affinity=IC50_nM",
            "--mhc-cache-tsv-column", "percentile_rank=Rank",
        ])
        assert len(out) == 1
        row = out.iloc[0]
        assert row["affinity"] == 125.0
        assert row["percentile_rank"] == 1.2
        assert row["prediction_method_name"] == "netchop"
        assert row["predictor_version"] == "3.1"
