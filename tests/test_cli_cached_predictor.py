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

    def test_file_and_directory_mutually_exclusive(self, tmp_path):
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        with pytest.raises(ValueError, match="mutually exclusive"):
            _run([
                "--peptide-csv", csv,
                "--mhc-cache-file", "a.out",
                "--mhc-cache-directory", "b",
                "--mhc-cache-format", "netmhcpan",
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
            "--mhc-cache-format", "netmhcpan",
        ])
        # NetMHCpan 4.1 -BA emits affinity + elution_score per
        # (peptide, allele) → 2 rows.
        assert len(df) == 2
        assert set(df["kind"]) == {"pMHC_affinity", "pMHC_presentation"}
        by_kind = df.set_index("kind")
        assert by_kind.loc["pMHC_affinity", "affinity"] == pytest.approx(
            8.82, rel=1e-3,
        )
        assert (df["prediction_method_name"] == "netmhcpan").all()
        assert (df["predictor_version"] == "4.1b").all()

    def test_netmhc_stdout_version_4(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhc_40_SLLQHLIGL.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhc",
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
            "--mhc-cache-format", "netmhcstabpan",
        ])
        assert len(df) == 1
        assert df.iloc[0]["prediction_method_name"] == "netmhcstabpan"

    def test_explicit_predictor_version_override(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan",
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
            "kind": "pMHC_affinity",
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
        """Generic TSV with a 'kind' column and non-canonical data-
        column names, mapped via --mhc-cache-tsv-column."""
        tsv_path = tmp_path / "third_party.tsv"
        tsv_path.write_text(
            "peptide\tallele\tIC50_nM\tRank\tkind\n"
            "SLLQHLIGL\tHLA-A*02:01\t125.0\t1.2\tpMHC_affinity\n"
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
        assert row["kind"] == "pMHC_affinity"

    def test_tsv_stability_kind(self, tmp_path):
        """TSV with a stability 'kind' value — DSL's Stability scope
        dispatches on it."""
        tsv_path = tmp_path / "stab.tsv"
        tsv_path.write_text(
            "peptide\tallele\tthalf_hours\tkind\n"
            "SLLQHLIGL\tHLA-A*02:01\t4.5\tpMHC_stability\n"
        )
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        out = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(tsv_path),
            "--mhc-cache-format", "tsv",
            "--mhc-cache-predictor-name", "custom-stab",
            "--mhc-cache-predictor-version", "1.0",
            "--mhc-cache-tsv-column", "value=thalf_hours",
        ])
        assert out.iloc[0]["kind"] == "pMHC_stability"

    def test_tsv_multi_kind_in_one_file(self, tmp_path):
        """Multi-kind TSV — both affinity and presentation rows for the
        same (peptide, allele) coexist via the 4-tuple index key."""
        tsv_path = tmp_path / "multi.tsv"
        tsv_path.write_text(
            "peptide\tallele\tkind\taffinity\tpercentile_rank\tscore\n"
            "SLLQHLIGL\tHLA-A*02:01\tpMHC_affinity\t42.0\t1.0\t\n"
            "SLLQHLIGL\tHLA-A*02:01\tpMHC_presentation\t\t0.5\t0.85\n"
        )
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        out = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(tsv_path),
            "--mhc-cache-format", "tsv",
            "--mhc-cache-predictor-name", "multi-kind-tool",
            "--mhc-cache-predictor-version", "1.0",
        ])
        assert len(out) == 2
        assert set(out["kind"]) == {"pMHC_affinity", "pMHC_presentation"}


# ---------------------------------------------------------------------------
# CLI directory sharding + filter/sort interaction
# ---------------------------------------------------------------------------


class TestCachedPredictorCliSharding:
    def test_mhc_cache_directory_merges_shards(self, tmp_path):
        """--mhc-cache-directory loads every matching file and concats them."""
        import pandas as pd
        from topiary import CachedPredictor

        def _make_shard(peptide, affinity):
            df = pd.DataFrame([{
                "peptide": peptide,
                "allele": "HLA-A*02:01",
                "peptide_length": 9,
                "kind": "pMHC_affinity",
                "affinity": affinity,
                "score": 0.5,
                "percentile_rank": 1.0,
                "prediction_method_name": "synthetic",
                "predictor_version": "1.0",
            }])
            return CachedPredictor.from_dataframe(df)

        cache_dir = tmp_path / "caches"
        cache_dir.mkdir()
        _make_shard("SLLQHLIGL", 42.0).save(cache_dir / "a.tsv")
        _make_shard("GILGFVFTL", 100.0).save(cache_dir / "b.tsv")

        # Peptides CSV covering BOTH shards
        pep_csv = tmp_path / "pep.csv"
        pep_csv.write_text("peptide\nSLLQHLIGL\nGILGFVFTL\n")

        df = _run([
            "--peptide-csv", str(pep_csv),
            "--mhc-cache-directory", str(cache_dir),
            "--mhc-cache-directory-pattern", "*.tsv",
        ])
        assert len(df) == 2
        assert set(df["peptide"]) == {"SLLQHLIGL", "GILGFVFTL"}
        by_pep = df.set_index("peptide")
        assert by_pep.loc["SLLQHLIGL", "affinity"] == 42.0
        assert by_pep.loc["GILGFVFTL", "affinity"] == 100.0


@pytest.mark.skipif(not _HAS_FIXTURES, reason="NetMHC fixtures missing")
class TestCachedPredictorCliFilterSort:
    def test_filter_by_applies_to_cached_predictions(self, tmp_path):
        """--filter-by runs on the joined cached-prediction DataFrame.
        DSL filters on the `affinity` column work against multi-kind
        cached output."""
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        # Permissive filter keeps rows where affinity passes (the
        # presentation row has affinity=NaN which is NaN-tolerant in
        # topiary's filter semantics — see
        # feedback_apply_filter_non_boolean).
        df = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan",
            "--filter-by", "affinity <= 500",
        ])
        assert len(df) >= 1
        # The affinity row (pMHC_affinity with affinity=8.82) survives.
        affinity_rows = df[df["kind"] == "pMHC_affinity"]
        assert len(affinity_rows) == 1
        assert affinity_rows.iloc[0]["affinity"] <= 500
        # Restrictive filter drops the affinity row too.
        df_empty = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan",
            "--filter-by", "affinity <= 1",
        ])
        assert not (df_empty["kind"] == "pMHC_affinity").any()

    def test_sort_by_applies_to_cached_predictions(self, tmp_path):
        """--sort-by runs on the joined cached-prediction DataFrame."""
        fixture = _FIXTURE_DIR / "netmhc_40_SLLQHLIGL.out"  # multi-allele
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhc",
            "--mhc-cache-netmhc-version", "4",
            "--sort-by", "affinity",  # ascending — strongest binder first
        ])
        assert len(df) == 3
        # Strongest binder is A*02:01 (~13 nM); sort pushes it to the top.
        assert df.iloc[0]["allele"] == "HLA-A*02:01"


# ---------------------------------------------------------------------------
# Auto-sniff: --mhc-cache-format can be omitted for formats with
# identifying content (NetMHC family, mhcflurry, topiary output).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_FIXTURES, reason="NetMHC fixtures missing")
class TestCachedPredictorCliAutoSniff:
    def test_netmhcpan_autodetected(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        # No --mhc-cache-format — sniffed from the "NetMHCpan version 4.1b"
        # preamble line.
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
        ])
        # NetMHCpan 4.1 -BA emits affinity + elution_score → 2 rows.
        assert len(df) == 2
        assert (df["prediction_method_name"] == "netmhcpan").all()
        assert set(df["kind"]) == {"pMHC_affinity", "pMHC_presentation"}

    def test_netmhcstabpan_autodetected(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhcstabpan_SLLQHLIGL_A0201.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
        ])
        assert len(df) == 1
        assert df.iloc[0]["prediction_method_name"] == "netmhcstabpan"

    def test_netmhc_classic_autodetected(self, tmp_path):
        fixture = _FIXTURE_DIR / "netmhc_40_SLLQHLIGL.out"
        csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        df = _run([
            "--peptide-csv", csv,
            "--mhc-cache-file", str(fixture),
        ])
        # NetMHC classic — multi-allele file → 3 rows
        assert len(df) == 3

    def test_topiary_output_autodetected(self, tmp_path):
        import pandas as pd
        from topiary import CachedPredictor

        df = pd.DataFrame([{
            "peptide": "SLLQHLIGL",
            "allele": "HLA-A*02:01",
            "peptide_length": 9,
            "affinity": 42.0,
            "score": 0.5,
            "percentile_rank": 1.0,
            "kind": "pMHC_affinity",
            "prediction_method_name": "synthetic",
            "predictor_version": "1.0",
        }])
        cache = CachedPredictor.from_dataframe(df)
        cache.save(tmp_path / "cache.tsv")
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        # No --mhc-cache-format — sniffed from the column headers.
        out = _run([
            "--peptide-csv", pep_csv,
            "--mhc-cache-file", str(tmp_path / "cache.tsv"),
        ])
        assert len(out) == 1
        assert out.iloc[0]["affinity"] == 42.0

    def test_tsv_not_autodetected(self, tmp_path):
        """Generic TSV has no identifying content; must be explicit."""
        tsv_path = tmp_path / "third_party.tsv"
        tsv_path.write_text(
            "peptide\tallele\tIC50_nM\n"
            "SLLQHLIGL\tHLA-A*02:01\t42.0\n"
        )
        pep_csv = _write_peptide_csv(tmp_path / "pep.csv", "SLLQHLIGL")
        with pytest.raises(ValueError, match="auto-detect"):
            _run([
                "--peptide-csv", pep_csv,
                "--mhc-cache-file", str(tsv_path),
            ])
