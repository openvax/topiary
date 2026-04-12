"""Tests for topiary.io — read/write with comment-block metadata."""

import numpy as np
import pandas as pd
import pytest

from topiary.io import (
    Metadata,
    _format_comment_block,
    _parse_comment_block,
    read_csv,
    read_tsv,
    to_csv,
    to_tsv,
)
from topiary.wide import to_wide


# ---------------------------------------------------------------------------
# Comment block parsing
# ---------------------------------------------------------------------------


class TestParseCommentBlock:
    def test_well_formed(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#form=long\n",
            "#model:netmhcpan=4.1b\n",
            "#model:mhcflurry=2.1.1\n",
            "peptide\tallele\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 4
        assert meta.topiary_version == "4.11.0"
        assert meta.form == "long"
        assert meta.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}

    def test_unknown_keys_in_extra(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#custom_key=custom_value\n",
            "#other_key=other_value\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 3
        assert meta.extra == {"custom_key": "custom_value", "other_key": "other_value"}

    def test_source_lines(self):
        lines = [
            "#source=patient01.tsv\n",
            "#source=patient02.tsv\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 2
        assert meta.sources == ["patient01.tsv", "patient02.tsv"]

    def test_no_comments(self):
        lines = ["peptide\tallele\n", "SIINFEKL\tHLA-A*02:01\n"]
        meta, n = _parse_comment_block(lines)
        assert n == 0
        assert meta.topiary_version is None
        assert meta.models == {}

    def test_empty_lines(self):
        meta, n = _parse_comment_block([])
        assert n == 0

    def test_malformed_comment_skipped(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#no-equals-sign\n",
            "#form=wide\n",
            "data\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 3
        assert meta.topiary_version == "4.11.0"
        assert meta.form == "wide"

    def test_model_entries(self):
        lines = [
            "#model:netmhcpan=4.1b\n",
            "#model:mhcflurry=2.1.1\n",
            "#model:netmhcstabpan=1.0\n",
            "data\n",
        ]
        meta, _ = _parse_comment_block(lines)
        assert len(meta.models) == 3
        assert meta.models["netmhcstabpan"] == "1.0"


# ---------------------------------------------------------------------------
# Comment block formatting
# ---------------------------------------------------------------------------


class TestFormatCommentBlock:
    def test_full_metadata(self):
        meta = Metadata(
            topiary_version="4.11.0",
            form="long",
            models={"netmhcpan": "4.1b"},
        )
        block = _format_comment_block(meta)
        assert "#topiary_version=4.11.0" in block
        assert "#form=long" in block
        assert "#model:netmhcpan=4.1b" in block

    def test_empty_metadata(self):
        meta = Metadata()
        block = _format_comment_block(meta)
        assert block == ""

    def test_extra_keys_preserved(self):
        meta = Metadata(extra={"custom_key": "custom_value"})
        block = _format_comment_block(meta)
        assert "#custom_key=custom_value" in block

    def test_sources_formatted(self):
        meta = Metadata(sources=["patient01.tsv", "patient02.tsv"])
        block = _format_comment_block(meta)
        assert "#source=patient01.tsv" in block
        assert "#source=patient02.tsv" in block

    def test_format_parse_roundtrip(self):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            sources=["lens-v1.9.tsv"],
            extra={"custom_key": "custom_value"},
        )
        block = _format_comment_block(meta)
        lines = [line + "\n" for line in block.split("\n")] + ["data\n"]
        parsed, _ = _parse_comment_block(lines)
        assert parsed.topiary_version == meta.topiary_version
        assert parsed.form == meta.form
        assert dict(parsed.models) == dict(meta.models)
        assert parsed.sources == meta.sources
        assert dict(parsed.extra) == dict(meta.extra)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_long_df():
    return pd.DataFrame([
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=3, peptide_length=8,
            kind="pMHC_affinity", score=0.85, value=120.0,
            percentile_rank=0.5, affinity=120.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
        dict(
            peptide="ELAGIGILT", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=10, peptide_length=9,
            kind="pMHC_affinity", score=0.3, value=5000.0,
            percentile_rank=15.0, affinity=5000.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
    ])


# ---------------------------------------------------------------------------
# Read/write round-trip tests
# ---------------------------------------------------------------------------


class TestReadWriteTSV:
    def test_long_form_roundtrip(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.tsv"
        to_tsv(df, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "long"
        assert meta.topiary_version is not None
        assert meta.models.get("netmhcpan") == "4.1b"
        assert len(df2) == len(df)
        assert list(df2["peptide"]) == list(df["peptide"])
        assert df2.iloc[0]["value"] == pytest.approx(120.0)

    def test_wide_form_roundtrip(self, tmp_path):
        df = _sample_long_df()
        wide = to_wide(df)
        path = tmp_path / "out.wide.tsv"
        to_tsv(wide, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "wide"
        assert "netmhcpan_affinity_value" in df2.columns
        assert len(df2) == len(wide)

    def test_metadata_preserved(self, tmp_path):
        df = _sample_long_df()
        meta = Metadata(
            sources=["test_cohort"],
            extra={"patient": "PT01"},
        )
        path = tmp_path / "out.tsv"
        to_tsv(df, path, metadata=meta)
        meta2 = read_tsv(path).metadata
        assert "test_cohort" in meta2.sources
        assert meta2.extra.get("patient") == "PT01"

    def test_model_versions_auto_extracted(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.tsv"
        to_tsv(df, path)
        meta = read_tsv(path).metadata
        assert meta.models.get("netmhcpan") == "4.1b"


class TestReadWriteCSV:
    def test_csv_roundtrip(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.csv"
        to_csv(df, path)
        result = read_csv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "long"
        assert len(df2) == len(df)
        assert df2.iloc[0]["value"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_file_without_comments(self, tmp_path):
        path = tmp_path / "plain.tsv"
        df = pd.DataFrame({"peptide": ["SIINFEKL"], "allele": ["A"]})
        df.to_csv(path, sep="\t", index=False)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.topiary_version is None
        assert meta.models == {}
        assert len(df2) == 1

    def test_empty_df_roundtrip(self, tmp_path):
        df = pd.DataFrame(columns=["peptide", "allele", "kind"])
        path = tmp_path / "empty.tsv"
        to_tsv(df, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert len(df2) == 0
        assert meta.form == "long"

    def test_header_only_file(self, tmp_path):
        path = tmp_path / "header_only.tsv"
        with open(path, "w") as f:
            f.write("#topiary_version=4.11.0\n")
            f.write("#form=long\n")
            f.write("peptide\tallele\tkind\n")
        result = read_tsv(path)
        df, meta = result.df, result.metadata
        assert len(df) == 0
        assert meta.topiary_version == "4.11.0"

    def test_all_metadata_fields(self, tmp_path):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            sources=["lens-v1.9.tsv"],
            extra={"patient": "PT01"},
        )
        df = pd.DataFrame({"peptide": ["A"], "netmhcpan_affinity_value": [100]})
        path = tmp_path / "full.tsv"
        to_tsv(df, path, metadata=meta)
        result = read_tsv(path)
        meta2 = result.metadata
        assert meta2.topiary_version == "4.11.0"
        assert meta2.form == "wide"
        assert meta2.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
        # The read function appends the filename to sources too
        assert "lens-v1.9.tsv" in meta2.sources
        assert meta2.extra == {"patient": "PT01"}

    def test_pandas_read_csv_with_comment_hash(self, tmp_path):
        """Standard pandas can still read our files (losing metadata)."""
        df = _sample_long_df()
        path = tmp_path / "compat.tsv"
        to_tsv(df, path)
        df2 = pd.read_csv(path, sep="\t", comment="#")
        assert len(df2) == 2
        assert "peptide" in df2.columns

    def test_write_index_false_by_default(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "no_idx.tsv"
        to_tsv(df, path)
        with open(path) as f:
            lines = f.readlines()
        # First non-comment line is the header; should not start with ","
        for line in lines:
            if not line.startswith("#"):
                assert not line.startswith(",")
                assert not line.startswith("\t,")
                break


# ---------------------------------------------------------------------------
# Full round-trip integration tests with real predictor
# ---------------------------------------------------------------------------


class TestRoundTripIntegration:
    """End-to-end: predict → to_wide → to_tsv → read_tsv → from_wide → verify."""

    def test_predict_to_wide_tsv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor
        from topiary.wide import to_wide, from_wide

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGGLLLAAA"}
        )
        assert len(long_orig) > 0

        # Long → wide
        wide = to_wide(long_orig)
        assert "kind" not in wide.columns

        # Wide → TSV → read back
        path = tmp_path / "roundtrip.wide.tsv"
        to_tsv(wide, path)
        read_result = read_tsv(path)
        wide_read, meta = read_result.df, read_result.metadata
        assert meta.form == "wide"
        assert len(wide_read) == len(wide)

        # Read back → long
        long_back = from_wide(wide_read, metadata=meta)
        assert "kind" in long_back.columns
        assert len(long_back) == len(long_orig)

        # Values round-trip correctly
        orig_values = sorted(long_orig["value"].dropna().tolist())
        back_values = sorted(long_back["value"].dropna().tolist())
        assert orig_values == pytest.approx(back_values)

    def test_predict_long_tsv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGGLLLAAA"}
        )

        # Long → TSV → read back
        path = tmp_path / "roundtrip.tsv"
        to_tsv(long_orig, path)
        read_result = read_tsv(path)
        long_read, meta = read_result.df, read_result.metadata
        assert meta.form == "long"
        assert len(long_read) == len(long_orig)
        assert list(long_read["peptide"]) == list(long_orig["peptide"])

    def test_predict_wide_csv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor
        from topiary.wide import to_wide, from_wide

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201", "B0702"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"braf": "MASIINFEKLGGG", "tp53": "MRKKLLQQREEY"}
        )
        wide = to_wide(long_orig)

        path = tmp_path / "roundtrip.wide.csv"
        to_csv(wide, path)
        read_result = read_csv(path)
        wide_read, meta = read_result.df, read_result.metadata
        long_back = from_wide(wide_read, metadata=meta)

        assert set(long_back["kind"].unique()) == set(long_orig["kind"].unique())
        assert len(long_back) == len(long_orig)

    def test_from_wide_to_wide_roundtrip(self, tmp_path):
        """Reverse direction: from_wide then to_wide should be identity."""
        from topiary.wide import to_wide, from_wide

        wide_orig = pd.DataFrame({
            "peptide": ["SIINFEKL", "ELAGIGILT"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "source_sequence_name": ["prot1", "prot1"],
            "netmhcpan_affinity_value": [120.0, 5000.0],
            "netmhcpan_affinity_score": [0.85, 0.3],
            "netmhcpan_affinity_rank": [0.5, 15.0],
        })
        long = from_wide(wide_orig)
        wide_back = to_wide(long)

        assert "netmhcpan_affinity_value" in wide_back.columns
        assert len(wide_back) == 2
        vals = sorted(wide_back["netmhcpan_affinity_value"].tolist())
        assert vals == pytest.approx([120.0, 5000.0])

    def test_sample_name_survives_roundtrip(self):
        """sample_name column from mhctools should survive wide/long."""
        from topiary.wide import to_wide, from_wide

        df = pd.DataFrame([
            dict(
                peptide="SIINFEKL", allele="HLA-A*02:01",
                sample_name="patient_01",
                kind="pMHC_affinity", score=0.85, value=120.0,
                percentile_rank=0.5, affinity=120.0,
                prediction_method_name="netmhcpan", predictor_version="4.1b",
            ),
        ])
        wide = to_wide(df)
        assert "sample_name" in wide.columns
        long = from_wide(wide)
        assert "sample_name" in long.columns
        assert long.iloc[0]["sample_name"] == "patient_01"

    def test_model_versions_in_metadata_after_write(self, tmp_path):
        """Model versions auto-extracted on write, available on read."""
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
        path = tmp_path / "models.tsv"
        to_tsv(df, path)
        meta = read_tsv(path).metadata
        assert len(meta.models) > 0
