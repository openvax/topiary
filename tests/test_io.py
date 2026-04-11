"""Tests for topiary.io — read/write with comment-block metadata."""

import math

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
            "#source=lens-v1.9\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 3
        assert meta.extra == {"custom_key": "custom_value", "source": "lens-v1.9"}

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
        meta = Metadata(extra={"source": "lens-v1.9"})
        block = _format_comment_block(meta)
        assert "#source=lens-v1.9" in block

    def test_format_parse_roundtrip(self):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            extra={"source": "lens-v1.9"},
        )
        block = _format_comment_block(meta)
        lines = [line + "\n" for line in block.split("\n")] + ["data\n"]
        parsed, _ = _parse_comment_block(lines)
        assert parsed.topiary_version == meta.topiary_version
        assert parsed.form == meta.form
        assert dict(parsed.models) == dict(meta.models)
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
        df2, meta = read_tsv(path)
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
        df2, meta = read_tsv(path)
        assert meta.form == "wide"
        assert "netmhcpan_affinity_value" in df2.columns
        assert len(df2) == len(wide)

    def test_metadata_preserved(self, tmp_path):
        df = _sample_long_df()
        meta = Metadata(
            extra={"source": "test", "patient": "PT01"},
        )
        path = tmp_path / "out.tsv"
        to_tsv(df, path, metadata=meta)
        _, meta2 = read_tsv(path)
        assert meta2.extra.get("source") == "test"
        assert meta2.extra.get("patient") == "PT01"

    def test_model_versions_auto_extracted(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.tsv"
        to_tsv(df, path)
        _, meta = read_tsv(path)
        assert meta.models.get("netmhcpan") == "4.1b"


class TestReadWriteCSV:
    def test_csv_roundtrip(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.csv"
        to_csv(df, path)
        df2, meta = read_csv(path)
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
        df2, meta = read_tsv(path)
        assert meta.topiary_version is None
        assert meta.models == {}
        assert len(df2) == 1

    def test_empty_df_roundtrip(self, tmp_path):
        df = pd.DataFrame(columns=["peptide", "allele", "kind"])
        path = tmp_path / "empty.tsv"
        to_tsv(df, path)
        df2, meta = read_tsv(path)
        assert len(df2) == 0
        assert meta.form == "long"

    def test_header_only_file(self, tmp_path):
        path = tmp_path / "header_only.tsv"
        with open(path, "w") as f:
            f.write("#topiary_version=4.11.0\n")
            f.write("#form=long\n")
            f.write("peptide\tallele\tkind\n")
        df, meta = read_tsv(path)
        assert len(df) == 0
        assert meta.topiary_version == "4.11.0"

    def test_all_metadata_fields(self, tmp_path):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            extra={"source": "lens-v1.9", "patient": "PT01"},
        )
        df = pd.DataFrame({"peptide": ["A"], "netmhcpan_affinity_value": [100]})
        path = tmp_path / "full.tsv"
        to_tsv(df, path, metadata=meta)
        _, meta2 = read_tsv(path)
        assert meta2.topiary_version == "4.11.0"
        assert meta2.form == "wide"
        assert meta2.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
        assert meta2.extra == {"source": "lens-v1.9", "patient": "PT01"}

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
