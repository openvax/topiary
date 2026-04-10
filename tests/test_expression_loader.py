"""Tests for topiary.rna.expression_loader — generic expression data loading."""

import os
import math

import pandas as pd
import pytest

from topiary.rna.expression_loader import (
    detect_format,
    load_expression,
    load_expression_from_spec,
    parse_expression_spec,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def test_detect_format_salmon():
    assert detect_format(os.path.join(DATA_DIR, "salmon_quant.sf")) == "salmon"


def test_detect_format_kallisto():
    assert detect_format(os.path.join(DATA_DIR, "kallisto_abundance.tsv")) == "kallisto"


def test_detect_format_cufflinks():
    assert detect_format(os.path.join(DATA_DIR, "genes.fpkm_tracking")) == "cufflinks"


def test_detect_format_stringtie_gtf():
    path = os.path.join(DATA_DIR, "B16-StringTie-chr1-subset.gtf")
    if os.path.exists(path):
        assert detect_format(path) == "stringtie_gtf"


def test_detect_format_unknown():
    assert detect_format("random_file.txt") is None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_salmon_defaults():
    df = load_expression(os.path.join(DATA_DIR, "salmon_quant.sf"))
    assert "Name" in df.columns
    assert "TPM" in df.columns
    assert len(df) == 4
    assert df.iloc[0]["TPM"] == 42.5


def test_load_salmon_explicit_columns():
    df = load_expression(
        os.path.join(DATA_DIR, "salmon_quant.sf"),
        id_col="Name",
        val_cols="NumReads",
    )
    assert "NumReads" in df.columns
    assert "TPM" not in df.columns


def test_load_salmon_all_numeric():
    df = load_expression(
        os.path.join(DATA_DIR, "salmon_quant.sf"),
        id_col="Name",
        val_cols="*",
    )
    # Should have all numeric columns except Name
    assert "TPM" in df.columns
    assert "NumReads" in df.columns
    assert "Length" in df.columns


def test_load_kallisto():
    df = load_expression(os.path.join(DATA_DIR, "kallisto_abundance.tsv"))
    assert "target_id" in df.columns
    assert "tpm" in df.columns
    assert len(df) == 3


def test_load_isovar_multi_column():
    """Isovar TSV has many numeric columns per variant."""
    df = load_expression(
        os.path.join(DATA_DIR, "isovar_output.tsv"),
        id_col="variant",
        val_cols="*",
    )
    assert "variant" in df.columns
    assert "num_alt_reads" in df.columns
    assert "fraction_alt_reads" in df.columns
    assert "num_alt_fragments" in df.columns
    assert len(df) == 3


def test_load_isovar_single_column():
    df = load_expression(
        os.path.join(DATA_DIR, "isovar_output.tsv"),
        id_col="variant",
        val_cols="num_alt_reads",
    )
    assert list(df.columns) == ["variant", "num_alt_reads"]


def test_load_missing_column_raises():
    with pytest.raises(ValueError, match="not found"):
        load_expression(
            os.path.join(DATA_DIR, "salmon_quant.sf"),
            id_col="nonexistent",
        )


def test_load_missing_value_column_raises():
    with pytest.raises(ValueError, match="not found"):
        load_expression(
            os.path.join(DATA_DIR, "salmon_quant.sf"),
            id_col="Name",
            val_cols="nonexistent",
        )


# ---------------------------------------------------------------------------
# Spec parsing
# ---------------------------------------------------------------------------


def test_parse_spec_file_only():
    assert parse_expression_spec("quant.sf") == (None, "quant.sf", None, None)


def test_parse_spec_name_and_file():
    assert parse_expression_spec("gene_tpm:quant.sf") == (
        "gene_tpm", "quant.sf", None, None
    )


def test_parse_spec_full():
    assert parse_expression_spec("gene_tpm:quant.sf:Name:TPM") == (
        "gene_tpm", "quant.sf", "Name", "TPM"
    )


def test_parse_spec_file_with_path():
    name, path, id_col, val_col = parse_expression_spec("data/salmon/quant.sf")
    assert name is None
    assert path == "data/salmon/quant.sf"


def test_parse_spec_file_and_id_col():
    name, path, id_col, val_col = parse_expression_spec("quant.sf:Name")
    assert name is None
    assert path == "quant.sf"
    assert id_col == "Name"


# ---------------------------------------------------------------------------
# End-to-end: spec → load
# ---------------------------------------------------------------------------


def test_load_from_spec_salmon():
    spec = os.path.join(DATA_DIR, "salmon_quant.sf")
    name, id_col, df = load_expression_from_spec(spec, default_name="gene")
    assert name == "gene"
    assert id_col == "Name"
    assert "TPM" in df.columns
    assert len(df) == 4


def test_load_from_spec_with_name():
    spec = f"gene_tpm:{os.path.join(DATA_DIR, 'salmon_quant.sf')}:Name:TPM"
    name, id_col, df = load_expression_from_spec(spec)
    assert name == "gene_tpm"
    assert id_col == "Name"
    assert list(df.columns) == ["Name", "TPM"]
