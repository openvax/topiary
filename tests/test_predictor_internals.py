"""Direct unit tests for TopiaryPredictor private methods."""

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import TopiaryPredictor
from topiary.predictor import _attach_expression_data


# ---------------------------------------------------------------------------
# _format_prediction_df
# ---------------------------------------------------------------------------


def _make_predictor():
    return TopiaryPredictor(models=RandomBindingPredictor, alleles=["A0201"])


def _raw_df(**overrides):
    """Minimal DataFrame mimicking mhctools output before formatting."""
    row = dict(
        peptide="SIINFEKL",
        allele="HLA-A*02:01",
        kind="pMHC_affinity",
        score=0.8,
        value=120.0,
        percentile_rank=0.5,
        offset=10,
        predictor_name="random",
    )
    row.update(overrides)
    return pd.DataFrame([row])


def test_format_renames_columns():
    df = _raw_df()
    result = _make_predictor()._format_prediction_df(df)
    assert "peptide_offset" in result.columns
    assert "prediction_method_name" in result.columns
    assert "offset" not in result.columns
    assert "predictor_name" not in result.columns


def test_format_adds_peptide_length():
    df = _raw_df(peptide="SIINFEKLAA")
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["peptide_length"] == 10


def test_format_adds_source_sequence_name_if_missing():
    df = _raw_df()
    df = df.drop(columns=["source_sequence_name"], errors="ignore")
    result = _make_predictor()._format_prediction_df(df)
    assert "source_sequence_name" in result.columns


def test_format_adds_peptide_offset_if_missing():
    df = _raw_df()
    df = df.drop(columns=["offset"], errors="ignore")
    df = df.drop(columns=["peptide_offset"], errors="ignore")
    result = _make_predictor()._format_prediction_df(df)
    assert "peptide_offset" in result.columns
    assert result.iloc[0]["peptide_offset"] == 0


def test_format_affinity_for_affinity_kind():
    df = _raw_df(kind="pMHC_affinity", value=85.0)
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["affinity"] == 85.0


def test_format_affinity_nan_for_non_affinity_kind():
    df = _raw_df(kind="pMHC_presentation", value=0.95)
    result = _make_predictor()._format_prediction_df(df)
    assert np.isnan(result.iloc[0]["affinity"])


def test_format_preserves_existing_affinity():
    df = _raw_df()
    df["affinity"] = 999.0
    result = _make_predictor()._format_prediction_df(df)
    assert result.iloc[0]["affinity"] == 999.0


def test_format_empty_df():
    df = pd.DataFrame()
    result = _make_predictor()._format_prediction_df(df)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# _attach_expression_data: additional edge cases
# ---------------------------------------------------------------------------


def _make_prediction_df():
    return pd.DataFrame([
        dict(
            source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            gene_id="ENSG00000157764", transcript_id="ENST00000288602",
            variant="chr7 g.140753336A>T",
        ),
    ])


def test_attach_empty_expression_noop():
    df = _make_prediction_df()
    n_cols = len(df.columns)
    result = _attach_expression_data(df, {"gene": [], "transcript": [], "variant": []})
    assert len(result.columns) == n_cols


def test_attach_duplicate_gene_ids_summed():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764", "ENSG00000157764"],
        "TPM": [20.0, 22.5],
    })
    expr_data = {"gene": [("gene", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "gene_tpm" in result.columns
    assert result.iloc[0]["gene_tpm"] == pytest.approx(42.5)


def test_attach_prefix_lowercasing():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764"],
        "TPM": [42.5],
        "NumReads": [1000],
    })
    expr_data = {"gene": [("salmon", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "salmon_tpm" in result.columns
    assert "salmon_numreads" in result.columns


def test_attach_no_prefix():
    """When name_prefix is None, columns keep original (lowercased) names."""
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "variant": ["chr7 g.140753336A>T"],
        "num_alt_reads": [15],
    })
    expr_data = {"gene": [], "transcript": [], "variant": [(None, "variant", expr_df)]}
    result = _attach_expression_data(df, expr_data)
    assert "num_alt_reads" in result.columns
