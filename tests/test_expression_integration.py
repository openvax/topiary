"""Integration tests: expression data loading → DataFrame join → DSL access."""

import math

import pandas as pd
import pytest

from topiary.predictor import _attach_expression_data
from topiary.ranking import Column, parse


def _make_prediction_df():
    """Minimal prediction DataFrame with gene_id, transcript_id, variant."""
    return pd.DataFrame([
        dict(
            source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            gene_id="ENSG00000157764", transcript_id="ENST00000288602",
            variant="chr7 g.140753336A>T",
        ),
        dict(
            source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_presentation",
            score=0.92, value=None, percentile_rank=0.3,
            gene_id="ENSG00000157764", transcript_id="ENST00000288602",
            variant="chr7 g.140753336A>T",
        ),
        dict(
            source_sequence_name="var2", peptide="ELAGIGIL", peptide_offset=20,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.3, value=5000.0, percentile_rank=15.0,
            gene_id="ENSG00000141510", transcript_id="ENST00000269305",
            variant="chr17 g.7577121G>A",
        ),
    ])


# ---------------------------------------------------------------------------
# Gene-level expression join
# ---------------------------------------------------------------------------


def test_attach_gene_expression():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764", "ENSG00000141510"],
        "TPM": [42.5, 15.3],
    })
    expr_data = {"gene": [("gene", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "gene_tpm" in result.columns
    # Both rows for var1 should have the same gene TPM
    assert result.iloc[0]["gene_tpm"] == 42.5
    assert result.iloc[1]["gene_tpm"] == 42.5
    assert result.iloc[2]["gene_tpm"] == 15.3


def test_attach_gene_expression_missing_gene_nan():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "gene_id": ["ENSG00000157764"],  # only one gene
        "TPM": [42.5],
    })
    expr_data = {"gene": [("gene", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert result.iloc[0]["gene_tpm"] == 42.5
    assert math.isnan(result.iloc[2]["gene_tpm"])


# ---------------------------------------------------------------------------
# Transcript-level expression join
# ---------------------------------------------------------------------------


def test_attach_transcript_expression():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "target_id": ["ENST00000288602", "ENST00000269305"],
        "tpm": [25.3, 12.1],
    })
    expr_data = {"gene": [], "transcript": [("tx", "target_id", expr_df)], "variant": []}
    result = _attach_expression_data(df, expr_data)
    assert "tx_tpm" in result.columns
    assert result.iloc[0]["tx_tpm"] == 25.3


# ---------------------------------------------------------------------------
# Variant-level expression join (isovar-style multi-column)
# ---------------------------------------------------------------------------


def test_attach_variant_expression_multi_column():
    df = _make_prediction_df()
    expr_df = pd.DataFrame({
        "variant": ["chr7 g.140753336A>T", "chr17 g.7577121G>A"],
        "num_alt_reads": [15, 8],
        "fraction_alt_reads": [0.143, 0.145],
    })
    expr_data = {"gene": [], "transcript": [], "variant": [(None, "variant", expr_df)]}
    result = _attach_expression_data(df, expr_data)
    assert "num_alt_reads" in result.columns
    assert "fraction_alt_reads" in result.columns
    assert result.iloc[0]["num_alt_reads"] == 15
    assert result.iloc[2]["num_alt_reads"] == 8


# ---------------------------------------------------------------------------
# Missing join column
# ---------------------------------------------------------------------------


def test_missing_join_column_warns(caplog):
    df = pd.DataFrame({
        "peptide": ["AAA"], "allele": ["A"], "kind": ["pMHC_affinity"],
        "score": [0.5], "value": [100.0], "percentile_rank": [1.0],
    })
    expr_df = pd.DataFrame({"gene_id": ["G1"], "TPM": [10.0]})
    expr_data = {"gene": [("gene", "gene_id", expr_df)], "transcript": [], "variant": []}
    result = _attach_expression_data(df, expr_data)
    # Should warn and return df unchanged (no gene_id column to join on)
    assert "gene_tpm" not in result.columns


# ---------------------------------------------------------------------------
# DSL access after join
# ---------------------------------------------------------------------------


def test_expression_column_in_dsl():
    df = _make_prediction_df()
    df["gene_tpm"] = [42.5, 42.5, 15.3]

    expr = parse("gene_tpm")
    assert isinstance(expr, Column)
    assert expr.evaluate(df) == 42.5


def test_expression_column_with_transform():
    df = _make_prediction_df()
    df["gene_tpm"] = [42.5, 42.5, 15.3]

    expr = parse("gene_tpm.log()")
    val = expr.evaluate(df)
    assert abs(val - math.log(42.5)) < 1e-9


def test_expression_in_composite_ranking():
    df = _make_prediction_df()
    df["gene_tpm"] = [42.5, 42.5, 15.3]

    expr = parse("0.5 * affinity.score + 0.5 * gene_tpm.log().ascending_cdf(2, 1)")
    val = expr.evaluate(df)
    assert isinstance(val, float)
    assert not math.isnan(val)


def test_isovar_columns_in_dsl():
    df = _make_prediction_df()
    df["num_alt_reads"] = [15, 15, 8]
    df["fraction_alt_reads"] = [0.143, 0.143, 0.145]

    expr = parse("num_alt_reads")
    assert expr.evaluate(df) == 15.0

    expr2 = parse("fraction_alt_reads")
    assert abs(expr2.evaluate(df) - 0.143) < 1e-6


def test_expression_column_repr_roundtrip():
    """Unknown identifiers produce Column with round-trippable repr."""
    expr = parse("gene_tpm")
    assert repr(expr) == "column(gene_tpm)"
    expr2 = parse(repr(expr))
    assert isinstance(expr2, Column)
    assert expr2.col_name == "gene_tpm"


def test_expression_transform_repr_roundtrip():
    expr = parse("gene_tpm.log()")
    text = repr(expr)
    assert "gene_tpm" in text
    assert "log" in text
    # Round-trip
    expr2 = parse(text)
    df = _make_prediction_df()
    df["gene_tpm"] = [42.5, 42.5, 15.3]
    assert abs(expr.evaluate(df) - expr2.evaluate(df)) < 1e-9


# ---------------------------------------------------------------------------
# The documented CLI examples produce the column the filter hints name (#571)
# ---------------------------------------------------------------------------


def _gene_expression_help_examples():
    """Every ``--gene-expression SPEC`` example in that option's help text."""
    import re

    from topiary.cli.args import create_arg_parser

    parser = create_arg_parser()
    action = next(
        action for action in parser._actions
        if "--gene-expression" in action.option_strings
    )
    return re.findall(r"--gene-expression (\S+)", action.help)


def test_gene_expression_help_examples_yield_the_gene_tpm_filter_column():
    import os

    from topiary.cli.args import create_arg_parser
    from topiary.cli.rna import expression_data_from_args
    from topiary.ranking import apply_filter

    salmon = os.path.join(os.path.dirname(__file__), "data", "salmon_quant.sf")
    examples = _gene_expression_help_examples()
    assert len(examples) == 2
    for example in examples:
        parts = example.split(":")
        file_index = next(i for i, part in enumerate(parts) if part.endswith(".sf"))
        parts[file_index] = salmon
        args = create_arg_parser().parse_args(
            ["--gene-expression", ":".join(parts)]
        )
        result = _attach_expression_data(
            _make_prediction_df(), expression_data_from_args(args),
        )
        assert "gene_tpm" in result.columns, example
        assert "gene_tpm_tpm" not in result.columns, example
        # The --gene-expression help and --filter-by examples point at this
        # exact expression.
        kept = apply_filter(result, parse("gene_tpm >= 20"))
        assert set(kept["gene_id"]) == {"ENSG00000157764"}, example


def test_gene_expression_name_is_a_prefix_not_a_column_name():
    import os

    from topiary.cli.args import create_arg_parser
    from topiary.cli.rna import expression_data_from_args

    salmon = os.path.join(os.path.dirname(__file__), "data", "salmon_quant.sf")
    args = create_arg_parser().parse_args(
        ["--gene-expression", f"gene_tpm:{salmon}:Name:TPM"]
    )
    result = _attach_expression_data(
        _make_prediction_df(), expression_data_from_args(args),
    )
    assert "gene_tpm_tpm" in result.columns
    assert "gene_tpm" not in result.columns
