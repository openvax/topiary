"""CLI integration tests for --rank-by expression parsing via _build_ranking_strategy."""

import math

import pandas as pd
import pytest
from mhctools import Kind

from topiary.cli.args import create_arg_parser, _build_ranking_strategy
from topiary.ranking import (
    Expr,
    Field,
    KindAccessor,
    RankingStrategy,
    _Const,
)


def _make_args(rank_by, extra_args=None):
    """Build a minimal parsed args namespace with --rank-by set."""
    parser = create_arg_parser(
        mhc=False, rna=False, output=False, variants=False,
        protein_changes=False, sequence_options=False, error_options=False,
        direct_inputs=False,
    )
    argv = []
    if rank_by is not None:
        argv += ["--rank-by", rank_by]
    if extra_args:
        argv += extra_args
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Simple kind names (backward compat)
# ---------------------------------------------------------------------------


def test_rank_by_simple_kind_name():
    args = _make_args("pMHC_affinity")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Field)


def test_rank_by_comma_separated_kinds():
    args = _make_args("pMHC_presentation,pMHC_affinity")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 2


# ---------------------------------------------------------------------------
# Expression strings via --rank-by
# ---------------------------------------------------------------------------


def test_rank_by_arithmetic_expression():
    args = _make_args("0.5 * affinity.score + 0.5 * presentation.score")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_rank_by_transform_expression():
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    expr = strategy.sort_by[0]
    assert isinstance(expr, Expr)


def test_rank_by_composite_expression():
    text = "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score.ascending_cdf(0.5, 0.3)"
    args = _make_args(text)
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_rank_by_parenthesized_expression():
    args = _make_args("(affinity.score + presentation.score) * 0.5")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_rank_by_aggregation_expression():
    args = _make_args("mean(affinity.score, presentation.score)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_rank_by_negation_expression():
    # Leading '-' looks like a flag to argparse, so use --rank-by=VALUE
    # form. This matches real CLI usage: --rank-by='-affinity.value'
    parser = create_arg_parser(
        mhc=False, rna=False, output=False, variants=False,
        protein_changes=False, sequence_options=False, error_options=False,
        direct_inputs=False,
    )
    args = parser.parse_args(["--rank-by=-affinity.value"])
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_rank_by_power_expression():
    args = _make_args("affinity.value ** 2")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


# ---------------------------------------------------------------------------
# Dot-transform without operators (comma-separated fallback path)
# ---------------------------------------------------------------------------


def test_rank_by_dot_transform_no_operators():
    """affinity.descending_cdf(500, 200) has parens so it hits the is_expr path,
    but verify it still produces a valid Expr."""
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


# ---------------------------------------------------------------------------
# Evaluation through the CLI path
# ---------------------------------------------------------------------------


PEPTIDE_A_ROWS = [
    dict(
        source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_affinity",
        score=0.8, value=120.0, percentile_rank=0.5,
    ),
    dict(
        source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_presentation",
        score=0.92, value=None, percentile_rank=0.3,
    ),
]


def test_rank_by_expression_evaluates_correctly():
    """End-to-end: parse via CLI, then evaluate expression against data."""
    args = _make_args("0.5 * affinity.score + 0.5 * presentation.score")
    strategy = _build_ranking_strategy(args)
    expr = strategy.sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = 0.5 * 0.8 + 0.5 * 0.92
    assert abs(val - expected) < 1e-9


def test_rank_by_expression_with_transform_evaluates():
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    expr = strategy.sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120, mean=500, std=200 → descending CDF should be high
    assert val > 0.9


# ---------------------------------------------------------------------------
# No --rank-by → None
# ---------------------------------------------------------------------------


def test_no_rank_by_returns_none():
    args = _make_args(None)
    strategy = _build_ranking_strategy(args)
    assert strategy is None


# ---------------------------------------------------------------------------
# Combined with filter args
# ---------------------------------------------------------------------------


def test_rank_by_with_ic50_cutoff():
    args = _make_args(
        "affinity.score",
        extra_args=["--ic50-cutoff", "500"],
    )
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 1
    assert len(strategy.sort_by) == 1


def test_rank_by_with_presentation_cutoff():
    args = _make_args(
        "0.5 * affinity.score + 0.5 * presentation.score",
        extra_args=["--presentation-cutoff", "2.0"],
    )
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 1
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)
