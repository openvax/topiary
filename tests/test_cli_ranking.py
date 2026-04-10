"""CLI integration tests for --sort-by expression parsing via _build_ranking_strategy."""

import pandas as pd

from topiary.cli.args import create_arg_parser, _build_ranking_strategy
from topiary.ranking import (
    Expr,
    Field,
    RankingStrategy,
    apply_ranking_strategy,
)


def _make_args(sort_by, extra_args=None):
    """Build a minimal parsed args namespace with --sort-by set."""
    parser = create_arg_parser(
        mhc=False, rna=False, output=False, variants=False,
        protein_changes=False, sequence_options=False, error_options=False,
        direct_inputs=False,
    )
    argv = []
    if sort_by is not None:
        argv += ["--sort-by", sort_by]
    if extra_args:
        argv += extra_args
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Simple kind names (backward compat)
# ---------------------------------------------------------------------------


def test_sort_by_simple_affinity_kind_name_uses_raw_value():
    args = _make_args("pMHC_affinity")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Field)
    assert repr(strategy.sort_by[0]) == "affinity.value"
    assert strategy.sort_direction == "auto"


def test_sort_by_simple_presentation_kind_name_uses_score():
    args = _make_args("pMHC_presentation")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Field)
    assert repr(strategy.sort_by[0]) == "presentation.score"


def test_sort_by_comma_separated_kinds():
    args = _make_args("pMHC_presentation,pMHC_affinity")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 2
    assert [repr(expr) for expr in strategy.sort_by] == [
        "presentation.score",
        "affinity.value",
    ]


# ---------------------------------------------------------------------------
# Expression strings via --sort-by
# ---------------------------------------------------------------------------


def test_sort_by_arithmetic_expression():
    args = _make_args("0.5 * affinity.score + 0.5 * presentation.score")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_transform_expression():
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    expr = strategy.sort_by[0]
    assert isinstance(expr, Expr)


def test_sort_by_composite_expression():
    text = "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score.ascending_cdf(0.5, 0.3)"
    args = _make_args(text)
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_parenthesized_expression():
    args = _make_args("(affinity.score + presentation.score) * 0.5")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_aggregation_expression():
    args = _make_args("mean(affinity.score, presentation.score)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_negation_expression():
    # Leading '-' looks like a flag to argparse, so use --sort-by=VALUE
    # form. This matches real CLI usage: --sort-by='-affinity.value'
    parser = create_arg_parser(
        mhc=False, rna=False, output=False, variants=False,
        protein_changes=False, sequence_options=False, error_options=False,
        direct_inputs=False,
    )
    args = parser.parse_args(["--sort-by=-affinity.value"])
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_power_expression():
    args = _make_args("affinity.value ** 2")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


# ---------------------------------------------------------------------------
# Dot-transform without operators (comma-separated fallback path)
# ---------------------------------------------------------------------------


def test_sort_by_dot_transform_no_operators():
    """affinity.descending_cdf(500, 200) has parens so it hits the is_expr path,
    but verify it still produces a valid Expr."""
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_by_comma_separated_expression_and_transform():
    args = _make_args("presentation.score,gene_tpm.log()")
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.sort_by) == 2
    assert isinstance(strategy.sort_by[0], Expr)
    assert isinstance(strategy.sort_by[1], Expr)


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


def test_sort_by_expression_evaluates_correctly():
    """End-to-end: parse via CLI, then evaluate expression against data."""
    args = _make_args("0.5 * affinity.score + 0.5 * presentation.score")
    strategy = _build_ranking_strategy(args)
    expr = strategy.sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = 0.5 * 0.8 + 0.5 * 0.92
    assert abs(val - expected) < 1e-9


def test_sort_by_expression_with_transform_evaluates():
    args = _make_args("affinity.descending_cdf(500, 200)")
    strategy = _build_ranking_strategy(args)
    expr = strategy.sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120, mean=500, std=200 → descending CDF should be high
    assert val > 0.9


# ---------------------------------------------------------------------------
# No --sort-by → None
# ---------------------------------------------------------------------------


def test_no_sort_by_returns_none():
    args = _make_args(None)
    strategy = _build_ranking_strategy(args)
    assert strategy is None


# ---------------------------------------------------------------------------
# Combined with filter args
# ---------------------------------------------------------------------------


def test_sort_by_with_ic50_cutoff():
    args = _make_args(
        "affinity.score",
        extra_args=["--ic50-cutoff", "500"],
    )
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 1
    assert len(strategy.sort_by) == 1


def test_sort_by_with_presentation_cutoff():
    args = _make_args(
        "0.5 * affinity.score + 0.5 * presentation.score",
        extra_args=["--presentation-cutoff", "2.0"],
    )
    strategy = _build_ranking_strategy(args)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 1
    assert len(strategy.sort_by) == 1
    assert isinstance(strategy.sort_by[0], Expr)


def test_sort_direction_explicit_desc():
    args = _make_args("ba", extra_args=["--sort-direction", "desc"])
    strategy = _build_ranking_strategy(args)
    assert strategy.sort_direction == "desc"


def test_sort_direction_auto_affinity_sorts_low_to_high():
    args = _make_args("ba")
    strategy = _build_ranking_strategy(args)
    df = pd.DataFrame([
        dict(
            source_sequence_name="v1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.1, value=5000.0,
            percentile_rank=10.0,
        ),
        dict(
            source_sequence_name="v1", peptide="BBB", peptide_offset=1,
            allele="A", kind="pMHC_affinity", score=0.9, value=50.0,
            percentile_rank=0.1,
        ),
    ])
    result = apply_ranking_strategy(df, strategy)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_direction_auto_presentation_rank_sorts_ascending():
    args = _make_args("presentation.rank")
    strategy = _build_ranking_strategy(args)
    assert strategy.sort_direction == "auto"
    df = pd.DataFrame([
        dict(
            source_sequence_name="v1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_presentation", score=0.1, value=None,
            percentile_rank=10.0,
        ),
        dict(
            source_sequence_name="v1", peptide="BBB", peptide_offset=1,
            allele="A", kind="pMHC_presentation", score=0.9, value=None,
            percentile_rank=0.1,
        ),
    ])
    result = apply_ranking_strategy(df, strategy)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_by_comma_separated_keys_breaks_ties():
    args = _make_args("presentation.score,gene_tpm.log()")
    strategy = _build_ranking_strategy(args)
    df = pd.DataFrame([
        dict(
            source_sequence_name="v1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_presentation", score=0.8, value=None,
            percentile_rank=1.0, gene_tpm=1.0,
        ),
        dict(
            source_sequence_name="v1", peptide="BBB", peptide_offset=1,
            allele="A", kind="pMHC_presentation", score=0.8, value=None,
            percentile_rank=1.0, gene_tpm=10.0,
        ),
    ])
    result = apply_ranking_strategy(df, strategy)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_by_comma_separated_keys_fall_through_on_missing():
    args = _make_args("presentation.score,gene_tpm.log()")
    strategy = _build_ranking_strategy(args)
    df = pd.DataFrame([
        dict(
            source_sequence_name="v1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_presentation", score=0.8, value=None,
            percentile_rank=1.0, gene_tpm=1.0,
        ),
        dict(
            source_sequence_name="v1", peptide="BBB", peptide_offset=1,
            allele="A", kind="pMHC_presentation", score=0.8, value=None,
            percentile_rank=1.0, gene_tpm=10.0,
        ),
        dict(
            source_sequence_name="v1", peptide="CCC", peptide_offset=2,
            allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
            percentile_rank=1.0, gene_tpm=100.0,
        ),
    ])
    result = apply_ranking_strategy(df, strategy)
    assert list(result["peptide"]) == ["CCC", "BBB", "AAA"]
