"""CLI integration tests for --sort-by / --filter-by parsing via _build_filter_and_sort."""

import pandas as pd

from topiary.cli import args as cli_args
from topiary.cli.args import create_arg_parser, _build_filter_and_sort
from topiary.ranking import (
    Affinity,
    BinOp,
    BoolOp,
    Comparison,
    DSLNode,
    Field,
    apply_filter,
    apply_sort,
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
    filter_by, sort_by, sort_direction = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], Field)
    assert repr(sort_by[0]) == "affinity.value"
    assert sort_direction == "auto"


def test_sort_by_string_style_affinity_kind_uses_raw_value(monkeypatch):
    monkeypatch.setattr(
        cli_args,
        "_resolve_qualified_kind",
        lambda text: ("pMHC_affinity", None),
    )
    expr = cli_args._parse_sort_expr("ba")
    assert repr(expr) == "affinity.value"


def test_sort_by_simple_presentation_kind_name_uses_score():
    args = _make_args("pMHC_presentation")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], Field)
    assert repr(sort_by[0]) == "presentation.score"


def test_sort_by_comma_separated_kinds():
    args = _make_args("pMHC_presentation,pMHC_affinity")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 2
    assert [repr(expr) for expr in sort_by] == [
        "presentation.score",
        "affinity.value",
    ]


# ---------------------------------------------------------------------------
# Expression strings via --sort-by
# ---------------------------------------------------------------------------


def test_sort_by_arithmetic_expression():
    args = _make_args("0.5 * affinity.score + 0.5 * presentation.score")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)
    assert isinstance(sort_by[0], BinOp)


def test_sort_by_transform_expression():
    args = _make_args("affinity.descending_cdf(500, 200)")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_composite_expression():
    text = "0.6 * affinity.descending_cdf(500, 200) + 0.4 * presentation.score.ascending_cdf(0.5, 0.3)"
    args = _make_args(text)
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_parenthesized_expression():
    args = _make_args("(affinity.score + presentation.score) * 0.5")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_aggregation_expression():
    args = _make_args("mean(affinity.score, presentation.score)")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_negation_expression():
    # Leading '-' looks like a flag to argparse, so use --sort-by=VALUE
    # form. This matches real CLI usage: --sort-by='-affinity.value'
    parser = create_arg_parser(
        mhc=False, rna=False, output=False, variants=False,
        protein_changes=False, sequence_options=False, error_options=False,
        direct_inputs=False,
    )
    args = parser.parse_args(["--sort-by=-affinity.value"])
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_power_expression():
    args = _make_args("affinity.value ** 2")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


# ---------------------------------------------------------------------------
# Dot-transform without operators (comma-separated fallback path)
# ---------------------------------------------------------------------------


def test_sort_by_dot_transform_no_operators():
    """affinity.descending_cdf(500, 200) has parens so it hits the is_expr path,
    but verify it still produces a valid DSLNode."""
    args = _make_args("affinity.descending_cdf(500, 200)")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_by_comma_separated_expression_and_transform():
    args = _make_args("presentation.score,gene_tpm.log()")
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert len(sort_by) == 2
    assert isinstance(sort_by[0], DSLNode)
    assert isinstance(sort_by[1], DSLNode)


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
    _, sort_by, _ = _build_filter_and_sort(args)
    expr = sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = 0.5 * 0.8 + 0.5 * 0.92
    assert abs(val - expected) < 1e-9


def test_sort_by_expression_with_transform_evaluates():
    args = _make_args("affinity.descending_cdf(500, 200)")
    _, sort_by, _ = _build_filter_and_sort(args)
    expr = sort_by[0]
    df = pd.DataFrame(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120, mean=500, std=200 → descending CDF should be high
    assert val > 0.9


# ---------------------------------------------------------------------------
# No --sort-by → empty list (and no filter)
# ---------------------------------------------------------------------------


def test_no_sort_by_and_no_filters_returns_empty():
    args = _make_args(None)
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert filter_by is None
    assert sort_by == []


# ---------------------------------------------------------------------------
# Combined with filter args
# ---------------------------------------------------------------------------


def test_sort_by_with_ic50_cutoff():
    args = _make_args(
        "affinity.score",
        extra_args=["--ic50-cutoff", "500"],
    )
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    # ic50_cutoff produces a single Comparison
    assert isinstance(filter_by, Comparison)
    assert len(sort_by) == 1


def test_sort_by_with_presentation_cutoff():
    args = _make_args(
        "0.5 * affinity.score + 0.5 * presentation.score",
        extra_args=["--presentation-cutoff", "2.0"],
    )
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert isinstance(filter_by, Comparison)
    assert len(sort_by) == 1
    assert isinstance(sort_by[0], DSLNode)


def test_sort_direction_explicit_desc():
    args = _make_args("ba", extra_args=["--sort-direction", "desc"])
    _, _, sort_direction = _build_filter_and_sort(args)
    assert sort_direction == "desc"


def test_sort_direction_auto_affinity_sorts_low_to_high():
    args = _make_args("ba")
    _, sort_by, sort_direction = _build_filter_and_sort(args)
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
    result = apply_sort(df, sort_by, sort_direction=sort_direction)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_direction_auto_presentation_rank_sorts_ascending():
    args = _make_args("presentation.rank")
    _, sort_by, sort_direction = _build_filter_and_sort(args)
    assert sort_direction == "auto"
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
    result = apply_sort(df, sort_by, sort_direction=sort_direction)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_by_comma_separated_keys_breaks_ties():
    args = _make_args("presentation.score,gene_tpm.log()")
    _, sort_by, sort_direction = _build_filter_and_sort(args)
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
    result = apply_sort(df, sort_by, sort_direction=sort_direction)
    assert list(result["peptide"]) == ["BBB", "AAA"]


def test_sort_by_comma_separated_keys_fall_through_on_missing():
    args = _make_args("presentation.score,gene_tpm.log()")
    _, sort_by, sort_direction = _build_filter_and_sort(args)
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
    result = apply_sort(df, sort_by, sort_direction=sort_direction)
    assert list(result["peptide"]) == ["CCC", "BBB", "AAA"]


# ---------------------------------------------------------------------------
# Filter-only paths (no sort)
# ---------------------------------------------------------------------------


def test_ic50_cutoff_only_produces_comparison_filter():
    args = _make_args(None, extra_args=["--ic50-cutoff", "500"])
    filter_by, sort_by, _ = _build_filter_and_sort(args)
    assert isinstance(filter_by, Comparison)
    assert sort_by == []


def test_multiple_cutoffs_combine_with_or_by_default():
    import operator
    args = _make_args(
        None,
        extra_args=["--ic50-cutoff", "500", "--presentation-cutoff", "2.0"],
    )
    filter_by, _, _ = _build_filter_and_sort(args)
    assert isinstance(filter_by, BoolOp)
    # Default filter_logic is "any" -> or_
    assert filter_by.op is operator.or_
    assert len(filter_by.children) == 2


def test_filter_by_string_is_parsed():
    import operator
    args = _make_args(
        None, extra_args=["--filter-by", "affinity <= 500 | el.rank <= 2"],
    )
    filter_by, _, _ = _build_filter_and_sort(args)
    assert isinstance(filter_by, BoolOp)
    assert filter_by.op is operator.or_
