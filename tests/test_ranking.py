"""Unit tests for topiary.ranking — filtering, ranking, and composite scoring."""

import math

import pandas as pd
from mhctools import Kind

from topiary.ranking import (
    Affinity,
    EpitopeFilter,
    KindAccessor,
    Presentation,
    RankingStrategy,
    Stability,
    affinity_filter,
    apply_ranking_strategy,
    presentation_filter,
)


def _make_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures: two peptides, each with affinity + presentation rows
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

PEPTIDE_B_ROWS = [
    dict(
        source_sequence_name="var1", peptide="ELAGIGIL", peptide_offset=20,
        allele="HLA-A*02:01", kind="pMHC_affinity",
        score=0.1, value=5000.0, percentile_rank=15.0,
    ),
    dict(
        source_sequence_name="var1", peptide="ELAGIGIL", peptide_offset=20,
        allele="HLA-A*02:01", kind="pMHC_presentation",
        score=0.05, value=None, percentile_rank=20.0,
    ),
]


def _two_peptide_df():
    return _make_df(PEPTIDE_A_ROWS + PEPTIDE_B_ROWS)


# ---------------------------------------------------------------------------
# Tests: affinity filter
# ---------------------------------------------------------------------------


def test_affinity_filter_ic50():
    df = _two_peptide_df()
    strategy = RankingStrategy(filters=[affinity_filter(ic50_cutoff=500)])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}
    assert len(result) == 2


def test_affinity_filter_percentile():
    df = _two_peptide_df()
    strategy = RankingStrategy(filters=[affinity_filter(percentile_cutoff=1.0)])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: presentation filter
# ---------------------------------------------------------------------------


def test_presentation_filter_rank():
    df = _two_peptide_df()
    strategy = RankingStrategy(filters=[presentation_filter(max_rank=1.0)])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_presentation_filter_score():
    df = _two_peptide_df()
    strategy = RankingStrategy(filters=[presentation_filter(min_score=0.5)])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: OR / AND logic
# ---------------------------------------------------------------------------


def test_or_logic():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[affinity_filter(ic50_cutoff=500), presentation_filter(min_score=0.01)],
        require_all=False,
    )
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL", "ELAGIGIL"}


def test_and_logic():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[affinity_filter(ic50_cutoff=500), presentation_filter(min_score=0.5)],
        require_all=True,
    )
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_and_logic_nothing_passes():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[affinity_filter(ic50_cutoff=10), presentation_filter(min_score=0.99)],
        require_all=True,
    )
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: sort_by with Field expressions
# ---------------------------------------------------------------------------


def test_sort_by_presentation_score():
    df = _two_peptide_df()
    strategy = RankingStrategy(sort_by=[Presentation.score])
    result = apply_ranking_strategy(df, strategy)
    assert result.iloc[0]["peptide"] == "SIINFEKL"


def test_sort_by_with_fallback():
    rows = [
        dict(source_sequence_name="v1", peptide="AAA", peptide_offset=0,
             allele="A", kind="pMHC_affinity", score=0.9, value=50.0,
             percentile_rank=0.1),
        dict(source_sequence_name="v1", peptide="BBB", peptide_offset=5,
             allele="A", kind="pMHC_affinity", score=0.1, value=5000.0,
             percentile_rank=15.0),
    ]
    df = _make_df(rows)
    strategy = RankingStrategy(sort_by=[Presentation.score, Affinity.score])
    result = apply_ranking_strategy(df, strategy)
    assert result.iloc[0]["peptide"] == "AAA"


# ---------------------------------------------------------------------------
# Tests: no filters / empty
# ---------------------------------------------------------------------------


def test_no_filters_passthrough():
    df = _two_peptide_df()
    result = apply_ranking_strategy(df, RankingStrategy())
    assert len(result) == len(df)


def test_empty_df():
    df = pd.DataFrame()
    result = apply_ranking_strategy(df, RankingStrategy(filters=[affinity_filter(ic50_cutoff=500)]))
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: convenience constructors
# ---------------------------------------------------------------------------


def test_affinity_filter_constructor():
    f = affinity_filter(ic50_cutoff=500, percentile_cutoff=2.0)
    assert f.kind == Kind.pMHC_affinity
    assert f.max_value == 500
    assert f.max_percentile_rank == 2.0


def test_presentation_filter_constructor():
    f = presentation_filter(max_rank=2.0, min_score=0.5)
    assert f.kind == Kind.pMHC_presentation
    assert f.max_percentile_rank == 2.0
    assert f.min_score == 0.5


# ---------------------------------------------------------------------------
# Tests: variant column grouping
# ---------------------------------------------------------------------------


def test_variant_column_grouping():
    rows = [
        dict(variant="chr7 p.V600E", peptide="SIINFEKL", peptide_offset=10,
             allele="HLA-A*02:01", kind="pMHC_affinity",
             score=0.8, value=120.0, percentile_rank=0.5),
    ]
    df = _make_df(rows)
    strategy = RankingStrategy(filters=[affinity_filter(ic50_cutoff=500)])
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: operator syntax
# ---------------------------------------------------------------------------


def test_operator_affinity_value_le():
    filt = Affinity.value <= 500
    assert isinstance(filt, EpitopeFilter)
    assert filt.kind == Kind.pMHC_affinity
    assert filt.max_value == 500


def test_operator_affinity_rank_le():
    filt = Affinity.rank <= 2.0
    assert isinstance(filt, EpitopeFilter)
    assert filt.max_percentile_rank == 2.0


def test_operator_presentation_score_ge():
    filt = Presentation.score >= 0.5
    assert isinstance(filt, EpitopeFilter)
    assert filt.kind == Kind.pMHC_presentation
    assert filt.min_score == 0.5


def test_operator_or():
    strategy = (Affinity.value <= 500) | (Presentation.rank <= 2.0)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 2
    assert strategy.require_all is False


def test_operator_and():
    strategy = (Affinity.value <= 500) & (Presentation.rank <= 2.0)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 2
    assert strategy.require_all is True


def test_operator_or_applied():
    df = _two_peptide_df()
    result = apply_ranking_strategy(
        df, (Affinity.value <= 500) | (Presentation.score >= 0.01)
    )
    assert set(result["peptide"]) == {"SIINFEKL", "ELAGIGIL"}


def test_operator_and_applied():
    df = _two_peptide_df()
    result = apply_ranking_strategy(
        df, (Affinity.value <= 500) & (Presentation.score >= 0.5)
    )
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_operator_rank_by():
    df = _two_peptide_df()
    strategy = (Affinity.value <= 50000).rank_by(Presentation.score, Affinity.score)
    result = apply_ranking_strategy(df, strategy)
    assert result.iloc[0]["peptide"] == "SIINFEKL"


def test_operator_chained_or():
    strategy = (Affinity.value <= 500) | (Presentation.rank <= 2.0) | (Stability.score >= 0.5)
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 3


def test_single_filter_as_ranking_param():
    from topiary import TopiaryPredictor
    from mhctools import RandomBindingPredictor

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
        filter=Affinity.value <= 500,
    )
    assert predictor.ranking_strategy is not None
    assert len(predictor.ranking_strategy.filters) == 1


# ---------------------------------------------------------------------------
# Tests: custom KindAccessor
# ---------------------------------------------------------------------------


def test_custom_kind_accessor():
    custom = KindAccessor(Kind.proteasome_cleavage)
    filt = custom.score >= 0.5
    assert filt.kind == Kind.proteasome_cleavage
    assert filt.min_score == 0.5


# ---------------------------------------------------------------------------
# Tests: arithmetic expressions
# ---------------------------------------------------------------------------


def test_field_multiply():
    expr = 0.5 * Affinity.score
    df = _make_df(PEPTIDE_A_ROWS)
    group = df[df["kind"] == "pMHC_affinity"]
    val = expr.evaluate(df)
    assert abs(val - 0.4) < 1e-9


def test_field_add():
    expr = Affinity.score + Presentation.score
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - (0.8 + 0.92)) < 1e-9


def test_field_sub():
    expr = Presentation.score - Affinity.score
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - 0.12) < 1e-9


def test_weighted_composite():
    expr = 0.5 * Affinity.score + 0.5 * Presentation.score
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - 0.86) < 1e-9


def test_negation():
    expr = -Affinity.value
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - (-120.0)) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Gaussian normalization
# ---------------------------------------------------------------------------


def test_norm_basic():
    """norm(mean=0, std=1) of 0 should be 0.5"""
    from topiary.ranking import _gauss_cdf
    assert abs(_gauss_cdf(0) - 0.5) < 1e-9


def test_field_norm():
    # IC50 = 120, norm(mean=500, std=200) -> CDF((120-500)/200) = CDF(-1.9) ≈ 0.0287
    expr = Affinity.value.norm(mean=500, std=200)
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = 0.5 * (1 + math.erf((120 - 500) / (200 * math.sqrt(2))))
    assert abs(val - expected) < 1e-6


def test_composite_norm_ranking():
    """Composite normalized score used in rank_by."""
    df = _two_peptide_df()
    composite = (
        0.5 * (1 - Affinity.value.norm(mean=500, std=200))
        + 0.5 * Presentation.score.norm(mean=0.5, std=0.3)
    )
    strategy = (Affinity.value <= 50000).rank_by(composite)
    result = apply_ranking_strategy(df, strategy)
    # Peptide A: low IC50 (good) + high presentation (good) → higher composite
    assert result.iloc[0]["peptide"] == "SIINFEKL"


def test_norm_missing_kind_returns_nan():
    """Normalization of a missing kind returns NaN."""
    expr = Stability.score.norm(mean=0.5, std=0.3)
    df = _make_df(PEPTIDE_A_ROWS)  # no stability rows
    val = expr.evaluate(df)
    assert math.isnan(val)


# ---------------------------------------------------------------------------
# Tests: default .value on KindAccessor (Affinity <= 500)
# ---------------------------------------------------------------------------


def test_kind_accessor_le_default_value():
    filt = Affinity <= 500
    assert isinstance(filt, EpitopeFilter)
    assert filt.kind == Kind.pMHC_affinity
    assert filt.max_value == 500


def test_kind_accessor_default_applied():
    df = _two_peptide_df()
    result = apply_ranking_strategy(df, (Affinity <= 500) | (Presentation.rank <= 2.0))
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: string parsing (parse_filter / parse_ranking)
# ---------------------------------------------------------------------------


def test_parse_filter_simple():
    from topiary.ranking import parse_filter
    f = parse_filter("affinity <= 500")
    assert f.kind == Kind.pMHC_affinity
    assert f.max_value == 500


def test_parse_filter_with_field():
    from topiary.ranking import parse_filter
    f = parse_filter("presentation.rank <= 2.0")
    assert f.kind == Kind.pMHC_presentation
    assert f.max_percentile_rank == 2.0


def test_parse_filter_score_ge():
    from topiary.ranking import parse_filter
    f = parse_filter("presentation.score >= 0.5")
    assert f.kind == Kind.pMHC_presentation
    assert f.min_score == 0.5


def test_parse_filter_aliases():
    from topiary.ranking import parse_filter
    # "ic50" -> pMHC_affinity, "el" -> pMHC_presentation
    f1 = parse_filter("ic50 <= 500")
    assert f1.kind == Kind.pMHC_affinity
    f2 = parse_filter("el.rank <= 2")
    assert f2.kind == Kind.pMHC_presentation


def test_parse_ranking_or():
    from topiary.ranking import parse_ranking
    strategy = parse_ranking("affinity <= 500 | presentation.rank <= 2")
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 2
    assert strategy.require_all is False


def test_parse_ranking_and():
    from topiary.ranking import parse_ranking
    strategy = parse_ranking("affinity <= 500 & presentation.score >= 0.5")
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 2
    assert strategy.require_all is True


def test_parse_ranking_single():
    from topiary.ranking import parse_ranking
    f = parse_ranking("affinity <= 500")
    assert isinstance(f, EpitopeFilter)


def test_parse_ranking_applied():
    from topiary.ranking import parse_ranking
    df = _two_peptide_df()
    strategy = parse_ranking("affinity <= 500 | presentation.rank <= 1")
    if isinstance(strategy, EpitopeFilter):
        strategy = RankingStrategy(filters=[strategy])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: clip, abs, log, exp, sqrt, pow
# ---------------------------------------------------------------------------


def test_clip_both():
    expr = Affinity.value.clip(lo=50, hi=200)
    df = _make_df(PEPTIDE_A_ROWS)  # value = 120
    assert abs(expr.evaluate(df) - 120.0) < 1e-9

    # Value below lo
    rows = [dict(PEPTIDE_A_ROWS[0], value=10.0)]
    assert abs(expr.evaluate(_make_df(rows)) - 50.0) < 1e-9

    # Value above hi
    rows = [dict(PEPTIDE_A_ROWS[0], value=999.0)]
    assert abs(expr.evaluate(_make_df(rows)) - 200.0) < 1e-9


def test_clip_lo_only():
    expr = Affinity.value.clip(lo=0)
    rows = [dict(PEPTIDE_A_ROWS[0], value=-5.0)]
    assert abs(expr.evaluate(_make_df(rows))) < 1e-9


def test_clip_hi_only():
    expr = Affinity.value.clip(hi=100)
    df = _make_df(PEPTIDE_A_ROWS)  # value = 120
    assert abs(expr.evaluate(df) - 100.0) < 1e-9


def test_abs_positive():
    expr = abs(Affinity.value)
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - 120.0) < 1e-9


def test_abs_negative():
    expr = abs(-Affinity.value)
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - 120.0) < 1e-9


def test_log():
    expr = Affinity.value.log()
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - math.log(120.0)) < 1e-9


def test_log10():
    expr = Affinity.value.log10()
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - math.log10(120.0)) < 1e-9


def test_exp():
    expr = Affinity.score.exp()  # score = 0.8
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - math.exp(0.8)) < 1e-9


def test_sqrt():
    expr = Affinity.value.sqrt()
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - math.sqrt(120.0)) < 1e-9


def test_pow():
    expr = Affinity.score ** 2
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - 0.64) < 1e-9


def test_log_negative_returns_nan():
    rows = [dict(PEPTIDE_A_ROWS[0], value=-5.0)]
    expr = Affinity.value.log()
    assert math.isnan(expr.evaluate(_make_df(rows)))


def test_composite_with_clip_and_log():
    """log(clip(IC50, 1, 50000)) is a common transform."""
    expr = Affinity.value.clip(lo=1, hi=50000).log()
    df = _make_df(PEPTIDE_A_ROWS)
    assert abs(expr.evaluate(df) - math.log(120.0)) < 1e-9


# ---------------------------------------------------------------------------
# Tests: >= and > on value/rank fields
# ---------------------------------------------------------------------------


def test_value_ge_filter():
    filt = Affinity.value >= 100
    assert isinstance(filt, EpitopeFilter)
    assert filt.min_value == 100


def test_rank_ge_filter():
    filt = Affinity.rank >= 1.0
    assert isinstance(filt, EpitopeFilter)
    assert filt.min_percentile_rank == 1.0


def test_gt_delegates_to_ge():
    filt = Presentation.score > 0.5
    assert isinstance(filt, EpitopeFilter)
    assert filt.min_score == 0.5


def test_lt_delegates_to_le():
    filt = Affinity < 500
    assert isinstance(filt, EpitopeFilter)
    assert filt.max_value == 500


def test_ge_filter_applied():
    """Keep only weak binders (IC50 >= 1000)."""
    df = _two_peptide_df()
    strategy = RankingStrategy(filters=[Affinity.value >= 1000])
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"ELAGIGIL"}


def test_parse_filter_value_ge():
    from topiary.ranking import parse_filter
    f = parse_filter("affinity >= 100")
    assert f.min_value == 100


def test_parse_filter_rank_ge():
    from topiary.ranking import parse_filter
    f = parse_filter("presentation.rank >= 5")
    assert f.min_percentile_rank == 5.0


# ---------------------------------------------------------------------------
# Tests: bug fixes from audit
# ---------------------------------------------------------------------------


def test_combine_mixed_operators_nesting():
    """(A | B) & C must be AND( OR(A, B), C ), not AND(A, B, C)."""
    df = _two_peptide_df()
    # A: affinity <= 500 (only peptide A passes)
    # B: affinity >= 1000 (only peptide B passes)
    # A | B: both pass
    # C: presentation.score >= 0.5 (only peptide A passes)
    # (A | B) & C: only peptide A should pass
    strategy = ((Affinity.value <= 500) | (Affinity.value >= 1000)) & (Presentation.score >= 0.5)
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_combine_same_operator_flattens():
    """A | B | C should flatten to OR(A, B, C), not OR(OR(A, B), C)."""
    strategy = (Affinity.value <= 500) | (Presentation.rank <= 2.0) | (Stability.score >= 0.5)
    assert len(strategy.filters) == 3
    assert all(isinstance(f, EpitopeFilter) for f in strategy.filters)


def test_norm_std_zero_returns_nan():
    expr = Affinity.value.norm(mean=500, std=0)
    df = _make_df(PEPTIDE_A_ROWS)
    assert math.isnan(expr.evaluate(df))


def test_field_evaluate_missing_column():
    from topiary.ranking import Field
    expr = Field(Kind.pMHC_affinity, "nonexistent_column")
    df = _make_df(PEPTIDE_A_ROWS)
    assert math.isnan(expr.evaluate(df))


def test_parse_filter_with_parentheses():
    from topiary.ranking import parse_filter
    f = parse_filter("(affinity <= 500)")
    assert f.kind == Kind.pMHC_affinity
    assert f.max_value == 500


def test_parse_ranking_mixed_operators_raises():
    import pytest
    from topiary.ranking import parse_ranking
    with pytest.raises(ValueError, match="Cannot mix"):
        parse_ranking("affinity <= 500 | presentation.rank <= 2 & stability.score >= 0.5")
