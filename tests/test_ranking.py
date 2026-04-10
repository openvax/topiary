"""Unit tests for topiary.ranking — filtering, ranking, and composite scoring."""

import math

import pandas as pd
import pytest
from mhctools import Kind

from topiary.ranking import (
    Affinity,
    Column,
    Count,
    EpitopeFilter,
    Field,
    KindAccessor,
    Len,
    Presentation,
    RankingStrategy,
    Stability,
    _Const,
    affinity_filter,
    apply_ranking_strategy,
    geomean,
    maximum,
    mean,
    median,
    minimum,
    parse_expr,
    presentation_filter,
    wt,
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


# ---------------------------------------------------------------------------
# Tests: method-qualified access (multi-model disambiguation)
# ---------------------------------------------------------------------------


def _multi_model_df():
    """Two models producing affinity for the same peptide-allele pair."""
    return _make_df([
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            prediction_method_name="netmhcpan",
        ),
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.6, value=350.0, percentile_rank=2.0,
            prediction_method_name="mhcflurry",
        ),
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_presentation",
            score=0.9, value=None, percentile_rank=0.3,
            prediction_method_name="mhcflurry",
        ),
    ])


def test_bracket_qualified_field_evaluate():
    df = _multi_model_df()
    assert Affinity["netmhcpan"].value.evaluate(df) == 120.0
    assert Affinity["mhcflurry"].value.evaluate(df) == 350.0
    assert Presentation["mhcflurry"].score.evaluate(df) == 0.9


def test_bracket_qualified_filter():
    f = Affinity["netmhcpan"] <= 500
    assert f.method == "netmhcpan"
    assert f.kind == Kind.pMHC_affinity
    assert f.max_value == 500


def test_bracket_qualified_ranking_apply():
    df = _multi_model_df()
    # Filter: only keep if mhcflurry affinity <= 200 (it's 350, so should drop)
    strategy = RankingStrategy(filters=[Affinity["mhcflurry"] <= 200])
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 0

    # Filter: netmhcpan affinity <= 200 (it's 120, so should keep)
    strategy = RankingStrategy(filters=[Affinity["netmhcpan"] <= 200])
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 3  # all rows for this group kept


def test_unqualified_ambiguity_raises():
    import pytest
    df = _multi_model_df()
    with pytest.raises(ValueError, match="Ambiguous.*multiple models"):
        Affinity.value.evaluate(df)


def test_unqualified_single_model_ok():
    """When only one model produces a kind, no qualification needed."""
    df = _multi_model_df()
    # Presentation only comes from mhcflurry — no ambiguity
    assert Presentation.score.evaluate(df) == 0.9


def test_bracket_norm_shorthand():
    """KindAccessor.norm() delegates to .value.norm()."""
    df = _multi_model_df()
    via_accessor = Affinity["netmhcpan"].norm(500, 200).evaluate(df)
    via_field = Affinity["netmhcpan"].value.norm(500, 200).evaluate(df)
    assert via_accessor == via_field


def test_bracket_arithmetic():
    """KindAccessor arithmetic delegates to .value."""
    df = _multi_model_df()
    expr = 0.5 * Affinity["netmhcpan"] + 0.5 * Affinity["mhcflurry"]
    result = expr.evaluate(df)
    assert result == 0.5 * 120.0 + 0.5 * 350.0


# ---------------------------------------------------------------------------
# Tests: CLI tool_kind parsing
# ---------------------------------------------------------------------------


def test_resolve_qualified_kind_plain():
    from topiary.ranking import _resolve_qualified_kind
    kind, method = _resolve_qualified_kind("affinity")
    assert kind == Kind.pMHC_affinity
    assert method is None


def test_resolve_qualified_kind_with_tool():
    from topiary.ranking import _resolve_qualified_kind
    kind, method = _resolve_qualified_kind("netmhcpan_affinity")
    assert kind == Kind.pMHC_affinity
    assert method == "netmhcpan"


def test_resolve_qualified_kind_short_aliases():
    from topiary.ranking import _resolve_qualified_kind
    kind, method = _resolve_qualified_kind("netmhcpan_ba")
    assert kind == Kind.pMHC_affinity
    assert method == "netmhcpan"

    kind, method = _resolve_qualified_kind("mhcflurry_el")
    assert kind == Kind.pMHC_presentation
    assert method == "mhcflurry"


def test_resolve_qualified_kind_underscore_kind():
    """Kind names with underscores (antigen_processing) resolve without a tool."""
    from topiary.ranking import _resolve_qualified_kind
    kind, method = _resolve_qualified_kind("antigen_processing")
    assert kind == Kind.antigen_processing
    assert method is None


def test_resolve_qualified_kind_tool_plus_underscore_kind():
    from topiary.ranking import _resolve_qualified_kind
    kind, method = _resolve_qualified_kind("netmhcpan_antigen_processing")
    assert kind == Kind.antigen_processing
    assert method == "netmhcpan"


def test_parse_filter_tool_qualified():
    from topiary.ranking import parse_filter
    f = parse_filter("netmhcpan_affinity <= 500")
    assert f.kind == Kind.pMHC_affinity
    assert f.method == "netmhcpan"
    assert f.max_value == 500.0


def test_parse_filter_tool_qualified_with_field():
    from topiary.ranking import parse_filter
    f = parse_filter("mhcflurry_el.rank <= 2")
    assert f.kind == Kind.pMHC_presentation
    assert f.method == "mhcflurry"
    assert f.max_percentile_rank == 2.0


def test_parse_ranking_tool_qualified():
    from topiary.ranking import parse_ranking
    strategy = parse_ranking(
        "netmhcpan_affinity <= 500 | mhcflurry_el.rank <= 2"
    )
    assert isinstance(strategy, RankingStrategy)
    assert len(strategy.filters) == 2
    assert strategy.filters[0].method == "netmhcpan"
    assert strategy.filters[1].method == "mhcflurry"


# ---------------------------------------------------------------------------
# Adversarial / edge case tests for method qualification
# ---------------------------------------------------------------------------


def test_qualified_nonexistent_method_raises():
    """Qualifying with a method that doesn't exist in data → ValueError."""
    import pytest
    df = _multi_model_df()
    with pytest.raises(ValueError, match="No pMHC_affinity predictions from method"):
        Affinity["nonexistent_tool"].value.evaluate(df)


def test_qualified_nonexistent_method_shows_available():
    """Error message lists available methods."""
    import pytest
    df = _multi_model_df()
    with pytest.raises(ValueError, match="Available.*mhcflurry.*netmhcpan"):
        Affinity["totally_wrong"].value.evaluate(df)


def test_qualified_typo_suggests_correction():
    """Close misspellings get 'Did you mean' suggestions."""
    import pytest
    df = _multi_model_df()
    with pytest.raises(ValueError, match="Did you mean.*netmhcpan"):
        Affinity["netmhcapn"].value.evaluate(df)


def test_qualified_typo_in_filter_suggests_correction():
    """Same suggestion works in the filter path."""
    import pytest
    df = _multi_model_df()
    strategy = RankingStrategy(filters=[Affinity["mchflurry"] <= 500])
    with pytest.raises(ValueError, match="Did you mean.*mhcflurry"):
        apply_ranking_strategy(df, strategy)


def test_method_match_is_case_insensitive():
    """Method matching should be case-insensitive."""
    df = _multi_model_df()
    assert Affinity["NETMHCPAN"].value.evaluate(df) == 120.0
    assert Affinity["NetMHCpan"].value.evaluate(df) == 120.0


def test_method_match_is_substring():
    """Method matching uses substring — 'pan' matches 'netmhcpan'."""
    df = _multi_model_df()
    assert Affinity["pan"].value.evaluate(df) == 120.0


def test_method_substring_ambiguity():
    """If substring matches multiple methods, picks first match (no error)."""
    df = _make_df([
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.8, value=100.0, percentile_rank=0.5,
            prediction_method_name="tool_alpha",
        ),
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.6, value=200.0, percentile_rank=1.0,
            prediction_method_name="tool_alphabeta",
        ),
    ])
    # "alpha" matches both — should get first match (100.0)
    val = Affinity["alpha"].value.evaluate(df)
    assert val == 100.0


def test_empty_dataframe_qualified():
    """Qualified field on empty DataFrame → NaN."""
    df = _make_df([])
    val = Affinity["netmhcpan"].value.evaluate(df)
    assert math.isnan(val)


def test_no_prediction_method_name_column():
    """Data without prediction_method_name column — unqualified works, qualified is NaN."""
    df = _make_df([
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.8, value=100.0, percentile_rank=0.5,
        ),
    ])
    # Unqualified works (no ambiguity check without column)
    assert Affinity.value.evaluate(df) == 100.0
    # Qualified — column doesn't exist, so method filter can't match
    # but the code guards with 'if col in kind_rows.columns', so falls through
    val = Affinity["netmhcpan"].value.evaluate(df)
    assert val == 100.0  # no column to filter on → uses all rows


def test_nan_prediction_method_name():
    """Rows with NaN prediction_method_name shouldn't crash or match."""
    df = _make_df([
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.8, value=100.0, percentile_rank=0.5,
            prediction_method_name=None,
        ),
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.6, value=200.0, percentile_rank=1.0,
            prediction_method_name="netmhcpan",
        ),
    ])
    # Qualified should skip None row and find netmhcpan
    assert Affinity["netmhcpan"].value.evaluate(df) == 200.0


def test_ambiguity_not_raised_when_qualified():
    """Even with multiple models, qualifying one side shouldn't error."""
    df = _multi_model_df()
    # Qualified access — no ambiguity
    assert Affinity["netmhcpan"].value.evaluate(df) == 120.0
    assert Affinity["mhcflurry"].value.evaluate(df) == 350.0


def test_composite_score_across_models():
    """Composite expression mixing qualified fields from different models."""
    df = _multi_model_df()
    expr = (
        0.5 * Affinity["netmhcpan"].norm(500, 200)
        + 0.5 * Presentation["mhcflurry"].score
    )
    val = expr.evaluate(df)
    assert isinstance(val, float)
    assert not math.isnan(val)
    assert val > 0


def test_sort_by_qualified_field():
    """Sorting by a qualified field across groups."""
    df = _make_df([
        # Group 1: netmhcpan says 500
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.5, value=500.0, percentile_rank=5.0,
            prediction_method_name="netmhcpan",
        ),
        # Group 2: netmhcpan says 100
        dict(
            source_sequence_name="seq1", peptide="BBB", peptide_offset=10,
            allele="A", kind="pMHC_affinity",
            score=0.9, value=100.0, percentile_rank=0.5,
            prediction_method_name="netmhcpan",
        ),
    ])
    strategy = RankingStrategy(sort_by=[Affinity["netmhcpan"].score])
    result = apply_ranking_strategy(df, strategy)
    # BBB has higher score (0.9) so should come first
    assert result.iloc[0]["peptide"] == "BBB"


def test_resolve_qualified_kind_unknown_raises():
    import pytest
    from topiary.ranking import _resolve_qualified_kind
    with pytest.raises(ValueError, match="Unknown prediction kind"):
        _resolve_qualified_kind("totally_bogus_nonsense")


def test_resolve_qualified_kind_empty_raises():
    import pytest
    from topiary.ranking import _resolve_qualified_kind
    with pytest.raises(ValueError):
        _resolve_qualified_kind("")


def test_filter_or_across_models():
    """OR filter: pass if netmhcpan OR mhcflurry affinity is low enough."""
    df = _make_df([
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.9, value=50.0, percentile_rank=0.1,
            prediction_method_name="netmhcpan",
        ),
        dict(
            source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity",
            score=0.1, value=9999.0, percentile_rank=50.0,
            prediction_method_name="mhcflurry",
        ),
    ])
    # netmhcpan passes, mhcflurry doesn't — OR should keep
    strategy = (Affinity["netmhcpan"] <= 500) | (Affinity["mhcflurry"] <= 500)
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 2  # both rows for the group kept

    # AND should fail (mhcflurry doesn't pass)
    strategy = (Affinity["netmhcpan"] <= 500) & (Affinity["mhcflurry"] <= 500)
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 0


def test_bracket_on_kind_accessor_returns_new_accessor():
    """Affinity["x"] returns a new KindAccessor, not a mutation."""
    original_method = Affinity.method
    qualified = Affinity["netmhcpan"]
    assert qualified.method == "netmhcpan"
    assert Affinity.method == original_method  # unchanged


# ---------------------------------------------------------------------------
# parse_expr tests — expression DSL string parsing
# ---------------------------------------------------------------------------


def test_parse_expr_number():
    expr = parse_expr("42.5")
    assert isinstance(expr, _Const)
    assert expr.val == 42.5


def test_parse_expr_kind_accessor_defaults_to_value():
    expr = parse_expr("affinity")
    assert isinstance(expr, Field)
    assert expr.kind == Kind.pMHC_affinity
    assert expr.field == "value"


def test_parse_expr_kind_field():
    expr = parse_expr("affinity.score")
    assert isinstance(expr, Field)
    assert expr.kind == Kind.pMHC_affinity
    assert expr.field == "score"


def test_parse_expr_kind_rank():
    expr = parse_expr("presentation.rank")
    assert isinstance(expr, Field)
    assert expr.kind == Kind.pMHC_presentation
    assert expr.field == "percentile_rank"


def test_parse_expr_arithmetic():
    expr = parse_expr("0.5 * affinity.score + 0.5 * presentation.score")
    # evaluate against a group
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = 0.5 * 0.8 + 0.5 * 0.92
    assert abs(val - expected) < 1e-9


def test_parse_expr_descending_cdf():
    expr = parse_expr("affinity.descending_cdf(500, 200)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120, mean=500, std=200 → descending CDF should be high (close to 1)
    assert val > 0.9


def test_parse_expr_ascending_cdf():
    expr = parse_expr("presentation.score.ascending_cdf(0.5, 0.3)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # score=0.92, mean=0.5, std=0.3 → ascending CDF should be high
    assert val > 0.9


def test_parse_expr_logistic():
    expr = parse_expr("affinity.logistic(350, 150)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120 < midpoint=350 → score > 0.5
    assert val > 0.5


def test_parse_expr_log():
    expr = parse_expr("affinity.value.log()")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - math.log(120.0)) < 1e-9


def test_parse_expr_log2():
    expr = parse_expr("affinity.value.log2()")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - math.log2(120.0)) < 1e-9


def test_parse_expr_clip():
    expr = parse_expr("affinity.value.clip(100, 200)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == 120.0  # 120 is within [100, 200]

    df_b = _make_df(PEPTIDE_B_ROWS)
    val_b = expr.evaluate(df_b)
    assert val_b == 200.0  # 5000 clipped to 200


def test_parse_expr_hinge():
    expr = parse_expr("affinity.value.hinge()")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == 120.0


def test_parse_expr_sqrt():
    expr = parse_expr("affinity.value.sqrt()")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - math.sqrt(120.0)) < 1e-9


def test_parse_expr_negation():
    expr = parse_expr("-affinity.value")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == -120.0


def test_parse_expr_abs():
    # abs of a negated expression
    expr = parse_expr("abs(-affinity.value)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == 120.0

    # abs of a positive expression
    expr2 = parse_expr("abs(affinity.value)")
    val2 = expr2.evaluate(df)
    assert val2 == 120.0


def test_parse_expr_power():
    expr = parse_expr("affinity.value ** 2")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - 120.0 ** 2) < 1e-6


def test_parse_expr_parentheses():
    expr = parse_expr("(affinity.score + presentation.score) * 0.5")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = (0.8 + 0.92) * 0.5
    assert abs(val - expected) < 1e-9


def test_parse_expr_mean():
    expr = parse_expr("mean(affinity.score, presentation.score)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    expected = (0.8 + 0.92) / 2
    assert abs(val - expected) < 1e-9


def test_parse_expr_minimum():
    expr = parse_expr("minimum(affinity.score, presentation.score)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == 0.8


def test_parse_expr_maximum():
    expr = parse_expr("maximum(affinity.score, presentation.score)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert val == 0.92


def test_parse_expr_column():
    expr = parse_expr("column(hydrophobicity)")
    assert isinstance(expr, Column)
    assert expr.col_name == "hydrophobicity"


def test_parse_expr_column_in_arithmetic():
    expr = parse_expr("0.5 * affinity.score - 0.2 * column(cysteine_count)")
    df = _make_df(PEPTIDE_A_ROWS)
    df["cysteine_count"] = 1
    val = expr.evaluate(df)
    expected = 0.5 * 0.8 - 0.2 * 1
    assert abs(val - expected) < 1e-9


def test_parse_expr_bracket_qualification():
    expr = parse_expr('affinity["netmhcpan"].score')
    assert isinstance(expr, Field)
    assert expr.method == "netmhcpan"


def test_parse_expr_norm_alias():
    """norm() is an alias for ascending_cdf()"""
    expr = parse_expr("affinity.norm(500, 200)")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    # IC50=120 with mean=500, std=200 → ascending CDF
    assert isinstance(val, float)
    assert 0 < val < 1


def test_parse_expr_complex_composite():
    """Test a realistic composite ranking expression."""
    text = (
        "0.6 * affinity.descending_cdf(500, 200) + "
        "0.4 * presentation.score.ascending_cdf(0.5, 0.3)"
    )
    expr = parse_expr(text)
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert isinstance(val, float)
    assert 0 < val  # both components should be positive


# ---------------------------------------------------------------------------
# AST ↔ string equivalence tests — verify parse_expr matches Python DSL
# ---------------------------------------------------------------------------


def _eval_both(ast_expr, text, df):
    """Evaluate both the Python AST and the parsed string expression."""
    parsed = parse_expr(text)
    ast_val = ast_expr.evaluate(df)
    parsed_val = parsed.evaluate(df)
    return ast_val, parsed_val


def _assert_equiv(ast_expr, text, df, tol=1e-9):
    """Assert AST and parsed string evaluate to the same value."""
    ast_val, parsed_val = _eval_both(ast_expr, text, df)
    assert abs(ast_val - parsed_val) < tol, (
        f"AST={ast_val}, parsed={parsed_val} for {text!r}"
    )


def test_equiv_bare_kind():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value, "affinity", df)


def test_equiv_kind_value():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value, "affinity.value", df)


def test_equiv_kind_score():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.score, "affinity.score", df)


def test_equiv_kind_rank():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.rank, "affinity.rank", df)


def test_equiv_presentation_score():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Presentation.score, "presentation.score", df)


def test_equiv_presentation_rank():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Presentation.rank, "presentation.rank", df)


def test_equiv_descending_cdf():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.descending_cdf(500, 200),
        "affinity.descending_cdf(500, 200)",
        df,
    )


def test_equiv_ascending_cdf():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Presentation.score.ascending_cdf(0.5, 0.3),
        "presentation.score.ascending_cdf(0.5, 0.3)",
        df,
    )


def test_equiv_norm_alias():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.value.norm(500, 200),
        "affinity.value.norm(500, 200)",
        df,
    )


def test_equiv_logistic():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.logistic(350, 150),
        "affinity.logistic(350, 150)",
        df,
    )


def test_equiv_log():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.log(), "affinity.value.log()", df)


def test_equiv_log2():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.log2(), "affinity.value.log2()", df)


def test_equiv_log10():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.log10(), "affinity.value.log10()", df)


def test_equiv_log1p():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.log1p(), "affinity.value.log1p()", df)


def test_equiv_exp():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.score.exp(), "affinity.score.exp()", df)


def test_equiv_sqrt():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.sqrt(), "affinity.value.sqrt()", df)


def test_equiv_clip():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.value.clip(100, 200),
        "affinity.value.clip(100, 200)",
        df,
    )


def test_equiv_clip_below():
    df = _make_df(PEPTIDE_B_ROWS)
    _assert_equiv(
        Affinity.value.clip(100, 200),
        "affinity.value.clip(100, 200)",
        df,
    )


def test_equiv_hinge():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value.hinge(), "affinity.value.hinge()", df)


def test_equiv_negation():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(-Affinity.value, "-affinity.value", df)


def test_equiv_abs():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(abs(-Affinity.value), "abs(-affinity.value)", df)


def test_equiv_power():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value ** 2, "affinity.value ** 2", df)


def test_equiv_addition():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.score + Presentation.score,
        "affinity.score + presentation.score",
        df,
    )


def test_equiv_subtraction():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.score - Presentation.score,
        "affinity.score - presentation.score",
        df,
    )


def test_equiv_multiplication():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        0.5 * Affinity.score,
        "0.5 * affinity.score",
        df,
    )


def test_equiv_division():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.value / 100,
        "affinity.value / 100",
        df,
    )


def test_equiv_weighted_sum():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        0.5 * Affinity.score + 0.5 * Presentation.score,
        "0.5 * affinity.score + 0.5 * presentation.score",
        df,
    )


def test_equiv_weighted_cdf_composite():
    df = _make_df(PEPTIDE_A_ROWS)
    ast_expr = (
        0.6 * Affinity.descending_cdf(500, 200)
        + 0.4 * Presentation.score.ascending_cdf(0.5, 0.3)
    )
    text = (
        "0.6 * affinity.descending_cdf(500, 200) + "
        "0.4 * presentation.score.ascending_cdf(0.5, 0.3)"
    )
    _assert_equiv(ast_expr, text, df)


def test_equiv_parenthesized():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        (Affinity.score + Presentation.score) * 0.5,
        "(affinity.score + presentation.score) * 0.5",
        df,
    )


def test_equiv_mean():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        mean(Affinity.score, Presentation.score),
        "mean(affinity.score, presentation.score)",
        df,
    )


def test_equiv_geomean():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        geomean(Affinity.score, Presentation.score),
        "geomean(affinity.score, presentation.score)",
        df,
    )


def test_equiv_minimum():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        minimum(Affinity.score, Presentation.score),
        "minimum(affinity.score, presentation.score)",
        df,
    )


def test_equiv_maximum():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        maximum(Affinity.score, Presentation.score),
        "maximum(affinity.score, presentation.score)",
        df,
    )


def test_equiv_median():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        median(Affinity.score, Presentation.score),
        "median(affinity.score, presentation.score)",
        df,
    )


def test_equiv_column():
    df = _make_df(PEPTIDE_A_ROWS)
    df["hydrophobicity"] = 0.75
    _assert_equiv(
        Column("hydrophobicity"),
        "column(hydrophobicity)",
        df,
    )


def test_equiv_column_in_arithmetic():
    df = _make_df(PEPTIDE_A_ROWS)
    df["cysteine_count"] = 2
    _assert_equiv(
        0.5 * Affinity.score - 0.1 * Column("cysteine_count"),
        "0.5 * affinity.score - 0.1 * column(cysteine_count)",
        df,
    )


def test_equiv_chained_transforms():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.value.clip(50, 500).log(),
        "affinity.value.clip(50, 500).log()",
        df,
    )


def test_equiv_chained_transforms_sqrt_hinge():
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(
        Affinity.value.hinge().sqrt(),
        "affinity.value.hinge().sqrt()",
        df,
    )


def test_equiv_complex_realistic():
    """A realistic multi-model composite score."""
    df = _make_df(PEPTIDE_A_ROWS)
    ast_expr = (
        0.4 * Affinity.descending_cdf(500, 200)
        + 0.4 * Presentation.score.ascending_cdf(0.5, 0.3)
        + 0.2 * Affinity.value.clip(1, 50000).log()
    )
    text = (
        "0.4 * affinity.descending_cdf(500, 200) + "
        "0.4 * presentation.score.ascending_cdf(0.5, 0.3) + "
        "0.2 * affinity.value.clip(1, 50000).log()"
    )
    _assert_equiv(ast_expr, text, df)


def test_equiv_kind_aliases():
    """ba, aff, ic50, el are aliases for affinity/presentation."""
    df = _make_df(PEPTIDE_A_ROWS)
    _assert_equiv(Affinity.value, "ba", df)
    _assert_equiv(Affinity.value, "aff", df)
    _assert_equiv(Affinity.value, "ic50", df)
    # el → presentation; use .score since presentation value is NaN in fixture
    _assert_equiv(Presentation.score, "el.score", df)


def test_equiv_operator_precedence():
    """Multiplication binds tighter than addition."""
    df = _make_df(PEPTIDE_A_ROWS)
    # 2 * a + 3 * p should be (2*a) + (3*p), not 2*(a+3)*p
    ast_expr = 2 * Affinity.score + 3 * Presentation.score
    _assert_equiv(ast_expr, "2 * affinity.score + 3 * presentation.score", df)


def test_equiv_right_associative_power():
    """2 ** 3 ** 2 should be 2 ** (3 ** 2) = 512, not (2 ** 3) ** 2 = 64."""
    expr = parse_expr("2 ** 3 ** 2")
    df = _make_df(PEPTIDE_A_ROWS)
    val = expr.evaluate(df)
    assert abs(val - 512.0) < 1e-9


def test_equiv_subtraction_vs_negation():
    """a - b should differ from a + (-b) only in AST shape, not value."""
    df = _make_df(PEPTIDE_A_ROWS)
    ast_sub = Affinity.score - Presentation.score
    ast_neg = Affinity.score + (-Presentation.score)
    sub_val = ast_sub.evaluate(df)
    neg_val = ast_neg.evaluate(df)
    parsed_val = parse_expr("affinity.score - presentation.score").evaluate(df)
    assert abs(sub_val - neg_val) < 1e-9
    assert abs(sub_val - parsed_val) < 1e-9


# ---------------------------------------------------------------------------
# Multi-dot chains, scopes, bracket+chain, peptide properties via parse_expr
# ---------------------------------------------------------------------------


WT_ROWS = [
    dict(
        source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_affinity",
        score=0.8, value=120.0, percentile_rank=0.5,
        wt_score=0.3, wt_value=800.0, wt_percentile_rank=5.0,
        prediction_method_name="netmhcpan",
    ),
    dict(
        source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_presentation",
        score=0.92, value=None, percentile_rank=0.3,
        wt_score=0.4, wt_value=None, wt_percentile_rank=3.0,
        prediction_method_name="netmhcpan",
    ),
]

PROPERTY_ROWS = [
    dict(
        source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_affinity",
        score=0.8, value=120.0, percentile_rank=0.5,
        prediction_method_name="netmhcpan",
        charge=-1.0, hydrophobicity=0.5, cysteine_count=0,
    ),
]


# --- Triple-chain transforms ---


def test_parse_triple_chain():
    """value.hinge().sqrt().log() — three chained transforms."""
    df = _make_df(PEPTIDE_A_ROWS)
    ast_expr = Affinity.value.hinge().sqrt().log()
    _assert_equiv(ast_expr, "affinity.value.hinge().sqrt().log()", df)


def test_parse_quad_chain():
    """value.clip().hinge().sqrt().log() — four chained transforms."""
    df = _make_df(PEPTIDE_A_ROWS)
    ast_expr = Affinity.value.clip(50, 1000).hinge().sqrt().log()
    _assert_equiv(
        ast_expr,
        "affinity.value.clip(50, 1000).hinge().sqrt().log()",
        df,
    )


def test_parse_chain_with_cdf():
    """descending_cdf after clip — chain CDF onto another transform."""
    df = _make_df(PEPTIDE_A_ROWS)
    ast_expr = Affinity.value.clip(50, 1000).descending_cdf(500, 200)
    _assert_equiv(
        ast_expr,
        "affinity.value.clip(50, 1000).descending_cdf(500, 200)",
        df,
    )


# --- Scope prefix (wt.) via parse_expr ---


def test_parse_wt_bare():
    """wt.affinity → wt.affinity.value"""
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.value
    _assert_equiv(ast_expr, "wt.affinity", df)


def test_parse_wt_score():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.score
    _assert_equiv(ast_expr, "wt.affinity.score", df)


def test_parse_wt_rank():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.rank
    _assert_equiv(ast_expr, "wt.affinity.rank", df)


def test_parse_wt_value():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.value
    _assert_equiv(ast_expr, "wt.affinity.value", df)


def test_parse_wt_descending_cdf():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.descending_cdf(500, 200)
    _assert_equiv(ast_expr, "wt.affinity.descending_cdf(500, 200)", df)


def test_parse_wt_logistic():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.logistic(350, 150)
    _assert_equiv(ast_expr, "wt.affinity.logistic(350, 150)", df)


def test_parse_wt_in_arithmetic():
    """affinity.score - wt.affinity.score — mutant vs wildtype delta."""
    df = _make_df(WT_ROWS)
    ast_expr = Affinity.score - wt.Affinity.score
    _assert_equiv(ast_expr, "affinity.score - wt.affinity.score", df)


def test_parse_wt_presentation_score():
    df = _make_df(WT_ROWS)
    ast_expr = wt.Presentation.score
    _assert_equiv(ast_expr, "wt.presentation.score", df)


def test_parse_wt_value_transform_chain():
    """wt.affinity.value.clip(100, 5000).log()"""
    df = _make_df(WT_ROWS)
    ast_expr = wt.Affinity.value.clip(100, 5000).log()
    _assert_equiv(
        ast_expr,
        "wt.affinity.value.clip(100, 5000).log()",
        df,
    )


def test_parse_wt_composite():
    """Realistic composite: mutant score weighted against WT delta."""
    df = _make_df(WT_ROWS)
    ast_expr = (
        0.7 * Affinity.descending_cdf(500, 200)
        + 0.3 * (Affinity.score - wt.Affinity.score)
    )
    text = (
        "0.7 * affinity.descending_cdf(500, 200) + "
        "0.3 * (affinity.score - wt.affinity.score)"
    )
    _assert_equiv(ast_expr, text, df)


# --- Bracket qualification + chaining ---


def test_parse_bracket_then_chain():
    """affinity["netmhcpan"].value.clip(50, 500).log()"""
    df = _make_df(WT_ROWS)
    ast_expr = Affinity["netmhcpan"].value.clip(50, 500).log()
    _assert_equiv(
        ast_expr,
        'affinity["netmhcpan"].value.clip(50, 500).log()',
        df,
    )


def test_parse_bracket_descending_cdf():
    """affinity["netmhcpan"].descending_cdf(500, 200)"""
    df = _make_df(WT_ROWS)
    ast_expr = Affinity["netmhcpan"].descending_cdf(500, 200)
    _assert_equiv(
        ast_expr,
        'affinity["netmhcpan"].descending_cdf(500, 200)',
        df,
    )


def test_parse_bracket_score():
    """affinity["netmhcpan"].score"""
    df = _make_df(WT_ROWS)
    ast_expr = Affinity["netmhcpan"].score
    _assert_equiv(ast_expr, 'affinity["netmhcpan"].score', df)


def test_parse_bracket_in_arithmetic():
    """Weighted sum with bracket-qualified accessors."""
    df = _make_df(WT_ROWS)
    ast_expr = (
        0.5 * Affinity["netmhcpan"].score
        + 0.5 * Presentation["netmhcpan"].score
    )
    text = '0.5 * affinity["netmhcpan"].score + 0.5 * presentation["netmhcpan"].score'
    _assert_equiv(ast_expr, text, df)


# --- Peptide properties via Column ---


def test_parse_column_property_hydrophobicity():
    df = _make_df(PROPERTY_ROWS)
    ast_expr = Column("hydrophobicity")
    _assert_equiv(ast_expr, "column(hydrophobicity)", df)


def test_parse_column_property_charge():
    df = _make_df(PROPERTY_ROWS)
    ast_expr = Column("charge")
    _assert_equiv(ast_expr, "column(charge)", df)


def test_parse_column_property_in_composite():
    """Realistic: score weighted by peptide properties."""
    df = _make_df(PROPERTY_ROWS)
    ast_expr = (
        0.6 * Affinity.score
        + 0.2 * Column("hydrophobicity")
        - 0.1 * Column("cysteine_count")
    )
    text = (
        "0.6 * affinity.score + "
        "0.2 * column(hydrophobicity) - "
        "0.1 * column(cysteine_count)"
    )
    _assert_equiv(ast_expr, text, df)


def test_parse_column_with_transform():
    """column(charge).abs() — transform on a column reference."""
    df = _make_df(PROPERTY_ROWS)
    ast_expr = abs(Column("charge"))
    _assert_equiv(ast_expr, "abs(column(charge))", df)


# --- Kitchen sink: everything combined ---


def test_parse_column_with_hinge():
    df = _make_df(PROPERTY_ROWS)
    ast_expr = Column("charge").hinge()
    _assert_equiv(ast_expr, "column(charge).hinge()", df)


def test_parse_kitchen_sink():
    """Combine WT, bracket, transforms, column, arithmetic in one expression."""
    df = _make_df(WT_ROWS)
    df["hydrophobicity"] = 0.5
    ast_expr = (
        0.4 * Affinity["netmhcpan"].descending_cdf(500, 200)
        + 0.3 * (Affinity.score - wt.Affinity.score)
        + 0.1 * Column("hydrophobicity")
    )
    text = (
        '0.4 * affinity["netmhcpan"].descending_cdf(500, 200) + '
        "0.3 * (affinity.score - wt.affinity.score) + "
        "0.1 * column(hydrophobicity)"
    )
    _assert_equiv(ast_expr, text, df)


# --- Underscore-qualified kind names (e.g. netmhcpan_affinity) ---


def test_parse_underscore_qualified_kind():
    """netmhcpan_affinity → Affinity["netmhcpan"].value"""
    df = _make_df(WT_ROWS)
    ast_expr = Affinity["netmhcpan"].value
    _assert_equiv(ast_expr, "netmhcpan_affinity", df)


def test_parse_underscore_qualified_ba():
    df = _make_df(WT_ROWS)
    ast_expr = Affinity["netmhcpan"].value
    _assert_equiv(ast_expr, "netmhcpan_ba", df)


def test_parse_underscore_qualified_el():
    df = _make_df(WT_ROWS)
    ast_expr = Presentation["netmhcpan"].value
    # presentation value is NaN, so use score
    _assert_equiv(Presentation["netmhcpan"].score, "netmhcpan_el.score", df)


def test_parse_underscore_qualified_with_field():
    df = _make_df(WT_ROWS)
    _assert_equiv(
        Affinity["netmhcpan"].score,
        "netmhcpan_affinity.score",
        df,
    )


def test_parse_underscore_qualified_with_rank():
    df = _make_df(WT_ROWS)
    _assert_equiv(
        Affinity["netmhcpan"].rank,
        "netmhcpan_affinity.rank",
        df,
    )


def test_parse_underscore_qualified_with_transform():
    df = _make_df(WT_ROWS)
    _assert_equiv(
        Affinity["netmhcpan"].descending_cdf(500, 200),
        "netmhcpan_affinity.descending_cdf(500, 200)",
        df,
    )


def test_parse_underscore_qualified_in_arithmetic():
    df = _make_df(WT_ROWS)
    _assert_equiv(
        0.5 * Affinity["netmhcpan"].score + 0.5 * Presentation["netmhcpan"].score,
        "0.5 * netmhcpan_affinity.score + 0.5 * netmhcpan_el.score",
        df,
    )


# --- Round-trip tests: parse → repr → parse → evaluate ---


def _assert_roundtrip(text, df, tol=1e-9):
    """Parse text, repr it, re-parse, compare evaluation."""
    expr1 = parse_expr(text)
    text2 = repr(expr1)
    expr2 = parse_expr(text2)
    v1 = expr1.evaluate(df)
    v2 = expr2.evaluate(df)
    assert abs(v1 - v2) < tol, (
        f"Round-trip mismatch: {text!r} -> {text2!r}, {v1} != {v2}"
    )


def test_roundtrip_bare_kind():
    _assert_roundtrip("affinity", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_kind_score():
    _assert_roundtrip("affinity.score", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_kind_rank():
    _assert_roundtrip("affinity.rank", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_descending_cdf():
    _assert_roundtrip("affinity.descending_cdf(500, 200)", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_ascending_cdf():
    _assert_roundtrip(
        "presentation.score.ascending_cdf(0.5, 0.3)",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_logistic():
    _assert_roundtrip("affinity.logistic(350, 150)", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_log():
    _assert_roundtrip("affinity.value.log()", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_log2():
    _assert_roundtrip("affinity.value.log2()", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_clip():
    _assert_roundtrip("affinity.value.clip(50, 500)", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_hinge():
    _assert_roundtrip("affinity.value.hinge()", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_sqrt():
    _assert_roundtrip("affinity.value.sqrt()", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_negation():
    _assert_roundtrip("-affinity.value", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_abs():
    _assert_roundtrip("abs(affinity.value)", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_power():
    _assert_roundtrip("affinity.value ** 2", _make_df(PEPTIDE_A_ROWS))


def test_roundtrip_arithmetic():
    _assert_roundtrip(
        "0.5 * affinity.score + 0.5 * presentation.score",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_parenthesized():
    _assert_roundtrip(
        "(affinity.score + presentation.score) * 0.5",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_mean():
    _assert_roundtrip(
        "mean(affinity.score, presentation.score)",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_geomean():
    _assert_roundtrip(
        "geomean(affinity.score, presentation.score)",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_minimum():
    _assert_roundtrip(
        "minimum(affinity.score, presentation.score)",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_column():
    df = _make_df(PEPTIDE_A_ROWS)
    df["hydrophobicity"] = 0.75
    _assert_roundtrip("column(hydrophobicity)", df)


def test_roundtrip_wt_score():
    _assert_roundtrip("wt.affinity.score", _make_df(WT_ROWS))


def test_roundtrip_wt_delta():
    _assert_roundtrip(
        "affinity.score - wt.affinity.score",
        _make_df(WT_ROWS),
    )


def test_roundtrip_chained_transforms():
    _assert_roundtrip(
        "affinity.value.clip(50, 500).log()",
        _make_df(PEPTIDE_A_ROWS),
    )


def test_roundtrip_bracket_qualified():
    _assert_roundtrip(
        'affinity["netmhcpan"].descending_cdf(500, 200)',
        _make_df(WT_ROWS),
    )


def test_roundtrip_complex_composite():
    df = _make_df(WT_ROWS)
    df["hydrophobicity"] = 0.5
    _assert_roundtrip(
        '0.4 * affinity["netmhcpan"].descending_cdf(500, 200) + '
        "0.3 * (affinity.score - wt.affinity.score) + "
        "0.1 * column(hydrophobicity)",
        df,
    )


# ---------------------------------------------------------------------------
# Len and Count expression tests
# ---------------------------------------------------------------------------

LEN_ROWS = [
    dict(
        source_sequence_name="var1", peptide="SIINFEKL", peptide_offset=10,
        allele="HLA-A*02:01", kind="pMHC_affinity",
        score=0.8, value=120.0, percentile_rank=0.5,
        peptide_length=8,
        wt_peptide="SIINFEAL", wt_peptide_length=8,
    ),
]


def test_len_default():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("len")
    assert isinstance(expr, Len)
    assert expr.evaluate(df) == 8.0


def test_len_wt_scope():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("wt.len")
    assert isinstance(expr, Len)
    assert expr.scope == "wt_"
    assert expr.evaluate(df) == 8.0


def test_len_missing_column_nan():
    df = _make_df(PEPTIDE_A_ROWS)
    expr = parse_expr("wt.len")
    assert math.isnan(expr.evaluate(df))


def test_len_in_arithmetic():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("len - 9")
    assert expr.evaluate(df) == -1.0


def test_len_repr_roundtrip():
    expr = parse_expr("len")
    assert repr(expr) == "len"
    expr2 = parse_expr(repr(expr))
    assert isinstance(expr2, Len)


def test_len_wt_repr_roundtrip():
    expr = parse_expr("wt.len")
    assert repr(expr) == "wt.len"
    expr2 = parse_expr(repr(expr))
    assert isinstance(expr2, Len)
    assert expr2.scope == "wt_"


def test_count_single_aa():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("count('K')")
    assert isinstance(expr, Count)
    assert expr.evaluate(df) == 1.0  # SIINFEKL has one K


def test_count_multiple_aa():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("count('KL')")
    # SIINFEKL: K=1, L=1
    assert expr.evaluate(df) == 2.0


def test_count_wt_scope():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("wt.count('K')")
    assert expr.scope == "wt_"
    # wt_peptide = SIINFEAL: no K
    assert expr.evaluate(df) == 0.0


def test_count_cysteine():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("count('C')")
    # SIINFEKL has no C
    assert expr.evaluate(df) == 0.0


def test_count_in_arithmetic():
    df = _make_df(LEN_ROWS)
    expr = parse_expr("count('K') - wt.count('K')")
    # SIINFEKL has K=1, SIINFEAL has K=0
    assert expr.evaluate(df) == 1.0


def test_count_double_quote():
    df = _make_df(LEN_ROWS)
    expr = parse_expr('count("KR")')
    # SIINFEKL: K=1, R=0
    assert expr.evaluate(df) == 1.0


def test_count_repr_roundtrip():
    expr = parse_expr("count('C')")
    assert repr(expr) == "count('C')"
    expr2 = parse_expr(repr(expr))
    assert isinstance(expr2, Count)
    assert expr2.chars == "C"


def test_count_wt_repr_roundtrip():
    expr = parse_expr("wt.count('KR')")
    assert repr(expr) == "wt.count('KR')"
    expr2 = parse_expr(repr(expr))
    assert isinstance(expr2, Count)
    assert expr2.chars == "KR"
    assert expr2.scope == "wt_"


def test_count_missing_peptide_nan():
    df = _make_df(PEPTIDE_A_ROWS)
    expr = parse_expr("wt.count('C')")
    assert math.isnan(expr.evaluate(df))


# ---------------------------------------------------------------------------
# Scope Python API tests
# ---------------------------------------------------------------------------


def test_scope_python_api_field():
    field = wt.Affinity.score
    assert isinstance(field, Field)
    assert field.scope == "wt_"
    assert field.field == "score"


def test_scope_python_api_value():
    field = wt.Affinity.value
    assert isinstance(field, Field)
    assert field.scope == "wt_"
    assert field.field == "value"


def test_scope_python_api_bracket():
    field = wt.Affinity["netmhcpan"].score
    assert isinstance(field, Field)
    assert field.scope == "wt_"
    assert field.method == "netmhcpan"


def test_scope_python_api_alias():
    field = wt.ba.score
    assert isinstance(field, Field)
    assert field.scope == "wt_"


def test_scope_python_api_len():
    expr = wt.len
    assert isinstance(expr, Len)
    assert expr.scope == "wt_"


def test_scope_python_api_count():
    expr = wt.count("C")
    assert isinstance(expr, Count)
    assert expr.scope == "wt_"
    assert expr.chars == "C"


def test_scope_repr():
    assert repr(wt) == "wt"


def test_scope_unknown_attr_raises():
    with pytest.raises(AttributeError, match="Unknown kind"):
        wt.nonexistent


def test_reserved_keyword_in_parser():
    """Context keywords can't be used as kind names."""
    with pytest.raises(ValueError, match="reserved context keyword"):
        parse_expr("wt")


def test_reserved_keyword_no_dot_raises():
    """Context keyword without dot raises."""
    with pytest.raises(ValueError, match="reserved context keyword"):
        parse_expr("self")


# ---------------------------------------------------------------------------
# Scope round-trip tests
# ---------------------------------------------------------------------------


def test_roundtrip_wt_len():
    df = _make_df(LEN_ROWS)
    _assert_roundtrip("wt.len", df)


def test_roundtrip_wt_count():
    df = _make_df(LEN_ROWS)
    _assert_roundtrip("wt.count('K')", df)


def test_roundtrip_len_arithmetic():
    df = _make_df(LEN_ROWS)
    _assert_roundtrip("len - 9", df)


def test_roundtrip_count_arithmetic():
    df = _make_df(LEN_ROWS)
    _assert_roundtrip("count('K') - wt.count('K')", df)


def test_roundtrip_wt_scope_composite():
    df = _make_df(WT_ROWS)
    df["peptide_length"] = 8
    df["wt_peptide_length"] = 8
    df["wt_peptide"] = "SIINFEAL"
    _assert_roundtrip(
        "0.5 * affinity.score - 0.3 * wt.affinity.score + 0.1 * len",
        df,
    )
