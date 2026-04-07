"""Unit tests for topiary.ranking — filtering and ranking of multi-kind predictions."""

import pandas as pd
from mhctools import Kind

from topiary.ranking import (
    EpitopeFilter,
    RankingStrategy,
    affinity_filter,
    apply_ranking_strategy,
    presentation_filter,
)


def _make_df(rows):
    """Build a small predictions DataFrame from a list of dicts."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures: two peptides, each with affinity + presentation rows
# ---------------------------------------------------------------------------

PEPTIDE_A_ROWS = [
    dict(
        source_sequence_name="var1",
        peptide="SIINFEKL",
        peptide_offset=10,
        allele="HLA-A*02:01",
        kind="pMHC_affinity",
        score=0.8,
        value=120.0,
        percentile_rank=0.5,
    ),
    dict(
        source_sequence_name="var1",
        peptide="SIINFEKL",
        peptide_offset=10,
        allele="HLA-A*02:01",
        kind="pMHC_presentation",
        score=0.92,
        value=None,
        percentile_rank=0.3,
    ),
]

PEPTIDE_B_ROWS = [
    dict(
        source_sequence_name="var1",
        peptide="ELAGIGIL",
        peptide_offset=20,
        allele="HLA-A*02:01",
        kind="pMHC_affinity",
        score=0.1,
        value=5000.0,  # weak binder
        percentile_rank=15.0,
    ),
    dict(
        source_sequence_name="var1",
        peptide="ELAGIGIL",
        peptide_offset=20,
        allele="HLA-A*02:01",
        kind="pMHC_presentation",
        score=0.05,
        value=None,
        percentile_rank=20.0,
    ),
]


def _two_peptide_df():
    return _make_df(PEPTIDE_A_ROWS + PEPTIDE_B_ROWS)


# ---------------------------------------------------------------------------
# Tests: affinity filter
# ---------------------------------------------------------------------------


def test_affinity_filter_ic50():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[affinity_filter(ic50_cutoff=500)],
    )
    result = apply_ranking_strategy(df, strategy)
    # Only peptide A has IC50 < 500; both its rows (affinity + presentation) kept
    assert set(result["peptide"]) == {"SIINFEKL"}
    assert len(result) == 2  # affinity + presentation rows


def test_affinity_filter_percentile():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[affinity_filter(percentile_cutoff=1.0)],
    )
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: presentation filter
# ---------------------------------------------------------------------------


def test_presentation_filter_rank():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[presentation_filter(max_rank=1.0)],
    )
    result = apply_ranking_strategy(df, strategy)
    # Only peptide A has presentation rank < 1.0
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_presentation_filter_score():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[presentation_filter(min_score=0.5)],
    )
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL"}


# ---------------------------------------------------------------------------
# Tests: OR / AND logic
# ---------------------------------------------------------------------------


def test_or_logic():
    """OR: peptide passes if ANY filter matches."""
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[
            affinity_filter(ic50_cutoff=500),
            presentation_filter(min_score=0.01),  # very permissive — both pass
        ],
        require_all=False,
    )
    result = apply_ranking_strategy(df, strategy)
    assert set(result["peptide"]) == {"SIINFEKL", "ELAGIGIL"}


def test_and_logic():
    """AND: peptide must pass ALL filters."""
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[
            affinity_filter(ic50_cutoff=500),
            presentation_filter(min_score=0.5),
        ],
        require_all=True,
    )
    result = apply_ranking_strategy(df, strategy)
    # Only peptide A passes both
    assert set(result["peptide"]) == {"SIINFEKL"}


def test_and_logic_nothing_passes():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        filters=[
            affinity_filter(ic50_cutoff=10),  # nothing this tight
            presentation_filter(min_score=0.99),
        ],
        require_all=True,
    )
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: sort_by
# ---------------------------------------------------------------------------


def test_sort_by_presentation_score():
    df = _two_peptide_df()
    strategy = RankingStrategy(
        sort_by=[(Kind.pMHC_presentation, "score")],
    )
    result = apply_ranking_strategy(df, strategy)
    # Peptide A has better presentation score, should come first
    assert result.iloc[0]["peptide"] == "SIINFEKL"


def test_sort_by_with_fallback():
    """When presentation rows are absent, fall back to affinity."""
    # Create a DataFrame with only affinity rows
    rows = [
        dict(
            source_sequence_name="v1", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.9, value=50.0,
            percentile_rank=0.1,
        ),
        dict(
            source_sequence_name="v1", peptide="BBB", peptide_offset=5,
            allele="A", kind="pMHC_affinity", score=0.1, value=5000.0,
            percentile_rank=15.0,
        ),
    ]
    df = _make_df(rows)
    strategy = RankingStrategy(
        sort_by=[
            (Kind.pMHC_presentation, "score"),  # not present
            (Kind.pMHC_affinity, "score"),  # fallback
        ],
    )
    result = apply_ranking_strategy(df, strategy)
    assert result.iloc[0]["peptide"] == "AAA"


# ---------------------------------------------------------------------------
# Tests: no filters = passthrough
# ---------------------------------------------------------------------------


def test_no_filters_passthrough():
    df = _two_peptide_df()
    strategy = RankingStrategy()
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == len(df)


# ---------------------------------------------------------------------------
# Tests: empty DataFrame
# ---------------------------------------------------------------------------


def test_empty_df():
    df = pd.DataFrame()
    strategy = RankingStrategy(filters=[affinity_filter(ic50_cutoff=500)])
    result = apply_ranking_strategy(df, strategy)
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
# Tests: variant column (renamed from source_sequence_name)
# ---------------------------------------------------------------------------


def test_variant_column_grouping():
    """Ranking works when column is 'variant' instead of 'source_sequence_name'."""
    rows = [
        dict(
            variant="chr7 p.V600E",
            peptide="SIINFEKL",
            peptide_offset=10,
            allele="HLA-A*02:01",
            kind="pMHC_affinity",
            score=0.8,
            value=120.0,
            percentile_rank=0.5,
        ),
    ]
    df = _make_df(rows)
    strategy = RankingStrategy(
        filters=[affinity_filter(ic50_cutoff=500)],
    )
    result = apply_ranking_strategy(df, strategy)
    assert len(result) == 1
