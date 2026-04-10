"""Tests for DSL extensions: logistic, Column, scopes, peptide properties."""

import math

import pandas as pd
import pytest
from mhctools import Kind

from topiary.ranking import (
    Affinity,
    Column,
    ColumnFilter,
    EpitopeFilter,
    KindAccessor,
    Presentation,
    RankingStrategy,
    Stability,
    apply_ranking_strategy,
    parse_filter,
    parse_ranking,
    wt,
)


def _make_df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _basic_group():
    """Single peptide-allele group with affinity + presentation."""
    return _make_df([
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            prediction_method_name="netmhcpan",
        ),
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_presentation",
            score=0.92, value=None, percentile_rank=0.3,
            prediction_method_name="netmhcpan",
        ),
    ])


def _group_with_wt():
    """Group with wt_* columns populated."""
    return _make_df([
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
    ])


def _group_with_properties():
    """Group with peptide property columns."""
    return _make_df([
        dict(
            source_sequence_name="seq1", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.8, value=120.0, percentile_rank=0.5,
            prediction_method_name="netmhcpan",
            charge=-1.0, hydrophobicity=0.5, cysteine_count=0,
        ),
    ])


# ---------------------------------------------------------------------------
# Tests: .logistic()
# ---------------------------------------------------------------------------


class TestLogistic:
    def test_basic(self):
        df = _basic_group()
        # IC50=120, midpoint=350, width=150 → strong binder → high score
        val = Affinity.logistic(midpoint=350, width=150).evaluate(df)
        assert 0.5 < val < 1.0  # 120 is well below midpoint

    def test_at_midpoint(self):
        """At the midpoint, logistic should return 0.5."""
        df = _make_df([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.5, value=350.0,
            percentile_rank=1.0,
        )])
        val = Affinity.logistic(midpoint=350, width=150).evaluate(df)
        assert abs(val - 0.5) < 1e-10

    def test_high_ic50_low_score(self):
        """Very high IC50 → near 0."""
        df = _make_df([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.1, value=10000.0,
            percentile_rank=50.0,
        )])
        val = Affinity.logistic(midpoint=350, width=150).evaluate(df)
        assert val < 0.01

    def test_nan_input(self):
        df = _make_df([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=float("nan"),
            value=float("nan"), percentile_rank=float("nan"),
        )])
        val = Affinity.logistic(350, 150).evaluate(df)
        assert math.isnan(val)

    def test_width_zero(self):
        df = _basic_group()
        val = Affinity.logistic(350, 0).evaluate(df)
        assert math.isnan(val)

    def test_overflow_protection(self):
        """Very large (x - midpoint) / width shouldn't crash."""
        df = _make_df([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.0, value=1e10,
            percentile_rank=99.0,
        )])
        val = Affinity.logistic(350, 150).evaluate(df)
        assert val == 0.0  # effectively 0

    def test_kind_accessor_delegation(self):
        """KindAccessor.logistic() should delegate to .value.logistic()."""
        df = _basic_group()
        via_accessor = Affinity.logistic(350, 150).evaluate(df)
        via_field = Affinity.value.logistic(350, 150).evaluate(df)
        assert via_accessor == via_field

    def test_qualified(self):
        """Bracket + logistic."""
        df = _basic_group()
        val = Affinity["netmhcpan"].logistic(350, 150).evaluate(df)
        assert 0.5 < val < 1.0

    def test_in_composite(self):
        """Logistic in arithmetic expression."""
        df = _basic_group()
        expr = 0.5 * Affinity.logistic(350, 150) + 0.5 * Presentation.score
        val = expr.evaluate(df)
        assert isinstance(val, float)
        assert not math.isnan(val)


# ---------------------------------------------------------------------------
# Tests: Column()
# ---------------------------------------------------------------------------


class TestColumn:
    def test_basic_read(self):
        df = _group_with_properties()
        assert Column("charge").evaluate(df) == -1.0
        assert Column("hydrophobicity").evaluate(df) == 0.5
        assert Column("cysteine_count").evaluate(df) == 0.0

    def test_missing_column_raises(self):
        df = _basic_group()
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            Column("nonexistent").evaluate(df)

    def test_missing_column_suggests_close_match(self):
        df = _group_with_properties()
        with pytest.raises(ValueError, match="Did you mean"):
            Column("hydrophobicty").evaluate(df)  # typo

    def test_in_arithmetic(self):
        df = _group_with_properties()
        expr = 0.5 * Affinity.score - 0.2 * Column("cysteine_count")
        val = expr.evaluate(df)
        assert val == 0.5 * 0.8 - 0.2 * 0.0

    def test_empty_df(self):
        df = _make_df([])
        val = Column("anything").evaluate(df)
        assert math.isnan(val)

    def test_nan_value(self):
        df = _make_df([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
            percentile_rank=1.0, charge=float("nan"),
        )])
        val = Column("charge").evaluate(df)
        assert math.isnan(val)

    def test_transforms(self):
        """Column supports all Expr transforms."""
        df = _group_with_properties()
        # charge = -1.0
        assert abs(Column("charge")).evaluate(df) == 1.0
        assert Column("charge").clip(lo=0).evaluate(df) == 0.0

    def test_non_numeric_column_raises(self):
        """String column gives clear error, not generic ValueError."""
        df = _make_df([dict(
            source_sequence_name="x", peptide="AAA", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
            percentile_rank=1.0, gene_name="BRAF",
        )])
        with pytest.raises(TypeError, match="non-numeric value"):
            Column("gene_name").evaluate(df)


class TestColumnFilter:
    def test_parse_column_filter(self):
        f = parse_filter("column(cysteine_count) <= 2")
        assert isinstance(f, ColumnFilter)
        assert f.col_name == "cysteine_count"
        assert f.max_value == 2.0

    def test_parse_column_filter_ge(self):
        f = parse_filter("column(hydrophobicity) >= -0.5")
        assert isinstance(f, ColumnFilter)
        assert f.col_name == "hydrophobicity"
        assert f.min_value == -0.5

    def test_column_filter_in_strategy(self):
        df = _make_df([
            dict(
                source_sequence_name="seq1", peptide="AAA", peptide_offset=0,
                allele="A", kind="pMHC_affinity", score=0.9, value=50.0,
                percentile_rank=0.1, cysteine_count=0,
            ),
            dict(
                source_sequence_name="seq1", peptide="CCC", peptide_offset=5,
                allele="A", kind="pMHC_affinity", score=0.8, value=80.0,
                percentile_rank=0.2, cysteine_count=3,
            ),
        ])
        strategy = RankingStrategy(
            filters=[ColumnFilter("cysteine_count", max_value=1)]
        )
        result = apply_ranking_strategy(df, strategy)
        assert set(result["peptide"]) == {"AAA"}

    def test_column_filter_combined_with_epitope_filter(self):
        f = parse_ranking("affinity <= 500 & column(cysteine_count) <= 2")
        assert isinstance(f, RankingStrategy)
        assert f.require_all is True
        assert len(f.filters) == 2

    def test_column_filter_or(self):
        f = parse_ranking("affinity <= 500 | column(charge) >= 0")
        assert isinstance(f, RankingStrategy)
        assert f.require_all is False

    def test_parse_column_empty_name_raises(self):
        with pytest.raises(ValueError):
            parse_filter("column() <= 5")

    def test_parse_column_nested_parens_raises(self):
        with pytest.raises(ValueError):
            parse_filter("column(func()) <= 5")


# ---------------------------------------------------------------------------
# Tests: wt scope
# ---------------------------------------------------------------------------


class TestWTScope:
    def test_basic_wt_access(self):
        df = _group_with_wt()
        assert wt.Affinity.value.evaluate(df) == 800.0
        assert wt.Affinity.score.evaluate(df) == 0.3
        assert wt.Affinity.rank.evaluate(df) == 5.0

    def test_differential_binding(self):
        df = _group_with_wt()
        diff = Affinity.score - wt.Affinity.score
        val = diff.evaluate(df)
        assert val == 0.8 - 0.3

    def test_wt_qualified(self):
        df = _group_with_wt()
        val = wt.Affinity["netmhcpan"].value.evaluate(df)
        assert val == 800.0

    def test_wt_bracket(self):
        df = _group_with_wt()
        val = wt.Affinity["netmhcpan"].value.evaluate(df)
        assert val == 800.0

    def test_wt_no_columns_returns_nan(self):
        """When wt_* columns don't exist, returns NaN."""
        df = _basic_group()
        val = wt.Affinity.value.evaluate(df)
        assert math.isnan(val)

    def test_wt_comparison_raises(self):
        """Scoped fields can't be used in filters — only in ranking expressions."""
        with pytest.raises(TypeError, match="Scoped fields"):
            wt.Affinity <= 1000

    def test_wt_logistic(self):
        df = _group_with_wt()
        val = wt.Affinity.logistic(350, 150).evaluate(df)
        assert 0 < val < 0.5  # WT IC50=800, above midpoint

    def test_wt_norm(self):
        df = _group_with_wt()
        val = wt.Affinity.norm(500, 200).evaluate(df)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_wt_arithmetic(self):
        df = _group_with_wt()
        expr = 0.5 * wt.Affinity + 0.5 * Affinity
        val = expr.evaluate(df)
        assert val == 0.5 * 800.0 + 0.5 * 120.0

    def test_wt_presentation(self):
        df = _group_with_wt()
        val = wt.Presentation.score.evaluate(df)
        assert val == 0.4


# ---------------------------------------------------------------------------
# Tests: peptide properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_add_all(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df)
        assert "charge" in result.columns
        assert "hydrophobicity" in result.columns
        assert "aromaticity" in result.columns
        assert "molecular_weight" in result.columns
        assert "cysteine_count" in result.columns
        assert "instability_index" in result.columns
        assert "tcr_charge" in result.columns

    def test_core_group(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, groups=["core"])
        assert "charge" in result.columns
        assert "hydrophobicity" in result.columns
        assert "aromaticity" in result.columns
        assert "molecular_weight" in result.columns
        assert "cysteine_count" not in result.columns
        assert "tcr_charge" not in result.columns

    def test_manufacturability_group(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, groups=["manufacturability"])
        assert "cysteine_count" in result.columns
        assert "asp_pro_bonds" in result.columns
        assert "difficult_nterm" in result.columns
        # core is included
        assert "charge" in result.columns
        # immunogenicity-only excluded
        assert "tcr_charge" not in result.columns

    def test_immunogenicity_group(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, groups=["immunogenicity"])
        assert "tcr_charge" in result.columns
        assert "tcr_aromaticity" in result.columns
        assert "tcr_hydrophobicity" in result.columns
        # core is included
        assert "charge" in result.columns

    def test_include_specific(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, include=["charge", "cysteine_count"])
        assert "charge" in result.columns
        assert "cysteine_count" in result.columns
        assert "hydrophobicity" not in result.columns

    def test_charge_values(self):
        from topiary.properties import add_peptide_properties
        # K and R are +1, E and D are -1
        df = _make_df([
            dict(peptide="KKRR"),    # +4
            dict(peptide="DDEE"),    # -4
            dict(peptide="AAAA"),    # 0
            dict(peptide="KRDE"),    # 0
        ])
        result = add_peptide_properties(df, include=["charge"])
        assert list(result["charge"]) == pytest.approx([4.0, -4.0, 0.0, 0.0], abs=0.2)

    def test_aromaticity_counts(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([
            dict(peptide="FWY"),     # 3
            dict(peptide="AAA"),     # 0
            dict(peptide="AFWY"),    # 3
        ])
        result = add_peptide_properties(df, include=["aromaticity"])
        assert list(result["aromaticity"]) == [3, 0, 3]

    def test_cysteine_count(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([
            dict(peptide="CCCC"),
            dict(peptide="AAAA"),
            dict(peptide="ACAC"),
        ])
        result = add_peptide_properties(df, include=["cysteine_count"])
        assert list(result["cysteine_count"]) == [4, 0, 2]

    def test_asp_pro_bonds(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([
            dict(peptide="ADPGDP"),   # 2 DP bonds
            dict(peptide="AAPAA"),    # 0
            dict(peptide="DPAAA"),    # 1
        ])
        result = add_peptide_properties(df, include=["asp_pro_bonds"])
        assert list(result["asp_pro_bonds"]) == [2, 0, 1]

    def test_difficult_nterm(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([
            dict(peptide="QAAA"),   # Q → difficult
            dict(peptide="EAAA"),   # E → difficult
            dict(peptide="CAAA"),   # C → difficult
            dict(peptide="AAAA"),   # A → fine
        ])
        result = add_peptide_properties(df, include=["difficult_nterm"])
        assert list(result["difficult_nterm"]) == [True, True, True, False]

    def test_difficult_cterm(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([
            dict(peptide="AAAP"),   # P → difficult
            dict(peptide="AAAC"),   # C → difficult
            dict(peptide="AAAA"),   # A → fine
        ])
        result = add_peptide_properties(df, include=["difficult_cterm"])
        assert list(result["difficult_cterm"]) == [True, True, False]

    def test_tcr_properties_9mer(self):
        from topiary.properties import add_peptide_properties
        # 9-mer: TCR positions are p4,p5,p6,p8 (0-indexed: 3,4,5,7)
        # SIINFEKL? → need 9-mer
        df = _make_df([dict(peptide="AAAAFFFAA")])
        # TCR-facing: positions 3,4,5,7 = A,F,F,A → 2 aromatic
        result = add_peptide_properties(df, include=["tcr_aromaticity"])
        assert result["tcr_aromaticity"].iloc[0] == 2

    def test_tcr_nan_for_unsupported_length(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="AAAAAAAAAAAAAAAA")])  # 16-mer, no TCR map
        result = add_peptide_properties(df, include=["tcr_aromaticity"])
        assert math.isnan(result["tcr_aromaticity"].iloc[0])

    def test_prefix(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="AAA", wt_peptide="KKK")])
        result = add_peptide_properties(
            df, include=["charge"], peptide_column="wt_peptide", prefix="wt_"
        )
        assert "wt_charge" in result.columns
        assert result["wt_charge"].iloc[0] == pytest.approx(3.0)

    def test_unknown_property_raises(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="AAA")])
        with pytest.raises(ValueError, match="Unknown properties"):
            add_peptide_properties(df, include=["nonexistent"])

    def test_unknown_group_raises(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="AAA")])
        with pytest.raises(ValueError, match="Unknown groups"):
            add_peptide_properties(df, groups=["bogus"])

    def test_missing_peptide_column_raises(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(sequence="AAA")])
        with pytest.raises(ValueError, match="Column 'peptide' not found"):
            add_peptide_properties(df)

    def test_does_not_mutate_input(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        original_cols = set(df.columns)
        add_peptide_properties(df, include=["charge"])
        assert set(df.columns) == original_cols

    def test_molecular_weight_reasonable(self):
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, include=["molecular_weight"])
        mw = result["molecular_weight"].iloc[0]
        # SIINFEKL is ~963 Da
        assert 900 < mw < 1050

    def test_instability_index_uses_dipeptides(self):
        from topiary.properties import add_peptide_properties
        # A sequence with known high instability
        df = _make_df([
            dict(peptide="AAAAAA"),  # mostly neutral dipeptides
            dict(peptide="HYHY"),    # HY has DIWV=44.94
        ])
        result = add_peptide_properties(df, include=["instability_index"])
        # HYHY should have higher instability than AAAAAA
        assert result["instability_index"].iloc[1] > result["instability_index"].iloc[0]


# ---------------------------------------------------------------------------
# Edge cases found during documentation review
# ---------------------------------------------------------------------------


class TestEdgeCasesFromDocs:
    def test_wt_bracket_qualified(self):
        """wt.Affinity["x"].value works correctly."""
        df = _group_with_wt()
        val = wt.Affinity["netmhcpan"].value.evaluate(df)
        assert val == 800.0

    def test_wt_ge_also_raises(self):
        """Scoped >= should also raise, not just <=."""
        with pytest.raises(TypeError, match="Scoped fields"):
            wt.Affinity >= 100

    def test_column_filter_both_bounds(self):
        """ColumnFilter with both min and max."""
        df = _make_df([
            dict(source_sequence_name="s", peptide="A", peptide_offset=0,
                 allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
                 percentile_rank=1.0, charge=-1.0),
            dict(source_sequence_name="s", peptide="B", peptide_offset=5,
                 allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
                 percentile_rank=1.0, charge=0.0),
            dict(source_sequence_name="s", peptide="C", peptide_offset=10,
                 allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
                 percentile_rank=1.0, charge=2.0),
        ])
        strategy = RankingStrategy(
            filters=[ColumnFilter("charge", min_value=-0.5, max_value=1.0)]
        )
        result = apply_ranking_strategy(df, strategy)
        assert set(result["peptide"]) == {"B"}

    def test_column_filter_or_with_epitope_filter(self):
        """ColumnFilter | EpitopeFilter works in apply_ranking_strategy."""
        df = _make_df([
            dict(source_sequence_name="s", peptide="AAA", peptide_offset=0,
                 allele="A", kind="pMHC_affinity", score=0.9, value=50.0,
                 percentile_rank=0.1, cysteine_count=5),
            dict(source_sequence_name="s", peptide="BBB", peptide_offset=5,
                 allele="A", kind="pMHC_affinity", score=0.1, value=9999.0,
                 percentile_rank=50.0, cysteine_count=0),
        ])
        # AAA passes affinity filter, BBB passes column filter
        strategy = (Affinity <= 500) | ColumnFilter("cysteine_count", max_value=1)
        result = apply_ranking_strategy(df, strategy)
        assert set(result["peptide"]) == {"AAA", "BBB"}

    def test_logistic_in_composite_with_column(self):
        """Logistic + Column in same expression."""
        df = _group_with_properties()
        expr = Affinity.logistic(350, 150) - 0.1 * Column("cysteine_count")
        val = expr.evaluate(df)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_wt_with_missing_kind(self):
        """Scoped kind with no rows returns NaN, not error."""
        df = _group_with_wt()
        # Stability kind doesn't exist in this df
        val = wt.Stability.value.evaluate(df)
        assert math.isnan(val)

    def test_multiple_groups_overlap(self):
        """Requesting overlapping groups doesn't duplicate columns."""
        from topiary.properties import add_peptide_properties
        df = _make_df([dict(peptide="SIINFEKL")])
        result = add_peptide_properties(df, groups=["core", "manufacturability"])
        # charge is in both groups but should appear only once
        assert list(result.columns).count("charge") == 1

    def test_parse_column_with_spaces(self):
        """column( name ) with spaces should work."""
        f = parse_filter("column( cysteine_count ) <= 2")
        assert isinstance(f, ColumnFilter)
        assert f.col_name == "cysteine_count"

    def test_available_properties(self):
        from topiary.properties import available_properties
        props = available_properties()
        assert "charge" in props
        assert "core" in props["charge"]
        assert "cysteine_count" in props
        assert "manufacturability" in props["cysteine_count"]
        assert "tcr_aromaticity" in props
        assert "immunogenicity" in props["tcr_aromaticity"]
