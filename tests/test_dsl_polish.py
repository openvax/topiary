"""Tests for DSL polish: norm/logistic direction, log2/log1p, filter_by, parser errors."""

import math

import pandas as pd
import pytest
from mhctools import Kind

from topiary.ranking import (
    Affinity,
    BoolOp,
    Column,
    Comparison,
    Presentation,
    parse,
)


def _aff_df(ic50):
    return pd.DataFrame([dict(
        source_sequence_name="x", peptide="A", peptide_offset=0,
        allele="A", kind="pMHC_affinity", score=0.9, value=ic50,
        percentile_rank=0.5,
    )])


# ---------------------------------------------------------------------------
# norm vs logistic direction
# ---------------------------------------------------------------------------


class TestDirection:
    def test_norm_higher_input_higher_output(self):
        """norm: higher input -> higher output (CDF)."""
        low = Affinity.ascending_cdf(500, 200).evaluate(_aff_df(100))
        high = Affinity.ascending_cdf(500, 200).evaluate(_aff_df(1000))
        assert high > low

    def test_norm_lower_lower_input_higher_output(self):
        """norm_lower: lower input -> higher output (1-CDF)."""
        strong = Affinity.descending_cdf(500, 200).evaluate(_aff_df(100))
        weak = Affinity.descending_cdf(500, 200).evaluate(_aff_df(1000))
        assert strong > weak

    def test_norm_lower_is_complement_of_norm(self):
        n = Affinity.ascending_cdf(500, 200).evaluate(_aff_df(300))
        nl = Affinity.descending_cdf(500, 200).evaluate(_aff_df(300))
        assert n + nl == pytest.approx(1.0)

    def test_logistic_lower_input_higher_output(self):
        """logistic: lower input -> higher output."""
        strong = Affinity.logistic(350, 150).evaluate(_aff_df(100))
        weak = Affinity.logistic(350, 150).evaluate(_aff_df(5000))
        assert strong > weak

    def test_norm_lower_for_ic50(self):
        """For IC50, norm_lower gives high scores for strong binders."""
        val = Affinity.descending_cdf(500, 200).evaluate(_aff_df(100))
        assert val > 0.5

    def test_logistic_for_ic50_works_directly(self):
        """For IC50 (lower=better), logistic works without inversion."""
        val = Affinity.logistic(350, 150).evaluate(_aff_df(100))
        assert val > 0.5

    def test_norm_and_logistic_are_different(self):
        """They give different values for the same input."""
        n = Affinity.ascending_cdf(350, 150).evaluate(_aff_df(200))
        l = Affinity.logistic(350, 150).evaluate(_aff_df(200))
        assert n != pytest.approx(l, abs=0.01)


# ---------------------------------------------------------------------------
# log2, log1p
# ---------------------------------------------------------------------------


class TestNewTransforms:
    def test_log2(self):
        val = Affinity.log2().evaluate(_aff_df(8.0))
        assert val == pytest.approx(3.0)

    def test_log2_accessor(self):
        """KindAccessor delegates log2."""
        val = Affinity.log2().evaluate(_aff_df(8.0))
        assert val == pytest.approx(3.0)

    def test_log1p(self):
        val = Affinity.log1p().evaluate(_aff_df(0.0))
        assert val == pytest.approx(0.0)  # log(1+0) = 0

    def test_log1p_small(self):
        """log1p is more accurate than log for small values."""
        val = Affinity.log1p().evaluate(_aff_df(1e-10))
        assert val == pytest.approx(1e-10, rel=1e-5)

    def test_log1p_nan_for_invalid(self):
        val = Affinity.log1p().evaluate(_aff_df(-2.0))
        assert math.isnan(val)

    def test_log2_nan_for_zero(self):
        val = Affinity.log2().evaluate(_aff_df(0.0))
        # log2(0) = -inf (numpy path). NaN is also acceptable.
        assert math.isinf(val) or math.isnan(val)

    def test_log2_in_composite(self):
        expr = Affinity.log2() + 1
        val = expr.evaluate(_aff_df(4.0))
        assert val == pytest.approx(3.0)  # log2(4) + 1

    def test_hinge(self):
        assert Affinity.value.hinge().evaluate(_aff_df(100)) == 100.0
        assert Affinity.value.hinge().evaluate(_aff_df(-5)) == 0.0
        assert Affinity.value.hinge().evaluate(_aff_df(0)) == 0.0

    def test_hinge_in_expression(self):
        """hinge of a difference: max(0, mutant - wt)."""
        expr = (Affinity.value - 200).hinge()
        assert expr.evaluate(_aff_df(300)) == 100.0
        assert expr.evaluate(_aff_df(100)) == 0.0


class TestAggregations:
    def test_mean(self):
        from topiary.ranking import mean
        expr = mean(Affinity.value, 200)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(150.0)

    def test_mean_three(self):
        from topiary.ranking import mean
        expr = mean(Affinity.value, 200, 300)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(200.0)

    def test_geomean(self):
        from topiary.ranking import geomean
        expr = geomean(Affinity.value, 400)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(200.0)  # sqrt(100 * 400)

    def test_geomean_skips_non_positive(self):
        from topiary.ranking import geomean
        expr = geomean(Affinity.value, -5, 400)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(200.0)  # -5 skipped

    def test_minimum(self):
        from topiary.ranking import minimum
        expr = minimum(Affinity.value, 500, 200)
        assert expr.evaluate(_aff_df(100)) == 100.0
        assert expr.evaluate(_aff_df(300)) == 200.0

    def test_maximum(self):
        from topiary.ranking import maximum
        expr = maximum(Affinity.value, 50, 200)
        assert expr.evaluate(_aff_df(100)) == 200.0
        assert expr.evaluate(_aff_df(300)) == 300.0

    def test_median_odd(self):
        from topiary.ranking import median
        expr = median(Affinity.value, 200, 300)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(200.0)  # sorted: [100, 200, 300]

    def test_median_even(self):
        from topiary.ranking import median
        expr = median(Affinity.value, 200, 300, 400)
        val = expr.evaluate(_aff_df(100))
        assert val == pytest.approx(250.0)  # sorted: [100, 200, 300, 400] → (200+300)/2

    def test_mean_with_qualified_fields(self):
        """mean() works with multi-model qualified fields."""
        from topiary.ranking import mean
        df = pd.DataFrame([
            dict(source_sequence_name="x", peptide="A", peptide_offset=0,
                 allele="A", kind="pMHC_affinity", score=0.8, value=100.0,
                 percentile_rank=0.5, prediction_method_name="netmhcpan"),
            dict(source_sequence_name="x", peptide="A", peptide_offset=0,
                 allele="A", kind="pMHC_affinity", score=0.6, value=300.0,
                 percentile_rank=2.0, prediction_method_name="mhcflurry"),
        ])
        expr = mean(
            Affinity["netmhcpan"].logistic(350, 150),
            Affinity["mhcflurry"].logistic(350, 150),
        )
        val = expr.evaluate(df)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_aggregation_all_nan(self):
        from topiary.ranking import mean
        df = pd.DataFrame([dict(
            source_sequence_name="x", peptide="A", peptide_offset=0,
            allele="A", kind="pMHC_affinity", score=float("nan"),
            value=float("nan"), percentile_rank=float("nan"),
        )])
        val = mean(Affinity.value, Affinity.score).evaluate(df)
        assert math.isnan(val)


# ---------------------------------------------------------------------------
# filter_by parameter and string parsing
# ---------------------------------------------------------------------------


class TestFilterBy:
    def test_filter_by_object(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by=Affinity <= 500,
        )
        assert p.filter_by is not None
        assert isinstance(p.filter_by, Comparison)

    def test_filter_by_string(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by="affinity <= 500",
        )
        assert p.filter_by is not None
        # Single-clause filter is a Comparison, not a BoolOp
        assert isinstance(p.filter_by, Comparison)

    def test_filter_by_complex_string(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by="affinity <= 500 | el.rank <= 2",
        )
        assert isinstance(p.filter_by, BoolOp)
        assert len(p.filter_by.children) == 2


# ---------------------------------------------------------------------------
# Parser error quality
# ---------------------------------------------------------------------------


class TestParserErrors:
    def test_kind_typo_becomes_column(self):
        """Typos in kind names become column references (error deferred to eval)."""
        f = parse("afinity <= 500")
        assert isinstance(f, Comparison)
        assert isinstance(f.left, Column)
        assert f.left.col_name == "afinity"

    def test_field_typo_suggests(self):
        with pytest.raises(ValueError, match="Unknown field.*rnk"):
            parse("affinity.rnk <= 2")

    def test_non_numeric_threshold(self):
        # 'abc' becomes a Column reference rather than a number — the
        # parser resolves it to Column('abc'). This is valid syntactically;
        # real errors emerge at eval time when the column is missing.
        f = parse("affinity <= abc")
        assert isinstance(f, Comparison)
        assert isinstance(f.right, Column)

    def test_missing_operator(self):
        # 'affinity' alone parses fine as a bare kind reference — there's
        # no longer a comparison-required check. It evaluates as the raw
        # affinity.value expression.
        result = parse("affinity")
        assert result is not None

    def test_empty_string(self):
        with pytest.raises(ValueError):
            parse("")

    def test_reversed_expression(self):
        """'500 <= affinity' is now valid: 500 <= affinity.value (comparison)."""
        f = parse("500 <= affinity")
        assert isinstance(f, Comparison)

    def test_space_in_operator(self):
        """'affinity < = 500' is a syntax error (= alone is not an operator)."""
        with pytest.raises(ValueError):
            parse("affinity < = 500")

    def test_qualified_kind_typo_becomes_column(self):
        """Qualified kind typos become column references (error deferred to eval)."""
        f = parse("netmhcpan_afinity <= 500")
        assert isinstance(f, Comparison)
        assert isinstance(f.left, Column)
        assert f.left.col_name == "netmhcpan_afinity"
