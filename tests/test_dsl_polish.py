"""Tests for DSL polish: norm/logistic direction, log2/log1p, filter_by, parser errors."""

import math

import pandas as pd
import pytest
from mhctools import Kind

from topiary.ranking import (
    Affinity,
    Presentation,
    Column,
    parse_filter,
    parse_ranking,
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
        low = Affinity.norm(500, 200).evaluate(_aff_df(100))
        high = Affinity.norm(500, 200).evaluate(_aff_df(1000))
        assert high > low

    def test_logistic_lower_input_higher_output(self):
        """logistic: lower input -> higher output."""
        strong = Affinity.logistic(350, 150).evaluate(_aff_df(100))
        weak = Affinity.logistic(350, 150).evaluate(_aff_df(5000))
        assert strong > weak

    def test_norm_for_ic50_needs_inversion(self):
        """For IC50 (lower=better), must use 1 - norm()."""
        raw = Affinity.norm(500, 200).evaluate(_aff_df(100))
        inverted = (1 - Affinity.norm(500, 200)).evaluate(_aff_df(100))
        assert raw < 0.5   # low IC50 -> low CDF
        assert inverted > 0.5  # inverted -> high score

    def test_logistic_for_ic50_works_directly(self):
        """For IC50 (lower=better), logistic works without inversion."""
        val = Affinity.logistic(350, 150).evaluate(_aff_df(100))
        assert val > 0.5   # low IC50 -> high logistic score

    def test_norm_and_logistic_are_different(self):
        """They give different values for the same input."""
        n = Affinity.norm(350, 150).evaluate(_aff_df(200))
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
        assert math.isnan(val)

    def test_log2_in_composite(self):
        expr = Affinity.log2() + 1
        val = expr.evaluate(_aff_df(4.0))
        assert val == pytest.approx(3.0)  # log2(4) + 1


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
        assert p.ranking_strategy is not None

    def test_filter_by_string(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by="affinity <= 500",
        )
        assert p.ranking_strategy is not None
        assert len(p.ranking_strategy.filters) == 1

    def test_filter_by_complex_string(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by="affinity <= 500 | el.rank <= 2",
        )
        assert len(p.ranking_strategy.filters) == 2

    def test_filter_by_takes_precedence_over_filter(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter_by=Affinity <= 500,
            filter=Affinity <= 1000,  # deprecated, ignored
        )
        assert p.ranking_strategy.filters[0].max_value == 500

    def test_deprecated_filter_still_works(self):
        from topiary import TopiaryPredictor
        from mhctools import RandomBindingPredictor
        p = TopiaryPredictor(
            models=RandomBindingPredictor(alleles=["A0201"]),
            filter=Affinity <= 500,
        )
        assert p.ranking_strategy is not None


# ---------------------------------------------------------------------------
# Parser error quality
# ---------------------------------------------------------------------------


class TestParserErrors:
    def test_kind_typo_suggests(self):
        with pytest.raises(ValueError, match="Did you mean.*affinity"):
            parse_filter("afinity <= 500")

    def test_field_typo_suggests(self):
        with pytest.raises(ValueError, match="Did you mean.*rank"):
            parse_filter("affinity.rnk <= 2")

    def test_non_numeric_threshold(self):
        with pytest.raises(ValueError, match="must be a number"):
            parse_filter("affinity <= abc")

    def test_missing_operator(self):
        with pytest.raises(ValueError, match="No comparison operator"):
            parse_filter("affinity")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="No comparison operator"):
            parse_filter("")

    def test_reversed_expression(self):
        """'500 <= affinity' — 'affinity' on right side fails as number."""
        with pytest.raises(ValueError, match="must be a number"):
            parse_filter("500 <= affinity")

    def test_space_in_operator(self):
        """'affinity < = 500' — the '= 500' part fails as threshold."""
        with pytest.raises(ValueError, match="must be a number"):
            parse_filter("affinity < = 500")

    def test_qualified_kind_typo_suggests(self):
        with pytest.raises(ValueError, match="Did you mean"):
            parse_filter("netmhcpan_afinity <= 500")
