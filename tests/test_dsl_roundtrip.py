"""DSL round-trip + vectorized-eval regression tests.

These exercise the cases called out in issue #111:

    - Simple comparisons (affinity <= 500, presentation.rank <= 2)
    - Compound & / | (including mixed nesting)
    - Arithmetic inside Comparison
    - Boolean-in-arithmetic composition
    - Transforms (descending_cdf, log, logistic)
    - Method + version qualifiers
    - Scoped fields
    - Vectorized evaluation across many groups (sanity benchmark)
"""

import math
import operator

import numpy as np
import pandas as pd
import pytest

from topiary.ranking import (
    Affinity,
    BinOp,
    BoolOp,
    Column,
    Comparison,
    Const,
    EvalContext,
    Field,
    Len,
    Presentation,
    Stability,
    UnaryOp,
    apply_filter,
    apply_sort,
    parse,
    wt,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _simple_df():
    """Two peptides with affinity + presentation rows apiece."""
    return pd.DataFrame([
        dict(source_sequence_name="v1", peptide="SIINFEKL", peptide_offset=10,
             allele="HLA-A*02:01", kind="pMHC_affinity",
             score=0.8, value=120.0, percentile_rank=0.5,
             prediction_method_name="netmhcpan", predictor_version="4.1b"),
        dict(source_sequence_name="v1", peptide="SIINFEKL", peptide_offset=10,
             allele="HLA-A*02:01", kind="pMHC_presentation",
             score=0.92, value=None, percentile_rank=0.3,
             prediction_method_name="netmhcpan", predictor_version="4.1b"),
        dict(source_sequence_name="v1", peptide="ELAGIGIL", peptide_offset=20,
             allele="HLA-A*02:01", kind="pMHC_affinity",
             score=0.1, value=5000.0, percentile_rank=15.0,
             prediction_method_name="netmhcpan", predictor_version="4.1b"),
        dict(source_sequence_name="v1", peptide="ELAGIGIL", peptide_offset=20,
             allele="HLA-A*02:01", kind="pMHC_presentation",
             score=0.05, value=None, percentile_rank=20.0,
             prediction_method_name="netmhcpan", predictor_version="4.1b"),
    ])


def _assert_roundtrips(node, df=None):
    """Parse-stringify-parse equivalence: evaluate the original and a
    re-parsed version of its string form, assert same values."""
    text = node.to_expr_string()
    reparsed = parse(text)
    if df is None:
        df = _simple_df()
    ctx = EvalContext(df)
    a = node.eval(ctx)
    b = reparsed.eval(ctx)
    # Align dtypes — bool vs float both fine for our comparison
    a_arr = a.astype(float).to_numpy(na_value=np.nan)
    b_arr = b.astype(float).to_numpy(na_value=np.nan)
    assert np.allclose(a_arr, b_arr, equal_nan=True), (
        f"Round-trip mismatch for {text!r}: {a_arr} vs {b_arr}"
    )


# ---------------------------------------------------------------------------
# Simple comparisons
# ---------------------------------------------------------------------------


class TestSimpleComparisons:
    def test_affinity_le(self):
        _assert_roundtrips(Affinity.value <= 500)

    def test_presentation_rank_le(self):
        _assert_roundtrips(Presentation.rank <= 2.0)

    def test_score_ge(self):
        _assert_roundtrips(Presentation.score >= 0.5)

    def test_value_ge(self):
        _assert_roundtrips(Affinity.value >= 100.0)

    def test_bare_affinity_delegates(self):
        # Affinity <= 500 delegates to .value <= 500
        node = Affinity <= 500
        assert isinstance(node, Comparison)
        assert isinstance(node.left, Field)
        assert node.left.field == "value"
        _assert_roundtrips(node)


# ---------------------------------------------------------------------------
# Compound & / | with mixed nesting
# ---------------------------------------------------------------------------


class TestCompoundBool:
    def test_simple_or(self):
        node = (Affinity.value <= 500) | (Presentation.rank <= 2.0)
        assert isinstance(node, BoolOp) and node.op is operator.or_
        _assert_roundtrips(node)

    def test_simple_and(self):
        node = (Affinity.value <= 500) & (Presentation.score >= 0.5)
        assert isinstance(node, BoolOp) and node.op is operator.and_
        _assert_roundtrips(node)

    def test_mixed_or_of_and(self):
        # (A & B) | (C & D)
        node = (
            ((Affinity.value <= 500) & (Presentation.score >= 0.5))
            | ((Affinity.rank <= 1.0) & (Presentation.rank <= 2.0))
        )
        _assert_roundtrips(node)

    def test_mixed_and_of_or(self):
        # (A | B) & C — the classic nesting case that must preserve
        # precedence through round-trip.
        node = (
            ((Affinity.value <= 500) | (Affinity.value >= 1000))
            & (Presentation.score >= 0.5)
        )
        _assert_roundtrips(node)

    def test_triple_or_flattens(self):
        node = (
            (Affinity.value <= 500)
            | (Presentation.rank <= 2.0)
            | (Stability.score >= 0.5)
        )
        assert isinstance(node, BoolOp)
        assert node.op is operator.or_
        assert len(node.children) == 3
        _assert_roundtrips(node)

    def test_not_roundtrip(self):
        node = ~(Affinity.value <= 500)
        assert isinstance(node, BoolOp) and node.op is operator.invert
        _assert_roundtrips(node)


# ---------------------------------------------------------------------------
# Arithmetic inside Comparison
# ---------------------------------------------------------------------------


class TestArithmeticInComparison:
    def test_add_then_le(self):
        node = (Affinity.score + 1) <= 5
        _assert_roundtrips(node)

    def test_scaled_sum_le(self):
        node = (0.5 * Affinity.score + 0.5 * Presentation.score) >= 0.8
        _assert_roundtrips(node)

    def test_column_scaled_le(self):
        df = _simple_df()
        df["gene_tpm"] = [10.0, 10.0, 2.0, 2.0]
        node = (Column("gene_tpm") * 2) >= 10
        _assert_roundtrips(node, df=df)


# ---------------------------------------------------------------------------
# Boolean-in-arithmetic (the hallmark of the unified tree)
# ---------------------------------------------------------------------------


class TestBooleanInArithmetic:
    def test_bool_times_numeric(self):
        # For SIINFEKL: value=120 ≤ 500 → 1, score=0.8 → result 0.8
        # For ELAGIGIL: value=5000 > 500 → 0, result 0
        node = (Affinity.value <= 500) * Affinity.score
        df = _simple_df()
        ctx = EvalContext(df)
        result = node.eval(ctx).sort_index()
        values = result.to_numpy(dtype=float)
        assert set(np.round(values, 4).tolist()) == {0.0, 0.8}
        _assert_roundtrips(node)

    def test_piecewise_bool_arithmetic(self):
        # Classic "bool as mask" idiom from the issue:
        #   (Affinity <= 500) * Affinity.score + (Affinity > 500) * 0.5 * Affinity.score
        node = (
            (Affinity <= 500) * Affinity.score
            + (Affinity > 500) * 0.5 * Affinity.score
        )
        df = _simple_df()
        ctx = EvalContext(df)
        result = node.eval(ctx)
        # SIINFEKL: 1*0.8 + 0*... = 0.8
        # ELAGIGIL: 0*... + 1*0.5*0.1 = 0.05
        values = sorted(result.to_numpy(dtype=float).tolist())
        assert values == pytest.approx([0.05, 0.8])
        _assert_roundtrips(node)

    def test_penalty_term(self):
        # "Subtract 0.3 if gene_tpm < 1" — ordinary bool-in-arith.
        df = _simple_df()
        df["gene_tpm"] = [0.5, 0.5, 10.0, 10.0]
        node = Affinity.score - 0.3 * (Column("gene_tpm") < 1)
        ctx = EvalContext(df)
        result = node.eval(ctx)
        # SIINFEKL: score=0.8, tpm=0.5 → 0.8 - 0.3 = 0.5
        # ELAGIGIL: score=0.1, tpm=10 → 0.1
        expected = sorted([0.5, 0.1])
        assert sorted(result.to_numpy(dtype=float).tolist()) == pytest.approx(expected)
        _assert_roundtrips(node, df=df)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class TestTransforms:
    def test_descending_cdf(self):
        _assert_roundtrips(Affinity.descending_cdf(500, 200))

    def test_ascending_cdf(self):
        _assert_roundtrips(Presentation.score.ascending_cdf(0.5, 0.3))

    def test_logistic(self):
        _assert_roundtrips(Affinity.logistic(350, 150))

    def test_logistic_normalized(self):
        _assert_roundtrips(Affinity.logistic_normalized(350, 150))

    def test_logistic_normalized_reaches_one(self):
        """Normalized variant approaches 1 for arbitrarily good (low IC50)."""
        df = pd.DataFrame([
            dict(source_sequence_name="s", peptide="A", peptide_offset=0,
                 allele="A", kind="pMHC_affinity",
                 score=0.9, value=0.0, percentile_rank=0.0),
        ])
        raw = Affinity.logistic(350, 150).evaluate(df)
        normed = Affinity.logistic_normalized(350, 150).evaluate(df)
        assert raw < 0.92   # raw sigmoid caps below 1
        assert normed == pytest.approx(1.0)

    def test_logistic_vs_normalized_ratio_is_constant(self):
        """normalized(x) / raw(x) is the normalizer, independent of x."""
        df = pd.DataFrame([
            dict(source_sequence_name=f"s{i}", peptide="A", peptide_offset=0,
                 allele="A", kind="pMHC_affinity",
                 score=0.5, value=float(v), percentile_rank=1.0)
            for i, v in enumerate([50, 200, 500, 1000, 5000])
        ])
        ctx = EvalContext(df)
        raw = Affinity.logistic(350, 150).eval(ctx).to_numpy()
        normed = Affinity.logistic_normalized(350, 150).eval(ctx).to_numpy()
        ratios = normed / raw
        assert np.allclose(ratios, ratios[0])
        expected_cap = 1.0 / (1.0 + math.exp(-350 / 150))
        assert ratios[0] == pytest.approx(1.0 / expected_cap)

    def test_parse_logistic_normalized(self):
        from topiary.ranking import LogisticNormalizedExpr
        node = parse("affinity.logistic_normalized(350, 150)")
        assert isinstance(node, LogisticNormalizedExpr)
        assert node.midpoint == 350.0
        assert node.width == 150.0

    def test_log(self):
        _assert_roundtrips(Affinity.value.log())

    def test_log2(self):
        _assert_roundtrips(Affinity.value.log2())

    def test_clip(self):
        _assert_roundtrips(Affinity.value.clip(1, 50000))

    def test_hinge(self):
        _assert_roundtrips(Affinity.value.hinge())

    def test_chained_transforms(self):
        _assert_roundtrips(Affinity.value.clip(1, 50000).log())

    def test_transform_inside_comparison(self):
        _assert_roundtrips(Affinity.value.log() <= 7.0)


# ---------------------------------------------------------------------------
# Method + version qualifiers
# ---------------------------------------------------------------------------


class TestMethodAndVersion:
    def _multi_version_df(self):
        return pd.DataFrame([
            dict(source_sequence_name="seq1", peptide="SIINFEKL",
                 peptide_offset=0, allele="A", kind="pMHC_affinity",
                 score=0.8, value=120.0, percentile_rank=0.5,
                 prediction_method_name="netmhcpan",
                 predictor_version="4.1b"),
            dict(source_sequence_name="seq1", peptide="SIINFEKL",
                 peptide_offset=0, allele="A", kind="pMHC_affinity",
                 score=0.6, value=300.0, percentile_rank=1.5,
                 prediction_method_name="netmhcpan",
                 predictor_version="4.0"),
        ])

    def test_method_only(self):
        node = Affinity["netmhcpan"].value
        _assert_roundtrips(node, df=self._multi_version_df()[:1])

    def test_method_plus_version_python_api(self):
        df = self._multi_version_df()
        node = Affinity["netmhcpan", "4.1b"].value
        assert isinstance(node, Field)
        assert node.method == "netmhcpan"
        assert node.version == "4.1b"
        val = node.evaluate(df[df["predictor_version"] == "4.1b"])
        assert val == 120.0

    def test_method_plus_version_parser(self):
        df = self._multi_version_df()
        node = parse("affinity['netmhcpan', '4.1b'].value")
        assert isinstance(node, Field)
        assert node.method == "netmhcpan"
        assert node.version == "4.1b"
        val = node.evaluate(df[df["predictor_version"] == "4.1b"])
        assert val == 120.0

    def test_version_disambiguates(self):
        df = self._multi_version_df()
        assert (
            Affinity["netmhcpan", "4.1b"].value.evaluate(df) == 120.0
        )
        assert (
            Affinity["netmhcpan", "4.0"].value.evaluate(df) == 300.0
        )

    def test_method_plus_version_roundtrip(self):
        df = self._multi_version_df()
        node = Affinity["netmhcpan", "4.1b"].value <= 200
        _assert_roundtrips(node, df=df[df["predictor_version"] == "4.1b"])

    def test_missing_version_raises(self):
        df = self._multi_version_df()
        with pytest.raises(ValueError, match="predictor_version"):
            Affinity["netmhcpan", "9.9.9"].value.evaluate(df)


# ---------------------------------------------------------------------------
# Scoped fields
# ---------------------------------------------------------------------------


class TestScopedFields:
    def _wt_df(self):
        return pd.DataFrame([
            dict(source_sequence_name="v1", peptide="SIINFEKL",
                 peptide_offset=0, allele="A", kind="pMHC_affinity",
                 score=0.8, value=120.0, percentile_rank=0.5,
                 wt_peptide="SIINFEKK", wt_value=1000.0, wt_score=0.3,
                 wt_peptide_length=8),
        ])

    def test_wt_field_roundtrip(self):
        node = wt.affinity.score
        _assert_roundtrips(node, df=self._wt_df())

    def test_wt_differential(self):
        node = Affinity.score - wt.affinity.score
        _assert_roundtrips(node, df=self._wt_df())

    def test_wt_len_roundtrip(self):
        node = wt.len
        _assert_roundtrips(node, df=self._wt_df())

    def test_wt_count_roundtrip(self):
        node = wt.count("K")
        _assert_roundtrips(node, df=self._wt_df())


# ---------------------------------------------------------------------------
# Vectorized evaluation across many groups — sanity test
# ---------------------------------------------------------------------------


class TestVectorizedEval:
    def _big_df(self, n_groups=500):
        """n_groups peptide/allele groups, each with affinity+presentation."""
        rows = []
        rng = np.random.default_rng(42)
        for i in range(n_groups):
            peptide = "".join(
                rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=9)
            )
            ic50 = float(rng.uniform(20, 20000))
            pres_score = float(rng.uniform(0, 1))
            rows.append(dict(
                source_sequence_name=f"s{i}", peptide=peptide,
                peptide_offset=0, allele="HLA-A*02:01",
                kind="pMHC_affinity",
                score=float(rng.uniform(0, 1)),
                value=ic50,
                percentile_rank=float(rng.uniform(0, 100)),
            ))
            rows.append(dict(
                source_sequence_name=f"s{i}", peptide=peptide,
                peptide_offset=0, allele="HLA-A*02:01",
                kind="pMHC_presentation",
                score=pres_score,
                value=None,
                percentile_rank=float(rng.uniform(0, 100)),
            ))
        return pd.DataFrame(rows)

    def test_vectorized_filter(self):
        df = self._big_df(500)
        node = (Affinity.value <= 500) & (Presentation.score >= 0.5)
        result = apply_filter(df, node)
        # Every surviving row belongs to a group that passes
        passing_groups = set(
            tuple(g) for g in result[
                ["source_sequence_name", "peptide", "peptide_offset", "allele"]
            ].apply(tuple, axis=1).tolist()
        )
        # Independent scalar check for a few groups
        for key in list(passing_groups)[:5]:
            group_df = df[
                (df["source_sequence_name"] == key[0])
                & (df["peptide"] == key[1])
            ]
            aff = group_df[group_df["kind"] == "pMHC_affinity"]["value"].iloc[0]
            pres = (
                group_df[group_df["kind"] == "pMHC_presentation"]["score"].iloc[0]
            )
            assert aff <= 500 and pres >= 0.5

    def test_vectorized_sort(self):
        df = self._big_df(100)
        sorted_df = apply_sort(df, [Presentation.score, Affinity.score])
        # First group's presentation score should be >= the last group's
        first_key = tuple(sorted_df.iloc[0][
            ["source_sequence_name", "peptide", "peptide_offset", "allele"]
        ])
        last_key = tuple(sorted_df.iloc[-1][
            ["source_sequence_name", "peptide", "peptide_offset", "allele"]
        ])
        first_pres = df[
            (df["source_sequence_name"] == first_key[0])
            & (df["peptide"] == first_key[1])
            & (df["kind"] == "pMHC_presentation")
        ]["score"].iloc[0]
        last_pres = df[
            (df["source_sequence_name"] == last_key[0])
            & (df["peptide"] == last_key[1])
            & (df["kind"] == "pMHC_presentation")
        ]["score"].iloc[0]
        assert first_pres >= last_pres

    def test_sort_nan_fallthrough(self):
        """When a sort key is NaN for a group, ordering falls through to
        the next tiebreaker instead of forcing the NaN group to an
        extreme position."""
        # Two peptides: only A has a presentation row; both have affinity.
        df = pd.DataFrame([
            dict(source_sequence_name="v1", peptide="AAA", peptide_offset=0,
                 allele="A", kind="pMHC_affinity",
                 score=0.9, value=100.0, percentile_rank=0.5),
            dict(source_sequence_name="v1", peptide="AAA", peptide_offset=0,
                 allele="A", kind="pMHC_presentation",
                 score=0.7, value=None, percentile_rank=0.8),
            dict(source_sequence_name="v1", peptide="BBB", peptide_offset=5,
                 allele="A", kind="pMHC_affinity",
                 score=0.1, value=5000.0, percentile_rank=20.0),
            # No presentation row for BBB — its presentation.score is NaN
        ])
        # First key (presentation.score) is NaN for BBB → falls through
        # to affinity.score which puts AAA first (0.9 > 0.1).
        sorted_df = apply_sort(df, [Presentation.score, Affinity.score])
        assert sorted_df.iloc[0]["peptide"] == "AAA"

    def test_child_nodes_generic_walk(self):
        """DSLNode.child_nodes() exposes direct children uniformly — used by
        generic walkers like column validation."""
        from topiary.ranking.apply import _collect_column_names
        node = (
            0.5 * Affinity.score
            + 0.3 * (Column("gene_tpm").log1p())
            - 0.2 * Column("cysteine_count")
        )
        names = _collect_column_names(node)
        assert names == {"gene_tpm", "cysteine_count"}

    def test_child_nodes_inside_comparison_and_bool(self):
        """Column references inside comparisons and BoolOps are also found."""
        from topiary.ranking.apply import _collect_column_names
        node = (
            (Column("gene_tpm") >= 5)
            & (Column("cysteine_count") <= 2)
            | (Column("hydrophobicity") > 0)
        )
        names = _collect_column_names(node)
        assert names == {"gene_tpm", "cysteine_count", "hydrophobicity"}

    def test_sort_stability(self):
        """Two groups tied on every sort key must retain input order."""
        df = pd.DataFrame([
            dict(source_sequence_name="v1", peptide="AAA", peptide_offset=0,
                 allele="A", kind="pMHC_affinity",
                 score=0.5, value=100.0, percentile_rank=1.0),
            dict(source_sequence_name="v1", peptide="BBB", peptide_offset=5,
                 allele="A", kind="pMHC_affinity",
                 score=0.5, value=100.0, percentile_rank=1.0),
        ])
        sorted_df = apply_sort(df, [Affinity.score])
        # Input order preserved — AAA first, BBB second
        assert sorted_df.iloc[0]["peptide"] == "AAA"
        assert sorted_df.iloc[1]["peptide"] == "BBB"

    def test_vectorized_composite(self):
        """Composite score evaluates elementwise without a Python group loop."""
        df = self._big_df(1000)
        composite = (
            0.5 * Affinity.value.descending_cdf(500, 200)
            + 0.5 * Presentation.score.ascending_cdf(0.5, 0.3)
        )
        ctx = EvalContext(df)
        result = composite.eval(ctx)
        # One value per group, finite
        assert len(result) == 1000
        finite = result.dropna()
        assert len(finite) == 1000
        assert (finite >= 0).all()
        assert (finite <= 1).all()


# ---------------------------------------------------------------------------
# apply_filter error on non-boolean input
# ---------------------------------------------------------------------------


class TestApplyFilterBooleanCheck:
    def test_accepts_bool(self):
        df = _simple_df()
        result = apply_filter(df, Affinity.value <= 500)
        assert set(result["peptide"]) == {"SIINFEKL"}

    def test_accepts_zero_one_arithmetic(self):
        """0/1 (from bool-times-numeric being 0*something=0 or 1*1=1) is allowed."""
        df = _simple_df()
        # Build a composition that yields integer 0/1 values
        node = (Affinity.value <= 500) * (Presentation.score >= 0.5)
        result = apply_filter(df, node)
        # SIINFEKL: 1*1=1 → pass; ELAGIGIL: 0*0=0 → drop
        assert set(result["peptide"]) == {"SIINFEKL"}

    def test_rejects_non_boolean_score(self):
        """A bare numeric node like Affinity.score (values in [0,1] but not {0,1}) must error."""
        df = _simple_df()
        with pytest.raises(ValueError, match="non-boolean"):
            apply_filter(df, Affinity.score)

    def test_rejects_scaled_bool(self):
        """(bool) * 2 produces 0/2, not 0/1 → error."""
        df = _simple_df()
        node = (Affinity.value <= 500) * 2
        with pytest.raises(ValueError, match="non-boolean"):
            apply_filter(df, node)

    def test_nan_treated_as_false(self):
        """Groups whose filter evaluates to NaN (e.g. missing kind) are dropped, not errored."""
        # Only affinity rows — Stability.score will be NaN for every group
        df = _simple_df()[_simple_df()["kind"] == "pMHC_affinity"]
        result = apply_filter(df, Stability.score >= 0.5)
        assert len(result) == 0

    def test_nan_in_and(self):
        """NaN inside an AND acts as False: (True & NaN) drops the row."""
        # Stability missing → (Affinity<=500) AND (Stability.score>=0.5) == False
        df = _simple_df()[_simple_df()["kind"] == "pMHC_affinity"]
        node = (Affinity <= 500) & (Stability.score >= 0.5)
        result = apply_filter(df, node)
        assert len(result) == 0

    def test_nan_in_or(self):
        """NaN inside an OR acts as False: (True | NaN) still passes, (False | NaN) drops."""
        df = _simple_df()[_simple_df()["kind"] == "pMHC_affinity"]
        # Stability is NaN; use OR so Affinity<=500 is what decides each group.
        node = (Affinity <= 500) | (Stability.score >= 0.5)
        result = apply_filter(df, node)
        assert set(result["peptide"]) == {"SIINFEKL"}

    def test_nan_in_invert(self):
        """NaN inside ~ acts as False: ~NaN = ~False = True."""
        # Construct a BoolOp(invert, [Field]) where Field evals to NaN.
        # Legal through the Python API: ~(some Stability-only comparison)
        df = _simple_df()[_simple_df()["kind"] == "pMHC_affinity"]
        node = ~(Stability.score >= 0.5)
        # Stability.score >= 0.5 evaluates to False (NaN comparison → False).
        # ~False → True, so every group passes.
        result = apply_filter(df, node)
        assert set(result["peptide"]) == {"SIINFEKL", "ELAGIGIL"}

    def test_nan_in_boolop_from_non_comparison(self):
        """Regression: passing a non-Comparison node (raw Field with NaN values)
        into a BoolOp must still honor the NaN → False policy.  The old
        implementation cast NaN to True via .astype(bool), letting missing
        data silently pass the filter."""
        df = _simple_df()[_simple_df()["kind"] == "pMHC_affinity"]
        # Build a BoolOp directly: AND( (Affinity<=500), Stability.score )
        # Stability.score is a Field returning NaN (no stability rows).
        node = BoolOp(operator.and_, [Affinity <= 500, Stability.score])
        result = apply_filter(df, node)
        # NaN → False, AND → drops everything
        assert len(result) == 0


