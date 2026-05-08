"""Tests for BestAlleleField — per-peptide max/min across alleles with
optional allele attribution. Motivated by issue #158 (haplotype-mode
presentation predictors like MHCflurry need a "best across alleles"
accessor)."""

import math

import numpy as np
import pandas as pd
import pytest

from topiary import Affinity, BestAlleleField, EvalContext, Presentation
from topiary.ranking import parse


def _row(
    peptide,
    allele,
    *,
    score=None,
    value=None,
    percentile_rank=None,
    kind="pMHC_presentation",
    method="mhcflurry",
    version="2.0",
    source_sequence_name="p1",
    peptide_offset=0,
):
    return {
        "source_sequence_name": source_sequence_name,
        "peptide": peptide,
        "peptide_offset": peptide_offset,
        "allele": allele,
        "kind": kind,
        "score": score,
        "value": value,
        "percentile_rank": percentile_rank,
        "prediction_method_name": method,
        "predictor_version": version,
    }


@pytest.fixture
def presentation_df():
    """Two peptides × two alleles, asymmetric scores so the 'best allele'
    differs between peptides — exercises the broadcast logic."""
    return pd.DataFrame([
        _row("SIINFEKLA", "HLA-A*02:01", score=0.70, percentile_rank=1.5),
        _row("SIINFEKLA", "HLA-B*07:02", score=0.30, percentile_rank=12.0),
        _row("YLLPRRGPRL", "HLA-A*02:01", score=0.40, percentile_rank=3.5,
             peptide_offset=5),
        _row("YLLPRRGPRL", "HLA-B*07:02", score=0.85, percentile_rank=0.5,
             peptide_offset=5),
    ])


class TestBestScoreBroadcasts:
    def test_best_score_max_across_alleles(self, presentation_df):
        ctx = EvalContext(presentation_df)
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        # Both alleles for each peptide see the same per-peptide max
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-A*02:01")] == 0.70
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-B*07:02")] == 0.70
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-A*02:01")] == 0.85
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-B*07:02")] == 0.85

    def test_best_score_allele_attribution(self, presentation_df):
        ctx = EvalContext(presentation_df)
        result = Presentation["mhcflurry"].best_score_allele.eval(ctx)
        # SIINFEKLA's max is on HLA-A*02:01; YLLPRRGPRL's max is on HLA-B*07:02
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-A*02:01")] == "HLA-A*02:01"
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-B*07:02")] == "HLA-A*02:01"
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-A*02:01")] == "HLA-B*07:02"
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-B*07:02")] == "HLA-B*07:02"


class TestBestRankIsMin:
    def test_best_rank_picks_min_per_peptide(self, presentation_df):
        ctx = EvalContext(presentation_df)
        result = Presentation["mhcflurry"].best_rank.eval(ctx)
        # Lower percentile rank = better. SIINFEKLA: 1.5 < 12.0; YLLPRRGPRL: 0.5 < 3.5.
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-A*02:01")] == 1.5
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-B*07:02")] == 1.5
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-A*02:01")] == 0.5
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-B*07:02")] == 0.5

    def test_best_rank_allele_attribution(self, presentation_df):
        ctx = EvalContext(presentation_df)
        result = Presentation["mhcflurry"].best_rank_allele.eval(ctx)
        # The allele with the *lowest* rank wins
        assert result.loc[("p1", "SIINFEKLA", 0, "HLA-A*02:01")] == "HLA-A*02:01"
        assert result.loc[("p1", "YLLPRRGPRL", 5, "HLA-A*02:01")] == "HLA-B*07:02"


class TestBestValueIsMin:
    """`value` follows the affinity convention (IC50 nM): lower is better."""

    def test_best_value_picks_min_per_peptide(self):
        df = pd.DataFrame([
            _row("PEPTIDE1", "HLA-A*02:01", value=120.0,
                 kind="pMHC_affinity", method="netmhcpan"),
            _row("PEPTIDE1", "HLA-B*07:02", value=850.0,
                 kind="pMHC_affinity", method="netmhcpan"),
        ])
        ctx = EvalContext(df)
        result = Affinity["netmhcpan"].best_value.eval(ctx)
        assert result.loc[("p1", "PEPTIDE1", 0, "HLA-A*02:01")] == 120.0
        assert result.loc[("p1", "PEPTIDE1", 0, "HLA-B*07:02")] == 120.0

        allele_result = Affinity["netmhcpan"].best_value_allele.eval(ctx)
        assert allele_result.loc[
            ("p1", "PEPTIDE1", 0, "HLA-B*07:02")
        ] == "HLA-A*02:01"


class TestEdgeCases:
    def test_empty_dataframe_returns_empty_series(self):
        df = pd.DataFrame(columns=[
            "source_sequence_name", "peptide", "peptide_offset", "allele",
            "kind", "score", "prediction_method_name", "predictor_version",
        ])
        ctx = EvalContext(df)
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        assert len(result) == 0

    def test_no_matching_kind_returns_nan_series(self, presentation_df):
        # Only one kind in the frame; ask for a different one
        df = presentation_df.copy()
        df["kind"] = "pMHC_affinity"
        ctx = EvalContext(df)
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        # Group index has 4 rows but no presentation rows survive the filter
        assert len(result) == 4
        assert result.isna().all()

    def test_all_nan_values_returns_nan_series(self, presentation_df):
        df = presentation_df.copy()
        df["score"] = np.nan
        ctx = EvalContext(df)
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        assert result.isna().all()

    def test_nan_score_skipped_when_picking_best(self):
        df = pd.DataFrame([
            _row("PEPTIDE1", "HLA-A*02:01", score=0.30),
            _row("PEPTIDE1", "HLA-B*07:02", score=np.nan),
        ])
        ctx = EvalContext(df)
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        # The NaN allele's row exists in group_index, but the per-peptide
        # max is computed from the one valid score (0.30) and broadcast.
        assert result.loc[("p1", "PEPTIDE1", 0, "HLA-A*02:01")] == 0.30
        assert result.loc[("p1", "PEPTIDE1", 0, "HLA-B*07:02")] == 0.30

    def test_no_allele_in_group_keys_degenerates_to_field(self):
        df = pd.DataFrame([
            _row("PEPTIDE1", "HLA-A*02:01", score=0.5),
        ])
        ctx = EvalContext(df, group_keys=["source_sequence_name", "peptide"])
        # No allele in group_keys → can't aggregate; should fall back to
        # ordinary Field semantics for the numeric variant.
        result = Presentation["mhcflurry"].best_score.eval(ctx)
        assert result.loc[("p1", "PEPTIDE1")] == 0.5


class TestParserSupport:
    def test_parses_to_best_allele_field(self):
        node = parse("presentation['mhcflurry'].best_score")
        assert isinstance(node, BestAlleleField)
        assert node.field == "score"
        assert node.return_allele is False
        assert node.method == "mhcflurry"

    def test_parses_allele_attribution_form(self):
        node = parse("presentation['mhcflurry'].best_score_allele")
        assert isinstance(node, BestAlleleField)
        assert node.return_allele is True

    def test_parses_unqualified_kind(self):
        node = parse("presentation.best_rank")
        assert isinstance(node, BestAlleleField)
        assert node.field == "percentile_rank"

    def test_parses_percentile_alias(self):
        # best_percentile is an accepted spelling for best_rank
        node = parse("presentation['mhcflurry'].best_percentile")
        assert isinstance(node, BestAlleleField)
        assert node.field == "percentile_rank"

    def test_unknown_field_raises_with_helpful_message(self):
        with pytest.raises(ValueError, match="best_value"):
            parse("presentation['mhcflurry'].best_xyz")

    def test_repr_roundtrips_through_parser(self, presentation_df):
        node = Presentation["mhcflurry"].best_score
        reparsed = parse(repr(node))
        assert isinstance(reparsed, BestAlleleField)
        assert reparsed.field == node.field
        assert reparsed.method == node.method
        assert reparsed.return_allele == node.return_allele

        # Functional equivalence: same result on a real frame
        ctx = EvalContext(presentation_df)
        pd.testing.assert_series_equal(node.eval(ctx), reparsed.eval(ctx))


class TestComposes:
    """`best_score` returns a regular numeric DSLNode and must compose
    with the rest of the DSL (arithmetic, comparisons, transforms)."""

    def test_composes_with_arithmetic(self, presentation_df):
        ctx = EvalContext(presentation_df)
        # Doubling the broadcast best_score should still broadcast
        node = Presentation["mhcflurry"].best_score * 2.0
        result = node.eval(ctx)
        assert math.isclose(
            result.loc[("p1", "SIINFEKLA", 0, "HLA-A*02:01")], 1.40
        )

    def test_composes_with_comparison(self, presentation_df):
        ctx = EvalContext(presentation_df)
        node = Presentation["mhcflurry"].best_score >= 0.5
        result = node.eval(ctx)
        # SIINFEKLA's max is 0.70 (>=0.5), YLLPRRGPRL's is 0.85 (>=0.5)
        assert result.all()


class TestKindDirectionTable:
    """`_best_direction(kind, field)` is per-(kind, field) — `value`
    direction depends on the kind (IC50 vs half-life)."""

    def test_value_undefined_for_presentation_raises(self, presentation_df):
        # presentation rows don't populate `value`, but more importantly:
        # value direction is kind-dependent and not registered for
        # pMHC_presentation. Asking for best_value should raise loudly.
        ctx = EvalContext(presentation_df)
        with pytest.raises(ValueError, match="best_value direction is undefined"):
            Presentation["mhcflurry"].best_value.eval(ctx)

    def test_affinity_best_value_is_min(self):
        df = pd.DataFrame([
            _row("PEP", "HLA-A*02:01", value=120.0,
                 kind="pMHC_affinity", method="netmhcpan"),
            _row("PEP", "HLA-B*07:02", value=850.0,
                 kind="pMHC_affinity", method="netmhcpan"),
        ])
        ctx = EvalContext(df)
        assert Affinity["netmhcpan"].best_value.eval(ctx).iloc[0] == 120.0

    def test_score_max_works_for_any_kind(self):
        # processing kinds populate `score`, no `value`; best_score is
        # well-defined (max) regardless of kind
        df = pd.DataFrame([
            _row("PEP", "HLA-A*02:01", score=0.6,
                 kind="proteasome_cleavage", method="pepsickle"),
            _row("PEP", "HLA-B*07:02", score=0.9,
                 kind="proteasome_cleavage", method="pepsickle"),
        ])
        ctx = EvalContext(df)
        from topiary.ranking import Processing
        from topiary.ranking.nodes import BestAlleleField
        # Processing's kind is antigen_processing, not proteasome_cleavage;
        # build the node directly to test the direction logic.
        node = BestAlleleField(
            "proteasome_cleavage", "score", method="pepsickle",
        )
        assert node.eval(ctx).iloc[0] == 0.9


class TestKindSupportWarning:
    """When ``ctx.kind_support`` reports ``mhc_dependence='single_allele'``
    for the matched (model, kind), ``best_*`` warns that the result is
    the best per-allele score, not a true joint multi-allele aggregate."""

    def test_warns_for_single_allele_predictor(self, presentation_df):
        ctx = EvalContext(
            presentation_df,
            kind_support={
                "mhcflurry": {
                    "pMHC_presentation": {
                        "mhc_dependence": "single_allele",
                        "mhc_class": "I",
                    }
                }
            },
        )
        with pytest.warns(UserWarning, match="single_allele"):
            Presentation["mhcflurry"].best_score.eval(ctx)

    def test_no_warning_for_haplotype_predictor(self, presentation_df):
        ctx = EvalContext(
            presentation_df,
            kind_support={
                "mhcflurry": {
                    "pMHC_presentation": {
                        "mhc_dependence": "haplotype",
                        "mhc_class": "I",
                    }
                }
            },
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            # Should not raise — no warning emitted
            Presentation["mhcflurry"].best_score.eval(ctx)

    def test_no_warning_when_kind_support_absent(self, presentation_df):
        ctx = EvalContext(presentation_df)  # no kind_support
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            Presentation["mhcflurry"].best_score.eval(ctx)

    def test_warning_filter_uses_method_substring(self, presentation_df):
        # kind_support has both a haplotype and a single_allele model;
        # asking specifically for the haplotype one should not warn.
        ctx = EvalContext(
            presentation_df,
            kind_support={
                "mhcflurry": {
                    "pMHC_presentation": {
                        "mhc_dependence": "haplotype",
                        "mhc_class": "I",
                    }
                },
                "netmhcpan": {
                    "pMHC_presentation": {
                        "mhc_dependence": "single_allele",
                        "mhc_class": "I",
                    }
                },
            },
        )
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error", UserWarning)
            Presentation["mhcflurry"].best_score.eval(ctx)


class TestBroadcastCorrectness:
    """The broadcast must align values to ctx.group_index by peptide_keys
    even when the group_index is permuted relative to the input rows.
    Catches any positional-merge regression."""

    def test_broadcast_ignores_input_row_order(self):
        # Input rows in (B, A, A, B) order; group_index will be in
        # (A, B, A, B) order per peptide because dedup preserves order.
        df = pd.DataFrame([
            _row("PEP1", "HLA-B*07:02", score=0.30),
            _row("PEP1", "HLA-A*02:01", score=0.70),
            _row("PEP2", "HLA-A*02:01", score=0.40, peptide_offset=5),
            _row("PEP2", "HLA-B*07:02", score=0.85, peptide_offset=5),
        ])
        ctx = EvalContext(df)
        scores = Presentation["mhcflurry"].best_score.eval(ctx)
        alleles = Presentation["mhcflurry"].best_score_allele.eval(ctx)
        # PEP1 max is 0.70 on HLA-A*02:01; both rows for PEP1 should see it
        for allele in ("HLA-A*02:01", "HLA-B*07:02"):
            assert scores.loc[("p1", "PEP1", 0, allele)] == 0.70
            assert alleles.loc[("p1", "PEP1", 0, allele)] == "HLA-A*02:01"
        # PEP2 max is 0.85 on HLA-B*07:02
        for allele in ("HLA-A*02:01", "HLA-B*07:02"):
            assert scores.loc[("p1", "PEP2", 5, allele)] == 0.85
            assert alleles.loc[("p1", "PEP2", 5, allele)] == "HLA-B*07:02"

    def test_tie_picks_first_input_row_deterministically(self):
        # Two alleles tie on score; idxmax returns the first row index.
        df = pd.DataFrame([
            _row("PEP", "HLA-A*02:01", score=0.5),
            _row("PEP", "HLA-B*07:02", score=0.5),
        ])
        ctx = EvalContext(df)
        attribution = Presentation["mhcflurry"].best_score_allele.eval(ctx)
        # Locks in the deterministic-first behavior — change with intent
        assert attribution.loc[("p1", "PEP", 0, "HLA-A*02:01")] == "HLA-A*02:01"


class TestApplyFilterIntegration:
    """The vaxrank use case (#158): filter peptides by 'presented anywhere'
    via ``apply_filter`` with a best_score threshold."""

    def test_apply_filter_keeps_peptides_with_high_best_score(self):
        from topiary import apply_filter
        df = pd.DataFrame([
            _row("KEEP1", "HLA-A*02:01", score=0.10),  # weak per-allele,
            _row("KEEP1", "HLA-B*07:02", score=0.85),  # strong on B → keep both
            _row("DROP1", "HLA-A*02:01", score=0.10, peptide_offset=5),
            _row("DROP1", "HLA-B*07:02", score=0.20, peptide_offset=5),
        ])
        result = apply_filter(df, Presentation["mhcflurry"].best_score >= 0.5)
        kept_peptides = set(result["peptide"])
        assert kept_peptides == {"KEEP1"}
        # both alleles for KEEP1 survive (broadcast keeps the per-peptide group)
        assert len(result) == 2

    def test_apply_filter_forwards_kind_support_warning(self):
        from topiary import apply_filter
        df = pd.DataFrame([
            _row("PEP", "HLA-A*02:01", score=0.7),
            _row("PEP", "HLA-B*07:02", score=0.3),
        ])
        kind_support = {
            "mhcflurry": {
                "pMHC_presentation": {
                    "mhc_dependence": "single_allele",
                    "mhc_class": "I",
                }
            }
        }
        with pytest.warns(UserWarning, match="single_allele"):
            apply_filter(
                df,
                Presentation["mhcflurry"].best_score >= 0.5,
                kind_support=kind_support,
            )
