"""peptide_view: one value per peptide, reduced for the kind's allele mode.

The DSL reads one value per (peptide, allele) group.  Which reduction is
*correct* depends on how the predictor treats alleles — best-across-alleles
for per-allele kinds, a direct read for peptide-level ones — and callers
had to know which by hand (topiary #169).
"""

import pandas as pd
import pytest

from topiary.ranking import (
    Affinity,
    EvalContext,
    Presentation,
    Processing,
    apply_filter,
    apply_sort,
    evaluate_scores,
    parse,
    peptide_view,
)

ALLELES = ("HLA-A*02:01", "HLA-B*07:02")


def _row(**kwargs):
    row = dict(
        source_sequence_name="ctx", peptide="SIINFEKL", peptide_offset=0,
        allele="", kind="pMHC_affinity", score=None, value=None,
        percentile_rank=None, prediction_method_name="netmhcpan",
    )
    row.update(kwargs)
    return row


def _affinity_rows(values=(50.0, 4000.0), peptide="SIINFEKL"):
    return [
        _row(peptide=peptide, allele=allele, kind="pMHC_affinity",
             value=value, score=0.9, percentile_rank=0.5)
        for allele, value in zip(ALLELES, values)
    ]


def _processing_row(score=0.77, peptide="SIINFEKL"):
    """Allele-free: the predictor never saw an allele."""
    return _row(peptide=peptide, allele="", kind="antigen_processing",
                score=score, prediction_method_name="mhcflurry")


def _mixed_df():
    return pd.DataFrame(_affinity_rows() + [_processing_row()])


# ---------------------------------------------------------------------------
# mhc_dependence='none' — the case that could not be written before
# ---------------------------------------------------------------------------


def test_allele_free_kind_is_nan_without_peptide_view():
    """The motivating gap: a processing row is in nobody's allele group."""
    df = _mixed_df()

    scores = evaluate_scores(df, Processing.score)

    # Two per-allele groups read NaN; only the processing row's own group
    # sees the value.
    assert scores.tolist()[:2] == [pytest.approx(float("nan"), nan_ok=True)] * 2
    assert scores.isna().tolist() == [True, True, False]


def test_peptide_view_broadcasts_allele_free_value_to_every_group():
    df = _mixed_df()

    scores = evaluate_scores(df, peptide_view(Processing.score))

    assert scores.tolist() == [0.77, 0.77, 0.77]


def test_peptide_view_composes_with_per_allele_clauses():
    """The expression vaxrank has to fan rows out to write today."""
    df = _mixed_df()
    node = parse("affinity <= 500 & peptide_view(processing.score) >= 0.5")

    kept = apply_filter(df, node)

    # Only the A*02:01 affinity row clears both clauses.
    assert kept["allele"].tolist() == ["HLA-A*02:01"]

    # Without the projection, the processing clause is NaN everywhere the
    # affinity clause is true, so nothing survives.
    assert len(apply_filter(df, parse("affinity <= 500 & processing.score >= 0.5"))) == 0


def test_peptide_view_processing_gate_can_reject_every_allele():
    df = pd.DataFrame(_affinity_rows() + [_processing_row(score=0.1)])
    node = parse("affinity <= 500 & peptide_view(processing.score) >= 0.5")

    assert len(apply_filter(df, node)) == 0


def test_allele_free_rows_of_two_peptides_stay_separate():
    df = pd.DataFrame(
        _affinity_rows(peptide="SIINFEKL")
        + _affinity_rows(peptide="ELAGIGILT", values=(20.0, 30.0))
        + [_processing_row(peptide="SIINFEKL", score=0.9),
           _processing_row(peptide="ELAGIGILT", score=0.2)]
    )

    scores = evaluate_scores(df, peptide_view(Processing.score))
    by_peptide = dict(zip(df["peptide"], scores))

    assert by_peptide["SIINFEKL"] == 0.9
    assert by_peptide["ELAGIGILT"] == 0.2


# ---------------------------------------------------------------------------
# mhc_dependence='single_allele' — best across alleles
# ---------------------------------------------------------------------------


def test_peptide_view_aggregates_per_allele_kinds():
    df = pd.DataFrame(_affinity_rows(values=(50.0, 4000.0)))

    scores = evaluate_scores(df, peptide_view(Affinity.value))

    # Best IC50 for the peptide, broadcast to both allele groups.
    assert scores.tolist() == [50.0, 50.0]


def test_peptide_view_matches_best_allele_field():
    df = pd.DataFrame(_affinity_rows(values=(50.0, 4000.0)))

    via_view = evaluate_scores(df, peptide_view(Affinity.value))
    via_best = evaluate_scores(df, Affinity.best_value)

    assert via_view.tolist() == via_best.tolist()


def test_peptide_view_uses_the_kinds_own_direction():
    """Scores are best-high, IC50 best-low; the caller shouldn't track that."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.2),
        _row(allele=ALLELES[1], kind="pMHC_presentation", score=0.95),
    ])

    scores = evaluate_scores(df, peptide_view(Presentation.score))

    assert scores.tolist() == [0.95, 0.95]


def test_peptide_view_accepts_a_kind_accessor():
    df = pd.DataFrame(_affinity_rows())

    assert (
        evaluate_scores(df, peptide_view(Affinity)).tolist()
        == evaluate_scores(df, peptide_view(Affinity.value)).tolist()
    )


# ---------------------------------------------------------------------------
# Dispatch: kind_support beats data inference
# ---------------------------------------------------------------------------


def _kind_support(model, kind_value, dependence):
    return {model: {kind_value: {"mhc_dependence": dependence}}}


def test_haplotype_row_is_read_directly_not_aggregated():
    """One row per peptide already IS the peptide-level value."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.8,
             prediction_method_name="mhcflurry"),
        # A second peptide's haplotype row, so the frame has two groups.
        _row(peptide="ELAGIGILT", allele=ALLELES[0], kind="pMHC_presentation",
             score=0.3, prediction_method_name="mhcflurry"),
    ])
    support = _kind_support("mhcflurry", "pMHC_presentation", "haplotype")

    scores = evaluate_scores(
        df, peptide_view(Presentation.score), kind_support=support,
    )

    assert scores.tolist() == [0.8, 0.3]


def test_kind_support_overrides_what_the_rows_look_like():
    """Rows carry alleles, so inference would say single_allele."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.8,
             prediction_method_name="mhcflurry"),
        _row(allele=ALLELES[1], kind="pMHC_presentation", score=0.2,
             prediction_method_name="mhcflurry"),
    ])
    support = _kind_support("mhcflurry", "pMHC_presentation", "haplotype")

    # Two haplotype rows for one peptide is a data error, not a max().
    with pytest.raises(ValueError, match="expects one score per peptide"):
        evaluate_scores(
            df, peptide_view(Presentation.score), kind_support=support,
        )

    # Without the metadata, the same frame aggregates as single_allele.
    assert evaluate_scores(df, peptide_view(Presentation.score)).tolist() == [0.8, 0.8]


def test_models_disagreeing_about_dependence_is_an_error():
    df = pd.DataFrame(_affinity_rows())
    support = {
        "netmhcpan": {"pMHC_affinity": {"mhc_dependence": "single_allele"}},
        "mhcflurry": {"pMHC_affinity": {"mhc_dependence": "haplotype"}},
    }

    with pytest.raises(ValueError, match="models disagree"):
        evaluate_scores(df, peptide_view(Affinity.value), kind_support=support)


def test_qualifying_by_method_resolves_the_disagreement():
    df = pd.DataFrame(_affinity_rows())
    support = {
        "netmhcpan": {"pMHC_affinity": {"mhc_dependence": "single_allele"}},
        "mhcflurry": {"pMHC_affinity": {"mhc_dependence": "haplotype"}},
    }

    # The single_allele caveat still applies: best-of-per-allele is not a
    # joint multi-allele aggregate, and peptide_view doesn't hide that.
    with pytest.warns(UserWarning, match="not a joint multi-allele aggregate"):
        scores = evaluate_scores(
            df, peptide_view(Affinity["netmhcpan"].value), kind_support=support,
        )

    assert scores.tolist() == [50.0, 50.0]


# ---------------------------------------------------------------------------
# Errors and edges
# ---------------------------------------------------------------------------


def test_conflicting_peptide_level_values_are_rejected():
    df = pd.DataFrame(_affinity_rows() + [
        _processing_row(score=0.77), _processing_row(score=0.11),
    ])

    with pytest.raises(ValueError, match="carry several different values"):
        evaluate_scores(df, peptide_view(Processing.score))


def test_duplicate_identical_peptide_level_rows_are_fine():
    df = pd.DataFrame(_affinity_rows() + [
        _processing_row(score=0.77), _processing_row(score=0.77),
    ])

    # One value per row, and the duplicate agrees with the original.
    assert evaluate_scores(df, peptide_view(Processing.score)).tolist() == (
        [0.77] * 4
    )


def test_best_field_on_an_allele_free_kind_is_rejected():
    df = _mixed_df()

    with pytest.raises(ValueError, match="nothing to aggregate across alleles"):
        evaluate_scores(df, peptide_view(Processing.best_score))


def test_peptide_view_rejects_a_composite_expression():
    with pytest.raises(TypeError, match="Wrap the field, not the expression"):
        peptide_view(Affinity.value * 2)


def test_peptide_view_result_composes_in_arithmetic():
    df = _mixed_df()

    scores = evaluate_scores(
        df, 0.5 * peptide_view(Processing.score) + 0.5 * Affinity.score,
    )

    # 0.5 * 0.77 + 0.5 * 0.9 for the two affinity groups.
    assert scores.tolist()[:2] == [pytest.approx(0.835)] * 2


def test_peptide_view_sorts_groups():
    df = pd.DataFrame(
        _affinity_rows(peptide="SIINFEKL")
        + _affinity_rows(peptide="ELAGIGILT", values=(10.0, 20.0))
        + [_processing_row(peptide="SIINFEKL", score=0.2),
           _processing_row(peptide="ELAGIGILT", score=0.9)]
    )

    ordered = apply_sort(df, [peptide_view(Processing.score)])

    assert ordered["peptide"].tolist()[0] == "ELAGIGILT"


def test_missing_kind_gives_nan_not_an_error():
    df = pd.DataFrame(_affinity_rows())

    scores = evaluate_scores(df, peptide_view(Processing.score))

    assert scores.isna().all()


def test_empty_frame_is_empty():
    df = _mixed_df().iloc[0:0]

    assert evaluate_scores(df, peptide_view(Processing.score)).tolist() == []


def test_group_keys_without_allele_degenerate_to_a_direct_read():
    df = _mixed_df()

    scores = evaluate_scores(
        df, peptide_view(Processing.score),
        group_keys=["source_sequence_name", "peptide", "peptide_offset"],
    )

    assert scores.tolist() == [0.77, 0.77, 0.77]


def test_dependence_inference_ignores_other_kinds():
    """An allele-free processing row must not make affinity look allele-free."""
    df = _mixed_df()

    ctx = EvalContext(df)
    from topiary.ranking.nodes import _resolve_mhc_dependence
    from mhctools import Kind

    assert _resolve_mhc_dependence(ctx, Kind.pMHC_affinity) == "single_allele"
    assert _resolve_mhc_dependence(ctx, Kind.antigen_processing) == "none"


# ---------------------------------------------------------------------------
# String form — vaxrank drives scoring from config strings
# ---------------------------------------------------------------------------


def test_string_form_parses():
    node = parse("peptide_view(processing.score)")

    assert node.to_ast_string() == (
        "PeptideView(Field(kind=antigen_processing, field='score'))"
    )


def test_string_form_round_trips():
    for text in (
        "peptide_view(processing.score)",
        "peptide_view(affinity.value)",
        "peptide_view(mhcflurry:el.score)",
        "0.5 * peptide_view(processing.score) + 0.5 * affinity.score",
    ):
        node = parse(text)
        assert parse(node.to_expr_string()).to_ast_string() == node.to_ast_string()


def test_string_form_evaluates_like_the_python_api():
    df = _mixed_df()

    assert (
        evaluate_scores(df, parse("peptide_view(processing.score)")).tolist()
        == evaluate_scores(df, peptide_view(Processing.score)).tolist()
    )
