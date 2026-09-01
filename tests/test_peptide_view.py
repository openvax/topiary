"""peptide_view: one value per peptide, reduced for the kind's allele mode.

The DSL reads one value per (peptide, allele) group.  Which reduction is
*correct* depends on how the predictor treats alleles — best-across-alleles
for per-allele kinds, a direct read for peptide-level ones — and callers
had to know which by hand (topiary #169).
"""

import warnings

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


def test_allele_free_kind_projects_even_unwrapped():
    """A bare reference means the same thing — and says so once.

    A plain read can only ever leave the allele groups NaN, since the
    row is in none of them, so the reference is projected and warns
    rather than quietly returning nothing (topiary #186).
    """
    df = _mixed_df()

    with pytest.warns(UserWarning, match="carries no allele"):
        scores = evaluate_scores(df, Processing.score)

    assert scores.tolist() == [0.77, 0.77, 0.77]


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

    # The unwrapped form means the same thing, with a warning.
    with pytest.warns(UserWarning, match="carries no allele"):
        unwrapped = apply_filter(
            df, parse("affinity <= 500 & processing.score >= 0.5"),
        )
    assert unwrapped["allele"].tolist() == kept["allele"].tolist()


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
    with pytest.raises(ValueError, match="means one row per peptide"):
        evaluate_scores(
            df, peptide_view(Presentation.score), kind_support=support,
        )

    # Without the metadata, the same frame aggregates as single_allele.
    assert evaluate_scores(df, peptide_view(Presentation.score)).tolist() == [0.8, 0.8]


def test_models_disagreeing_about_dependence_is_an_error():
    """Both models are read here, so neither mode can be assumed.

    One method per allele group, so nothing narrows the read to a single
    model the way an ambiguity default would.
    """
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_affinity", value=50.0,
             prediction_method_name="netmhcpan"),
        _row(allele=ALLELES[1], kind="pMHC_affinity", value=90.0,
             prediction_method_name="mhcflurry"),
    ])
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

    with pytest.raises(ValueError, match="carry several different score"):
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
    from topiary.ranking.nodes import (
        _filter_kind_method_version, _resolve_mhc_dependence,
    )
    from mhctools import Kind

    for kind, expected in (
        (Kind.pMHC_affinity, "single_allele"),
        (Kind.antigen_processing, "none"),
    ):
        sub = _filter_kind_method_version(ctx, kind, None, None)
        assert _resolve_mhc_dependence(ctx, kind, sub) == expected


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


def test_group_keys_of_allele_alone_is_rejected():
    """No peptide dimension: a plain read would be NaN nearly everywhere."""
    df = _mixed_df()

    with pytest.raises(ValueError, match="needs a peptide dimension"):
        evaluate_scores(
            df, peptide_view(Processing.score), group_keys=["allele"],
        )


def test_two_key_grouping_projects_the_same_way():
    """Peptide + allele only — the flat-index shape for the peptide level."""
    df = _mixed_df()

    allele_free = evaluate_scores(
        df, peptide_view(Processing.score), group_keys=["peptide", "allele"],
    )
    per_allele = evaluate_scores(
        df, peptide_view(Affinity.value), group_keys=["peptide", "allele"],
    )

    assert allele_free.tolist() == [0.77, 0.77, 0.77]
    assert per_allele.tolist() == [50.0, 50.0, 50.0]


def test_null_spellings_in_peptide_keys_still_project():
    df = _mixed_df()
    df.loc[0, "source_sequence_name"] = None
    df.loc[1, "source_sequence_name"] = float("nan")
    df.loc[2, "source_sequence_name"] = float("nan")

    scores = evaluate_scores(df, peptide_view(Processing.score))

    assert scores.tolist() == [0.77, 0.77, 0.77]


def test_null_allele_counts_as_allele_free():
    """Loaders write NaN where others write the empty string."""
    df = _mixed_df()
    df.loc[2, "allele"] = float("nan")

    assert evaluate_scores(df, peptide_view(Processing.score)).tolist() == [
        0.77, 0.77, 0.77,
    ]


# ---------------------------------------------------------------------------
# Sort direction: the wrappers change which row is read, not which end is best
# ---------------------------------------------------------------------------


def _two_peptide_affinity_df():
    return pd.DataFrame(
        _affinity_rows(peptide="AAA", values=(50.0, 60.0))
        + _affinity_rows(peptide="BBB", values=(5000.0, 6000.0))
    )


def test_peptide_view_sorts_affinity_ascending():
    """auto direction must survive the wrapper: strong binders first."""
    df = _two_peptide_affinity_df()

    ordered = apply_sort(df, [peptide_view(Affinity.value)])

    assert ordered["peptide"].tolist()[0] == "AAA"


def test_best_allele_field_sorts_affinity_ascending():
    """Regression: BestAlleleField isn't a Field either, and sorted desc."""
    df = _two_peptide_affinity_df()

    ordered = apply_sort(df, [Affinity.best_value])

    assert ordered["peptide"].tolist()[0] == "AAA"


def test_peptide_view_sorts_percentile_rank_ascending():
    df = pd.DataFrame([
        _row(peptide="AAA", allele=ALLELES[0], kind="pMHC_affinity",
             value=10.0, score=0.9, percentile_rank=0.1),
        _row(peptide="BBB", allele=ALLELES[0], kind="pMHC_affinity",
             value=20.0, score=0.1, percentile_rank=30.0),
    ])

    ordered = apply_sort(df, [peptide_view(Affinity.rank)])

    assert ordered["peptide"].tolist()[0] == "AAA"


def test_peptide_view_sorts_scores_descending():
    df = _two_peptide_affinity_df()
    df.loc[df["peptide"] == "AAA", "score"] = 0.1
    df.loc[df["peptide"] == "BBB", "score"] = 0.9

    ordered = apply_sort(df, [peptide_view(Affinity.score)])

    assert ordered["peptide"].tolist()[0] == "BBB"


# ---------------------------------------------------------------------------
# Method binding: the resolver must see what the DSL already resolved
# ---------------------------------------------------------------------------


def _two_model_presentation_df():
    return pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.9,
             prediction_method_name="netmhcpan"),
        _row(allele=ALLELES[1], kind="pMHC_presentation", score=0.4,
             prediction_method_name="netmhcpan"),
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.8,
             prediction_method_name="mhcflurry"),
    ])


_SPLIT_SUPPORT = {
    "netmhcpan": {"pMHC_presentation": {"mhc_dependence": "single_allele"}},
    "mhcflurry": {"pMHC_presentation": {"mhc_dependence": "haplotype"}},
}


def test_default_methods_settles_the_dependence():
    """An explicit default names one model; that model's mode applies."""
    df = _two_model_presentation_df()

    scores = evaluate_scores(
        df, peptide_view(Presentation.score), kind_support=_SPLIT_SUPPORT,
        default_methods={"pMHC_presentation": "netmhcpan"},
    )

    # netmhcpan is single_allele: best of 0.9 / 0.4 across the peptide.
    assert scores.tolist()[:2] == [0.9, 0.9]


def test_filter_auto_aggregation_binds_the_method_for_dependence():
    """apply_filter binds one method per iteration; the resolver must see it."""
    df = _two_model_presentation_df()

    kept = apply_filter(
        df, parse("peptide_view(el.score) >= 0.5"),
        kind_support=_SPLIT_SUPPORT,
    )

    assert len(kept) > 0


# ---------------------------------------------------------------------------
# Frames and kinds without an allele dimension
# ---------------------------------------------------------------------------


def test_frame_without_an_allele_column_is_allele_free():
    df = pd.DataFrame([
        _row(peptide="AAA", kind="antigen_processing", score=0.77,
             prediction_method_name="mhcflurry"),
        _row(peptide="BBB", kind="antigen_processing", score=0.2,
             prediction_method_name="mhcflurry"),
    ]).drop(columns=["allele"])
    group_keys = ["source_sequence_name", "peptide", "peptide_offset"]

    scores = evaluate_scores(
        df, peptide_view(Processing.score), group_keys=group_keys,
    )

    assert scores.tolist() == [0.77, 0.2]


def test_field_without_a_best_direction_reads_per_peptide():
    """Legacy layout: processing rows duplicated across alleles.

    `value` has no per-kind ordering for processing, so there is no
    "best" to pick — reading the peptide's single value is the only
    sensible answer, and it must not crash.
    """
    df = pd.DataFrame([
        _row(allele=allele, kind="antigen_processing", value=0.77,
             prediction_method_name="mhcflurry")
        for allele in ALLELES
    ])

    assert evaluate_scores(df, peptide_view(Processing.value)).tolist() == [
        0.77, 0.77,
    ]


def test_missing_kind_without_a_direction_gives_nan():
    df = pd.DataFrame(_affinity_rows())

    scores = evaluate_scores(df, peptide_view(Processing.value))

    assert scores.isna().all()


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------


def test_allele_returning_field_is_rejected():
    with pytest.raises(TypeError, match="returns an allele name"):
        peptide_view(Affinity.best_score_allele)


def test_unknown_mhc_dependence_is_rejected():
    df = pd.DataFrame(_affinity_rows())
    support = _kind_support("netmhcpan", "pMHC_affinity", "supertype")

    with pytest.raises(ValueError, match="unknown mhc_dependence 'supertype'"):
        evaluate_scores(
            df, peptide_view(Affinity.value), kind_support=support,
        )


def test_models_sharing_a_method_name_get_an_honest_error():
    """TopiaryPredictor keys same-named models __1/__2; rows can't say which."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.8,
             prediction_method_name="mhcflurry"),
    ])
    support = {
        "mhcflurry__1": {"pMHC_presentation": {"mhc_dependence": "single_allele"}},
        "mhcflurry__2": {"pMHC_presentation": {"mhc_dependence": "haplotype"}},
    }

    with pytest.raises(ValueError, match="cannot separate them"):
        evaluate_scores(
            df, peptide_view(Presentation.score), kind_support=support,
        )


def test_a_scoped_field_filters_through_peptide_view():
    """Scoped fields are filterable (#192); the wrapper doesn't change that."""
    from topiary.ranking import wt

    df = pd.DataFrame([
        _row(allele=allele, kind="pMHC_affinity", value=100.0, score=0.5,
             wt_value=wt_value)
        for allele, wt_value in zip(ALLELES, (9000.0, 9000.0))
    ])

    kept = apply_filter(df, peptide_view(wt.Affinity.value) >= 500)

    assert len(kept) == len(df)
    assert parse("peptide_view(wt.affinity.value) >= 500") is not None


def test_values_agreeing_within_float_noise_are_accepted():
    """One row round-tripped through a CSV, one computed in-process."""
    df = pd.DataFrame(_affinity_rows() + [
        _processing_row(score=0.1 + 0.2), _processing_row(score=0.3),
    ])

    scores = evaluate_scores(df, peptide_view(Processing.score))

    assert scores.tolist() == [pytest.approx(0.3)] * 4


def test_values_that_really_differ_are_still_rejected():
    df = pd.DataFrame(_affinity_rows() + [
        _processing_row(score=0.30), _processing_row(score=0.31),
    ])

    with pytest.raises(ValueError, match="carry several different score"):
        evaluate_scores(df, peptide_view(Processing.score))


# ---------------------------------------------------------------------------
# The mode is resolved from the rows the expression actually reads
# ---------------------------------------------------------------------------


def test_default_methods_naming_an_absent_model_keeps_kind_support():
    """A pipeline-wide default for another kind must not void the metadata."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.8,
             prediction_method_name="mhcflurry"),
        _row(allele=ALLELES[1], kind="pMHC_presentation", score=0.2,
             prediction_method_name="mhcflurry"),
    ])
    support = _kind_support("mhcflurry", "pMHC_presentation", "haplotype")

    # Two haplotype rows for one peptide is a data error either way — the
    # default naming a model absent from this frame must not turn it into
    # a silent max() across alleles.
    with pytest.raises(ValueError, match="means one row per peptide"):
        evaluate_scores(
            df, peptide_view(Presentation.score), kind_support=support,
            default_methods={"pMHC_presentation": "netmhcpan"},
        )


def test_another_models_alleles_do_not_reclassify_this_one():
    """Row-based inference must look at the selected rows, not the kind."""
    conflicting = [
        _row(kind="antigen_processing", score=score,
             prediction_method_name="netchop")
        for score in (0.30, 0.31)
    ]
    unrelated = _row(kind="antigen_processing", score=0.5,
                     allele=ALLELES[0], prediction_method_name="otherpred")

    for rows in (conflicting, conflicting + [unrelated]):
        with pytest.raises(ValueError, match="carry several different score"):
            evaluate_scores(
                pd.DataFrame(rows), peptide_view(Processing["netchop"].score),
            )


def test_a_model_with_no_rows_here_does_not_conflict():
    """Metadata for a model that produced nothing must not veto the frame."""
    df = pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_presentation", score=0.9,
             prediction_method_name="netmhcpan"),
        _row(allele=ALLELES[1], kind="pMHC_presentation", score=0.4,
             prediction_method_name="netmhcpan"),
    ])
    support = {
        "netmhcpan": {"pMHC_presentation": {"mhc_dependence": "single_allele"}},
        "mhcflurry": {"pMHC_presentation": {"mhc_dependence": "haplotype"}},
    }

    with pytest.warns(UserWarning, match="not a joint multi-allele aggregate"):
        scores = evaluate_scores(
            df, peptide_view(Presentation.score), kind_support=support,
        )

    assert scores.tolist() == [0.9, 0.9]


def test_unknown_dependence_is_named_even_alongside_a_known_one():
    """Version skew can't be resolved by picking the other model's mode."""
    df = pd.DataFrame([
        _row(kind="antigen_processing", score=0.5,
             prediction_method_name="netchop"),
        _row(allele=ALLELES[0], kind="antigen_processing", score=0.6,
             prediction_method_name="otherpred"),
    ])
    support = {
        "netchop": {"antigen_processing": {"mhc_dependence": "none"}},
        "otherpred": {"antigen_processing": {"mhc_dependence": "supertype"}},
    }

    with pytest.raises(ValueError, match="unknown mhc_dependence 'supertype'"):
        evaluate_scores(
            df, peptide_view(Processing.score), kind_support=support,
        )


# ---------------------------------------------------------------------------
# The one-value-per-peptide check runs on every peptide-level path
# ---------------------------------------------------------------------------


def test_disagreeing_values_are_rejected_without_an_allele_group_key():
    """Same data, same node: the grouping must not decide whether it raises."""
    df = pd.DataFrame([
        _row(kind="antigen_processing", value=value,
             prediction_method_name="mhcflurry")
        for value in (0.9, 0.1)
    ])

    with pytest.raises(ValueError, match="carry several different value"):
        evaluate_scores(df, peptide_view(Processing.value))

    with pytest.raises(ValueError, match="carry several different value"):
        evaluate_scores(
            df, peptide_view(Processing.value),
            group_keys=["source_sequence_name", "peptide", "peptide_offset"],
        )


def test_best_field_without_a_direction_is_rejected_not_downgraded():
    """peptide_view(processing.best_value) must not quietly read `value`."""
    df = pd.DataFrame([
        _row(allele=allele, kind="antigen_processing", value=0.77,
             prediction_method_name="mhcflurry")
        for allele in ALLELES
    ])

    with pytest.raises(ValueError, match="no defined best direction"):
        evaluate_scores(df, peptide_view(Processing.best_value))


def test_no_direction_error_explains_the_real_cause():
    """Not 'mhc_dependence=single_allele means one row per peptide'.

    The disagreeing rows must be **allele-free** to be a disagreement.
    Two allele-restricted rows are two answers to two questions (#232),
    not one peptide contradicting itself.
    """
    df = pd.DataFrame([
        _row(allele=None, kind="antigen_processing", value=value,
             prediction_method_name="mhcflurry")
        for value in (0.9, 0.1)
    ])

    with pytest.raises(ValueError, match="no defined best direction"):
        evaluate_scores(df, peptide_view(Processing.value))


# ---------------------------------------------------------------------------
# The warning names what the user wrote
# ---------------------------------------------------------------------------


def test_single_allele_warning_names_the_wrapping_expression():
    df = pd.DataFrame(_affinity_rows())
    support = _kind_support("netmhcpan", "pMHC_affinity", "single_allele")

    with pytest.warns(UserWarning, match=r"peptide_view\(affinity\.value\)"):
        evaluate_scores(
            df, peptide_view(Affinity.value), kind_support=support,
        )

    # A bare best_* still names itself.
    with pytest.warns(UserWarning, match="best_value on"):
        evaluate_scores(df, Affinity.best_value, kind_support=support)


# ---------------------------------------------------------------------------
# A genotype-level score is peptide-level too (auto-projection, haplotype)
# ---------------------------------------------------------------------------


HAPLOTYPE_SUPPORT = {
    "mhcflurry": {
        "pMHC_presentation": {"mhc_dependence": "haplotype"},
        "pMHC_affinity": {"mhc_dependence": "single_allele"},
    },
}


def _haplotype_df(presentation_allele):
    """One genotype-level presentation row plus per-allele affinity.

    ``presentation_allele`` is how the row names itself: mhctools stamps
    the deconvolved best allele today, and a faithful writer might leave
    it blank.  Neither should decide whether the genotype's score
    reaches the genotype's alleles.
    """
    return pd.DataFrame([
        _row(allele=ALLELES[0], kind="pMHC_affinity", value=50.0, score=0.5,
             prediction_method_name="mhcflurry"),
        _row(allele=ALLELES[1], kind="pMHC_affinity", value=4000.0, score=0.4,
             prediction_method_name="mhcflurry"),
        _row(allele=presentation_allele, kind="pMHC_presentation", score=0.9,
             prediction_method_name="mhcflurry"),
    ])


@pytest.mark.parametrize("presentation_allele", ["", ALLELES[0]])
def test_haplotype_score_reaches_the_whole_genotype(presentation_allele):
    df = _haplotype_df(presentation_allele)

    with pytest.warns(UserWarning, match="whole genotype"):
        scores = evaluate_scores(
            df, Presentation.score, kind_support=HAPLOTYPE_SUPPORT,
        )

    # Every row's group carries the genotype's score, not just the one
    # allele the predictor deconvolved as best.
    assert scores.tolist() == [0.9] * len(df)


def test_metadata_decides_what_a_blank_allele_row_means():
    """Rows alone can't tell genotype-level from malformed.

    With ``kind_support`` saying haplotype, a blank-allele presentation
    row is peptide-level and projects across the genotype. Without it,
    the kind's own default applies — ``pMHC_presentation`` describes a
    peptide-MHC pair — so a blank allele reads as a per-allele row
    missing its allele, and projecting it would invent evidence for
    alleles no model scored (topiary #195).

    Neither case is silent, which is the invariant that matters: more
    metadata must never make the bare read fail more quietly.
    """
    df = _haplotype_df("")

    with pytest.warns(UserWarning, match="whole genotype"):
        labeled = evaluate_scores(
            df, Presentation.score, kind_support=HAPLOTYPE_SUPPORT,
        )
    with pytest.warns(UserWarning, match="carry no allele"):
        unlabeled = evaluate_scores(df, Presentation.score)

    assert labeled.tolist() == [0.9] * len(df)
    assert unlabeled.isna().tolist() == [True, True, False]


def test_haplotype_warning_names_the_deconvolution():
    df = _haplotype_df(ALLELES[0])

    with pytest.warns(UserWarning) as caught:
        evaluate_scores(df, Presentation.score, kind_support=HAPLOTYPE_SUPPORT)

    message = str(caught[0].message)
    assert "deconvolved" in message and "peptide_view(" in message


def test_explicit_peptide_view_on_a_haplotype_kind_is_silent():
    df = _haplotype_df(ALLELES[0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = evaluate_scores(
            df, peptide_view(Presentation.score),
            kind_support=HAPLOTYPE_SUPPORT,
        )

    assert scores.tolist() == [0.9] * len(df)


def test_single_allele_kinds_are_still_left_alone():
    """Choosing which allele's row to read remains the caller's decision."""
    df = _haplotype_df(ALLELES[0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        affinity = evaluate_scores(
            df, Affinity.value, kind_support=HAPLOTYPE_SUPPORT,
        )

    assert affinity.tolist()[:2] == [50.0, 4000.0]


def test_no_haplotype_projection_without_an_allele_group_key():
    df = _haplotype_df(ALLELES[0])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = evaluate_scores(
            df, Presentation.score, kind_support=HAPLOTYPE_SUPPORT,
            group_keys=["source_sequence_name", "peptide", "peptide_offset"],
        )

    assert scores.tolist() == [0.9] * len(df)
