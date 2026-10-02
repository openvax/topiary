"""Coverage denominators and common-set comparisons retain unknown evidence."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from topiary import (
    SelectionCriterion, SelectionPolicy, TopiaryResult, compare_policy_evaluations,
    evaluate_selection_policy, policy_coverage, replay_selection_policy, summarize_policy_coverage,
)
from .test_candidate_tables import source
from .test_twin_conformance import DELIMITED_IO_TWINS

KEYS = ["prediction_id", "peptide", "allele"]


def evaluation(values=(0., np.nan), policy=None, **columns):
    frame = source(values=values, prediction_id=["zero", "missing"], **columns)
    policy = policy or SelectionPolicy("coverage", 'criterion("binding")', score_fill=0., strata=(),
        criteria=(SelectionCriterion("binding", "affinity.value", "score"),))
    return evaluate_selection_policy(frame, policy, group_keys=KEYS)


def requests(evaluated, **columns):
    frame = evaluated.occurrences[KEYS].copy()
    for key, value in dict(kind="pMHC_affinity", prediction_method_name="original", predictor_version="1",
                           field="value", **columns).items():
        frame[key] = value
    return frame


def test_zero_is_assessed_but_filled_zero_remains_missing():
    evaluated = evaluation()
    coverage = policy_coverage(evaluated, prediction_requests=requests(evaluated))
    assert coverage.df.score.eq(0).all()
    assert coverage.df.eligible.all()
    assert coverage.df.query("prediction_id == 'zero'").status.eq("assessed").all()
    assert coverage.df.query("prediction_id == 'missing'").status.eq("missing").all()
    summary = summarize_policy_coverage(coverage)
    assert summary.n_assessments.tolist() == [2, 2, 2]
    assert summary.n_assessed.eq(1).all() and summary.n_missing.eq(1).all()
    assert summary.assessed_fraction.eq(.5).all()
    assert summary.n_prediction_rows.tolist() == [0, 0, 2]
    assert coverage.extra["policy_coverage"]["evaluation"]["definition"]["score_fill"] == 0.


def test_missing_inputs_requests_and_diagnostics_are_in_denominator():
    evaluated = evaluation()
    expected = pd.concat([evaluated.occurrences[KEYS], pd.DataFrame([dict(
        prediction_id="absent", peptide="ELAGIGILT", allele="HLA-A*02:01")])], ignore_index=True)
    requested = requests(evaluated).reindex(range(3))
    requested.loc[2] = [*expected.iloc[2], "pMHC_affinity", "original", "1", "value"]
    requested["status"] = [None, "missing", "failed"]
    requested["reason"] = [None, "insufficient_context", "backend_failure"]
    coverage = policy_coverage(evaluated, universe=expected, prediction_requests=requested)
    absent = coverage.df.query("prediction_id == 'absent'")
    assert absent.status.tolist() == ["not_evaluated", "not_evaluated", "failed"]
    summary = summarize_policy_coverage(coverage, by=["level", "reason"])
    assert summary.query("reason == 'insufficient_context'").n_missing.item() == 1
    assert summary.query("reason == 'backend_failure'").n_failed.item() == 1
    assert summarize_policy_coverage(coverage, by=["level"]).n_assessments.eq(3).all()
    requested.loc[0, ["status", "reason"]] = ["failed", "backend_failure"]
    with pytest.raises(ValueError, match="contradicts"):
        policy_coverage(evaluated, universe=expected, prediction_requests=requested)


def test_native_failure_inapplicability_and_unused_are_not_missing():
    policy = SelectionPolicy("filter", 'criterion("binding")', filter_by='criterion("keep")',
        criteria=(SelectionCriterion("keep", "n_rna_alt > 10", "eligibility"),
                  SelectionCriterion("binding", "affinity.value", "score", applies_to="n_rna_alt < 0"),
                  SelectionCriterion("unused", "affinity.value", "score")))
    coverage = policy_coverage(evaluation(policy=policy))
    by = coverage.df.set_index(["prediction_id", "level", "criterion"])
    assert by.loc[("zero", "criterion", "keep"), "native_status"] == "fail"
    assert by.loc[("zero", "criterion", "keep"), "status"] == "assessed"
    assert by.loc[("missing", "criterion", "binding"), "status"] == "not_applicable"
    assert by.loc[("zero", "criterion", "binding"), "reason"] == "filtered_before_scoring"
    assert by.loc[("missing", "criterion", "unused"), "reason"] == "not_referenced"


def test_model_version_field_and_allele_are_exact():
    evaluated = evaluation(values=(0., 3.))
    requested = requests(evaluated)
    requested.loc[0, "predictor_version"] = "2"
    requested.loc[1, "field"] = "absent_column"
    coverage = policy_coverage(evaluated, prediction_requests=requested)
    assert coverage.df.query("level == 'prediction'").reason.tolist() == ["no_prediction_rows", "missing_field"]
    frame = evaluated.evidence.df.copy()
    second_allele = frame.iloc[[0]].assign(allele="HLA-B*07:02", value=np.nan)
    frame = pd.concat([frame, second_allele], ignore_index=True)
    evaluated = evaluate_selection_policy(TopiaryResult(frame), evaluated.policy, group_keys=KEYS)
    coverage = policy_coverage(evaluated, prediction_requests=requests(evaluated))
    row = coverage.df.query("level == 'prediction' and allele == 'HLA-B*07:02'").iloc[0]
    assert row.status == "missing" and row.evidence_rows == [2]


def test_projection_and_duplicate_rows_do_not_inflate_assessments():
    frame = source(kind="proteasome_cleavage", allele="", prediction_id=["one", "two"], score=[.2, .8]).df
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    evaluated = evaluate_selection_policy(TopiaryResult(frame),
        SelectionPolicy("processing", "peptide_view(proteasome_cleavage.score)"),
        group_keys=KEYS, alleles=["HLA-A*02:01", "HLA-B*07:02"])
    requested = requests(evaluated)
    requested["kind"] = "proteasome_cleavage"
    requested["field"] = "score"
    coverage = policy_coverage(evaluated, prediction_requests=requested)
    summary = summarize_policy_coverage(coverage).query("level == 'prediction'").iloc[0]
    assert summary.n_assessed == summary.n_occurrences == 6
    assert summary.n_prediction_rows == 3  # three stored rows, not nine projected links
    assert coverage.df.query("level == 'prediction' and prediction_id == 'one'").evidence_rows.tolist() == [[0, 2]] * 3


def test_haplotype_is_not_borrowed_from_presenter():
    frame = source(prediction_id=["zero", "missing"], allele_set="HLA-A*02:01,HLA-B*07:02").df
    evaluated = evaluate_selection_policy(TopiaryResult(frame), SelectionPolicy("haplotype", "peptide_view(affinity.value)"),
                                           group_keys=KEYS + ["allele_set"])
    keys = KEYS + ["allele_set"]
    requested = evaluated.occurrences[keys].assign(kind="pMHC_affinity", prediction_method_name="original",
                                                   predictor_version="1", field="value")
    coverage = policy_coverage(evaluated, prediction_requests=requested)
    assert coverage.df.query("level == 'prediction'").status.eq("assessed").all()
    # An explicit narrower grouping cannot silently treat a missing genotype as per-allele.
    requested = requested.drop(columns="allele_set")
    coverage = policy_coverage(evaluated, keys=KEYS, prediction_requests=requested)
    assert coverage.df.query("level == 'prediction'").status.eq("missing").all()


def test_conflicting_predictions_remain_unknown():
    evaluated = evaluation()
    frame = pd.concat([evaluated.evidence.df, evaluated.evidence.df.iloc[[0]].assign(value=100.)], ignore_index=True)
    evaluated = evaluate_selection_policy(TopiaryResult(frame), evaluated.policy, group_keys=KEYS)
    coverage = policy_coverage(evaluated, prediction_requests=requests(evaluated))
    assert coverage.df.query("level == 'prediction' and prediction_id == 'zero'").reason.item() == "conflicting_evidence"


def test_comparison_preserves_union_and_only_compares_common_assessable_scores():
    left = evaluation(values=(0., 3.))
    right = evaluation(values=(1., np.nan))
    comparison = compare_policy_evaluations(left, right)
    assert comparison.df.assessment_set.tolist() == ["both", "left_only"]
    assert comparison.df.raw_score_delta.iloc[0] == 1.
    assert pd.isna(comparison.df.raw_score_delta.iloc[1])
    assert comparison.df.right_score.iloc[1] == 0.
    assert comparison.df.right_unknown_criteria.iloc[1] == ["binding"]
    assert comparison.extra["policy_comparison"]["left"]["sha256"] == left.policy.sha256
    frame = right.evidence.df.copy()
    frame.loc[1, "prediction_id"] = "other"
    right = evaluate_selection_policy(TopiaryResult(frame), right.policy, group_keys=KEYS)
    comparison = compare_policy_evaluations(left, right)
    assert comparison.df.left_present.tolist() == [True, True, False]
    assert comparison.df.right_present.tolist() == [True, False, True]
    assert comparison.df.assessment_set.tolist() == ["both", "left_only", "neither"]


def test_unknown_included_filter_is_not_fully_assessable():
    policy = SelectionPolicy("include", "1", filter_by='criterion("binding")', unknown="include",
        criteria=(SelectionCriterion("binding", "affinity.value < 100", "eligibility"),))
    evaluated = evaluation(policy=policy)
    assert evaluated.occurrences.score.eq(1).all()
    comparison = compare_policy_evaluations(evaluated, evaluated)
    assert comparison.df.assessment_set.tolist() == ["both", "neither"]


@pytest.mark.parametrize("suffix,writer,_,reader", DELIMITED_IO_TWINS)
def test_coverage_and_comparison_roundtrip_replay(tmp_path, suffix, writer, _, reader, pandas_string_inference):
    evaluated = evaluation()
    requested = requests(evaluated)
    coverage = policy_coverage(evaluated, prediction_requests=requested)
    path = tmp_path / ("coverage." + suffix)
    writer(coverage, path)
    restored = reader(path)
    pd.testing.assert_frame_equal(restored.df[coverage.df.columns], coverage.df, check_exact=True)
    pd.testing.assert_frame_equal(summarize_policy_coverage(restored), summarize_policy_coverage(coverage))
    evidence_path = tmp_path / ("evidence." + suffix)
    writer(evaluated.evidence, evidence_path)
    replay = replay_selection_policy(reader(evidence_path))
    metadata = restored.extra["policy_coverage"]
    regenerated = policy_coverage(replay, keys=metadata["keys"],
                                  prediction_requests=pd.DataFrame(metadata["prediction_requests"]))
    pd.testing.assert_frame_equal(regenerated.df, coverage.df, check_exact=True)
    comparison = compare_policy_evaluations(evaluated, replay)
    writer(comparison, path)
    pd.testing.assert_frame_equal(reader(path).df[comparison.df.columns], comparison.df, check_exact=True)


def test_empty_and_invalid_denominators():
    evaluated = evaluation()
    empty = evaluate_selection_policy(TopiaryResult(evaluated.evidence.df.iloc[:0]), evaluated.policy, group_keys=KEYS)
    assert policy_coverage(empty).df.empty
    assert summarize_policy_coverage(policy_coverage(empty)).empty
    assert compare_policy_evaluations(empty, empty).df.empty
    with pytest.raises(ValueError, match="include every"):
        policy_coverage(evaluated, universe=evaluated.occurrences[KEYS].iloc[:1])
    with pytest.raises(ValueError, match="uniquely"):
        policy_coverage(evaluated, keys=["allele"])
    requested = requests(evaluated)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        policy_coverage(evaluated, prediction_requests=pd.concat([requested, requested]))
    with pytest.raises(ValueError, match="Missing prediction"):
        policy_coverage(evaluated, prediction_requests=requested.drop(columns="field"))
    with pytest.raises(ValueError, match="universe"):
        policy_coverage(evaluated, prediction_requests=requested.assign(prediction_id="absent"))
    with pytest.raises(ValueError, match="groupings differ"):
        compare_policy_evaluations(evaluated, evaluate_selection_policy(evaluated.evidence, evaluated.policy))


def short_context_inputs():
    """Synthetic retained outputs: the short source has no processing result."""
    frame = source(peptides=("SIINFEKL", "SIINFEKL"), values=(50., 100.), prediction_id=["short", "long"],
                   peptide_offset=[0, 3], n_flank=["", "AAA"], c_flank=["", "GGG"],
                   source_sequence=["SIINFEKL", "AAASIINFEKLGGG"]).df
    processing = frame.iloc[[1]].assign(kind="proteasome_cleavage", prediction_method_name="context-model",
                                       allele="", value=1., score=1.)
    return TopiaryResult(pd.concat([frame, processing], ignore_index=True))


@pytest.mark.parametrize("values", [(0., np.nan), (2., 3.)])
@pytest.mark.parametrize("filtered", [False, True])
def test_report_and_comparison_share_assessability(values, filtered):
    from .test_twin_conformance import POLICY_COVERAGE_TWINS
    report, compare = POLICY_COVERAGE_TWINS
    evaluated = evaluation(values=values)
    if filtered:
        evaluated = evaluation(values=values, policy=replace(evaluated.policy, filter_by="n_rna_alt > 10"))
    coverage = report(evaluated).df
    comparable = compare(evaluated, evaluated).df
    for row in comparable.to_dict("records"):
        group = coverage[coverage.prediction_id.eq(row["prediction_id"])]
        expected = group.loc[group.level.eq("score"), "status"].eq("assessed").all()
        expected &= ~group.loc[group.level.eq("criterion"), "status"].eq("missing").any()
        assert row["left_assessable"] == row["right_assessable"] == bool(expected)


def test_allele_credited_peptide_rows_do_not_broadcast():
    frame = source(kind="proteasome_cleavage", prediction_id=["one", "two"], score=[.2, .8]).df
    evaluated = evaluate_selection_policy(TopiaryResult(frame),
        SelectionPolicy("processing", "peptide_view(proteasome_cleavage.score)"),
        group_keys=KEYS, alleles=["HLA-A*02:01", "HLA-B*07:02"])
    requested = requests(evaluated)
    requested["kind"], requested["field"] = "proteasome_cleavage", "score"
    coverage = policy_coverage(evaluated, prediction_requests=requested)
    predictions = coverage.df.query("level == 'prediction'")
    assert predictions.query("allele == 'HLA-A*02:01'").status.eq("assessed").all()
    assert predictions.query("allele == 'HLA-B*07:02'").status.eq("missing").all()


def test_repeated_discoveries_keep_occurrences_but_share_candidate_count():
    from topiary import combine_sources
    combined = combine_sources({"one": source(), "two": source(allele="A0201")}, sample_name="patient")
    evaluated = evaluate_selection_policy(combined, SelectionPolicy("binding", "affinity.value"),
        group_keys=["candidate_sample", "source_label", "peptide", "allele"], source_contexts={"one": {}, "two": {}})
    coverage = policy_coverage(evaluated)
    summary = summarize_policy_coverage(coverage).iloc[0]
    assert summary.n_occurrences == summary.n_assessments == 4
    assert summary.n_candidates == 2
    assert coverage.extra["policy_coverage"]["evaluation"]["evidence_extra"]["combined_sources"]


def test_report_retains_filter_and_acceptance_decisions():
    evaluated = evaluation()
    coverage = policy_coverage(evaluated).df
    assert coverage.query("prediction_id == 'missing'").decision_reason.eq("missing_score").all()
    assert coverage.filter_value.eq(True).all()
    comparison = compare_policy_evaluations(evaluated, evaluated).df
    assert comparison.left_filter_value.tolist() == evaluated.occurrences.filter_value.tolist()
    frame = evaluated.evidence.df.assign(left_score=["one", "two"])
    renamed = evaluate_selection_policy(TopiaryResult(frame), evaluated.policy,
                                        group_keys=["left_score", "peptide", "allele"])
    with pytest.raises(ValueError, match="reserved comparison"):
        compare_policy_evaluations(renamed, renamed)


def test_haplotype_request_scope_is_retained_and_canonical_duplicates_rejected():
    frame = source(prediction_id=["zero", "missing"], allele_set="HLA-A*02:01,HLA-B*07:02").df
    evaluated = evaluate_selection_policy(TopiaryResult(frame), SelectionPolicy("haplotype", "peptide_view(affinity.value)"),
                                           group_keys=KEYS)
    requested = requests(evaluated).assign(allele_set="B0702,A0201")
    coverage = policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'")
    assert coverage.requested_allele_set.eq("HLA-A*02:01,HLA-B*07:02").all()
    with pytest.raises(ValueError, match="Duplicate prediction"):
        policy_coverage(evaluated, prediction_requests=pd.concat([
            requested, requested.assign(allele_set="HLA-A*02:01,HLA-B*07:02")]))


def test_missing_model_request_uses_its_own_scope_declaration():
    frame = source(prediction_id=["zero", "missing"])
    support = {"requested": {"pMHC_presentation": {"mhc_dependence": "none"}},
               "unrelated": {"pMHC_presentation": {"mhc_dependence": "single_allele"}}}
    evaluated = evaluate_selection_policy(frame, SelectionPolicy("constant", "1"), group_keys=KEYS,
                                           kind_support=support)
    requested = requests(evaluated)
    requested["kind"], requested["prediction_method_name"] = "pMHC_presentation", "requested"
    coverage = policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'")
    assert coverage.status.eq("missing").all()
    assert coverage.mhc_dependence.eq("none").all()
    assert coverage.evidence_rows.tolist() == [[], []]


@pytest.mark.parametrize("version", [None, "", "nan", "null", "<NA>"])
def test_unnamed_versions_use_the_existing_public_identity_rule(version):
    from topiary import is_named_version
    assert not is_named_version(version)
    evaluated = evaluation(values=(0., 3.), predictor_version=version)
    requested = requests(evaluated)
    requested["predictor_version"] = None
    coverage = policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'")
    assert coverage.status.eq("assessed").all()
    assert coverage.predictor_version.isna().all()
    requested["predictor_version"] = "1"
    assert policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'").status.eq("missing").all()


def test_absent_version_column_is_unnamed_and_duplicate_spellings_are_rejected():
    original = evaluation(values=(0., 3.))
    evaluated = evaluate_selection_policy(TopiaryResult(original.evidence.df.drop(columns="predictor_version")),
                                           original.policy, group_keys=KEYS)
    requested = requests(evaluated)
    requested["predictor_version"] = None
    assert policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'").status.eq("assessed").all()
    with pytest.raises(ValueError, match="Duplicate prediction"):
        policy_coverage(evaluated, prediction_requests=pd.concat([requested, requested.assign(predictor_version="")]))


@pytest.mark.parametrize("version", ["NA", "unknown"])
def test_literal_named_versions_are_not_collapsed_to_missing(version):
    evaluated = evaluation(values=(0., 3.), predictor_version=version)
    requested = requests(evaluated)
    requested["predictor_version"] = version
    assert policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'").status.eq("assessed").all()
    requested["predictor_version"] = None
    assert policy_coverage(evaluated, prediction_requests=requested).df.query("level == 'prediction'").status.eq("missing").all()
