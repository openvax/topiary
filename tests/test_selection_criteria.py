"""Named criteria compose explicitly and preserve evidence states in audits."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from topiary import (
    EvalContext, RankingTerm, SelectionCriterion, SelectionPolicy, TopiaryResult,
    evaluate_selection_policy, evaluate_selection_criteria, parse,
    replay_selection_policy, resolve_selection_policy, select_policy_representatives,
)
from .test_candidate_tables import source
from .test_twin_conformance import DELIMITED_IO_TWINS


BINDING = SelectionCriterion("binding", "affinity.value < 500", "eligibility")
PROCESSING = SelectionCriterion("processing", "peptide_view(proteasome_cleavage.score)", "score")


def test_reusable_binding_and_opt_in_processing_change_only_referencing_policy(tmp_path):
    frame = pd.concat([
        source(values=(50., 100.)).df,
        source(kind="proteasome_cleavage", allele="", score=[.1, .9]).df,
    ], ignore_index=True)
    policy = SelectionPolicy("binding-v1", "1 / affinity.value", filter_by='criterion("binding")',
                             criteria=(BINDING, PROCESSING), strata=())
    baseline = evaluate_selection_policy(TopiaryResult(frame), policy)
    changed = evaluate_selection_policy(TopiaryResult(frame), replace(
        policy, name="processing-v2", score_by='criterion("processing")'))
    assert select_policy_representatives(baseline, candidate_keys=["peptide", "allele"]).peptide.iloc[0] == "SIINFEKL"
    assert select_policy_representatives(changed, candidate_keys=["peptide", "allele"]).peptide.iloc[0] == "GILGFVFTL"
    assert baseline.audit.query("criterion == 'processing'").status.eq("not_evaluated").all()
    assert changed.audit.query("criterion == 'processing'").status.eq("value").all()
    for suffix, writer, _, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("criteria." + suffix)
        writer(changed.evidence, path)
        restored = replay_selection_policy(reader(path))
        pd.testing.assert_frame_equal(restored.occurrences, changed.occurrences, check_exact=True)
        assert restored.policy.sha256 == changed.policy.sha256


@pytest.mark.parametrize("unknown,expected", [("exclude", [0]), ("include", [0, 2, 3])])
def test_pass_fail_unknown_zero_and_not_applicable_remain_distinct(unknown, expected):
    frame = pd.concat([source().df.iloc[[0]]] * 4, ignore_index=True)
    frame["prediction_id"] = range(4)
    frame["measurement"] = [0., -1., np.nan, 1.]
    frame["applicable"] = [1, 1, 1, 0]
    criterion = SelectionCriterion("measurement", "measurement >= 0", "eligibility", "applicable == 1")
    score = SelectionCriterion("observed", "measurement", "score")
    policy = SelectionPolicy("states", 'criterion("observed")', filter_by='criterion("measurement")',
                             criteria=(criterion, score), unknown=unknown)
    keys = ["prediction_id", "peptide", "allele"]
    evaluation = evaluate_selection_policy(TopiaryResult(frame), policy, group_keys=keys)
    audit = evaluation.audit
    assert audit.query("criterion == 'measurement'").status.tolist() == ["pass", "fail", "unknown", "not_applicable"]
    assert audit.query("criterion == 'measurement'").reason.tolist() == [
        "predicate_true", "predicate_false", "missing_evidence", "applicability_false"]
    observed = audit.query("criterion == 'observed'")
    assert observed.iloc[0].status == "value" and observed.iloc[0].reason == "observed_zero"
    assert observed.iloc[1].status == "not_evaluated" and observed.iloc[1].reason == "filtered_before_scoring"
    assert evaluation.selected.prediction_id.tolist() == expected
    assert len(evaluation.evidence.df) == 4
    with pytest.raises(ValueError, match="Unknown eligibility"):
        evaluate_selection_policy(TopiaryResult(frame), replace(policy, unknown="error"), group_keys=keys)


@pytest.mark.parametrize("expression,reason", [
    ("not_supplied >= 0", "missing_column"),
    ("affinity.value >= 0", "missing_evidence"),
    ("(-1).sqrt() >= 0", "out_of_domain"),
    ("(1 / 0) > 0", "out_of_domain"),
])
def test_unknown_diagnostics(expression, reason):
    frame = source(values=(np.nan, np.nan)).df
    policy = SelectionPolicy("unknown", "1", filter_by='criterion("check")',
                             criteria=(SelectionCriterion("check", expression, "eligibility"),))
    evaluation = evaluate_selection_policy(TopiaryResult(frame), policy)
    assert evaluation.audit.status.eq("unknown").all()
    assert evaluation.audit.reason.eq(reason).all()
    assert evaluation.selected.empty


def test_ambiguity_and_conflicting_measurements_are_not_observed_false():
    for frame, reason in (
        (pd.concat([source().df, source(prediction_method_name="other").df]), "ambiguous_model"),
        (pd.concat([source().df, source(values=(40., 400.)).df]), "conflicting_evidence"),
    ):
        # Numeric criteria do not inherit directional filter auto-aggregation.
        policy = SelectionPolicy("audit", 'criterion("binding_value")',
                                 criteria=(SelectionCriterion("binding_value", "affinity.value", "score"),))
        evaluation = evaluate_selection_policy(TopiaryResult(frame), policy)
        assert evaluation.audit.status.eq("unknown").all()
        assert evaluation.audit.reason.eq(reason).all()


def test_transitive_and_or_criteria_and_ordered_ties():
    criteria = (BINDING,
                SelectionCriterion("rna", "n_rna_alt >= 10", "eligibility"),
                SelectionCriterion("both", 'criterion("binding") & criterion("rna")', "eligibility"),
                SelectionCriterion("reads", "n_rna_alt", "ranking"))
    policy = SelectionPolicy("compose", "1", filter_by='criterion("binding") | criterion("rna")',
                             criteria=criteria, ranking_by=(RankingTerm('criterion("reads")'),),
                             strata=(), duplicates="best")
    frame = source(peptides=("SIINFEKL", "SIINFEKL"), values=(50., 100.), peptide_offset=[0, 20]).df
    evaluation = evaluate_selection_policy(TopiaryResult(frame), policy)
    best = select_policy_representatives(evaluation, candidate_keys=["peptide", "allele"])
    assert best.peptide_offset.tolist() == [20]
    assert len(best.alternative_occurrences.iloc[0]) == 2
    assert evaluation.audit.query("criterion == 'both'").status.eq("not_evaluated").all()
    both = evaluate_selection_policy(TopiaryResult(frame), replace(policy, filter_by='criterion("both")'))
    assert both.selected.peptide_offset.tolist() == [20]
    resolved = resolve_selection_policy({k: v for k, v in policy.to_dict().items() if k != "expanded"})
    assert resolved.sha256 == policy.sha256
    saved = policy.to_dict()
    saved["expanded"]["score_by"]["expression"] = "999"
    with pytest.raises(ValueError, match="expanded"):
        SelectionPolicy.from_dict(saved)


@pytest.mark.parametrize("criteria,expression,role", [
    ((BINDING, BINDING), 'criterion("binding")', "eligibility"),
    ((BINDING,), 'criterion("missing")', "eligibility"),
    ((BINDING,), 'criterion("binding")', "score"),
    ((SelectionCriterion("a", 'criterion("b")', "score"),
      SelectionCriterion("b", 'criterion("a")', "score")), 'criterion("a")', "score"),
    ((SelectionCriterion("a", "affinity.value < 500", "score"),), 'criterion("a")', "score"),
])
def test_duplicate_unresolved_cyclic_and_wrong_role_references_rejected(criteria, expression, role):
    from topiary import resolve_selection_expression
    with pytest.raises(ValueError):
        resolve_selection_expression(expression, criteria, role=role)


def test_named_references_do_not_shadow_input_columns_and_direct_dsl_stays_unchanged():
    frame = source(values=(np.nan, 50.), binding=[5., 10.]).df
    policy = SelectionPolicy("columns", "binding", criteria=(BINDING,))
    result = evaluate_selection_policy(TopiaryResult(frame), policy)
    assert result.occurrences.score.tolist() == [5., 10.]
    ctx = EvalContext(frame)
    predicate = parse("~(affinity.value < 500)")
    assert predicate.eval(ctx).tolist() == [True, False]
    assert pd.isna(predicate.eval(ctx.derive(preserve_unknown=True)).iloc[0])
    audit = evaluate_selection_criteria(policy, ctx, references=["binding"])
    assert audit.decisions.status.tolist() == ["unknown", "pass"]
