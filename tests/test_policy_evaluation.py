"""Policy evaluation retains occurrences and reuses the public DSL semantics."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from topiary import (
    EvalContext, SelectionPolicy, TopiaryResult, apply_filter, combine_sources,
    evaluate_filter, evaluate_selection_policy, parse, replay_selection_policy,
    select_policy_representatives,
)
from .test_candidate_tables import source
from .test_twin_conformance import DELIMITED_IO_TWINS, OCCURRENCE_POLICY_TWINS, FILTER_DECISION_TWINS


@pytest.mark.parametrize("filter_by", [None, "n_rna_alt >= 10", "affinity.value < 100", "n_rna_alt > 100"])
@pytest.mark.parametrize("score_fill,min_score", [(None, None), (0., 1e-5), (0., 0.)])
def test_occurrence_and_existing_dsl_paths_agree(filter_by, score_fill, min_score):
    frame = source(values=(50., np.nan)).df
    policy = SelectionPolicy("baseline", "affinity.value", filter_by=filter_by,
                             score_fill=score_fill, min_score=min_score)
    keys = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
    evaluate, make_context = OCCURRENCE_POLICY_TWINS
    decide_filter, filter_rows = FILTER_DECISION_TWINS
    evaluation = evaluate(TopiaryResult(frame), policy, group_keys=keys)
    kept = filter_rows(frame, filter_by, group_keys=keys)
    direct_ctx = make_context(kept, group_keys=keys)
    direct = parse(policy.score_by).eval(direct_ctx).reindex(direct_ctx.group_index)
    if score_fill is not None:
        direct = direct.fillna(score_fill)
    decisions = evaluation.occurrences.set_index(keys)
    pd.testing.assert_series_equal(decisions["score"].reindex(direct.index), direct, check_names=False)
    expected = direct.index if min_score is None else direct[direct.ge(min_score)].index
    assert set(evaluation.selected.set_index(keys).index) == set(expected)
    pd.testing.assert_frame_equal(evaluation.evidence.df, frame)
    filters = decide_filter(frame, filter_by, group_keys=keys)
    ctx = EvalContext(frame, group_keys=keys)
    pd.testing.assert_frame_equal(frame[filters.retained.to_numpy()[ctx.row_group_codes()]].reset_index(drop=True), kept)


def test_different_flanks_offsets_and_genes_keep_alternatives_and_replay(tmp_path):
    first = source(peptides=("SIINFEKL", "SIINFEKL"), values=(50., 50.),
                   peptide_offset=[0, 20], n_flank=["A", "GG"], gene=["G1", "G2"],
                   protein_sequence=["SIINFEKL", "M" * 20 + "SIINFEKL"])
    combined = combine_sources({"first": first, "repeat": first}, sample_name="p")
    policy = SelectionPolicy("context", "n_rna_alt", duplicates="best")
    evaluation = evaluate_selection_policy(combined, policy)
    best = select_policy_representatives(evaluation)
    assert best.score.tolist() == [15.]
    assert len(best.alternative_occurrences.iloc[0]) == 4
    assert len(evaluation.occurrences) == 4
    row = best.evidence_rows.iloc[0][0]
    assert combined.df.iloc[row].peptide_offset == 20
    assert combined.df.iloc[row].n_rna_alt == 15  # never 30 from repeated discovery
    changed = evaluate_selection_policy(combined, replace(policy, filter_by="gene == 'G1'"))
    assert select_policy_representatives(changed).score.tolist() == [5.]
    for suffix, writer, _, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("retained." + suffix)
        writer(evaluation.evidence, path)
        restored = reader(path)
        assert len(restored.df) == len(combined.df)
        replay = replay_selection_policy(restored)
        pd.testing.assert_frame_equal(replay.occurrences, evaluation.occurrences, check_exact=True)
        pd.testing.assert_frame_equal(select_policy_representatives(replay), best, check_exact=True)


def test_sparse_genotypes_materialize_once_and_project_without_borrowing(tmp_path):
    frame = source(kind="proteasome_cleavage", allele="", values=(.2, .8), score=[.2, .8]).df
    frame["prediction_id"] = ["one", "two"]
    keys = ["prediction_id", "peptide", "peptide_offset", "allele"]
    calls = []
    def genotype(row):
        calls.append(row)
        return ["HLA-A*02:01"] if row["prediction_id"] == "one" else ["HLA-B*07:02"]
    policy = SelectionPolicy("processing", "peptide_view(proteasome_cleavage.score)")
    evaluation = evaluate_selection_policy(TopiaryResult(frame), policy, group_keys=keys, alleles=genotype)
    assert len(calls) == 2
    projected = evaluation.occurrences.query("allele != ''")
    assert projected[["prediction_id", "allele", "score"]].values.tolist() == [
        ["one", "HLA-A*02:01", .2], ["two", "HLA-B*07:02", .8]]
    assert projected.evidence_rows.tolist() == [[], []]
    assert projected.supporting_rows.tolist() == [[0], [1]]
    assert len(evaluation.evidence.df) == 2
    replay = replay_selection_policy(evaluation.evidence)
    assert len(calls) == 2
    pd.testing.assert_frame_equal(replay.occurrences, evaluation.occurrences)
    assert json.dumps(evaluation.evidence.extra, allow_nan=False)


def test_source_local_models_versions_used_for_filter_and_score(tmp_path):
    frames = pd.concat([source(values=(50., 500.)).df,
                        source(values=(500., 50.), predictor_version="2").df,
                        source(values=(20., 800.), prediction_method_name="other").df], ignore_index=True)
    evidence = combine_sources({"one": TopiaryResult(frames), "two": TopiaryResult(frames)}, sample_name="p")
    policy = SelectionPolicy("local", "affinity.value", filter_by="affinity.value < 100", ascending=True)
    contexts = {
        "one": dict(default_methods={"affinity": "original"}, default_versions={("affinity", "original"): "2"}),
        "two": dict(default_methods={"affinity": "other"}, default_versions={("affinity", "other"): "1"}),
    }
    evaluation = evaluate_selection_policy(evidence, policy, source_contexts=contexts)
    assert evaluation.selected[["source_label", "peptide", "score"]].values.tolist() == [
        ["one", "GILGFVFTL", 50.], ["two", "SIINFEKL", 20.]]
    for suffix, writer, _, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("models." + suffix)
        writer(evaluation.evidence, path)
        pd.testing.assert_frame_equal(replay_selection_policy(reader(path)).occurrences,
                                      evaluation.occurrences, check_exact=True)


def test_v1_policy_preserves_definition_and_digest():
    original = SelectionPolicy("v1", "affinity.value").to_dict()
    original["schema_version"] = 1
    for field in ("score_fill", "min_score", "criteria", "ranking_by", "unknown", "expanded"):
        del original[field]
    restored = SelectionPolicy.from_dict(original)
    assert restored.to_dict() == original
    assert restored.score_fill is None and restored.min_score is None


def test_empty_evidence_and_invalid_runtime_context():
    evidence = combine_sources({}, sample_name="p")
    evaluation = evaluate_selection_policy(evidence, SelectionPolicy("empty", "affinity.value"))
    assert evaluation.selected.empty
    assert replay_selection_policy(evaluation.evidence).occurrences.empty
    with pytest.raises(ValueError, match="every source"):
        evaluate_selection_policy(combine_sources({"one": source()}, sample_name="p"),
                                  evaluation.policy, source_contexts={})


def test_context_derivation_accepts_callbacks_and_rebuilds_changed_genotypes():
    frame = source(allele="", kind="proteasome_cleavage").df
    callback = lambda keys: ["HLA-A*02:01"]
    ctx = EvalContext(frame, alleles=callback)
    expected = ctx.group_index
    derived = ctx.derive(filter_context=True)
    assert derived.group_index.equals(expected)
    apply_filter(frame, "n_rna_alt >= 0", context=ctx)
    first = {"SIINFEKL": ["HLA-A*02:01"]}
    ctx = EvalContext(frame, alleles=first)
    assert "HLA-A*02:01" in ctx.group_index.get_level_values("allele")
    changed = ctx.derive(alleles={"SIINFEKL": ["HLA-B*07:02"]})
    assert "HLA-A*02:01" not in changed.group_index.get_level_values("allele")
    assert "HLA-B*07:02" in changed.group_index.get_level_values("allele")


def test_projected_combined_candidates_use_the_same_public_identity():
    from topiary import candidate_identifier
    combined = combine_sources({"processing": source(allele="", kind="proteasome_cleavage")}, sample_name="p")
    evaluation = evaluate_selection_policy(combined, SelectionPolicy("projection", "peptide_view(proteasome_cleavage.score)"),
                                           alleles=["HLA-A*02:01"])
    representatives = select_policy_representatives(evaluation)
    assert len(representatives) == 2
    assert set(representatives.candidate_id) == {
        candidate_identifier("p", peptide, "A0201") for peptide in combined.df.peptide}
    assert representatives.source_label.eq("processing").all()


@pytest.mark.parametrize("expression", ["(n_rna_alt + 3).sqrt()", "n_rna_alt / (2 * 3)",
                                         "n_rna_alt - (2 - 3)", "(1 + 2).logistic(2, 1)",
                                         "affinity.value < 100"])
def test_saved_expansion_preserves_original_arithmetic_and_legacy_boolean_scores(expression):
    from topiary import evaluate_scores
    frame = source().df
    policy = SelectionPolicy("grouping", expression)
    actual = evaluate_selection_policy(TopiaryResult(frame), policy).occurrences.score
    expected = evaluate_scores(frame, expression)
    pd.testing.assert_series_equal(actual, expected, check_names=False, check_exact=True)
