"""Table-only combination and explicit enrichment use the same ranking DSL."""

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from topiary import (
    Affinity, Column, TopiaryResult, apply_filter, combine_sources,
    evaluate_scores, parse, protein_evidence_view, rank_candidates, read_tsv, rescore_candidates,
)


def source(peptides=("SIINFEKL", "GILGFVFTL"), values=(50., 500.), **columns):
    frame = pd.DataFrame(dict(
        peptide=peptides, value=values, affinity=values, score=[0.5] * len(peptides),
        percentile_rank=[1.] * len(peptides), allele="HLA-A*02:01",
        kind="pMHC_affinity", prediction_method_name="original",
        predictor_version="1", source_sequence_name="shared", peptide_offset=0,
        n_flank="AAA", c_flank="GGG", gene="TEST", original_rank=[2, 1][:len(peptides)],
        n_rna_alt=[5, 15][:len(peptides)],
    ))
    for key, value in columns.items():
        frame[key] = value
    return TopiaryResult(frame, sources=["input.tsv"], extra={"keep": {"source": "original"}})


class Model:
    """Deterministic numerical backend; real normalized rows and DSL around it."""

    uses_flanking_sequences = True
    alleles = ["HLA-A*02:01"]

    def __init__(self, *, haplotype=False, other=False):
        self.calls = []
        self.haplotype = haplotype
        self.other = other

    def kind_support(self):
        return {"pMHC_affinity": {"mhc_dependence": "haplotype" if self.haplotype else "single_allele"}}

    def predict_dataframe(self, peptides, **kwargs):
        self.calls.append((list(peptides), deepcopy(kwargs)))
        flank = (kwargs.get("n_flanks") or [""])[0]
        rows = []
        for peptide in peptides:
            value = (800. if peptide == "SIINFEKL" else 10.) + len(flank)
            rows.append(dict(peptide="ELAGIGILT" if self.other else peptide,
                             allele=self.alleles[0], kind="pMHC_affinity", value=value,
                             score=1 / value, percentile_rank=2., value_unit="nM",
                             prediction_method_name="testmodel", predictor_version="2"))
        return pd.DataFrame(rows)


def combined(**kwargs):
    return combine_sources({"one": source(), "two": source(values=(60., 600.), **kwargs)}, sample_name="patient")


def test_preserves_all_original_cells_and_each_sources_metadata():
    original = source()
    copy = original.df.copy(deep=True)
    output = combine_sources({"one": original, "two": original}, sample_name="patient")
    assert output.df.candidate_id.nunique() == 2
    assert output.df.source_observation_id.nunique() == 4
    for label, rows in output.df.groupby("source_label"):
        pd.testing.assert_frame_equal(rows[copy.columns].reset_index(drop=True), copy)
        assert output.extra["combined_sources"][label]["extra"] == original.extra
    pd.testing.assert_frame_equal(original.df, copy)
    assert evaluate_scores(output.df, parse("n_rna_alt")).tolist() == [5, 15, 5, 15]


def test_samples_and_canonical_alleles_define_candidate_identity_without_rewriting_inputs():
    output = combine_sources({
        "one": source(sample_name="T1"),
        "two": source(sample_name="T2"),
        "three": source(allele="A0201"),
    }, sample_name="T1")
    assert output.df.candidate_id.nunique() == 4
    assert output.df.loc[output.df.source_label.eq("three"), "allele"].eq("A0201").all()
    assert set(output.df.candidate_allele) == {"HLA-A*02:01"}
    with pytest.raises(ValueError, match="sample_name"):
        combine_sources({"unknown": source()})


def test_dsl_default_grouping_preserves_different_sources_and_transcripts():
    output = combined(transcript_id="other", n_rna_alt=[0, 1])
    kept = output.filter_by("n_rna_alt >= 5")
    assert kept.df.source_label.tolist() == ["one", "one"]
    for node in (parse("affinity.value"), Affinity.value):
        assert evaluate_scores(output.df, node).tolist() == [50, 500, 60, 600]


def test_rank_duplicates_are_explicit_and_do_not_sum_counts():
    output = combined(n_rna_alt=[50, 60])
    with pytest.raises(ValueError, match="conflicting"):
        rank_candidates(output, "affinity.value", ascending=True)
    best = rank_candidates(output, "affinity.value", ascending=True, duplicates="best")
    worst = rank_candidates(output, "affinity.value", ascending=True, duplicates="worst")
    assert best.candidate_score.tolist() == [50, 500]
    assert best.n_rna_alt.tolist() == [5, 15]
    assert worst.candidate_score.tolist() == [60, 600]
    assert all(len(json.loads(ids)) == 2 for ids in best.candidate_observations)
    assert best.candidate_rank.tolist() == [1, 2]
    strata = rank_candidates(output, "affinity.value", ascending=True, strata=["source_label", "candidate_mhc_class"])
    assert len(strata) == 4
    assert strata.groupby("source_label").candidate_rank.apply(list).tolist() == [[1, 2], [1, 2]]


def test_sparse_scores_are_visible_and_mhc_classes_rank_separately():
    output = combine_sources({"one": source(values=(50., np.nan)),
                              "two": source(allele="HLA-DRB1*01:01")}, sample_name="p")
    ranked = rank_candidates(output, "affinity.value", ascending=True)
    assert ranked.candidate_rank.dropna().tolist() == [1, 1, 2]
    assert ranked.iloc[-1].ranking_status == "missing_score"
    assert pd.isna(ranked.iloc[-1].candidate_rank)


def test_rescoring_only_selected_subset_preserves_original_ranking_and_adds_dsl_feature(tmp_path):
    output = combined()
    before = output.df.copy(deep=True)
    model = Model()
    enriched = rescore_candidates(output, model, prefix="fresh", select="source_label == 'one'")
    feature = "fresh__testmodel__pMHC_affinity__value"
    assert [call[0] for call in model.calls] == [["SIINFEKL"], ["GILGFVFTL"]]
    assert enriched.df.loc[enriched.df.source_label.eq("two"), feature].isna().all()
    pd.testing.assert_frame_equal(enriched.df[before.columns], before)
    pd.testing.assert_frame_equal(output.df, before)
    for expr in ("affinity.value", Affinity.value):
        pd.testing.assert_frame_equal(
            rank_candidates(output, expr, ascending=True, duplicates="best"),
            rank_candidates(enriched, expr, ascending=True, duplicates="best")[
                rank_candidates(output, expr, ascending=True, duplicates="best").columns],
        )
    ranks = rank_candidates(enriched, feature, ascending=True, duplicates="best")
    assert ranks.peptide.tolist() == ["GILGFVFTL", "SIINFEKL"]
    assert enriched.filter_by(f"{feature} < 20").df.peptide.tolist() == ["GILGFVFTL"]
    for form in (enriched, enriched.to_wide()):
        path = tmp_path / f"{form.form}.tsv"
        form.to_tsv(path)
        restored = read_tsv(path).to_long()
        assert set(restored.df.prediction_method_name) == {"original"}
        assert restored.extra["candidate_rescoring"] == enriched.extra["candidate_rescoring"]
        assert set(restored.df[feature].dropna()) == {803, 13}


def test_contexts_are_not_reused_and_flanks_reach_the_predictor():
    output = combined(n_flank="OTHER")
    model = Model()
    scored = rescore_candidates(output, model, prefix="new")
    assert len(model.calls) == 4
    assert {c[1]["n_flanks"][0] for c in model.calls} == {"AAA", "OTHER"}
    assert scored.df.new__testmodel__pMHC_affinity__value.tolist() == [803, 13, 805, 15]


def test_deduplicated_calls_keep_distinct_observations_and_evidence():
    output = combined()
    model = Model()
    scored = rescore_candidates(output, model, prefix="new")
    assert len(model.calls) == 2
    assert len(scored.df) == len(output.df)
    assert scored.df.source_observation_id.nunique() == 4


def test_missing_flanks_and_genotype_require_explicit_resolution():
    output = combine_sources({"one": source(n_flank=None)}, sample_name="p")
    with pytest.raises(ValueError, match="both flanks"):
        rescore_candidates(output, Model(), prefix="new")
    assert len(rescore_candidates(output, Model(), prefix="new", use_flanks=False)) == 2
    output = combine_sources({"one": source()}, sample_name="p")
    with pytest.raises(ValueError, match="allele_set"):
        rescore_candidates(output, Model(haplotype=True), prefix="new")
    output = combine_sources({"one": source(allele_set="HLA-A*02:01")}, sample_name="p")
    assert len(rescore_candidates(output, Model(haplotype=True), prefix="new")) == 2


def test_invalid_output_and_existing_features_raise_without_mutating_input():
    output = combined()
    with pytest.raises(ValueError, match="different peptide"):
        rescore_candidates(output, Model(other=True), prefix="new")
    first = rescore_candidates(output, Model(), prefix="new")
    with pytest.raises(ValueError, match="already exists"):
        rescore_candidates(first, Model(), prefix="new")
    assert "candidate_rescoring" not in output.extra


def test_empty_selection_never_calls_a_model_and_empty_combination_ranks():
    output = combined()
    model = Model()
    rescore_candidates(output, model, prefix="new", select="n_rna_alt > 100")
    assert model.calls == []
    empty = combine_sources({})
    assert rank_candidates(empty, "affinity.value").empty
    assert rescore_candidates(empty, model, prefix="new").empty


@pytest.mark.parametrize("expr", ["n_rna_alt > 10", "affinity.value < 100"])
def test_filter_string_and_ast_doors_agree(expr):
    output = combined()
    pd.testing.assert_frame_equal(output.filter_by(expr).df, apply_filter(output.df, parse(expr)))
    left = rank_candidates(output, "affinity.value", filter_by=expr, duplicates="best")
    right = rank_candidates(output, Affinity.value, filter_by=parse(expr), duplicates="best")
    pd.testing.assert_frame_equal(left, right)
    assert left.peptide.nunique() == 1


def test_new_features_are_column_dsl_nodes():
    enriched = rescore_candidates(combined(), Model(), prefix="new")
    name = "new__testmodel__pMHC_affinity__value"
    pd.testing.assert_series_equal(evaluate_scores(enriched.df, parse(name)),
                                   evaluate_scores(enriched.df, Column(name)))


def orfs():
    return pd.DataFrame(dict(
        protein_sequence=["MAAASIINFEKL", "MQQQSIINFEKL"],
        event_id=["GRCh38:example-event"] * 2,
        transcript_expression=[11., 22.],
        rna_evidence_method=["source_reported"] * 2,
    ))


def test_same_orf_and_alternative_orfs_keep_independent_rna_and_prediction_evidence():
    frame = orfs()
    lens = source(protein_sequence=frame.protein_sequence.iloc[0],
                  event_id=frame.event_id.iloc[0], transcript_expression=30.)
    combined = combine_sources({"exacto_normalized": frame, "lens_normalized": lens}, sample_name="p")
    assert len(combined) == 4
    assert combined.df.protein_sequence_id.nunique() == 2
    assert combined.df.candidate_id.notna().sum() == 2
    assert combined.df.loc[combined.df.source_label.eq("exacto_normalized"), "prediction_method_name"].isna().all()
    view = protein_evidence_view(combined)
    assert len(view) == 2
    same = view[view.protein_sequence.eq("MAAASIINFEKL")].iloc[0]
    assert json.loads(same.source_labels) == ["exacto_normalized", "lens_normalized"]
    assert len(json.loads(same.candidate_ids)) == 2
    assert combined.df.transcript_expression.tolist() == [11, 22, 30, 30]
    ranked = rank_candidates(combined, "affinity.value", ascending=True)
    assert len(ranked) == 2
    assert ranked.candidate_score.tolist() == [50, 500]


def test_full_orf_is_not_inferred_from_peptide_or_local_context_and_unknown_events_are_separate():
    proteins = orfs().drop(columns="event_id").iloc[:1]
    combined = combine_sources({"one": proteins, "two": proteins,
                                "peptides": source(pep_context="MAAASIINFEKL")}, sample_name="p")
    assert len(protein_evidence_view(combined)) == 2
    assert combined.df.protein_sequence_id.nunique() == 1
    assert combined.df.loc[combined.df.source_label.eq("peptides"), "protein_sequence_id"].isna().all()


@pytest.mark.parametrize("wide", [False, True])
def test_mixed_orf_and_sparse_predictions_roundtrip_without_invented_model_rows(tmp_path, wide):
    combined = combine_sources({"orf": orfs(), "one": source(),
                                "two": source(prediction_method_name="other")}, sample_name="p")
    saved = combined.to_wide() if wide else combined
    path = tmp_path / "combined.tsv"
    saved.to_tsv(path)
    restored = read_tsv(path).to_long()
    assert len(restored) == len(combined)
    restored_rows = restored.df.sort_values(["source_label", "source_row"])
    original_rows = combined.df.sort_values(["source_label", "source_row"])
    for col in ("source_label", "source_observation_id", "protein_sequence_id", "candidate_id",
                "prediction_method_name", "kind", "value", "transcript_expression"):
        left, right = restored_rows[col], original_rows[col]
        pd.testing.assert_series_equal(left.where(left.notna(), np.nan).reset_index(drop=True),
                                       right.where(right.notna(), np.nan).reset_index(drop=True), check_dtype=False)
    assert len(protein_evidence_view(restored)) == 2


def test_conflicting_abundance_inside_one_source_is_not_silently_collapsed():
    a = source()
    b = source(n_rna_alt=[100, 200])
    frame = pd.concat([a.df, b.df], ignore_index=True)
    combined = combine_sources({"one": frame}, sample_name="p")
    assert combined.df.source_observation_id.nunique() == 4
    assert evaluate_scores(combined.df, parse("n_rna_alt")).tolist() == [5, 15, 100, 200]
    assert combined.filter_by("n_rna_alt > 50").df.n_rna_alt.tolist() == [100, 200]


def test_protein_only_input_remains_useful_without_any_predictor(tmp_path):
    combined = combine_sources({"exacto_normalized": orfs()}, sample_name="p")
    assert rank_candidates(combined, "affinity.value").empty
    assert len(combined.filter_by("transcript_expression > 15")) == 1
    model = Model()
    rescored = rescore_candidates(combined, model, prefix="new")
    assert model.calls == []
    assert len(rescored) == 2
    for result in (combined, combined.to_wide()):
        path = tmp_path / f"orf-{result.form}.tsv"
        result.to_tsv(path)
        restored = read_tsv(path).to_long()
        assert restored.df.kind.isna().all()
        assert len(restored) == 2


@pytest.mark.parametrize("with_other_source", [False, True])
def test_wide_roundtrip_preserves_prediction_identity_when_all_scores_are_missing(tmp_path, with_other_source):
    missing = source(values=(np.nan, np.nan), score=np.nan, percentile_rank=np.nan,
                     prediction_method_name="missing", predictor_version=None,
                     prediction_run_name="original-run")
    sources = {"missing": missing}
    if with_other_source:
        sources["available"] = source()
        sources["orf"] = orfs()
    combined = combine_sources(sources, sample_name="p")
    path = tmp_path / "missing-wide.tsv"
    combined.to_wide().to_tsv(path)
    restored = read_tsv(path).to_long()
    rows = restored.df[restored.df.source_label.eq("missing")]
    assert len(restored) == len(combined)
    assert len(rows) == 2
    assert rows.kind.eq("pMHC_affinity").all()
    assert rows.prediction_method_name.eq("missing").all()
    assert rows.predictor_version.isna().all()
    assert rows.prediction_run_name.eq("original-run").all()
    assert rows.value.isna().all()


def test_selective_rescoring_does_not_leak_across_genotypes():
    first = source(allele_set="HLA-A*02:01")
    second = source(allele_set="HLA-A*02:01,HLA-B*07:02")
    combined = combine_sources({"one": pd.concat([first.df, second.df])}, sample_name="p")
    model = Model()
    scored = rescore_candidates(combined, model, prefix="new", select="allele_set == 'HLA-A*02:01'")
    feature = "new__testmodel__pMHC_affinity__value"
    assert scored.df[feature].iloc[:2].notna().all()
    assert scored.df[feature].iloc[2:].isna().all()
    assert len(model.calls) == 2


def test_conflicting_predictions_need_source_or_run_identity_before_dsl_evaluation():
    rows = pd.concat([source().df, source(values=(60., 600.)).df], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting predictions"):
        combine_sources({"one": rows}, sample_name="p")
    rows["prediction_run_name"] = ["first", "first", "second", "second"]
    combined = combine_sources({"one": rows}, sample_name="p")
    assert combined.df.source_observation_id.nunique() == 4
    with pytest.raises(ValueError, match="conflicting scores"):
        rank_candidates(combined, "affinity.value")
    assert rank_candidates(combined, "affinity.value", duplicates="best").candidate_score.tolist() == [600., 60.]


def test_source_mhc_scope_survives_combination_and_roundtrip(tmp_path):
    per_allele = source(kind="pMHC_presentation")
    per_allele.extra["kind_support"] = {"original": {"pMHC_presentation": {"mhc_dependence": "single_allele"}}}
    haplotype = source(kind="pMHC_presentation", allele_set="HLA-A*02:01,HLA-B*07:02")
    haplotype.extra["kind_support"] = {"original": {"pMHC_presentation": {"mhc_dependence": "haplotype"}}}
    combined = combine_sources({"per_allele": per_allele, "joint": haplotype}, sample_name="p")
    path = tmp_path / "scope.tsv"
    combined.to_wide().to_tsv(path)
    for result in (combined, read_tsv(path).to_long()):
        assert set(result.df.source_prediction_mhc_dependence) == {"single_allele", "haplotype"}
        with pytest.raises(ValueError, match="different MHC dependence"):
            rank_candidates(result, "presentation.score")
        original_only = result.filter_by("source_label == 'per_allele'")
        assert rank_candidates(original_only, "presentation.score").candidate_score.tolist() == [.5, .5]


def test_equal_protein_products_preserve_distinct_orf_hypotheses():
    rows = pd.DataFrame(dict(protein_sequence=["MA", "MA"],
                             event_id=["event", "event"],
                             orf_id=["orf-one", "orf-two"],
                             coding_sequence=["ATGGCT", "ATGGCC"],
                             transcript_expression=[11., 22.]))
    combined = combine_sources({"one": rows}, sample_name="p")
    assert combined.df.source_observation_id.nunique() == 2
    assert combined.df.protein_sequence_id.nunique() == 1
    proteins = protein_evidence_view(combined)
    assert len(proteins) == 1
    assert len(json.loads(proteins.iloc[0].source_observations)) == 2
    assert combined.df.coding_sequence.tolist() == ["ATGGCT", "ATGGCC"]


def test_empty_source_table_retains_its_metadata():
    combined = combine_sources({"empty": source().df.iloc[:0]}, sample_name="p")
    assert combined.empty
    assert "empty" in combined.extra["combined_sources"]
