"""Explicit occurrence inference preserves context without expanding peptides."""
import random

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    CachedPredictor, PartialPredictionWarning, SelectionPolicy, TopiaryPredictor,
    TopiaryResult, combine_sources, evaluate_selection_policy, predict_peptide_occurrences,
    rescore_candidates, read_tsv,
)
from .test_candidate_tables import Model, MultiKindModel, source
from .test_twin_conformance import PEPTIDE_OCCURRENCE_TWINS, NAMED_OCCURRENCE_TWINS


def occurrences():
    return pd.DataFrame([
        dict(prediction_id="one", peptide="SIINFEKL", n_flank="AAA", c_flank="GG", peptide_offset=3, gene="G1", n_rna_alt=7),
        dict(prediction_id="other-gene", peptide="SIINFEKL", n_flank="AAA", c_flank="GG", peptide_offset=30, gene="G2", n_rna_alt=7),
        dict(prediction_id="other-context", peptide="SIINFEKL", n_flank="T", c_flank="GG", peptide_offset=1, gene="G3", n_rna_alt=2),
        dict(prediction_id="other-peptide", peptide="GILGFVFTL", n_flank="AAA", c_flank="GG", peptide_offset=3, gene="G4", n_rna_alt=3),
    ])


@pytest.mark.parametrize("predict", PEPTIDE_OCCURRENCE_TWINS)
def test_contexts_batch_once_and_keep_every_occurrence(predict):
    inputs = occurrences()
    before = inputs.copy(deep=True)
    model = Model()
    result = predict(inputs, model)
    pd.testing.assert_frame_equal(inputs, before)
    pd.testing.assert_frame_equal(result[inputs.columns], inputs)
    assert result.value.tolist() == [803., 803., 801., 13.]
    assert sum(len(peptides) for peptides, _ in model.calls) == 3
    assert len(model.calls) == 2
    assert result.prediction_flanks_supplied.all()
    assert result.prediction_mhc_dependence.eq("single_allele").all()
    # Batched and independent calls must have the same scores and evidence.
    independent = pd.concat([predict(inputs.iloc[[index]], Model()) for index in range(len(inputs))], ignore_index=True)
    pd.testing.assert_frame_equal(result, independent)
    evaluated = evaluate_selection_policy(TopiaryResult(result), SelectionPolicy("context", "affinity.value"),
                                           group_keys=["prediction_id", "peptide", "peptide_offset", "allele"])
    assert evaluated.occurrences.prediction_id.tolist() == inputs.prediction_id.tolist()
    assert evaluated.occurrences.score.tolist() == result.value.tolist()


@pytest.mark.parametrize("peptides", [[], ["SIINFEKL"], ["SIINFEKL", "SIINFEKL", "GILGFVFTL"]])
def test_real_mhctools_named_and_occurrence_doors_agree(peptides):
    names = {str(index): peptide for index, peptide in enumerate(peptides)}
    frames = []
    for predict in NAMED_OCCURRENCE_TWINS:
        random.seed(123)
        frames.append(predict(names, RandomBindingPredictor(alleles=["HLA-A*02:01", "HLA-B*07:02"])))
    columns = ["source_sequence_name", "peptide", "peptide_offset", "allele", "kind", "value", "score"]
    left, right = (frame[columns].sort_values(columns[:4]).reset_index(drop=True) for frame in frames)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


@pytest.mark.parametrize("change,match", [
    ({"prediction_id": "same"}, "unique"),
    ({"prediction_id": None}, "prediction_id"),
    ({"peptide": ""}, "peptide"),
    ({"peptide_offset": -1}, "peptide_offset"),
    ({"peptide_offset": 1.5}, "peptide_offset"),
    ({"peptide_offset": True}, "peptide_offset"),
    ({"peptide_length": 99}, "peptide_length"),
    ({"value": 123.}, "prediction columns"),
    ({"candidate_id": "old-candidate"}, "prediction columns"),
    ({"n_flank": None}, "both flanks"),
])
@pytest.mark.parametrize("predict", PEPTIDE_OCCURRENCE_TWINS)
def test_invalid_inputs_raise_before_inference(change, match, predict):
    inputs = occurrences().assign(**change)
    model = Model()
    with pytest.raises(ValueError, match=match):
        predict(inputs, model)
    assert not model.calls


def test_unknown_and_empty_flanks_are_distinct_and_positions_not_invented(tmp_path):
    inputs = pd.DataFrame([dict(prediction_id="known", peptide="SIINFEKL", n_flank="", c_flank=""),
                           dict(prediction_id="unknown", peptide="SIINFEKL")])
    output = predict_peptide_occurrences(inputs, Model(), use_flanks=False)
    assert output.peptide_offset.isna().all()
    assert output.n_flank.iloc[0] == "" and pd.isna(output.n_flank.iloc[1])
    assert not output.prediction_flanks_supplied.any()
    path = tmp_path / "context.tsv"
    TopiaryResult(output).to_tsv(path)
    reloaded = read_tsv(path).df
    assert reloaded.n_flank.iloc[0] == "" and pd.isna(reloaded.n_flank.iloc[1])
    assert reloaded.peptide_offset.isna().all()
    with pytest.raises(ValueError, match="both flanks"):
        predict_peptide_occurrences(inputs, Model())


def test_model_scope_and_coverage_are_explicit():
    inputs = occurrences()
    kinds = ("pMHC_affinity", "antigen_processing")
    output = predict_peptide_occurrences(inputs, MultiKindModel(declared=kinds, emitted=kinds))
    assert len(output) == 8
    assert output.query("kind == 'antigen_processing'").allele.eq("").all()
    assert output.query("kind == 'antigen_processing'").prediction_mhc_dependence.eq("none").all()
    haplotype = Model(haplotype=True)
    output = predict_peptide_occurrences(inputs, haplotype)
    assert output.allele_set.eq("HLA-A*02:01").all()
    with pytest.raises(ValueError, match="allele_set"):
        predict_peptide_occurrences(inputs.assign(allele_set="HLA-B*07:02"), haplotype)
    with pytest.raises(ValueError, match="pMHC_presentation"):
        predict_peptide_occurrences(inputs, MultiKindModel(declared=(*kinds, "pMHC_presentation")))
    missing_allele = Model()
    missing_allele.alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    with pytest.raises(ValueError, match="HLA-B\\*07:02"):
        predict_peptide_occurrences(inputs, missing_allele)
    with pytest.raises(ValueError, match="different peptide"):
        predict_peptide_occurrences(inputs, Model(other=True))


def test_filters_distinguish_occurrences_in_the_same_source():
    inputs = occurrences().assign(source_sequence_name="same-protein", peptide_offset=0)
    predictor = TopiaryPredictor(models=Model(), filter_by="affinity.value < 802")
    output = predictor.predict_from_peptide_occurrences(inputs)
    assert output.prediction_id.tolist() == ["other-context", "other-peptide"]


@pytest.mark.parametrize("predict", PEPTIDE_OCCURRENCE_TWINS)
def test_empty_input_calls_no_model(predict):
    model = Model()
    assert predict([], model).empty
    assert not model.calls


def test_explicit_comparators_use_only_their_own_context():
    inputs = occurrences().iloc[:2].copy()
    inputs["wt_peptide"] = ["GILGFVFTL", None]
    inputs["wt_n_flank"], inputs["wt_c_flank"] = "TTTTT", ""
    model = Model()
    output = TopiaryPredictor(models=model, predict_wt=True).predict_from_peptide_occurrences(inputs)
    assert output.value.tolist() == [803., 803.]
    assert output.wt_value.iloc[0] == 15. and pd.isna(output.wt_value.iloc[1])
    assert [peptides for peptides, _ in model.calls] == [["SIINFEKL"], ["GILGFVFTL"]]
    assert model.calls[-1][1] == dict(n_flanks=["TTTTT"], c_flanks=[""])
    with pytest.raises(ValueError, match="both flanks"):
        predict_peptide_occurrences(inputs.drop(columns="wt_n_flank"), Model(), predict_wt=True)
    assert predict_peptide_occurrences(inputs.drop(columns="wt_n_flank"), Model(), predict_wt=True, use_flanks=False).wt_value.iloc[0] == 10.


def test_additive_rescoring_and_occurrence_predictions_agree():
    original = combine_sources({"one": source(), "two": source(n_flank="T")}, sample_name="p")
    requests = original.df[["peptide", "n_flank", "c_flank", "candidate_sample"]].copy()
    requests["prediction_id"] = original.df.source_observation_id
    direct = predict_peptide_occurrences(requests, Model())
    enriched = rescore_candidates(original, Model(), prefix="fresh")
    pd.testing.assert_series_equal(direct.value, enriched.df.fresh__testmodel__pMHC_affinity__value, check_names=False)
    pd.testing.assert_frame_equal(enriched.df[original.df.columns], original.df)


def test_conflicting_outputs_cannot_be_relabelled_as_one_occurrence():
    class Conflicting(Model):
        def predict_dataframe(self, peptides, **kwargs):
            rows = super().predict_dataframe(peptides, **kwargs)
            return pd.concat([rows, rows.assign(value=rows.value + 1)], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting or ambiguous"):
        predict_peptide_occurrences(occurrences(), Conflicting())


def test_genotype_aliases_and_missing_values_use_mhcgnomes():
    model = Model(haplotype=True)
    model.alleles = ["A0201"]
    result = predict_peptide_occurrences(occurrences().assign(allele_set="HLA-A*02:01"), model)
    assert result.allele_set.eq("HLA-A*02:01").all()
    result = predict_peptide_occurrences(occurrences().assign(allele_set=pd.NA), model)
    assert result.allele_set.eq("HLA-A*02:01").all()
    with pytest.raises(TypeError, match="explicit booleans"):
        predict_peptide_occurrences(occurrences(), model, use_flanks=None)


def test_partial_cache_coverage_is_reported_by_occurrence_and_empty_schema_survives():
    cache = CachedPredictor(source().df.iloc[:1].assign(n_flank="", c_flank="", peptide_length=8))
    inputs = occurrences()
    reports = []
    predictor = TopiaryPredictor(models=cache, cache_miss_handler=reports.append)
    with pytest.warns(PartialPredictionWarning, match="skipped 1"):
        output = predictor.predict_from_peptide_occurrences(inputs, use_flanks=False)
    assert output.prediction_id.tolist() == inputs.prediction_id.tolist()[:3]
    assert reports[0]["source_sequence_name"] == "other-peptide"
    assert reports[0]["stage"] == "occurrence"
    assert reports[0]["model_key"]
    assert reports[0]["configured_alleles"] == ["HLA-A*02:01"]
    with pytest.warns(PartialPredictionWarning, match="skipped 1"):
        empty = predictor.predict_from_peptide_occurrences(inputs.iloc[3:], use_flanks=False)
    assert empty.empty and {"peptide", "prediction_id", "value", "kind", "allele"} <= set(empty)


def test_haplotype_comparator_can_have_a_different_deconvolved_presenter():
    class Haplotype(Model):
        alleles = ["HLA-A*02:01", "HLA-B*07:02"]
        def predict_dataframe(self, peptides, **kwargs):
            output = super().predict_dataframe(peptides, **kwargs)
            output["allele"] = np.where(output.peptide.eq("SIINFEKL"), self.alleles[0], self.alleles[1])
            return output
    inputs = occurrences().iloc[:1].assign(wt_peptide="GILGFVFTL", wt_n_flank="T", wt_c_flank="")
    output = predict_peptide_occurrences(inputs, Haplotype(haplotype=True), predict_wt=True)
    assert output.allele.tolist() == ["HLA-A*02:01"]
    assert output.allele_set.tolist() == ["HLA-A*02:01,HLA-B*07:02"]
    assert output.wt_value.tolist() == [11.]


def test_shared_sequences_in_distinct_samples_remain_distinct_inference_contexts():
    inputs = occurrences().assign(sample_name=["one", "two", "one", "one"])
    model = Model()
    output = predict_peptide_occurrences(inputs, model)
    assert len(model.calls) == 3
    assert sum(len(peptides) for peptides, _ in model.calls) == 4
    pd.testing.assert_frame_equal(output[inputs.columns], inputs)
