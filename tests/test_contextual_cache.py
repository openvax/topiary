"""File-backed exact cache queries preserve inference context and coverage."""

import numpy as np
import pandas as pd
import pytest

from topiary import (
    CachedPredictor, CachedPredictorCoverageError, EvalContext, parse,
    predict_peptide_occurrences, prediction_flanks_match,
)
from .test_candidate_tables import Model, MultiKindModel, source
from .test_peptide_occurrences import occurrences
from .test_twin_conformance import CACHE_FALLBACK_IDENTITY_TWINS, CACHE_PROVENANCE_TWINS

pytestmark = pytest.mark.usefixtures("pandas_string_inference")


@pytest.mark.parametrize("kind", ["unknown", "termini", "peptide_only", "supplied"])
@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_known_termini_unknown_context_and_peptide_only_stay_distinct(tmp_path, kind, suffix):
    row = source().df.iloc[:1].copy()
    row["n_flank"] = row["c_flank"] = ""
    if kind == "termini":
        row["prediction_flanks_supplied"] = True
    elif kind == "peptide_only":
        row["prediction_flanks_supplied"] = False
        row["n_flank"], row["c_flank"] = "AAA", "GG"
    elif kind == "supplied":
        row["n_flank"], row["c_flank"] = "AAA", "GG"
    cache = CachedPredictor.from_dataframe(row)
    path = tmp_path / ("context" + suffix)
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    for n_flank, c_flank, expected in [
        (None, None, kind in {"unknown", "peptide_only"}),
        ("", "", kind == "termini"),
        ("AAA", "GG", kind == "supplied"),
        ("T", "GG", False),
    ]:
        assert prediction_flanks_match(
            restored.to_dataframe().iloc[0], n_flank=n_flank, c_flank=c_flank,
        ) is expected
        kwargs = {} if n_flank is None else dict(n_flanks=[n_flank], c_flanks=[c_flank])
        if expected:
            assert restored.predict_contextual_peptides_dataframe(["SIINFEKL"], **kwargs).value.tolist() == [50.]
        else:
            with pytest.raises(CachedPredictorCoverageError):
                restored.predict_contextual_peptides_dataframe(["SIINFEKL"], **kwargs)


def test_unknown_occurrence_context_requires_explicit_peptide_only_mode():
    cache = CachedPredictor.from_dataframe(source().df.iloc[:1].assign(n_flank="", c_flank=""))
    inputs = [dict(prediction_id="one", peptide="SIINFEKL")]
    with pytest.raises(ValueError, match="both flanks"):
        predict_peptide_occurrences(inputs, cache)
    assert predict_peptide_occurrences(inputs, cache, use_flanks=False).value.tolist() == [50.]


def test_peptide_only_inference_replays_without_promoting_source_annotations():
    inputs = occurrences().iloc[:1]
    fresh = predict_peptide_occurrences(inputs, Model(), use_flanks=False)
    cache = CachedPredictor.from_dataframe(fresh)
    replay = predict_peptide_occurrences(inputs, cache, use_flanks=False)
    assert replay.value.tolist() == fresh.value.tolist() == [800.]
    assert replay.n_flank.tolist() == ["AAA"]
    assert cache.to_dataframe().n_flank.tolist() == ["AAA"]
    with pytest.raises(CachedPredictorCoverageError):
        predict_peptide_occurrences(inputs, cache)


@pytest.mark.parametrize("missing", ["n_flank", "c_flank"])
@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_one_unknown_legacy_flank_does_not_become_a_known_terminus(tmp_path, missing, suffix):
    row = source().df.iloc[:1].assign(**{missing: None})
    cache = CachedPredictor.from_dataframe(row)
    path = tmp_path / ("partial-context" + suffix)
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    inputs = [dict(prediction_id="one", peptide="SIINFEKL",
                   n_flank="" if missing == "n_flank" else "AAA",
                   c_flank="" if missing == "c_flank" else "GGG")]
    for candidate in (cache, restored):
        with pytest.raises(CachedPredictorCoverageError):
            predict_peptide_occurrences(inputs, candidate)
        with pytest.raises(CachedPredictorCoverageError):
            predict_peptide_occurrences(inputs, candidate, use_flanks=False)


def test_known_empty_and_unknown_predictions_do_not_collide_in_cache_keys():
    rows = source().df.iloc[:1].assign(n_flank="", c_flank="")
    known = rows.assign(prediction_flanks_supplied=True, value=100.)
    cache = CachedPredictor.from_dataframe(pd.concat([rows, known], ignore_index=True))
    assert cache.predict_contextual_peptides_dataframe(["SIINFEKL"]).value.tolist() == [50.]
    assert cache.predict_contextual_peptides_dataframe(
        ["SIINFEKL"], n_flanks=[""], c_flanks=[""]).value.tolist() == [100.]
    assert sorted(cache.predict_peptides_dataframe(["SIINFEKL"]).value) == [50., 100.]


@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_missing_context_falls_back_once_then_replays_from_file(tmp_path, suffix):
    inputs = occurrences()
    expected = predict_peptide_occurrences(inputs, Model())
    fallback = Model()
    cache = CachedPredictor.from_dataframe(expected.iloc[:1], fallback=fallback)
    actual = predict_peptide_occurrences(inputs, cache)
    assert actual.value.tolist() == expected.value.tolist()
    assert sum(len(peptides) for peptides, _ in fallback.calls) == 2
    calls = len(fallback.calls)
    again = predict_peptide_occurrences(inputs, cache)
    assert len(fallback.calls) == calls
    pd.testing.assert_frame_equal(actual, again)
    path = tmp_path / ("filled" + suffix)
    cache.save(path)
    restored = predict_peptide_occurrences(inputs, CachedPredictor.from_topiary_output(path))
    assert restored.value.tolist() == expected.value.tolist()


def test_empty_cache_uses_declared_fallback_kinds():
    inputs = occurrences()
    model = Model()
    cache = CachedPredictor(fallback=model)
    assert predict_peptide_occurrences(inputs, cache).value.tolist() == [803., 803., 801., 13.]
    assert cache.predictor_version == "2"


@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_primary_and_comparator_use_their_own_cached_context(tmp_path, suffix):
    inputs = occurrences().iloc[:2].assign(
        wt_peptide="GILGFVFTL", wt_n_flank="T", wt_c_flank="G", wt_peptide_offset=1,
    )
    expected = predict_peptide_occurrences(inputs, Model(), predict_wt=True)
    primary = predict_peptide_occurrences(inputs, Model())
    comparator = predict_peptide_occurrences([
        dict(prediction_id="wt", peptide="GILGFVFTL", n_flank="T", c_flank="G"),
    ], Model())
    cache = CachedPredictor.from_dataframe(pd.concat([primary, comparator], ignore_index=True))
    path = tmp_path / ("comparators" + suffix)
    cache.save(path)
    actual = predict_peptide_occurrences(inputs, CachedPredictor.from_topiary_output(path), predict_wt=True)
    columns = ["prediction_id", "peptide", "peptide_offset", "value", "wt_value", "wt_score",
               "wt_prediction_method_name", "wt_predictor_version"]
    pd.testing.assert_frame_equal(expected[columns], actual[columns], check_exact=True)
    assert actual.value.tolist() == [803., 803.]
    assert actual.wt_value.tolist() == [11., 11.]


class ChangingModel(Model):
    version = "2"
    method = "testmodel"
    mixed = False

    def predict_dataframe(self, peptides, **kwargs):
        frame = super().predict_dataframe(peptides, **kwargs).assign(
            predictor_version=self.version, prediction_method_name=self.method)
        if self.mixed and len(frame) > 1:
            frame.loc[frame.index[-1], "predictor_version"] = "different"
        return frame


@pytest.mark.parametrize("query", CACHE_FALLBACK_IDENTITY_TWINS)
@pytest.mark.parametrize("changed", ["version", "method", "mixed"])
def test_every_fallback_batch_verifies_identity(query, changed):
    fallback = ChangingModel()
    cache = CachedPredictor.from_dataframe(fallback.predict_dataframe(["SIINFEKL"]), fallback=fallback)
    query(cache, ["GILGFVFTL"])
    before = cache.to_dataframe()
    setattr(fallback, changed, True if changed == "mixed" else "different")
    with pytest.raises(ValueError, match="mismatch|exactly one"):
        query(cache, ["ELAGIGILT", "SIINFEKK"])
    pd.testing.assert_frame_equal(cache.to_dataframe(), before)


def test_all_table_cache_loaders_preserve_threshold_decisions(tmp_path):
    boundary = 1 / 803
    frame = source().df.assign(score=[boundary, np.nextafter(boundary, -np.inf)])
    path = tmp_path / "predictions.tsv"
    frame.to_csv(path, sep="\t", index=False)
    expression = parse(f"affinity.score >= {boundary!r}")
    for name, loader in CACHE_PROVENANCE_TWINS:
        supplied = frame if name == "dataframe" else tmp_path if name == "directory" else path
        restored = loader(supplied).to_dataframe()
        assert restored.score.tolist() == frame.score.tolist(), name
        assert expression.eval(EvalContext(restored)).tolist() == [True, False], name


@pytest.mark.parametrize("suffix", [".tsv", ".parquet"])
def test_genotypes_sharing_presenter_have_distinct_replay_and_comparators(tmp_path, suffix):
    class GenotypeModel(Model):
        def predict_dataframe(self, peptides, **kwargs):
            frame = super().predict_dataframe(peptides, **kwargs)
            frame["value"] += 100 * len(self.alleles)
            frame["score"] = 1 / frame.value
            frame.loc[frame.peptide.eq("GILGFVFTL"), "allele"] = self.alleles[-1]
            return frame

    inputs = occurrences().iloc[:1].assign(
        wt_peptide="GILGFVFTL", wt_n_flank="T", wt_c_flank="G")
    models = [GenotypeModel(haplotype=True), GenotypeModel(haplotype=True)]
    models[1].alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    cache = CachedPredictor.concat([
        CachedPredictor.from_dataframe(pd.concat([
            predict_peptide_occurrences(inputs.drop(columns=["wt_peptide", "wt_n_flank", "wt_c_flank"]), model),
            predict_peptide_occurrences([dict(prediction_id="wt", peptide="GILGFVFTL", n_flank="T", c_flank="G")], model),
        ])) for model in models])
    path = tmp_path / ("genotypes" + suffix)
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    for model in models:
        restored.alleles = model.alleles
        actual = predict_peptide_occurrences(inputs, restored, predict_wt=True)
        expected = predict_peptide_occurrences(inputs, model, predict_wt=True)
        columns = ["value", "score", "allele", "allele_set", "wt_value", "wt_score"]
        pd.testing.assert_frame_equal(actual[columns], expected[columns], check_exact=True)
    assert actual.wt_value.item() == 211.
    restored.alleles = ["A0201", "B0801"]
    with pytest.raises(CachedPredictorCoverageError):
        predict_peptide_occurrences(inputs, restored)
    restored.fallback = models[1]
    with pytest.raises(ValueError, match="Fallback genotype"):
        predict_peptide_occurrences(inputs, restored)


@pytest.mark.parametrize("use_flanks", [True, False])
def test_every_kind_and_allele_is_required_and_fallback_retains_existing_measurements(use_flanks):
    inputs = occurrences().iloc[:1]
    model = MultiKindModel()
    complete = predict_peptide_occurrences(inputs, model, use_flanks=use_flanks)
    incomplete = complete.loc[complete.kind.eq("antigen_processing")].drop(
        columns=["prediction_flanks_supplied", "prediction_mhc_dependence"]).assign(value=123.)
    if not use_flanks:
        incomplete = incomplete.assign(n_flank="", c_flank="")
    cache = CachedPredictor.from_dataframe(incomplete, fallback=model)
    actual = predict_peptide_occurrences(inputs, cache, use_flanks=use_flanks)
    assert set(actual.kind) == {"pMHC_affinity", "antigen_processing"}
    assert len(model.calls) == 2
    assert cache.to_dataframe().loc[lambda df: df.kind.eq("antigen_processing"), "value"].tolist() == [123.]
    cache.fallback = None
    cache.alleles = ["A0201", "B0702"]
    with pytest.raises(CachedPredictorCoverageError):
        predict_peptide_occurrences(inputs, cache, use_flanks=use_flanks)
    assert len(cache.to_dataframe()) == 2


@pytest.mark.parametrize("invalid", ["peptide", "allele", "kind", "flank", "empty", "duplicate"])
def test_invalid_fallback_output_does_not_enter_cache(invalid):
    class InvalidModel(Model):
        def predict_dataframe(self, peptides, **kwargs):
            frame = super().predict_dataframe(peptides, **kwargs)
            if invalid == "empty":
                return frame.iloc[:0]
            if invalid == "duplicate":
                return pd.concat([frame, frame.assign(value=1.)], ignore_index=True)
            column, value = {"peptide": ("peptide", "ELAGIGILT"),
                             "allele": ("allele", "HLA-B*07:02"),
                             "kind": ("kind", "undeclared"), "flank": ("n_flank", "ZZZ")}[invalid]
            return frame.assign(**{column: value})
    rows = predict_peptide_occurrences(occurrences().iloc[:1], Model())
    cache = CachedPredictor.from_dataframe(rows, fallback=InvalidModel())
    before = cache.to_dataframe()
    with pytest.raises(ValueError):
        predict_peptide_occurrences(occurrences().iloc[-1:], cache)
    pd.testing.assert_frame_equal(cache.to_dataframe(), before)


@pytest.mark.parametrize("allele", ["A0201", "H2-Kb"])
def test_aliases_and_nonhuman_alleles_use_canonical_query_scope(allele):
    frame = source().df.iloc[:1].assign(allele=allele, c_flank="GG")
    cache = CachedPredictor.from_dataframe(frame)
    cache.alleles = [allele]
    actual = predict_peptide_occurrences(occurrences().iloc[:1], cache)
    assert actual.value.tolist() == [50.]
    assert actual.allele.tolist() == cache.alleles


@pytest.mark.parametrize("kwargs", [dict(n_flanks=["A"]), dict(n_flanks="A", c_flanks="B"),
                                    dict(n_flanks=[], c_flanks=[]),
                                    dict(n_flanks=[None], c_flanks=[""])])
def test_malformed_context_queries_are_refused_before_fallback(kwargs):
    model = Model()
    cache = CachedPredictor(fallback=model)
    with pytest.raises(ValueError):
        cache.predict_contextual_peptides_dataframe(["SIINFEKL"], **kwargs)
    assert model.calls == []
