"""Protein fallback preserves the inference occurrence, then replays offline."""

import pandas as pd
import pytest
from mhctools.base_predictor import BasePredictor
from mhctools import Prediction, PeptideResult

from topiary import (
    CachedPredictor, CachedPredictorCoverageError, from_predictions,
    prediction_matches_protein_context, validate_protein_predictions,
)

PEPTIDE = "SIINFEKLA"
A, B = "HLA-A*02:01", "HLA-B*07:02"


class ProteinModel(BasePredictor):
    """Use mhctools' real scanner with deterministic flank-dependent scores."""
    uses_flanking_sequences = True
    flank_length = 2

    def __init__(self, alleles=(A,), lengths=(9,), modes=None):
        super().__init__(alleles=alleles, default_peptide_lengths=lengths)
        self.scans = []
        self.modes = modes or {"pMHC_affinity": "single_allele"}
        self.method, self.version = "protein-test", "1"
        self.corrupt = lambda frame: frame

    def kind_support(self):
        return {kind: {"mhc_dependence": mode} for kind, mode in self.modes.items()}

    def predict_with_flanks(self, peptides, n_flanks, c_flanks):
        results = []
        for peptide, n, c in zip(peptides, n_flanks, c_flanks):
            value = float(100 + sum(map(ord, n + c)))
            preds = []
            for kind, mode in self.modes.items():
                for allele in self.alleles if mode == "single_allele" else [self.alleles[-1] if mode == "haplotype" else None]:
                    preds.append(Prediction(
                        peptide=peptide, allele=allele, kind=kind, value=value if kind == "pMHC_affinity" else None,
                        score=1 / value, predictor_name=self.method, predictor_version=self.version,
                        n_flank=n, c_flank=c))
            results.append(PeptideResult(preds=tuple(preds)))
        return results

    def predict_dataframe(self, peptides, **kwargs):
        return from_predictions([p for result in self.predict_with_flanks(
            peptides, kwargs.get("n_flanks") or [""] * len(peptides),
            kwargs.get("c_flanks") or [""] * len(peptides)) for p in result.preds])

    def predict_proteins_dataframe(self, sequences, peptide_lengths=None):
        self.scans.append(dict(sequences))
        return self.corrupt(super().predict_proteins_dataframe(sequences, peptide_lengths))


def filled_cache(model=None, sequences=None):
    model = model or ProteinModel()
    sequences = sequences or {"original": "MA" + PEPTIDE + "GG"}
    frame = validate_protein_predictions(model.predict_proteins_dataframe(sequences), sequences,
                                         model, peptide_lengths=model.default_peptide_lengths)
    model.scans.clear()
    return CachedPredictor(frame, fallback=model), model


@pytest.mark.parametrize("suffix", ["tsv", "csv.gz", "parquet"])
def test_scan_uses_whole_affected_proteins_then_replays_offline(tmp_path, suffix):
    cache, model = filled_cache()
    original = cache.to_dataframe()
    sequences = {"same": "MA" + PEPTIDE + "GG", "changed": "WW" + PEPTIDE + "CC" + PEPTIDE + "TT"}
    result = cache.predict_proteins_dataframe(sequences)
    assert model.scans == [{"changed": sequences["changed"]}]
    target = result[result.peptide == PEPTIDE]
    assert set(zip(target.source_sequence_name, target.offset)) == {("same", 2), ("changed", 2), ("changed", 13)}
    assert target.value.nunique() == 3
    pd.testing.assert_frame_equal(cache.to_dataframe().iloc[:len(original)], original)
    pd.testing.assert_frame_equal(cache.predict_proteins_dataframe(sequences), result)
    assert len(model.scans) == 1
    path = tmp_path / f"cache.{suffix}"
    cache.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    pd.testing.assert_frame_equal(restored.predict_proteins_dataframe(sequences), result, check_dtype=False)


def test_partial_allele_hit_is_authoritative_even_when_fallback_recomputes_it():
    cache, _ = filled_cache()
    model = ProteinModel(alleles=(A, B))
    cache.fallback = model
    before = cache.to_dataframe()
    model.corrupt = lambda df: df.assign(value=df.value * 2, score=df.score / 2)
    result = cache.predict_proteins_dataframe({"original": "MA" + PEPTIDE + "GG"})
    assert set(result.allele) == {A, B}
    assert result.loc[result.allele == A, "value"].tolist() == before.sort_values("peptide").value.tolist()
    assert result.loc[result.allele == B, "value"].tolist() == (before.sort_values("peptide").value * 2).tolist()


@pytest.mark.parametrize("selected", [False, True])
def test_selected_lengths_do_not_change_fallback_configuration(selected):
    model = ProteinModel(lengths=(8, 9))
    if selected:
        original_scan = model.predict_proteins_dataframe
        model.predict_proteins_dataframe = lambda sequences: original_scan(sequences)
    cache = CachedPredictor(fallback=model)
    cache.default_peptide_lengths = [9]
    result = cache.predict_proteins_dataframe({"p": PEPTIDE})
    assert result.peptide_length.tolist() == [9]
    assert cache.to_dataframe().peptide_length.tolist() == [9]
    assert model.default_peptide_lengths == [8, 9]


def test_scan_retains_stored_scopes_without_mutating_model():
    cache, model = filled_cache(ProteinModel(alleles=(A, B)))
    original_alleles = list(model.alleles)
    cache.alleles = [B]
    result = cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    assert set(result.allele) == {A, B}
    assert model.alleles == original_alleles


def test_haplotype_presenter_does_not_change_measurement_identity():
    model = ProteinModel(alleles=(A, B), modes={"pMHC_presentation": "haplotype", "antigen_processing": "none"})
    model.alleles = [A, B]
    cache, model = filled_cache(model)
    model.corrupt = lambda df: df.assign(allele=df.allele.where(df.kind != "pMHC_presentation", A))
    result = cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    assert set(result.kind) == {"pMHC_presentation", "antigen_processing"}
    assert set(result.loc[result.kind == "pMHC_presentation", "allele"]) == {A}
    assert set(result.loc[result.kind == "pMHC_presentation", "allele_set"]) == {f"{A},{B}"}
    assert result.loc[result.kind == "antigen_processing", "allele"].eq("").all()


CORRUPTIONS = [
    lambda df: df.iloc[0:0],
    lambda df: df.iloc[1:],
    lambda df: pd.concat([df, df.iloc[:1]], ignore_index=True),
    lambda df: df.assign(source_sequence_name="wrong"),
    lambda df: df.assign(offset=-1),
    lambda df: df.assign(offset=.5),
    lambda df: df.assign(offset=True),
    lambda df: df.assign(peptide_offset=df.offset),
    lambda df: df.drop(columns="offset"),
    lambda df: df.assign(peptide="ELAGIGILT"),
    lambda df: df.assign(length=8),
    lambda df: df.assign(allele=B),
    lambda df: df.assign(kind="unknown"),
    lambda df: df.assign(prediction_mhc_dependence="none"),
    lambda df: df.assign(n_flank="ZZ"),
    lambda df: df.assign(n_flank=None, prediction_flanks_supplied=True),
    lambda df: df.assign(n_flank="", c_flank="", prediction_flanks_supplied=True),
    lambda df: df.assign(predictor_version=None),
    lambda df: df.assign(predictor_version=["1"] + ["other"] * (len(df) - 1)),
    lambda df: df.assign(score="invalid"),
]


@pytest.mark.parametrize("corrupt", CORRUPTIONS, ids=lambda f: str(CORRUPTIONS.index(f)))
@pytest.mark.parametrize("empty", [False, True])
def test_invalid_batches_are_atomic(corrupt, empty):
    cache, model = filled_cache()
    if empty:
        cache = CachedPredictor(fallback=model)
    before, identity = cache.to_dataframe(), (cache.prediction_method_name, cache.predictor_version)
    model.corrupt = corrupt
    with pytest.raises((ValueError, CachedPredictorCoverageError)):
        cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(cache.to_dataframe(), before)
    assert (cache.prediction_method_name, cache.predictor_version) == identity


def test_wrong_genotype_cannot_fill_another_genotypes_gap():
    cache, model = filled_cache(ProteinModel(alleles=(A, B), modes={"pMHC_presentation": "haplotype"}))
    model.corrupt = lambda df: df.assign(allele_set=A)
    before = cache.to_dataframe()
    with pytest.raises(ValueError, match="genotype"):
        cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(cache.to_dataframe(), before)


def test_peptide_only_fallback_handles_new_peptides_but_not_existing_flank_misses():
    cache, model = filled_cache()
    model.predict_proteins_dataframe = None
    with pytest.raises(CachedPredictorCoverageError, match="flanking context"):
        cache.predict_proteins_dataframe({"p": PEPTIDE})
    assert not model.scans
    result = cache.predict_proteins_dataframe({"new": "ELAGIGILT"})
    assert result.peptide.tolist() == ["ELAGIGILT"]


def test_explicit_peptide_only_annotations_rebind_without_changing_stored_inputs():
    cache, _ = filled_cache(sequences={"old": PEPTIDE})
    frame = cache.to_dataframe().assign(n_flank="MA", c_flank="GG", prediction_flanks_supplied=False)
    cache = CachedPredictor(frame)
    # Other windows are absent: directly exercise the public context predicate.
    assert prediction_matches_protein_context(frame.iloc[0], "WW" + PEPTIDE + "CC", 2)
    result = cache.predict_proteins_dataframe({"p": PEPTIDE})
    assert result.n_flank.tolist() == [""]
    assert result.c_flank.tolist() == [""]
    assert cache.to_dataframe().n_flank.tolist() == ["MA"]


def test_empty_and_short_inputs_never_infer():
    cache = CachedPredictor(fallback=ProteinModel())
    for sequences in ({}, {"short": "MA"}, {"empty": ""}):
        assert cache.predict_proteins_dataframe(sequences).empty
    assert not cache.fallback.scans


@pytest.mark.parametrize("change", ["method", "version"])
def test_contextual_fallback_doors_check_every_response(change):
    from .test_twin_conformance import CONTEXTUAL_FALLBACK_TWINS
    for door in CONTEXTUAL_FALLBACK_TWINS:
        cache, model = filled_cache()
        door(cache, "ELAGIGILT")
        before = cache.to_dataframe()
        setattr(model, change, "different")
        with pytest.raises(ValueError, match="mismatch|exactly one"):
            door(cache, "GILGFVFTL")
        pd.testing.assert_frame_equal(cache.to_dataframe(), before)


def test_nested_cache_preserves_legacy_context_metadata():
    model = ProteinModel()
    proteins = {"original": "MA" + PEPTIDE + "GG"}
    frame = from_predictions(model.predict_proteins_dataframe(proteins))
    # Real archived scanner output predates explicit inference flags.
    inner = CachedPredictor(frame)
    outer = CachedPredictor(fallback=inner)
    result = outer.predict_proteins_dataframe(proteins)
    assert result.loc[result.peptide == PEPTIDE, "prediction_flanks_supplied"].tolist() == [True]
    with pytest.raises(CachedPredictorCoverageError):
        outer.predict_proteins_dataframe({"changed": "WW" + PEPTIDE + "CC"})


def test_uncovered_old_genotype_cannot_be_filled_by_configured_new_genotype():
    old, _ = filled_cache(ProteinModel(alleles=(A,), modes={"pMHC_presentation": "haplotype"}))
    old.fallback = ProteinModel(alleles=(B,), modes={"pMHC_presentation": "haplotype"})
    before = old.to_dataframe()
    with pytest.raises(CachedPredictorCoverageError):
        old.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(before, old.to_dataframe())


def test_missing_kind_cannot_hide_behind_another_kinds_coverage():
    cache, model = filled_cache(ProteinModel(modes={"pMHC_affinity": "single_allele", "antigen_processing": "none"}))
    model.corrupt = lambda df: df[df.kind == "pMHC_affinity"]
    before = cache.to_dataframe()
    with pytest.raises(ValueError, match="coverage"):
        cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(before, cache.to_dataframe())


def test_more_general_fallback_cannot_add_a_second_answer_to_an_existing_hit():
    cache, model = filled_cache()
    before = cache.to_dataframe()
    # A producer changes which inputs it uses while keeping the same identity.
    # The new generic row would cover an already cached, different measurement.
    model.corrupt = lambda df: df.assign(prediction_flanks_supplied=False)
    with pytest.raises(ValueError, match="conflicts"):
        cache.predict_proteins_dataframe({"same": "MA" + PEPTIDE + "GG", "changed": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(before, cache.to_dataframe())


def test_peptide_only_allele_miss_cannot_mask_an_existing_flank_miss():
    cache, _ = filled_cache()
    model = ProteinModel(alleles=(A, B))
    model.predict_proteins_dataframe = None
    model.predict_dataframe = lambda peptides: pytest.fail("Must reject the incompatible context before peptide inference")
    cache.fallback = model
    before = cache.to_dataframe()
    with pytest.raises(CachedPredictorCoverageError, match="flanking context"):
        cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(before, cache.to_dataframe())
