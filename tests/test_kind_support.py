"""Tests for kind_support / supported_kinds plumbing on TopiaryPredictor
and CachedPredictor (mhctools >=3.13.7 metadata API)."""

import pandas as pd
from mhctools import (
    Kind,
    MHC_CLASS_VALUES,
    MHC_DEPENDENCE_VALUES,
    RandomBindingPredictor,
)

from topiary import CachedPredictor, TopiaryPredictor


class FakeProcessingPredictor:
    """Minimal mhctools-shaped predictor with no allele dependence.

    Mirrors what real processing predictors (Pepsickle, NetChop) report
    via ``kind_support()`` without needing the external binary.
    """
    prediction_method_name = "fake-processing"
    predictor_version = "0.0.0"
    default_peptide_lengths = [9]
    alleles = []

    def kind_support(self):
        return {
            Kind.proteasome_cleavage: {
                "mhc_dependence": "none",
                "mhc_class": "none",
            }
        }

    @property
    def supported_kinds(self):
        return tuple(self.kind_support())


def _cache_row(
    peptide="SIINFEKLA",
    allele="HLA-A*02:01",
    *,
    kind="pMHC_affinity",
    score=0.5,
    affinity=150.0,
    percentile_rank=2.0,
):
    return {
        "peptide": peptide,
        "allele": allele,
        "peptide_length": len(peptide),
        "kind": kind,
        "score": score,
        "affinity": affinity,
        "percentile_rank": percentile_rank,
        "prediction_method_name": "random",
        "predictor_version": "1.0",
    }


class TestTopiaryPredictorKindSupport:
    def test_single_model_reports_kind_support(self):
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"]
        )
        support = predictor.kind_support
        assert list(support.keys()) == ["RandomBindingPredictor"]
        entry = support["RandomBindingPredictor"]
        assert entry == {
            Kind.pMHC_affinity: {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            }
        }

    def test_supported_kinds_dedupes_across_models(self):
        a = RandomBindingPredictor(alleles=["A0201"])
        b = RandomBindingPredictor(alleles=["A0201"])
        predictor = TopiaryPredictor(models=[a, b], alleles=["A0201"])
        # Two instances of the same predictor still emit the same kind once
        assert predictor.supported_kinds == (Kind.pMHC_affinity,)
        # But kind_support keeps both model entries (uniquified keys)
        assert len(predictor.kind_support) == 2

    def test_kind_support_keys_match_model_keys(self):
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"]
        )
        assert set(predictor.kind_support) == set(predictor._model_keys)

    def test_kind_support_returns_dict_copies(self):
        """Mutating returned dicts must not poison subsequent reads."""
        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"]
        )
        first = predictor.kind_support
        first["RandomBindingPredictor"][Kind.pMHC_affinity]["mhc_class"] = "II"
        second = predictor.kind_support
        assert (
            second["RandomBindingPredictor"][Kind.pMHC_affinity]["mhc_class"]
            == "I"
        )

    def test_processing_predictor_reports_none_dependence(self):
        """Allele-independent predictors flow through with mhc_dependence='none'."""
        binding = RandomBindingPredictor(alleles=["A0201"])
        processing = FakeProcessingPredictor()
        predictor = TopiaryPredictor(
            models=[binding, processing], alleles=["A0201"]
        )
        support = predictor.kind_support
        # Both predictors present, each with their own kind/dependence
        binding_entry = support["RandomBindingPredictor"]
        processing_entry = support["fake-processing"]
        assert binding_entry[Kind.pMHC_affinity]["mhc_dependence"] == "single_allele"
        assert processing_entry[Kind.proteasome_cleavage] == {
            "mhc_dependence": "none",
            "mhc_class": "none",
        }
        assert set(predictor.supported_kinds) == {
            Kind.pMHC_affinity,
            Kind.proteasome_cleavage,
        }


class TestCompatibilityWithMhctools:
    """The (model_key, kind) inner dict must stay byte-compatible with what
    mhctools' own ``kind_support()`` returns — same keys, same value
    vocabulary — so consumers can pass topiary's metadata anywhere mhctools'
    is accepted without translation."""

    def test_inner_dict_matches_mhctools_for_each_model(self):
        model = RandomBindingPredictor(alleles=["A0201"])
        predictor = TopiaryPredictor(models=model, alleles=["A0201"])
        wrapped = predictor.kind_support
        assert len(wrapped) == 1
        ((_, support),) = wrapped.items()
        assert support == model.kind_support()

    def test_values_are_in_mhctools_vocabulary(self):
        predictor = TopiaryPredictor(
            models=[
                RandomBindingPredictor(alleles=["A0201"]),
                FakeProcessingPredictor(),
            ],
            alleles=["A0201"],
        )
        for model_support in predictor.kind_support.values():
            for meta in model_support.values():
                assert meta["mhc_dependence"] in MHC_DEPENDENCE_VALUES
                assert meta["mhc_class"] in MHC_CLASS_VALUES

    def test_cached_predictor_values_are_in_mhctools_vocabulary(self):
        cache = CachedPredictor(
            pd.DataFrame([
                _cache_row(kind="pMHC_affinity"),
                _cache_row(peptide="ELAGIGILTV", kind="proteasome_cleavage"),
            ])
        )
        for meta in cache.kind_support().values():
            assert meta["mhc_dependence"] in MHC_DEPENDENCE_VALUES
            assert meta["mhc_class"] in MHC_CLASS_VALUES


class TestCachedPredictorKindSupport:
    def test_kinds_from_cache_default_to_single_allele(self):
        cache = CachedPredictor(
            pd.DataFrame([
                _cache_row(kind="pMHC_affinity"),
                _cache_row(peptide="ELAGIGILTV", kind="pMHC_presentation"),
            ])
        )
        support = cache.kind_support()
        assert support == {
            "pMHC_affinity": {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            },
            "pMHC_presentation": {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            },
        }
        assert set(cache.supported_kinds) == {
            "pMHC_affinity",
            "pMHC_presentation",
        }

    def test_fallback_kind_support_overrides_for_shared_kinds(self):
        """If the fallback predictor reports a richer mapping, prefer it
        for kinds the cache also carries — single-allele vs haplotype is
        a property of (predictor, kind), not just kind."""

        class FakeFallback:
            prediction_method_name = "random"
            predictor_version = "1.0"
            alleles = ["HLA-A*02:01"]
            default_peptide_lengths = [9]

            def kind_support(self):
                return {
                    "pMHC_affinity": {
                        "mhc_dependence": "single_allele",
                        "mhc_class": "I",
                    },
                    "pMHC_presentation": {
                        "mhc_dependence": "haplotype",
                        "mhc_class": "I",
                    },
                }

        cache = CachedPredictor(
            pd.DataFrame([_cache_row(kind="pMHC_presentation")]),
            fallback=FakeFallback(),
        )
        support = cache.kind_support()
        assert support["pMHC_presentation"]["mhc_dependence"] == "haplotype"
        # Cache had no affinity rows, so fallback's affinity entry isn't surfaced
        assert "pMHC_affinity" not in support

    def test_proteasome_cleavage_rows_default_to_single_allele(self):
        """No-fallback default for any kind is single_allele/I (the
        cache stores rows per (peptide, allele, kind), so we can't
        downgrade to 'none' without external context). This is the
        documented conservative behavior — a fallback is the way to
        report 'none' faithfully."""
        cache = CachedPredictor(
            pd.DataFrame([_cache_row(kind="proteasome_cleavage")])
        )
        support = cache.kind_support()
        assert support == {
            "proteasome_cleavage": {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            }
        }

    def test_proteasome_cleavage_via_fallback_reports_none(self):
        """A fallback that knows it's a processing predictor flows
        ``mhc_dependence='none'``/``mhc_class='none'`` through to the
        cache's reported metadata."""

        class FakeProcessingFallback:
            prediction_method_name = "fake-processing"
            predictor_version = "0.0.0"
            alleles = []
            default_peptide_lengths = [9]

            def kind_support(self):
                return {
                    "proteasome_cleavage": {
                        "mhc_dependence": "none",
                        "mhc_class": "none",
                    }
                }

        cache = CachedPredictor(
            pd.DataFrame([_cache_row(kind="proteasome_cleavage")]),
            fallback=FakeProcessingFallback(),
        )
        assert cache.kind_support()["proteasome_cleavage"] == {
            "mhc_dependence": "none",
            "mhc_class": "none",
        }
