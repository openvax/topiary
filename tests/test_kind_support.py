"""Tests for kind_support / supported_kinds plumbing on TopiaryPredictor
and CachedPredictor (mhctools >=3.13.7 metadata API)."""

import pandas as pd
import pytest
from mhctools import Kind, RandomBindingPredictor

from topiary import CachedPredictor, TopiaryPredictor


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
