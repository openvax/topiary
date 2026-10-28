"""Configured model facts have one public, serializable representation."""

import json

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import CachedPredictor, TopiaryPredictor, describe_prediction_models


def test_real_duplicate_models_have_distinct_keys_and_json_facts():
    predictor = TopiaryPredictor(models=[
        RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
        RandomBindingPredictor(alleles=["B0702"], default_peptide_lengths=[10]),
    ])

    inventory = describe_prediction_models(predictor)

    assert [record["model_key"] for record in inventory] == [
        "RandomBindingPredictor__1", "RandomBindingPredictor__2",
    ]
    assert [record["prediction_method_name"] for record in inventory] == [
        None, None,
    ]
    assert inventory[0] == {
        "model_key": "RandomBindingPredictor__1",
        "model_class": "RandomBindingPredictor",
        "prediction_method_name": None,
        "predictor_version": None,
        "alleles": ["HLA-A*02:01"],
        "default_peptide_lengths": [9],
        "min_peptide_length": 8,
        "max_peptide_length": None,
        "uses_flanking_sequences": False,
        "protein_flank_lengths": {"n": 0, "c": 0},
        "kinds": [{
            "kind": "pMHC_affinity",
            "mhc_dependence": "single_allele",
            "mhc_class": "I",
        }],
    }
    assert inventory[1]["alleles"] == ["HLA-B*07:02"]
    assert inventory[1]["default_peptide_lengths"] == [10]
    assert json.loads(json.dumps(inventory)) == inventory


def test_mixed_endpoint_scope_and_numpy_metadata_are_normalized():
    class MixedModel:
        prediction_method_name = "mixed"
        predictor_version = "2"
        alleles = ["HLA-A*02:01", "HLA-B*07:02"]
        default_peptide_lengths = [np.int64(8), np.int64(9)]
        min_peptide_length = np.int64(8)
        max_peptide_length = np.int64(15)
        uses_flanking_sequences = np.bool_(True)

        def protein_flank_lengths(self):
            return np.int64(5), np.int64(7)

        def kind_support(self):
            return {
                "pMHC_presentation": {
                    "mhc_dependence": "haplotype", "mhc_class": "I",
                },
                "antigen_processing": {
                    "mhc_dependence": "none", "mhc_class": "none",
                },
            }

    inventory = describe_prediction_models(TopiaryPredictor(models=MixedModel()))

    assert inventory[0]["default_peptide_lengths"] == [8, 9]
    assert inventory[0]["protein_flank_lengths"] == {"n": 5, "c": 7}
    assert inventory[0]["kinds"] == [
        {"kind": "antigen_processing", "mhc_dependence": "none", "mhc_class": "none"},
        {"kind": "pMHC_presentation", "mhc_dependence": "haplotype", "mhc_class": "I"},
    ]
    json.dumps(inventory)


def test_cached_and_undeclared_metadata_are_reported_without_inference():
    frame = pd.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFTL"],
        "peptide_length": [8, 9],
        "allele": ["HLA-A*02:01"] * 2,
        "kind": ["pMHC_affinity"] * 2,
        "value": [10.0, 20.0],
        "prediction_method_name": ["archive"] * 2,
        "predictor_version": ["1"] * 2,
    })

    record = describe_prediction_models(
        TopiaryPredictor(models=CachedPredictor(frame))
    )[0]

    assert record["prediction_method_name"] == "archive"
    assert record["model_class"] == "CachedPredictor"
    assert record["predictor_version"] == "1"
    assert record["alleles"] == ["HLA-A*02:01"]
    assert record["default_peptide_lengths"] == [8, 9]
    assert record["min_peptide_length"] is None
    assert record["max_peptide_length"] is None
    assert record["uses_flanking_sequences"] is None
    assert record["protein_flank_lengths"] is None
    assert record["kinds"][0]["kind"] == "pMHC_affinity"

    class UndeclaredModel:
        pass

    unknown = describe_prediction_models(
        TopiaryPredictor(models=UndeclaredModel())
    )[0]
    assert unknown["model_class"] == "UndeclaredModel"
    assert unknown["prediction_method_name"] is None
    assert unknown["predictor_version"] is None
    assert unknown["alleles"] is None
    assert unknown["default_peptide_lengths"] is None
    assert unknown["uses_flanking_sequences"] is None
    assert unknown["kinds"] is None

    class MissingNameModel:
        prediction_method_name = np.nan

    missing_name = describe_prediction_models(
        TopiaryPredictor(models=MissingNameModel())
    )[0]
    assert missing_name["model_key"] == "MissingNameModel"
    assert missing_name["prediction_method_name"] is None


@pytest.mark.parametrize("value,label", [
    ([0], "default_peptide_lengths"),
    ([8.5], "default_peptide_lengths"),
])
def test_invalid_declared_length_metadata_cannot_be_summarized(value, label):
    class InvalidModel:
        default_peptide_lengths = value

    with pytest.raises(ValueError, match=label):
        describe_prediction_models(TopiaryPredictor(models=InvalidModel()))


def test_only_topiary_predictors_are_accepted():
    with pytest.raises(TypeError, match="TopiaryPredictor"):
        describe_prediction_models(RandomBindingPredictor(alleles=["A0201"]))
