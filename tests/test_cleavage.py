"""Native site evidence retains its meaning through both input forms."""

from dataclasses import replace

import pandas as pd
import pytest
from mhctools import CleavageModel, CleavageResult, CleavageSite, PeptideInput

from topiary import from_cleavage_results
from .test_twin_conformance import CLEAVAGE_RESULT_TWINS


def reference_model():
    return CleavageModel(
        name="test-reference", version="1", enzyme="test-enzyme", uniprot=None,
        species="human", compartments=("serum",), evidence="substrate_reference",
        references=("https://example.org/synthetic-test-fixture",),
        assay="synthetic fixture", limitations="Not a biological observation",
    )


@pytest.mark.parametrize("outcome", ["sites", "unsupported", "unassessed",
                                    "cleavage_reported", "no_cleavage_detected"])
def test_native_and_serialized_evidence_agree(outcome):
    peptide = PeptideInput("APAA", occurrence_id="first", source_start=20)
    result = CleavageResult(peptide, reference_model(), conditions=(("pH", "7"),))
    if outcome == "sites":
        result = replace(result, sites=(CleavageSite(2, "reported", "fixture"),))
    elif outcome == "unsupported":
        result = replace(result, unsupported_reason="outside the fixture domain")
    elif outcome != "unassessed":
        result = replace(result, substrate_observation=outcome)
    frames = [door([result]) for _, door in CLEAVAGE_RESULT_TWINS]
    pd.testing.assert_frame_equal(*frames)
    row = frames[0].iloc[0]
    assert row.conditions == {"pH": "7"}
    assert row.model_metadata["name"] == result.model.name
    assert row.peptide_input["occurrence_id"] == "first"
    if outcome == "sites":
        assert (row.bond, row.source_bond, row.status) == (2, 22, "reported")
    else:
        assert pd.isna(row.bond) and pd.isna(row.source_bond)
        assert row.status == outcome
    assert pd.isna(row.score) and pd.isna(row["match"])


def test_empty_inputs_and_invalid_records():
    frames = [door([]) for _, door in CLEAVAGE_RESULT_TWINS]
    pd.testing.assert_frame_equal(*frames)
    assert frames[0].empty and "bond" in frames[0]
    with pytest.raises(TypeError, match="CleavageResult"):
        from_cleavage_results(["not a record"])
    record = CleavageResult(
        PeptideInput("APAA"), reference_model(),
        (CleavageSite(2, "reported", "fixture"),),
    ).to_dict()
    record["sites"][0]["source_bond"] = 3
    with pytest.raises(ValueError, match="source_bond"):
        from_cleavage_results([record])


def test_site_rows_do_not_share_mutable_metadata():
    result = CleavageResult(
        PeptideInput("APAA"), reference_model(),
        (CleavageSite(1, "reported", "fixture"), CleavageSite(2, "reported", "fixture")),
    )
    frame = from_cleavage_results([result])
    frame.at[0, "peptide_input"]["sequence"] = "AAAA"
    assert frame.at[1, "peptide_input"]["sequence"] == result.peptide.sequence
