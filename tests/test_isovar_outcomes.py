"""Negative reconstruction outcomes remain useful, explicit results."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from topiary import describe_isovar_result


def result(**changes):
    fields = dict(
        variant=None, top_protein_sequence=None, predicted_effect=None,
        num_total_reads=12, num_alt_reads=3, num_ref_reads=8, num_other_reads=1,
        num_total_fragments=6, num_alt_fragments=2, num_ref_fragments=3, num_other_fragments=1,
        filter_values={"has_mutant_protein_sequence_from_rna": False},
        passes_all_filters=False,
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("changes,status", [
    ({"num_total_reads": 0, "num_alt_reads": 0}, "no_usable_reads"),
    ({"num_alt_reads": 0}, "no_alt_reads"),
    ({"predicted_effect": SimpleNamespace(modifies_protein_sequence=False)}, "no_predicted_coding_change"),
    ({}, "no_protein_sequence"),
])
def test_negative_outcomes_preserve_counts_and_named_filters(changes, status):
    upstream = result(**changes)
    actual = describe_isovar_result(upstream)
    assert actual["status"] == status
    assert actual["num_total_reads"] == upstream.num_total_reads
    assert actual["num_total_fragments"] == upstream.num_total_fragments
    assert actual["failed_filters"] == ["has_mutant_protein_sequence_from_rna"]
    assert actual["protein_sequence"] is None
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual


@pytest.mark.parametrize("passing", [True, False, np.bool_(True), np.bool_(False)])
def test_filtered_sequences_remain_visible_without_becoming_passing(passing):
    upstream = result(
        top_protein_sequence=SimpleNamespace(amino_acids="KRFHATISF", mutation_start_idx=np.int64(3),
                                            mutation_end_idx=np.int64(4), transcript_ids=["ENST1"],
                                            num_supporting_reads=6, num_supporting_fragments=3),
        filter_values={"min_ratio_alt_to_other_fragments": passing}, passes_all_filters=passing)
    actual = describe_isovar_result(upstream)
    assert actual["status"] == ("passing" if passing else "filtered")
    assert actual["protein_sequence"] == "KRFHATISF"
    assert (actual["mutation_start"], actual["mutation_end"]) == (3, 4)
    assert actual["failed_filters"] == ([] if passing else ["min_ratio_alt_to_other_fragments"])
    assert (actual["protein_supporting_reads"], actual["protein_supporting_fragments"]) == (6, 3)
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual


def test_absent_counts_are_not_reported_as_zero_coverage():
    actual = describe_isovar_result(SimpleNamespace())
    assert actual["status"] == "no_protein_sequence"
    assert actual["num_total_reads"] is actual["num_alt_reads"] is None
    assert actual["passes_all_filters"] is None
    assert actual["variant"] is None


def test_sequence_without_filter_disposition_is_not_promoted():
    actual = describe_isovar_result(SimpleNamespace(
        top_protein_sequence=SimpleNamespace(amino_acids="SIINFEKLL")))
    assert actual["status"] == "filter_status_unavailable"
    assert actual["protein_sequence"] == "SIINFEKLL"
    assert actual["passes_all_filters"] is None
