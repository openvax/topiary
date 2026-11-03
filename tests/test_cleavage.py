"""Native site evidence retains its meaning through both input forms."""

from dataclasses import replace

import pandas as pd
import pytest
from mhctools import CleavageModel, CleavageResult, CleavageSite, PeptideInput

from topiary import (
    cleavage_interval_coverage, from_cleavage_results, read_tsv,
    select_cleavage_sites, to_tsv,
)
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


def scored(sequence, source_start, bonds, occurrence="a", model=None):
    peptide = PeptideInput(
        sequence, occurrence_id=occurrence, source_sequence_name="src",
        source_start=source_start)
    sites = tuple(CleavageSite(b, "reported", "fixture") for b in bonds)
    return CleavageResult(peptide, model or reference_model(), sites)


def test_interval_excludes_its_own_boundaries():
    table = from_cleavage_results([scored("APAAKLMN", 10, range(1, 8))])
    chosen = select_cleavage_sites(table, 12, 15)
    assert list(chosen.source_bond) == [13, 14]
    assert list(chosen.columns) == list(table.columns)
    assert select_cleavage_sites(table, 12, 13).empty


def test_coverage_differs_between_short_and_full_context():
    full = from_cleavage_results([scored("A" * 20, 10, range(1, 20))])
    short = from_cleavage_results([scored("A" * 8, 10, range(1, 8))])
    (full_row,) = cleavage_interval_coverage(full, 10, 30).itertuples()
    (short_row,) = cleavage_interval_coverage(short, 10, 30).itertuples()
    assert (full_row.n_internal_bonds, full_row.n_assessed) == (19, 19)
    assert full_row.fraction_assessed == 1.0
    assert (short_row.n_assessed, short_row.n_unassessed) == (7, 12)
    assert short_row.fraction_assessed == pytest.approx(7 / 19)
    # Unreached bonds are absent evidence, not rows carrying a score.
    assert set(select_cleavage_sites(short, 10, 30).source_bond) == set(range(11, 18))


def test_repeated_occurrences_must_be_chosen_explicitly():
    table = from_cleavage_results([
        scored("APAA", 0, [2], occurrence="first"),
        scored("APAA", 50, [2], occurrence="second"),
    ])
    with pytest.raises(ValueError, match="2 source occurrences"):
        select_cleavage_sites(table, 0, 60)
    with pytest.raises(ValueError, match="2 source occurrences"):
        cleavage_interval_coverage(table, 0, 60)
    first = select_cleavage_sites(table, 0, 60, occurrence_id="first")
    second = select_cleavage_sites(table, 0, 60, occurrence_id="second")
    assert (list(first.source_bond), list(second.source_bond)) == ([2], [52])


def test_repeated_results_for_one_bond_count_once():
    table = from_cleavage_results([scored("APAA", 0, [2]), scored("APAA", 0, [2])])
    (row,) = cleavage_interval_coverage(table, 0, 4).itertuples()
    assert (row.n_internal_bonds, row.n_assessed) == (3, 1)
    assert len(select_cleavage_sites(table, 0, 4)) == 2


def test_unsupported_model_is_visible_as_zero_coverage():
    other = replace(reference_model(), name="other-model")
    unsupported = replace(
        scored("APAA", 0, [], model=other), unsupported_reason="out of domain")
    table = from_cleavage_results([scored("APAA", 0, [1, 2, 3]), unsupported])
    coverage = cleavage_interval_coverage(table, 0, 4).set_index(
        "prediction_method_name")
    assert coverage.n_assessed.to_dict() == {"test-reference": 3, "other-model": 0}
    assert select_cleavage_sites(table, 0, 4).prediction_method_name.unique() == [
        "test-reference"]


def test_selection_survives_a_tsv_round_trip(tmp_path):
    table = from_cleavage_results([scored("APAAKLMN", 10, range(1, 8))])
    to_tsv(table, tmp_path / "sites.tsv")
    restored = read_tsv(tmp_path / "sites.tsv").df.drop(columns="source")
    before = select_cleavage_sites(table, 11, 16)
    after = select_cleavage_sites(restored, 11, 16)
    assert list(before.source_bond) == list(after.source_bond) == [12, 13, 14, 15]
    assert list(before.status) == list(after.status)
    assert cleavage_interval_coverage(table, 11, 16).fillna(-1).equals(
        cleavage_interval_coverage(restored, 11, 16).fillna(-1))


def test_interval_arguments_are_validated():
    table = from_cleavage_results([scored("APAA", 0, [2])])
    with pytest.raises(ValueError, match="greater than start"):
        select_cleavage_sites(table, 3, 3)
    with pytest.raises(TypeError, match="integer"):
        select_cleavage_sites(table, True, 3)
    with pytest.raises(TypeError, match="integer"):
        cleavage_interval_coverage(table, 0, 3.0)
    with pytest.raises(ValueError, match="from_cleavage_results"):
        select_cleavage_sites(pd.DataFrame({"bond": [1]}), 0, 3)
