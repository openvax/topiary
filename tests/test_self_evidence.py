"""All-match reference searches preserve origins, scope and evidence facts."""

import pandas as pd
import pytest
import numpy as np

from topiary import SelfProteome, match_self_peptides, self_matches_in_windows, read_tsv
from .test_twin_conformance import SELF_MATCH_TWINS, SELF_WINDOW_TWINS


def reference():
    return SelfProteome.from_peptides({"CTA": "SIINFEKL", "healthy": "SIINFEKL",
                                      "paralog": "SIINFEKM"}, peptide_lengths=[8])


def observations():
    return [dict(evidence_id="ms-confirmed", peptide="SIINFEKL", source="synthetic-ms", tissue="heart",
                 evidence_kind="observed", allele_assignment="confirmed", allele="A0201"),
            dict(evidence_id="ms-predicted", peptide="SIINFEKL", source="synthetic-ms", tissue="liver",
                 evidence_kind="observed", allele_assignment="predicted", allele="B0702"),
            dict(evidence_id="unresolved", peptide="SIINFEKM", source="synthetic-ms", tissue="lung",
                 evidence_kind="observed", allele_assignment="unknown", allele_set="A0201,B0702")]


def predictions():
    return [dict(peptide="SIINFEKL", allele="A0201", kind="pMHC_affinity", value=value,
                 prediction_method_name="synthetic", predictor_version=None) for value in [0., 50.]]


@pytest.mark.parametrize("search", SELF_MATCH_TWINS)
def test_all_origins_and_candidate_sequences_are_retained(search):
    result = search(reference(), ["SIINFEKL", "SIINFEKL"], max_mismatches=1, excluded_gene_ids={"CTA"})
    assert result.df.self_peptide.tolist() == ["SIINFEKL", "SIINFEKL", "SIINFEKM"] * 2
    assert result.df.self_gene_id.tolist() == ["CTA", "healthy", "paralog"] * 2
    assert result.df.self_in_scope.tolist() == [False, True, True] * 2
    assert result.df.self_mismatches.tolist() == [0, 0, 1] * 2
    assert result.df.query_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert result.extra["self_search"]["include_indels"] is False
    assert result.df.self_match_id.nunique() == 3


@pytest.mark.parametrize("search", SELF_WINDOW_TWINS)
def test_window_and_direct_exact_queries_agree(search):
    result = search(reference(), ["SIINFEKL", "AAAAAAAA", "SIINFEKL"], excluded_gene_ids={"CTA"})
    assert result.df.self_gene_id.tolist() == ["CTA", "healthy", None, "CTA", "healthy"]
    assert result.df.self_search_status.tolist() == ["matched", "matched", "no_match_in_scope", "matched", "matched"]
    assert result.df.query_index.tolist() == [0, 0, 1, 2, 2]


def test_every_window_position_and_explicit_alleles_survive():
    result = self_matches_in_windows({"repeated": "SIINFEKLSIINFEKL", "short": "AC"}, reference(),
                                     peptide_lengths=[8, 9], alleles={"repeated": ["A0201", "B0702"]})
    matched = result.df[result.df.self_peptide.notna()]
    assert matched.peptide_offset.tolist() == [0] * 4 + [8] * 4
    assert set(matched.allele) == {"HLA-A*02:01", "HLA-B*07:02"}
    unavailable = result.df[result.df.peptide.str.len().eq(9)]
    assert unavailable.self_search_reason.eq("reference_length_unavailable").all()
    assert result.extra["self_window_coverage"][-1] == dict(
        window_id="short", peptide_length=9, n_occurrences=0, reason="window_shorter_than_peptide_length")


def test_observed_presented_and_predicted_facts_do_not_become_recognition():
    obs, pred = observations(), predictions()
    before = repr((obs, pred))
    result = match_self_peptides(reference(), ["SIINFEKL", "SIINFEKL", "SIINFEKM", "SIINFEKL"],
                                 alleles=["A0201", "B0702", "A0201", None], observations=obs, predictions=pred)
    by_query = result.df.groupby("query_index", sort=False).first()
    assert by_query.self_observed.tolist() == [True] * 4
    assert bool(by_query.self_observed_confirmed_same_allele.iloc[0])
    assert not bool(by_query.self_observed_confirmed_same_allele.iloc[1])
    assert bool(by_query.self_observed_predicted_same_allele.iloc[1])
    # A candidate genotype does not establish restriction to this allele.
    assert not bool(by_query.self_observed_confirmed_same_allele.iloc[2])
    assert pd.isna(by_query.self_observed_confirmed_same_allele.iloc[3])
    assert by_query.self_prediction_status.tolist() == [
        "reported_for_allele", "not_reported_for_allele", "not_reported_for_allele", "query_allele_unknown"]
    assert [row["value"] for row in by_query.self_same_allele_predictions.iloc[0]] == [0., 50.]
    assert by_query.self_same_allele_predictions.iloc[1] == []
    assert all(row["predictor_version"] is None for row in by_query.self_predictions.iloc[0])
    assert "gene_id" not in by_query.self_observations.iloc[0][0]  # no inferred gene attribution
    assert "recognition" not in " ".join(result.df.columns)
    assert repr((obs, pred)) == before


def test_gene_attributed_observations_do_not_leak_to_another_origin():
    obs = [dict(observations()[0], gene_id="healthy")]
    result = match_self_peptides(reference(), ["SIINFEKL"], alleles=["A0201"], observations=obs)
    assert result.df.self_observation_status.tolist() == ["not_reported", "reported"]
    assert pd.isna(result.df.self_observed.iloc[0])


def test_haplotype_presenter_is_not_same_allele_binding_evidence():
    pred = [dict(predictions()[0], kind="pMHC_presentation", allele_set="A0201,B0702",
                 prediction_mhc_dependence="haplotype")]
    result = match_self_peptides(reference(), ["SIINFEKL"], alleles=["A0201"], predictions=pred)
    assert result.df.self_prediction_status.eq("not_reported_for_allele").all()
    assert len(result.df.self_predictions.iloc[0]) == 1
    assert result.df.self_predictions.iloc[0][0]["allele_set"] == "HLA-A*02:01,HLA-B*07:02"


def test_mouse_alleles_and_missing_values_are_explicit():
    obs = [dict(observations()[0], allele="H-2-Kb")]
    result = match_self_peptides(reference(), ["SIINFEKL", "AAAAAAAA", "SIINFEKLL", "SIINFEKX"],
                                 alleles=["H-2-K*b", None, None, None], observations=obs, predictions=[])
    assert result.df.self_observed_confirmed_same_allele.iloc[0]
    assert result.df.self_search_status.tolist() == ["matched", "matched", "no_match_in_scope", "unassessed", "unassessed"]
    assert result.df.self_search_reason.iloc[-1] == "query_sequence_unsupported"
    assert result.df.self_observation_status.iloc[-1] == "not_reported"


def test_incomplete_reference_cannot_produce_a_complete_negative():
    ref = SelfProteome.from_peptides({"unknown": "XIINFEKL"}, peptide_lengths=[8])
    for radius in (0, 1):
        result = match_self_peptides(ref, ["SIINFEKL"], max_mismatches=radius)
        assert result.df.self_search_status.tolist() == ["unassessed"]
        assert result.df.self_search_reason.tolist() == ["unsupported_reference_sequences"]
        assert result.extra["self_search"]["reference_coverage"]["8"]["n_unsupported"] == 1


def test_search_finds_a_candidate_beyond_the_first_reference_chunk():
    ref = SelfProteome.from_peptides({"background": "AAAAAAAA", "hit": "SIINFEKL"}, peptide_lengths=[8])
    ref._reference_arrays[8] = np.concatenate([np.repeat(ref._reference_arrays[8][:1], 65536, axis=0),
                                             ref._reference_arrays[8][1:]])
    ref._reference_peptides[8] = ["AAAAAAAA"] * 65536 + ["SIINFEKL"]
    result = match_self_peptides(ref, ["SIINFEKM"], max_mismatches=1)
    assert result.df.self_peptide.tolist() == ["SIINFEKL"]
    assert result.df.self_mismatches.tolist() == [1]


def test_search_evidence_roundtrip_and_identity(tmp_path, pandas_string_inference):
    result = match_self_peptides(reference(), ["SIINFEKL", "SIINFEKM", "AAAAAAAA"], max_mismatches=1,
                                 alleles=["A0201"] * 3, observations=observations(), predictions=predictions())
    path = tmp_path / "self.tsv"
    result.to_tsv(path)
    restored = read_tsv(path)
    # Readers add the input filename as source; every original fact survives.
    assert restored.df.source.eq("self.tsv").all()
    pd.testing.assert_frame_equal(restored.df[result.df.columns].convert_dtypes(),
                                  result.df.convert_dtypes(), check_dtype=False)
    assert restored.extra == result.extra
    changed = match_self_peptides(reference(), ["SIINFEKL"], observations=[dict(observations()[0], tissue="kidney")])
    assert changed.extra["self_search"]["observation_sha256"] != result.extra["self_search"]["observation_sha256"]
    assert match_self_peptides(reference(), [], observations=[]).empty
    assert self_matches_in_windows({}, reference(), peptide_lengths=[8]).empty


@pytest.mark.parametrize("options,match", [
    ({"max_mismatches": -1}, "nonnegative"),
    ({"max_mismatches": True}, "nonnegative"),
    ({"alleles": []}, "one allele"),
    ({"observations": observations() * 2}, "unique"),
    ({"observations": [dict(observations()[0], allele=None)]}, "assigned"),
    ({"observations": [dict(observations()[0], allele_assignment="certain") ]}, "Unknown"),
])
def test_invalid_evidence_and_search_settings_fail(options, match):
    with pytest.raises(ValueError, match=match):
        match_self_peptides(reference(), ["SIINFEKL"], **options)
