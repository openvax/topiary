"""Binding axes retain bounded uncertainty, ties and all reference origins."""

import pandas as pd
import pytest

from topiary import SelfProteome, read_tsv
from .test_twin_conformance import SELF_BINDING_TWINS, SELF_MATCH_TWINS


QUERY = "SIINFEKL"
NEAR = "SIINFEKM"
FAR = "SIINFEAM"
MODEL = dict(prediction_method_name="test-binding", predictor_version="1")


def prediction(peptide, rank, allele="A0201", **extra):
    return dict(peptide=peptide, allele=allele, kind="pMHC_affinity", value=100.,
                percentile_rank=rank, score=.5, value_unit="nM", **MODEL) | extra


def reference():
    return SelfProteome.from_peptides({"CTA": NEAR, "healthy": NEAR, "strongest": FAR}, peptide_lengths=[8])


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_nearest_equally_strong_and_strongest_are_distinct_with_all_origins(evidence):
    result = evidence(reference(), [QUERY, QUERY], alleles=["A0201", "B0702"], max_mismatches=2,
                      predictions=[prediction(QUERY, 2), prediction(NEAR, 2), prediction(FAR, .1),
                                   prediction(QUERY, .5, "B0702"), prediction(NEAR, 4, "B0702"),
                                   prediction(FAR, 3, "B0702")], **MODEL)
    first, second = result.df.to_dict("records")
    assert first["self_mimic_peptide"] == NEAR
    assert first["self_strongest_nearby_peptide"] == FAR
    assert [c["self_gene_id"] for c in first["self_mimic_candidates"]] == ["CTA", "healthy"]
    assert first["self_mimic_complete"] and first["self_strongest_nearby_complete"]
    assert second["self_mimic_status"] == "no_match_within_scope"
    assert second["self_strongest_nearby_peptide"] == FAR
    assert len(first["self_nearest_candidates"]) == 3
    assert first["allele"] == "HLA-A*02:01" and second["allele"] == "HLA-B*07:02"


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_missing_query_and_missing_self_have_separate_axis_completeness(evidence):
    records = [prediction(NEAR, .1)]
    row = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=2,
                   predictions=records, **MODEL).df.iloc[0]
    assert row.self_mimic_status == "unassessed"
    assert row.self_strongest_nearby_status == "matched"
    assert not row.self_strongest_nearby_complete
    records.append(prediction(FAR, .3))
    row = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=2,
                   predictions=records, **MODEL).df.iloc[0]
    assert row.self_strongest_nearby_complete
    assert not row.self_mimic_complete


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_tied_sequences_and_excluded_origins_remain_structured(evidence):
    ref = SelfProteome.from_peptides({"cta": "SIINFEKM", "normal": "SIINFEKM",
                                     "other": "SIINFEKA"}, peptide_lengths=[8])
    result = evidence(ref, [QUERY], alleles=["A0201"], max_mismatches=1,
                      predictions=[prediction(p, 1) for p in [QUERY, "SIINFEKM", "SIINFEKA"]],
                      excluded_gene_ids=["cta"], **MODEL)
    row = result.df.iloc[0]
    assert row.self_mimic_peptide is None  # No arbitrary representative of tied sequences.
    assert row.self_mimic_percentile_rank == 1
    assert {c["self_gene_id"] for c in row.self_mimic_candidates} == {"normal", "other"}
    assert {c["self_gene_id"] for c in row.self_nearest_candidates} == {"cta", "normal", "other"}


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
@pytest.mark.parametrize("change", [
    dict(prediction_method_name="other"), dict(predictor_version="2"),
    dict(kind="pMHC_presentation"), dict(allele="B0702"),
    dict(measurement_context={"assay": "other"}),
    dict(prediction_mhc_dependence="haplotype", allele_set="A0201,B0702"),
])
def test_other_measurements_never_supply_selected_coverage(evidence, change):
    result = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                      predictions=[prediction(QUERY, 1), prediction(NEAR, .1, **change)], **MODEL)
    assert result.df.self_mimic_status.tolist() == ["unassessed"]
    assert not result.df.self_mimic_complete.iloc[0]
    assert result.df.self_nearest_candidates.iloc[0][0]["self_predictions"]


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
@pytest.mark.parametrize("extra", [dict(percentile_rank=3), dict(n_flank="A")])
def test_conflicts_do_not_become_complete_predictions(evidence, extra):
    result = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                      predictions=[prediction(QUERY, 1), prediction(NEAR, .1),
                                   prediction(NEAR, .1) | extra], **MODEL)
    candidate = result.df.self_nearest_candidates.iloc[0][0]
    assert candidate["binding_status"] == "ambiguous"
    assert result.df.self_mimic_status.tolist() == ["unassessed"]


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_explicit_metric_context_and_nonhuman_allele(evidence):
    records = [prediction(QUERY, 1, "H-2-Kb", score=.4, measurement_context={"assay": "x"}),
               prediction(NEAR, 99, "H-2-Kb", score=.8, measurement_context={"assay": "x"})]
    row = evidence(reference(), [QUERY], alleles=["H-2-K*b"], predictions=records,
                   max_mismatches=1, binding_metric="score", lower_is_better=False,
                   measurement_context={"assay": "x"}, **MODEL).df.iloc[0]
    assert row.self_mimic_status == "matched" and row.self_mimic_score == .8
    assert row.allele == "H2-K*b"


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_typed_roundtrip_retains_bounded_policy_and_missing_values(evidence, tmp_path):
    result = evidence(reference(), [QUERY, "AAAAAAAA", QUERY], alleles=["A0201", None, "A0201"],
                      max_mismatches=2, predictions=[prediction(QUERY, 1), prediction(NEAR, .1)], **MODEL)
    path = tmp_path / "self.tsv"
    result.to_tsv(path)
    restored = read_tsv(path)
    pd.testing.assert_frame_equal(restored.df[result.df.columns].convert_dtypes(),
                                  result.df.convert_dtypes(), check_dtype=False)
    assert restored.extra == result.extra
    assert not restored.df.self_mimic_complete.any()
    empty = evidence(reference(), [], alleles=[], predictions=[], **MODEL)
    assert empty.empty and list(empty.df) == list(result.df)


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
@pytest.mark.parametrize("options", [
    dict(predictor_version="None"), dict(prediction_method_name=""),
    dict(binding_metric="value"), dict(binding_metric="score"),
    dict(binding_metric="value", lower_is_better=True),
    dict(binding_metric="unsupported"), dict(lower_is_better=False),
    dict(kind="antigen_processing"),
])
def test_comparison_requires_explicit_unambiguous_policy(evidence, options):
    with pytest.raises(ValueError):
        evidence(reference(), [QUERY], alleles=["A0201"], predictions=[], **(MODEL | options))


@pytest.mark.parametrize("search", SELF_MATCH_TWINS)
def test_indel_enumeration_is_bounded_and_retains_all_origins(search):
    ref = SelfProteome.from_peptides({"two_deletions": "SINFEL", "one_deletion": "SINFEKL",
                                     "same_length": NEAR, "inserted": "SIINAFEAKL",
                                     "shared": "SINFEL"}, peptide_lengths=[6, 7, 8, 9, 10])
    result = search(ref, [QUERY], max_edits=2)
    matched = result.df
    assert {"SINFEL", "SINFEKL", NEAR, "SIINAFEAKL"} <= set(matched.self_peptide)
    assert matched.loc[matched.self_peptide.eq("SINFEL"), "self_gene_id"].tolist() == ["two_deletions", "shared"]
    assert matched.loc[matched.self_peptide.eq("SINFEL"), "self_edit_distance"].tolist() == [2, 2]
    assert matched.self_search_complete.all()
    assert result.extra["self_search"]["search"] == "bounded_levenshtein"
    assert set(search(ref, [QUERY], max_edits=1).df.self_peptide) < set(matched.self_peptide)


@pytest.mark.parametrize("search", SELF_MATCH_TWINS)
def test_missing_indel_lengths_cannot_produce_complete_negative(search):
    result = search(reference(), ["AAAAAAAA"], max_edits=2)
    assert not result.df.self_search_complete.any()
    assert result.df.self_search_reason.tolist() == ["reference_length_unavailable"]
    with pytest.raises(ValueError, match="not both"):
        search(reference(), [QUERY], max_edits=2, max_mismatches=1)


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
@pytest.mark.parametrize("rank", [None, float('nan'), -1, 101, True])
def test_invalid_rank_is_unknown_not_a_negative(evidence, rank):
    result = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                      predictions=[prediction(QUERY, 1), prediction(NEAR, rank)], **MODEL)
    assert result.df.self_mimic_status.tolist() == ["unassessed"]
    assert not result.df.self_mimic_complete.any()


@pytest.mark.parametrize("evidence", SELF_BINDING_TWINS)
def test_raw_value_comparison_needs_matching_units_and_changes_the_axis(evidence):
    records = [prediction(QUERY, 1, value=5), prediction(NEAR, 2, value=1)]
    rank = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                    predictions=records, **MODEL)
    value = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                     predictions=records, binding_metric="value", value_unit="nM", lower_is_better=True, **MODEL)
    assert rank.df.self_mimic_status.tolist() == ["no_match_within_scope"]
    assert value.df.self_mimic_status.tolist() == ["matched"]
    wrong_units = evidence(reference(), [QUERY], alleles=["A0201"], max_mismatches=1,
                           predictions=records, binding_metric="value", value_unit="uM", lower_is_better=True, **MODEL)
    assert wrong_units.df.self_mimic_status.tolist() == ["unassessed"]
