"""Biological identities must not collapse alternative hypotheses or evidence."""

from copy import deepcopy

import pandas as pd
import pytest

from topiary import (
    combine_sources, evidence_views, normalize_rna_observation,
    reconcile_evidence, union_rna_observations,
)


def rna(**updates):
    return dict(dict(sample_name="p", entity_type="transcript", entity_id="t",
                     quantity="count", unit="reads", value=2,
                     library_id="lib", read_set_id="alignment-1",
                     evidence_unit_ids=["r1", "r2"], method="caller"), **updates)


def biological_rows():
    base = dict(sample_name="p", reference_name="GRCh38", event_id="chr1:1:A:T",
                orf_id="local-1", coding_sequence="ATGGCT", transcript_id="t",
                orf_start=0, orf_end=6, reading_frame=0, orf_completeness="start_to_stop",
                protein_sequence="MA", rna_observations=[rna()],
                transcript_expression=10., starts_at_start_codon=True, ends_with_stop_codon=True)
    return pd.DataFrame([
        base,
        dict(base, orf_id="synonymous", coding_sequence="ATGGCC"),
        dict(base, orf_id="alternative", coding_sequence="ATGGGT", protein_sequence="MG"),
        dict(base, orf_id="partial", coding_sequence=None, protein_sequence=None,
             protein_hypothesis_sequence="MA", orf_completeness="partial", ends_with_stop_codon=None),
    ])


def test_cross_caller_orf_agreement_preserves_synonymous_and_alternative_hypotheses():
    original = biological_rows()
    other = original.iloc[:1].copy()
    other["orf_id"] = "different-local-name"
    other["transcript_expression"] = 99.
    combined = combine_sources({"one": original, "two": other})
    result = reconcile_evidence(combined)
    assert result.df.orf_hypothesis_id.nunique() == 4
    assert result.df.protein_sequence_id.nunique() == 2
    assert result.df.iloc[0].orf_hypothesis_id == result.df.iloc[-1].orf_hypothesis_id
    assert result.df.iloc[0].orf_hypothesis_id != result.df.iloc[1].orf_hypothesis_id
    assert result.df.transcript_expression.tolist() == [10., 10., 10., 10., 99.]
    assert pd.isna(result.df.iloc[3].protein_sequence_id)
    assert result.df.candidate_id.isna().all()
    assert "orf_hypothesis_id" not in combined.df
    pd.testing.assert_frame_equal(reconcile_evidence(result).df, result.df)
    views = evidence_views(result)
    assert len(views["events"]) == 1
    assert len(views["orfs"]) == 4
    assert len(views["proteins"]) == 2
    assert len(views["rna_observations"]) == 2
    assert len(views["links"]) == 5
    assert len(evidence_views(result, source_labels=["two"])["orfs"]) == 1
    assert evidence_views(result, source_labels=[])["orfs"].empty
    with pytest.raises(ValueError, match="Unknown source"):
        evidence_views(result, source_labels=["missing"])


@pytest.mark.parametrize("field,value", [
    ("reference_name", "GRCh37"), ("reference_name", None),
    ("sample_name", "other-patient"), ("transcript_id", "other-locus"),
    ("reading_frame", 1), ("orf_completeness", "partial"),
    ("linked_variants", ["second-variant"]),
])
def test_reference_sample_path_frame_and_completeness_boundaries(field, value):
    first = biological_rows().iloc[:1].drop(columns="rna_observations")
    second = first.copy()
    second[field] = pd.Series([value], dtype=object)
    result = reconcile_evidence(combine_sources({"one": first, "two": second}))
    assert result.df.orf_hypothesis_id.nunique() == 2
    if field in {"reference_name", "sample_name"}:
        assert result.df.biological_event_ids.iloc[0] != result.df.biological_event_ids.iloc[1]


def test_missing_reference_or_coding_path_cannot_prove_cross_source_identity():
    for missing in ("reference_name", "coding_sequence", "transcript_id", "orf_end"):
        rows = biological_rows().iloc[:1].drop(columns=missing)
        result = reconcile_evidence(combine_sources({"one": rows, "two": rows}))
        assert result.df.orf_hypothesis_id.nunique() == 2
        assert result.df.protein_sequence_id.nunique() == 1


def test_local_id_cannot_hide_conflicting_orfs_and_absent_assertions_can_be_filled():
    rows = biological_rows().iloc[:2].copy()
    rows["orf_id"] = "shared"
    with pytest.raises(ValueError, match="Conflicting coding_sequence"):
        reconcile_evidence(combine_sources({"one": rows}))
    rows.loc[1, "coding_sequence"] = None
    result = reconcile_evidence(combine_sources({"one": rows}))
    assert result.df.orf_hypothesis_id.nunique() == 1
    assert pd.isna(result.df.iloc[1].coding_sequence)
    assert result.df.iloc[1].orf_descriptor["coding_sequence"] == "ATGGCT"


def test_occurrences_keep_flanks_loci_and_unknown_geometry():
    row = dict(sample_name="p", peptide="MA", allele="HLA-A*02:01", protein_sequence="MAMA",
               orf_id="one", peptide_start=0, peptide_end=2)
    rows = pd.DataFrame([row, dict(row, peptide_start=2, peptide_end=4),
                         dict(row, orf_id="two"), dict(row, peptide_start=None, peptide_end=None),
                         dict(row, n_flank="", c_flank=""), dict(row, gene_id="other")])
    result = reconcile_evidence(combine_sources({"one": rows}))
    assert result.df.candidate_id.nunique() == 1
    assert result.df.peptide_occurrence_id.nunique() == 6
    assert len(evidence_views(result)["candidates"].iloc[0].source_observations) == 6


@pytest.mark.parametrize("updates,message", [
    ({"peptide_start": 1, "peptide_end": 3}, "supporting protein"),
    ({"peptide_start": 0}, "supplied together"),
    ({"peptide_start": -1, "peptide_end": 1}, "length"),
    ({"peptide_start": .5, "peptide_end": 2.5}, "integers"),
    ({"reading_frame": 3}, "reading_frame"),
    ({"orf_start": 2, "orf_end": 1}, "orf_start"),
    ({"starts_at_start_codon": "yes"}, "boolean"),
    ({"event_ids": ["different"]}, "event_id disagrees"),
    ({"rna_observations": {}}, "rna_observations"),
])
def test_invalid_assertions_fail(updates, message):
    row = biological_rows().iloc[0].to_dict()
    row.update(peptide="MA", **updates)
    with pytest.raises((ValueError, TypeError), match=message):
        reconcile_evidence(combine_sources({"one": pd.DataFrame([row])}))


def test_rna_union_deduplicates_only_known_membership():
    one, two = rna(), rna(evidence_unit_ids=["r2", "r3"], method="second")
    result = union_rna_observations([one, two, one])
    assert result["value"] == 3
    assert result["evidence_unit_ids"] == ["r1", "r2", "r3"]
    assert len(result["observations"]) == 3
    assert union_rna_observations([rna(value=0, evidence_unit_ids=[])])["value"] == 0
    assert one == rna()


@pytest.mark.parametrize("changes", [
    {"quantity": "abundance", "unit": "TPM", "evidence_unit_ids": None},
    {"evidence_unit_ids": None}, {"library_id": "other"}, {"read_set_id": "other"},
    {"sample_name": "other"}, {"entity_type": "gene"}, {"entity_id": "other"}, {"unit": "fragments"},
])
def test_rna_union_refuses_unknown_overlap_or_different_subject(changes):
    with pytest.raises(ValueError):
        union_rna_observations([rna(), rna(**changes)])


@pytest.mark.parametrize("changes", [
    {"value": -1}, {"value": True}, {"value": float("inf")}, {"value": .5},
    {"value": 3}, {"entity_type": "protein"}, {"entity_id": ""}, {"library_id": None},
    {"evidence_unit_ids": ["r1", "r1"]},
])
def test_invalid_rna_measurement(changes):
    with pytest.raises(ValueError):
        normalize_rna_observation(rna(**changes))


def test_rna_unknown_measurement_is_not_zero_and_original_annotations_survive():
    record = rna(value=None, evidence_unit_ids=None, extra={"keep": True})
    copy = deepcopy(record)
    result = normalize_rna_observation(record)
    assert result["value"] is None
    result["extra"]["keep"] = False
    assert record == copy
    with pytest.raises(ValueError, match="sample_name"):
        normalize_rna_observation(record, sample_name="other")
    with pytest.raises(ValueError):
        union_rna_observations([])


def test_empty_reconciliation():
    result = reconcile_evidence(combine_sources({}))
    assert result.empty
    assert all(table.empty for table in evidence_views(result).values())


def test_allele_free_occurrences_link_to_queries_without_fabricating_predictions():
    rows = pd.DataFrame([
        dict(sample_name="p", peptide="SIINFEKL", allele="HLA-A*02:01", value=50.),
        dict(sample_name="p", peptide="SIINFEKL", allele="HLA-B*07:02", value=90.),
        dict(sample_name="p", peptide="SIINFEKL", protein_hypothesis_sequence="AAASIINFEKL", orf_id="local"),
        dict(sample_name="other", peptide="SIINFEKL", protein_hypothesis_sequence="AAASIINFEKL", orf_id="local"),
    ])
    result = reconcile_evidence(combine_sources({"one": rows}))
    relations = evidence_views(result)["candidate_occurrences"]
    # HLA is a query axis: the first two rows share one occurrence. Both
    # queries also link to the allele-free sequence report from this patient.
    assert len(relations) == 4
    assert relations.reported_candidate.sum() == 2
    assert set(relations.peptide_occurrence_id) == set(result.df.iloc[:3].peptide_occurrence_id)
    assert result.df.iloc[2:].candidate_id.isna().all()
    assert result.df.iloc[2:].value.isna().all()
