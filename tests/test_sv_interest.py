"""All nominations survive; support stays scoped and aliases never add counts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from topiary import build_sv_interest_report, write_sv_interest_report


def catalogue():
    return dict(targets={
        "UTR": dict(genes=["GENE"], dna_samples=["T2"], adjacency_group_id="same-junction"),
        "INTRON": dict(genes=["GENE"], dna_samples=["T2"], adjacency_group_id="same-junction"),
        "NO_ORF": dict(genes=[], dna_samples=["T1"],
                       rna_evidence=[dict(source_id="T1-long", split_path_templates=3)]),
        "SAME": dict(genes=[], dna_samples=["T2"], gene_expression=[dict(sample_id="T2", tpm=1)]),
        "CROSS": dict(genes=[], dna_samples=["T0"], gene_expression=[dict(sample_id="T2", tpm=10)]),
        "UNKNOWN": dict(genes=[], dna_samples=[], kind="unresolved",
                        rna_evidence=[dict(source_id="not-acquired", split_path_templates=None)]),
    })


def export(event="UTR", *, priority=2, fragments=("a", "b"), source="library", sample="T2",
           reverse=False, relation="breakpoint_junction", complete=True):
    return dict(schema="isovar.sv_rna_orfs.v2", event_id=event, sample_id=sample, source=source, candidates=[dict(
        candidate_id="orf-" + event, amino_acids="MK", nucleotide_sequence="ATGAAATAA",
        ends_with_stop_codon=complete, start_evidence_summary=dict(priority=priority),
        rna_support=dict(fragments=len(fragments), fragment_ids=list(fragments)),
        uncertainty_flags=["rna_strand_unresolved"] + (["reverse_complement_support_only"] if reverse else []),
        occurrences=[dict(junctions=[dict(relation=relation)])])])


def test_interest_report_keeps_unresolved_and_unproductive_candidates(tmp_path):
    cat = catalogue()
    before = deepcopy(cat)
    result = build_sv_interest_report(cat, [export(), export("INTRON", priority=4, reverse=True)])
    assert cat == before
    assert [c["event_id"] for c in result["candidates"]] == ["UTR", "INTRON", "NO_ORF", "SAME", "CROSS", "UNKNOWN"]
    assert all(c["protein_abundance"] is None for c in result["candidates"])
    assert result["candidates"][-1]["rna_evidence"][0]["split_path_templates"] is None
    paths = write_sv_interest_report(result, tmp_path / "report")
    assert json.loads(paths["json"].read_text())["candidates"] == result["candidates"]
    assert paths["fasta"].read_text().count(">") == 1  # Aliases do not multiply proteins.


def test_duplicate_reconstructions_union_membership_and_never_mix_libraries():
    rows = [export(), export(fragments=("b", "c")), export(source="other", fragments=("b",)),
            export("INTRON", priority=4, fragments=("d",), reverse=True)]
    report = build_sv_interest_report(catalogue(), rows + [deepcopy(rows[0])])
    proteins = report["protein_hypotheses"]
    assert len(proteins) == 3
    union, = [r for r in proteins if r["event_id"] == "UTR" and r["source"] == "library"]
    assert union["full_orf_templates"] == 3 and len(union["source_observations"]) == 2
    other, = [r for r in proteins if r["source"] == "other"]
    assert other["full_orf_templates"] == 1 and other["rna_support_rank_within_source"] == 1
    intron, = [r for r in proteins if r["event_id"] == "INTRON"]
    assert intron["rna_support_rank_within_source"] == 2
    plain = build_sv_interest_report(catalogue(), [export("INTRON", priority=4)])
    reverse = build_sv_interest_report(catalogue(), [export("INTRON", priority=4, reverse=True)])
    assert plain["protein_hypotheses"][0]["evidence_priority"] == reverse["protein_hypotheses"][0]["evidence_priority"]


def test_ambiguous_event_and_start_assessments_remain_explicit():
    observations = [export(priority=2), export(priority=4)]
    result = build_sv_interest_report(catalogue(), observations)
    row, = result["protein_hypotheses"]
    assert row["start_priority"] is None and row["evidence_priority"] == 50
    result = build_sv_interest_report(catalogue(), [export(relation="splice_ambiguous_event_junction")])
    assert result["protein_hypotheses"][0]["evidence_priority"] == 75


def test_explicit_aliases_and_invalid_evidence():
    old = export("legacy-event")
    result = build_sv_interest_report(catalogue(), [old], event_aliases={"legacy-event": "UTR"})
    assert result["protein_hypotheses"][0]["event_id"] == "UTR"
    assert result["protein_hypotheses"][0]["source_observations"][0]["original_event_id"] == "legacy-event"
    with pytest.raises(ValueError, match="absent"):
        build_sv_interest_report(catalogue(), [old])
    bad = export()
    bad["candidates"][0]["rna_support"]["fragments"] = 10
    with pytest.raises(ValueError, match="membership"):
        build_sv_interest_report(catalogue(), [bad])
    bad = export()
    bad["candidates"][0]["amino_acids"] = "MKK"
    with pytest.raises(ValueError, match="Conflicting"):
        build_sv_interest_report(catalogue(), [export(), bad])
    bad = export()
    bad["reference_name"] = "GRCh37"
    with pytest.raises(ValueError, match="assemblies"):
        build_sv_interest_report(dict(catalogue(), assembly="GRCh38"), [bad])


def test_annotated_frame_never_substitutes_junction_count_for_full_orf_support(tmp_path):
    comparison = dict(schema="isovar.sv_rna_prediction_comparison.v1", event_id="UTR", sample_id="T2",
                      rna_source="library", hypotheses=[dict(kind="annotated_frame", hypothesis_id="frame",
                      amino_acids="MKK", nucleotide_sequence="ATGAAAAAA", complete_candidate=False,
                      paths={"path": dict(linkage="event_compatible_junction",
                             candidate=dict(departure_relations=["event_compatible_junction"]))})])
    report = build_sv_interest_report(catalogue(), comparisons=[comparison])
    row, = report["protein_hypotheses"]
    assert row["evidence_priority"] == 10
    assert row["full_orf_templates"] is None and row["rna_support_rank_within_source"] is None
    assert not row["complete"]
    write_sv_interest_report(report, tmp_path / "annotated")


def test_published_current_exports_keep_labels_lineage_and_annotated_frames(tmp_path):
    root = Path(__file__).parent / "data" / "isovar_exports"
    orfs = json.loads((root / "orfs-v4.json").read_text())
    comparison = json.loads((root / "comparison-v3.json").read_text())
    cat = dict(targets={orfs["event_id"]: {}, comparison["event_id"]: {}})
    before = deepcopy(orfs)
    report = build_sv_interest_report(cat, [orfs, deepcopy(orfs)], comparisons=[comparison])
    assert orfs == before
    rows = {r["kind"]: r for r in report["protein_hypotheses"]}
    orf = rows["exploratory_orf"]
    assert orf["full_orf_templates"] == 1
    observation, = orf["source_observations"]
    assert observation["candidate"] == orfs["candidates"][0]
    support = observation["rna_support"]
    assert (support["reads"], support["fragments"], support["umis"], support["cells"]) == (1, 1, 1, 1)
    assert support["umis_complete"] is support["cells_complete"] is True
    assert support["read_lineage"] == orfs["candidates"][0]["rna_support"]["read_lineage"]
    assert orf["translation_observed"] is False
    annotated = rows["annotated_frame"]
    assert annotated["evidence_priority"] == 10
    assert annotated["full_orf_templates"] is None
    paths = write_sv_interest_report(report, tmp_path / "current")
    assert json.loads(paths["json"].read_text()) == json.loads(json.dumps(report))


@pytest.mark.parametrize("schema", ["isovar.sv_rna_orfs.v1", "isovar.sv_rna_orfs.v2",
                                    "isovar.sv_rna_orfs.v3", "isovar.sv_rna_orfs.v4"])
def test_orf_versions_keep_the_same_evidence_priority(schema):
    value = export()
    value["schema"] = schema
    if schema.endswith(("v3", "v4")):
        value["candidates"][0]["rna_support"]["evidence_scope"] = [value["sample_id"], value["source"]]
    row, = build_sv_interest_report(catalogue(), [value])["protein_hypotheses"]
    assert row["evidence_priority"] == 30 and row["full_orf_templates"] == 2


@pytest.mark.parametrize("schema", ["isovar.sv_rna_prediction_comparison.v1",
                                    "isovar.sv_rna_prediction_comparison.v2",
                                    "isovar.sv_rna_prediction_comparison.v3"])
def test_comparison_versions_keep_annotated_frame_support_unknown(schema):
    root = Path(__file__).parent / "data" / "isovar_exports"
    comparison = json.loads((root / "comparison-v3.json").read_text())
    comparison["schema"] = schema
    row, = build_sv_interest_report(dict(targets={comparison["event_id"]: {}}),
                                   comparisons=[comparison])["protein_hypotheses"]
    assert row["evidence_priority"] == 10 and row["full_orf_templates"] is None
