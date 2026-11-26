"""Regression tests for compact Varcode provenance (#624) and --ensembl-release forwarding (#610)."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd

from topiary.varcode_evidence import _plain_record


def test_plain_record_compacts_genomic_positions_and_preserves_exon_runs():
    positions = list(range(100000, 120000))
    record = {
        "__class__": {"__module__": "varcode.transcript_model", "__name__": "RealizedTranscriptProduct"},
        "cdna_sequence": "ACGT" * 5000,
        "evidence": {
            "exon_runs": [[100000, 110000], [115000, 120000]],
            "genomic_positions": positions,
        },
    }
    plain = json.loads(json.dumps(record), object_hook=_plain_record)
    assert plain["record_type"] == "varcode.transcript_model.RealizedTranscriptProduct"
    evidence = plain["evidence"]
    assert "genomic_positions" not in evidence
    assert evidence["genomic_positions_count"] == 20000
    expected_sha = hashlib.sha256(
        json.dumps(positions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert evidence["genomic_positions_sha256"] == expected_sha
    assert evidence["exon_runs"] == [[100000, 110000], [115000, 120000]]


def test_ensembl_release_reaches_variant_loading_when_genome_omitted(monkeypatch):
    from topiary.cli import args as cli_args

    sentinel_genome = SimpleNamespace(release=93, species=SimpleNamespace(latin_name="homo_sapiens"))
    captured = {}

    monkeypatch.setattr("topiary.sources.ensembl_genome", lambda **kwargs: sentinel_genome)

    def fake_variant_collection(vargs):
        captured["genome"] = vargs.genome
        return [SimpleNamespace(genome=sentinel_genome)]

    monkeypatch.setattr(cli_args, "variant_collection_from_args", fake_variant_collection)
    monkeypatch.setattr(cli_args, "expression_data_from_args", lambda _a: None)
    monkeypatch.setattr(cli_args, "predictors_from_args", lambda _a: [SimpleNamespace()])

    class FakePredictor:
        def __init__(self, **_kwargs):
            pass

        def predict_from_variants(self, **_kwargs):
            return pd.DataFrame()

    monkeypatch.setattr(cli_args, "TopiaryPredictor", FakePredictor)

    ns = SimpleNamespace(
        mhc_predictor="netmhcpan",
        mhc_cache_file=None,
        mhc_cache_directory=None,
        vcf=["dummy.vcf"],
        maf=[],
        variant=[],
        json_variants=[],
        protein_change=None,
        genome=None,
        species=None,
        ensembl_release=93,
        peptide_csv=None,
        sequence_csv=None,
        fasta=None,
        peptide_fasta=None,
        ensembl_proteome=False,
        gene_names=None,
        gene_ids=None,
        transcript_ids=None,
        cta=False,
        exclude_fasta=None,
        exclude_ensembl=False,
        exclude_non_cta=False,
        exclude_tissues=None,
        exclude_species=None,
        exclude_ensembl_release=None,
        filter_by=None,
        sort_by=None,
        sort_direction="auto",
        ic50_cutoff=None,
        percentile_cutoff=None,
        padding_around_mutation=None,
        only_novel_epitopes=False,
        skip_variant_errors=False,
        protein_x_policy="skip",
        predict_wt=False,
        cache_miss_report=None,
        gene_expression=None,
        transcript_expression=None,
        variant_expression=None,
    )
    cli_args.predict_epitopes_from_args(ns)
    assert captured["genome"] is sentinel_genome
    # Original args namespace is not mutated
    assert ns.genome is None


def test_varcode_candidate_provenance_reads_varcode13_effect_candidate():
    from varcode import EffectCandidate, MutantTranscript
    from topiary.varcode_evidence import varcode_candidate_provenance

    model = MutantTranscript(
        reference_transcript=None,
        reference_segments=(),
        cdna_sequence=None,
        evidence={
            "novel_junctions": (("7", 55181379, 55191718, "+"),),
            "back_junctions": (("7", 55181379, 55200317, "+"),),
            "mate_search_region": ("7", 55191000, 55195000, "+"),
            "exit_frame": 1,
        },
    )
    effect = SimpleNamespace(
        mutant_transcript=model,
        original_protein_sequence="MAAA",
        mutant_protein_sequence=None,
        transcript=SimpleNamespace(sequence="ATGGCTGCTGCTTAA"),
    )
    candidate = EffectCandidate(
        effect=effect,
        source="varcode.dna",
        evidence={"n_merged_outcomes": 1},
        mechanisms=("exon_skipping",),
        phase=(("chr7 g.1A>G", "cis"),),
        uncertainty=("splice",),
    )
    provenance = varcode_candidate_provenance(candidate)
    assert provenance["source"] == "varcode.dna"
    ev = provenance["evidence"]
    assert ev["mechanisms"] == ["exon_skipping"]
    assert ev["phase"] == [["chr7 g.1A>G", "cis"]]
    assert ev["uncertainty"] == ["splice"]
    outcome = ev["outcomes"][0]
    assert outcome["baseline"]["protein_sequence"] == "MAAA"
    assert outcome["mutant"]["cdna_sequence"] is None
    assert outcome["is_complete"] is False
    assert outcome["novel_junctions"] == [["7", 55181379, 55191718, "+"]]
    assert outcome["back_junctions"] == [["7", 55181379, 55200317, "+"]]
    assert outcome["mate_region"] == ["7", 55191000, 55195000, "+"]
    assert outcome["exit_frame"] == 1
    assert json.loads(json.dumps(provenance)) == provenance
