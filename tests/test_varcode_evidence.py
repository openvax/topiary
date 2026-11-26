"""Regression tests for compact Varcode provenance (#624, #635) and --ensembl-release forwarding (#610)."""

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from topiary.varcode_evidence import _plain_record, varcode_candidate_provenance


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


def test_effect_candidate_provenance_retains_dna_fields_and_mutant_transcript_metadata():
    from varcode import EffectCandidate, MutantTranscript
    from varcode.mutant_transcript import AssembledAllele, ReferenceSegment

    positions = list(range(100000, 100500))
    baseline_mt = MutantTranscript(
        reference_segments=(
            ReferenceSegment(source=AssembledAllele("ATGGTGGCT"), start=0, end=9),
        ),
        cdna_sequence="ATGGTGGCT",
        mutant_protein_sequence="MVA",
        annotator_name="dna",
        evidence={"genomic_positions": positions[:9]},
    )
    mutant_mt = MutantTranscript(
        reference_segments=(
            ReferenceSegment(source=AssembledAllele("ATGGAGGCT"), start=0, end=9),
        ),
        cdna_sequence="ATGGAGGCT",
        mutant_protein_sequence="MEA",
        annotator_name="dna",
        evidence={
            "exon_runs": [[1, 1], [3, 1]],
            "genomic_positions": positions,
            "novel_junctions": (("7", 117149197, 117174329, "+"),),
            "back_junctions": (("7", 55241615, 55269047, "+"),),
            "wildtype_junctions": (("7", 117149197, 117170952, "+"),),
            "mate_search_region": ("1", 1000000, 1024999, "+"),
            "exit_frame": 1,
            "template_segments": (("five_prime_host", "7", 100, 200, "+"),),
        },
    )
    effect = SimpleNamespace(
        baseline_transcript=baseline_mt,
        mutant_transcript=mutant_mt,
        original_protein_sequence="MVA",
        mutant_protein_sequence="MEA",
        mechanism_rank=(1,),
        probability=None,
    )
    candidate = EffectCandidate(
        effect=effect,
        source="varcode.dna",
        evidence={"n_merged_outcomes": 2, "phase_mass": 0.5},
        mechanisms=("exon_skip",),
        phase=(("chr7:117171155 T>C", "cis"),),
        uncertainty=("phasing", "splicing"),
    )
    # Create a back-reference cycle from effect -> candidates to verify omission.
    effect.candidates = (candidate,)

    provenance = varcode_candidate_provenance(candidate)
    assert "effect" not in provenance
    assert json.loads(json.dumps(provenance)) == provenance
    assert provenance["source"] == "varcode.dna"
    assert provenance["mechanisms"] == ["exon_skip"]
    assert provenance["phase"] == [["chr7:117171155 T>C", "cis"]]
    assert provenance["uncertainty"] == ["phasing", "splicing"]
    assert provenance["ordinal_key"] == [1]
    assert provenance["probability"] is None
    assert provenance["evidence"] == {"n_merged_outcomes": 2, "phase_mass": 0.5}

    baseline = provenance["baseline"]
    assert baseline["cdna_sequence"] == "ATGGTGGCT"
    assert baseline["protein_sequence"] == "MVA"
    assert baseline["is_complete"] is True
    assert baseline["evidence"]["genomic_positions_count"] == 9
    assert "genomic_positions" not in baseline["evidence"]

    mutant = provenance["mutant"]
    assert mutant["cdna_sequence"] == "ATGGAGGCT"
    assert mutant["protein_sequence"] == "MEA"
    assert mutant["is_complete"] is True
    assert mutant["is_structural"] is True
    assert mutant["annotator_name"] == "dna"
    assert mutant["novel_junctions"] == [["7", 117149197, 117174329, "+"]]
    assert mutant["back_junctions"] == [["7", 55241615, 55269047, "+"]]
    assert mutant["wildtype_junctions"] == [["7", 117149197, 117170952, "+"]]
    assert mutant["mate_region"] == ["1", 1000000, 1024999, "+"]
    assert mutant["exit_frame"] == 1
    assert "genomic_positions" not in mutant["evidence"]
    assert mutant["evidence"]["genomic_positions_count"] == 500
    expected_sha = hashlib.sha256(
        json.dumps(positions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert mutant["evidence"]["genomic_positions_sha256"] == expected_sha
    assert mutant["evidence"]["exon_runs"] == [[1, 1], [3, 1]]


def test_effect_candidate_provenance_handles_open_template_and_fast_path():
    from varcode import EffectCandidate, MutantTranscript

    open_mt = MutantTranscript(
        reference_segments=(),
        cdna_sequence=None,
        mutant_protein_sequence=None,
        annotator_name="dna",
        evidence={
            "sequence_status": "open_search_template",
            "wildtype_junctions": (("7", 55241737, 55242414, "+"),),
            "mate_search_region": ("1", 1000000, 1024999, "+"),
            "exit_frame": 2,
        },
    )
    open_effect = SimpleNamespace(
        baseline_transcript=None,
        mutant_transcript=open_mt,
        original_protein_sequence="MVA",
        mutant_protein_sequence=None,
    )
    open_candidate = EffectCandidate(
        effect=open_effect,
        source="varcode.dna",
        mechanisms=("normal",),
        uncertainty=("structural",),
    )
    open_prov = varcode_candidate_provenance(open_candidate)
    assert "ordinal_key" not in open_prov
    assert "probability" not in open_prov
    assert open_prov["uncertainty"] == ["structural"]
    assert open_prov["mutant"]["is_complete"] is False
    assert open_prov["mutant"]["cdna_sequence"] is None
    assert open_prov["mutant"]["protein_sequence"] is None
    assert open_prov["mutant"]["wildtype_junctions"] == [["7", 55241737, 55242414, "+"]]
    assert open_prov["mutant"]["mate_region"] == ["1", 1000000, 1024999, "+"]
    assert open_prov["mutant"]["exit_frame"] == 2

    fast_effect = SimpleNamespace(
        baseline_transcript=None,
        mutant_transcript=None,
        original_protein_sequence="MVALK",
        mutant_protein_sequence="MEALK",
        mechanism_rank=(),
        probability=1.0,
    )
    fast_candidate = EffectCandidate(
        effect=fast_effect,
        source="varcode.dna",
        mechanisms=("normal",),
        uncertainty=(),
    )
    fast_prov = varcode_candidate_provenance(fast_candidate)
    assert fast_prov["source"] == "varcode.dna"
    assert fast_prov["ordinal_key"] == []
    assert fast_prov["probability"] == 1.0
    assert fast_prov["baseline"] == {"cdna_sequence": None, "protein_sequence": "MVALK"}
    assert fast_prov["mutant"] == {"cdna_sequence": None, "protein_sequence": "MEALK"}


def test_realized_effect_candidate_provenance_and_unsupported_candidate_type():
    from varcode.effect_hypotheses import (
        ClassifiedOutcome,
        EffectHypothesis,
        RealizedEffectCandidate,
        RealizedTranscriptProduct,
    )

    baseline = RealizedTranscriptProduct(
        transcript=None,
        cdna_sequence="ATGGTGGCT",
        protein_sequence="MVA",
        evidence={"genomic_positions": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    )
    mutant = RealizedTranscriptProduct(
        transcript=None,
        cdna_sequence="ATGGAGGCT",
        protein_sequence="MEA",
        evidence={"genomic_positions": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
    )
    outcome = ClassifiedOutcome(
        hypothesis=EffectHypothesis(),
        baseline=baseline,
        mutant=mutant,
        effect=SimpleNamespace(short_description="p.V2E"),
        change_aa_offset=1,
        change_cdna_offset=4,
    )
    realized = RealizedEffectCandidate(
        effect=outcome.effect,
        outcomes=(outcome,),
        ordinal_key=(0,),
        probability=None,
        phase_mass=1.0,
        first_enumeration_index=0,
    )

    provenance = varcode_candidate_provenance(realized)
    assert provenance["source"] == "varcode.transcript_model"
    assert "effect" not in provenance
    evidence = provenance["evidence"]
    assert evidence["ordinal_key"] == [0]
    assert evidence["probability"] is None
    assert evidence["phase_mass"] == 1.0
    assert evidence["first_enumeration_index"] == 0
    assert evidence["outcomes"][0]["baseline"]["protein_sequence"] == "MVA"
    assert evidence["outcomes"][0]["mutant"]["protein_sequence"] == "MEA"
    assert evidence["outcomes"][0]["mutant"]["evidence"]["genomic_positions_count"] == 9
    assert "genomic_positions" not in evidence["outcomes"][0]["mutant"]["evidence"]

    with pytest.raises(TypeError, match="Unsupported Varcode candidate"):
        varcode_candidate_provenance(object())
