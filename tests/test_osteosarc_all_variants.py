"""Every website entry survives; every literal allele exercises the real workflow."""

from collections import Counter
from functools import lru_cache
import gzip
import json
from pathlib import Path

from .sid_data import sid_data_root

import pandas as pd
import pytest

from scripts.osteosarc_rna_overlay import POLICY, digest, panel
from scripts.osteosarc_variant_audit import (
    audit_variant, check_mutation_windows, reference_genome, variant_inventory,
)


ROOT = sid_data_root("osteosarc_all_variants")


def records():
    return json.loads((ROOT / "checked-inventory.json").read_text())["variants"]


@lru_cache(maxsize=None)
def pinned_outcomes():
    """The pinned outcome per variant ID, parsed once for every case."""
    return {r["input"]["variant_id"]: r for r in json.loads((ROOT / "expected.json").read_text())}


@pytest.fixture(scope="module")
def all_variant_reference(tmp_path_factory):
    from .osteosarc_all_helpers import reference_models

    return reference_genome(ROOT, tmp_path_factory.mktemp("all-variant-reference")), reference_models(ROOT / "reference")


def test_all_original_assets_and_inventory_membership():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        assert digest(ROOT / name) == checksum
    with gzip.open(ROOT / "source/variants.html.gz", "rt") as handle:
        html = handle.read()
    with gzip.open(ROOT / "source/vafs.tsv.gz", "rt") as handle:
        vafs = handle.read()
    website = variant_inventory(html, vafs)
    assert len(website) == 182
    assert sum(r["vaccine_count"] > 0 for r in website) == 44
    # The pinned inventory is exactly what the parser derives from the pinned
    # sources — every allele, candidate list and status, not merely every ID —
    # followed only by the named additional candidates.
    pinned = json.loads((ROOT / "inventory.json").read_text())["variants"]
    assert pinned[:len(website)] == json.loads(json.dumps(website))
    assert {r.get("membership") for r in pinned[len(website):]} == {"additional_candidate_report"}
    # Reference checking only adds its evidence, and can only demote a ready
    # allele to a mismatch; it never rewrites or drops an input.
    current = records()
    assert len(current) == len(pinned) == len({r["variant_id"] for r in current}) == 184
    for before, after in zip(pinned, current):
        assert set(after) - set(before) <= {"reference_check", "observed_reference"}
        assert {k: after[k] for k in before if k != "input_status"} == {
            k: v for k, v in before.items() if k != "input_status"}
        assert after["input_status"] in {before["input_status"], "reference_mismatch"}
    assert Counter(r["input_status"] for r in current) == {
        "ready": 174, "non_literal_allele": 7, "missing_literal_allele": 3}
    assert {"GRCh38:" + key for key in panel()[1].allele_key} <= {r.get("allele_key") for r in current}
    assert all(r["observed_reference"] == r["ref"] for r in current if r["input_status"] == "ready")


@pytest.mark.parametrize("entry", [r for r in records() if r["input_status"] != "ready"], ids=lambda r: r["variant_id"])
def test_incomplete_alleles_have_no_invented_rna_outcome(entry):
    expected = pinned_outcomes()[entry["variant_id"]]
    assert expected["status"] == entry["input_status"]
    assert expected["rna"] is None
    assert "allele_key" not in entry


@pytest.mark.parametrize("strand", ["+", "-"])
@pytest.mark.parametrize("exons", [[(101, 110)], [(101, 105), (201, 210)]], ids=["one-exon", "spliced"])
def test_the_translation_oracle_places_edits_by_varcodes_convention(strand, exons):
    """Independent of the corpus, which has no insertion that reaches protein.

    Varcode anchors an insertion on the base before it, so ``G>GTT`` at 105
    inserts between 105 and 106; in transcript order that is after the
    earlier of the two. Substitutions start at their own base.
    """
    from varcode import Variant
    from .osteosarc_all_helpers import edit_offset
    from .osteosarc_helpers import transcript_offset

    def at(position):
        return transcript_offset(exons, strand, position)

    insertion = Variant("1", 104, "G", "GTT", ensembl=None)
    substitution = Variant("1", 104, "G", "T", ensembl=None)
    deletion = Variant("1", 103, "GAC", "G", ensembl=None)

    assert edit_offset(exons, strand, insertion) == max(at(104), at(105))
    assert edit_offset(exons, strand, substitution) == at(104)
    assert edit_offset(exons, strand, deletion) == min(at(104), at(105))
    assert at(105) - at(104) == (1 if strand == "+" else -1)


@pytest.mark.isovar
@pytest.mark.parametrize("entry", [r for r in records() if r["input_status"] == "ready"], ids=lambda r: r["variant_id"])
def test_every_literal_allele_has_a_checked_rna_and_consumer_outcome(entry, all_variant_reference, tmp_path):
    import isovar
    import pysam
    from mhctools import RandomBindingPredictor
    from topiary import (
        TopiaryPredictor, fragments_from_variants,
        read_fragments, read_tsv, write_fragments, TopiaryResult,
    )
    from .osteosarc_all_helpers import validate_rna_protein
    from .test_twin_conformance import ISOVAR_RESULT_TWINS

    genome, models = all_variant_reference
    variant = audit_variant(entry, genome)
    bam_path = ROOT / "source/t2-all-variant-regions.bam"
    options = dict(read_collector=isovar.ReadCollector(**POLICY),
                   protein_sequence_creator=isovar.ProteinSequenceCreator(
                       protein_context_peptide_length=25, variant_sequence_assembly=True))
    with pysam.AlignmentFile(bam_path) as bam:
        result, = isovar.run_isovar([variant], bam, **options)
    describe, adapt = ISOVAR_RESULT_TWINS
    observed = describe(result)
    diagnostic_fragment = adapt(result)
    assert bool(diagnostic_fragment) == bool(observed["protein_sequence"])
    if diagnostic_fragment is not None:
        assert diagnostic_fragment.sequence == observed["protein_sequence"]
        assert diagnostic_fragment.target_intervals == [(observed["mutation_start"], observed["mutation_end"])]
        assert diagnostic_fragment.n_rna_alt_reads == observed["num_alt_reads"]
        assert diagnostic_fragment.n_rna_alt_fragments == observed["num_alt_fragments"]
        assert diagnostic_fragment.transcript_id == min(observed["transcript_ids"], default=None)
    # Pinned counts are evidence produced by one Isovar release. A different
    # installed release can legitimately change them; say so rather than
    # letting an upstream change read as a Topiary regression.
    run = json.loads((ROOT / "run.json").read_text())
    pinned = pinned_outcomes()[entry["variant_id"]]["rna"]
    # Supporting transcripts are an unordered set upstream; outcomes sort them.
    assert observed == dict(pinned, transcript_ids=sorted(pinned["transcript_ids"])), (
        f"Differs from the outcome pinned with Isovar {run['isovar_version']} (installed "
        f"{isovar.__version__}). If the installed release changed this evidence, re-run the "
        "audit and re-pin; otherwise Topiary regressed.")
    validate_rna_protein(result, models)
    # Both doors receive the SAME policy, including the normal result filters.
    with pysam.AlignmentFile(bam_path) as bam:
        fragments = fragments_from_variants([variant], bam, **options)
    adapted = diagnostic_fragment if observed["status"] == "passing" else None
    assert bool(fragments) == (adapted is not None) == (observed["status"] == "passing")
    if not fragments:
        assert observed["failed_filters"] or observed["protein_sequence"] is None
        return
    fragment, = fragments
    assert fragment.sequence == adapted.sequence == observed["protein_sequence"]
    assert fragment.target_intervals == adapted.target_intervals
    assert fragment.n_rna_alt_reads == observed["num_alt_reads"]
    assert fragment.n_rna_alt_fragments == observed["num_alt_fragments"]
    saved = tmp_path / "fragment.tsv"
    write_fragments(fragments, saved)
    restored, = read_fragments(saved)
    assert restored.to_dict() == fragment.to_dict()
    # Synthetic numbers test transport/selection, never binding accuracy.
    predictor = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*01:01"], default_peptide_lengths=[9]), only_novel_epitopes=True)
    frame = predictor.predict_from_fragments([restored])
    assert len(frame) > 0
    check_mutation_windows(frame, [restored])
    output = tmp_path / "predictions.tsv"
    TopiaryResult(frame).to_tsv(output)
    reread = read_tsv(output)
    pd.testing.assert_series_equal(reread.df.peptide.reset_index(drop=True), frame.peptide.reset_index(drop=True))
    assert set(reread.filter_by("n_rna_alt_reads >= 1").df.peptide) == set(frame.peptide)
    assert len(reread.filter_by("n_rna_alt_reads > 1000000")) == 0
