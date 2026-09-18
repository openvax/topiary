"""Every website entry survives; every literal allele exercises the real workflow."""

from collections import Counter
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.osteosarc_rna_overlay import POLICY, digest, panel
from scripts.osteosarc_variant_audit import reference_genome, variant_inventory


ROOT = Path(__file__).parent / "data/osteosarc_all_variants"


def records():
    return json.loads((ROOT / "checked-inventory.json").read_text())["variants"]


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
    current = records()
    assert len(current) == len({r["variant_id"] for r in current}) == 184
    assert {r["variant_id"] for r in website} < {r["variant_id"] for r in current}
    assert Counter(r["input_status"] for r in current) == {
        "ready": 174, "non_literal_allele": 7, "missing_literal_allele": 3}
    assert {"GRCh38:" + key for key in panel()[1].allele_key} <= {r.get("allele_key") for r in current}
    assert all(r["observed_reference"] == r["ref"] for r in current if r["input_status"] == "ready")


@pytest.mark.parametrize("entry", [r for r in records() if r["input_status"] != "ready"], ids=lambda r: r["variant_id"])
def test_incomplete_alleles_have_no_invented_rna_outcome(entry):
    expected = next(r for r in json.loads((ROOT / "expected.json").read_text())
                    if r["input"]["variant_id"] == entry["variant_id"])
    assert expected["status"] == entry["input_status"]
    assert expected["rna"] is None
    assert "allele_key" not in entry


@pytest.mark.isovar
@pytest.mark.parametrize("entry", [r for r in records() if r["input_status"] == "ready"], ids=lambda r: r["variant_id"])
def test_every_literal_allele_has_a_checked_rna_and_consumer_outcome(entry, all_variant_reference, tmp_path):
    import isovar
    import pysam
    from mhctools import RandomBindingPredictor
    from varcode import Variant
    from topiary import (
        TopiaryPredictor, fragments_from_variants,
        read_fragments, read_tsv, write_fragments, TopiaryResult,
    )
    from .osteosarc_all_helpers import validate_rna_protein
    from .test_twin_conformance import ISOVAR_RESULT_TWINS

    genome, models = all_variant_reference
    contig = "MT" if entry["chrom"] == "chrM" else entry["chrom"].removeprefix("chr")
    variant = Variant(contig, entry["pos"], entry["ref"], entry["alt"], ensembl=genome)
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
    expected = next(r["rna"] for r in json.loads((ROOT / "expected.json").read_text())
                    if r["input"]["variant_id"] == entry["variant_id"])
    # Supporting transcript IDs are an unordered collection upstream, not a
    # ranked isoform choice. Check exact membership without imposing hash order.
    assert dict(observed, transcript_ids=sorted(observed["transcript_ids"])) == dict(
        expected, transcript_ids=sorted(expected["transcript_ids"]))
    validate_rna_protein(result, models)
    # Both doors receive the SAME policy, including the normal result filters.
    with pysam.AlignmentFile(bam_path) as bam:
        fragments = fragments_from_variants([variant], bam, **options)
    adapted = diagnostic_fragment if result.passes_all_filters else None
    assert bool(fragments) == (adapted is not None) == (observed["status"] == "passing")
    if not fragments:
        assert observed["failed_filters"] or observed["protein_sequence"] is None
        return
    fragment, = fragments
    assert fragment.sequence == adapted.sequence == observed["protein_sequence"]
    assert fragment.target_intervals == adapted.target_intervals
    assert fragment.n_rna_alt_reads == observed["num_alt_reads"]
    assert fragment.n_rna_alt_fragments == observed["num_alt_fragments"]
    saved = tmp_path / "fragment.json"
    write_fragments(fragments, saved)
    restored, = read_fragments(saved)
    assert restored.to_dict() == fragment.to_dict()
    # Synthetic numbers test transport/selection, never binding accuracy.
    predictor = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*01:01"], default_peptide_lengths=[9]), only_novel_epitopes=True)
    frame = predictor.predict_from_fragments([restored])
    assert len(frame) > 0
    assert frame.contains_mutant_residues.all()
    for row in frame.itertuples():
        assert row.peptide == fragment.sequence[row.peptide_offset:row.peptide_offset + 9]
        start, end = fragment.target_intervals[0]
        assert row.peptide_offset < end and row.peptide_offset + 9 > start
    output = tmp_path / "predictions.tsv"
    TopiaryResult(frame).to_tsv(output)
    reread = read_tsv(output)
    pd.testing.assert_series_equal(reread.df.peptide.reset_index(drop=True), frame.peptide.reset_index(drop=True))
    assert set(reread.filter_by("n_rna_alt_reads >= 1").df.peptide) == set(frame.peptide)
    assert len(reread.filter_by("n_rna_alt_reads > 1000000")) == 0
