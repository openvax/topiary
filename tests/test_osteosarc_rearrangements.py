"""Preserve unresolved frame status for actual observed rearrangement RNA."""

import gzip
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parent / "data" / "osteosarc_rearrangements"


@pytest.mark.isovar
@pytest.mark.parametrize("event,sample,length,inserted,support", [
    ("GABBR1--SLC29A1", "T1", 120, "", 2),
    ("GABBR1--SLC29A1", "T2", 120, "", 3),
    ("OTUD7A--FMN1", "T2", 122, "AG", 4),
])
def test_observed_rearrangement_rna_does_not_imply_a_coding_peptide(
    event, sample, length, inserted, support,
):
    import pysam
    from isovar.fusion import fusion_from_dict, reconstruct_fusion

    manifest = json.loads((ROOT / "manifest.json").read_text())
    entry = next(e for e in manifest["inputs"] if (e["event"], e["sample"]) == (event, sample))
    packed = (ROOT / entry["file"]).read_bytes()
    assert hashlib.sha256(packed).hexdigest() == entry["sha256"]
    raw = gzip.decompress(packed)
    assert hashlib.sha256(raw).hexdigest() == entry["input_sha256"]
    supplied = json.loads(raw)
    fusion, refs, reads = fusion_from_dict(supplied)
    assert len(fusion.sequence) == length
    assert fusion.sequence[fusion.junction_start:fusion.junction_end] == inserted
    assert len(reads) == entry["selected_paths"] == support
    assert len(supplied["original_records"]) == support
    header = pysam.AlignmentHeader.from_references(
        ["chr6", "chr15"], [170805979, 101991189])
    for original, read in zip(supplied["original_records"], supplied["reads"]):
        first, partner = [pysam.AlignedSegment.fromstring(original[k], header)
                          for k in ("sam", "partner_sam")]
        assert first.query_name == partner.query_name == read["read_id"]
        assert first.is_supplementary != partner.is_supplementary
        assert first.has_tag("SA") and partner.has_tag("SA")
        assert (first.get_tag("CB"), first.get_tag("UB")) == (
            partner.get_tag("CB"), partner.get_tag("UB"))
        sequence = first.query_sequence
        if original["source_reverse_complement"]:
            sequence = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        offset = read["source_query_start"]
        assert sequence[offset:offset + length] == read["sequence"]
        assert read["sequence"] == fusion.sequence
        pinned = next(p for p in manifest["observed_paths"]
                      if p["read_id"] == read["read_id"] and
                      p["event"] == event and p["source"] == sample + "-ONT-tagged")
        assert all(hashlib.sha256(original[k].encode()).hexdigest() in pinned["sam_sha256"]
                   for k in ("sam", "partner_sam"))
    result = reconstruct_fusion(fusion, refs, reads)
    assert result["status"] == "unresolved_frame"
    assert result["reasons"] == ["no_exact_collinear_annotated_donor"]
    assert result["translations"] == []
    paths = [p for p in manifest["observed_paths"]
             if p["event"] == event and p["source"] == sample + "-ONT-tagged"]
    assert len(paths) == (8 if event == "OTUD7A--FMN1" else support)
    assert len({p["cell_umi"] for p in paths}) == (6 if event == "OTUD7A--FMN1" else support)
