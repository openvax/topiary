"""Preserve unresolved frame status for actual observed rearrangement RNA."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from .sid_data import sid_data_root

import pytest


ROOT = sid_data_root("osteosarc_rearrangements")


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
    # Original SAM records are the fixture's audit envelope, not Isovar input.
    # Keep them for the provenance assertions below while passing the payload.
    fusion_input = supplied.copy()
    original_records = fusion_input.pop("original_records")
    fusion, refs, reads = fusion_from_dict(fusion_input)
    assert len(fusion.sequence) == length
    assert fusion.sequence[fusion.junction_start:fusion.junction_end] == inserted
    assert len(reads) == entry["selected_paths"] == support
    assert len(original_records) == support
    header = pysam.AlignmentHeader.from_references(
        ["chr6", "chr15"], [170805979, 101991189])
    for original, read in zip(original_records, supplied["reads"]):
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
    assert result["reasons"] == ["no_exact_collinear_annotated_donor"]
    assert result["schema"] == "isovar.fusion_rna.v3"
    assert result["status"] == "unresolved"
    assert len(result["paths"]) == 1
    path = result["paths"][0]
    assert path["sequence"] == fusion.sequence
    assert path["frame_status"] == "unresolved"
    assert path["translations"] == []
    paths = [p for p in manifest["observed_paths"]
             if p["event"] == event and p["source"] == sample + "-ONT-tagged"]
    assert len(paths) == (8 if event == "OTUD7A--FMN1" else support)
    assert len({p["cell_umi"] for p in paths}) == (6 if event == "OTUD7A--FMN1" else support)


@pytest.mark.osteosarc
def test_embedded_original_records_are_their_openvax_v1_members(tmp_path):
    """Each junction read path's SAM records, as the openvax-v1 member named after it.

    The inputs keep their original records as an audit envelope beside
    Isovar's supplied-fusion payload; openvax-v1 holds the same records.
    """
    import osteosarc

    embedded = {}
    for entry in json.loads((ROOT / "manifest.json").read_text())["inputs"]:
        for record in json.loads(gzip.decompress((ROOT / entry["file"]).read_bytes()))["original_records"]:
            read_id = record["sam"].split("\t", 1)[0]
            member = f"topiary/osteosarc_rearrangements/{entry['file']}#{read_id}"
            embedded[member] = Counter([record["sam"], record["partner_sam"]])
    exported = osteosarc.export_bundle(osteosarc.fetch_bundle("openvax-v1"), tmp_path,
                                       members=list(embedded), format="sam")
    assert len(exported) == len(embedded) == 9
    for member, path in exported.items():
        lines = [line for line in Path(path).read_text().splitlines() if not line.startswith("@")]
        assert Counter(lines) == embedded[member], member
