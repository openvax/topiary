"""Regional annotation gaps must survive the entire audit and report workflow."""

import json
import gzip
from pathlib import Path
import shutil

import pytest

from scripts.osteosarc_rna_overlay import digest, write_json
from scripts.osteosarc_variant_audit import audit, build_reference, report


DATA = Path(__file__).parent / "data/osteosarc_all_variants"


@pytest.mark.isovar
@pytest.mark.parametrize("annotated_later", [True, False], ids=["continue", "entirely-unannotated"])
def test_reference_build_to_report_keeps_annotation_gaps(tmp_path, annotated_later):
    import pysam

    source = tmp_path / "ensembl"
    source.mkdir()
    for kind, filename in (
        ("gtf", "Homo_sapiens.GRCh38.87.gtf.gz"),
        ("cdna", "Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        ("pep", "Homo_sapiens.GRCh38.pep.all.fa.gz"),
    ):
        suffix = ".gz" if kind == "gtf" else ".fa.gz"
        shutil.copyfile(DATA / "reference" / ("reference." + kind + suffix), source / filename)
    fam_source = DATA.parent / "osteosarc_shared/fam157a-ensembl87.gtf"
    assert digest(fam_source) == json.loads(fam_source.with_suffix(".json").read_text())["sha256"]
    assert "\t198153287\t198222513\t" in fam_source.read_text()
    full_gtf = source / "Homo_sapiens.GRCh38.87.gtf.gz"
    full_gtf.write_bytes(gzip.compress(gzip.decompress(full_gtf.read_bytes()) + fam_source.read_bytes(), mtime=0))
    # The recovered allele in #348 is upstream of Ensembl 87's FAM157A
    # interval. Retain its RefSeq assertion as source text, never an invented
    # Ensembl transcript or a claim of zero RNA support.
    fam = dict(variant_id="FAM157A-p_W70_Q71ins_14", gene="FAM157A", chrom="chr3",
               pos=198153259, ref="G", alt="GGCGGCGGCGGCGGCAGCAGCAGCAGCAGCAGCAGCAGCAGCA",
               input_status="ready", assembly="GRCh38",
               source_protein_label="NM_001145248.1:p.W70_Q71ins14",
               source_url="https://osteosarc.com/variant/FAM157A-p_W70_Q71ins_14/")
    invalid = dict(fam, variant_id="missing-contig", chrom="chrNotInBam")
    outside = dict(fam, variant_id="outside-contig", pos=198295560)
    variants = [fam, invalid, outside]
    if annotated_later:
        ntf = next(v for v in json.loads((DATA / "checked-inventory.json").read_text())["variants"]
                   if v["variant_id"] == "DYNC1H1-chr14-101980529")
        variants.append(ntf)
    write_json(tmp_path / "checked-inventory.json", dict(variants=variants))
    alignments = tmp_path / "source"
    alignments.mkdir()
    for suffix in ("", ".bai"):
        filename = "t2-all-variant-regions.bam" + suffix
        shutil.copyfile(DATA / "source" / filename, alignments / filename)
    bam = alignments / "t2-all-variant-regions.bam"
    write_json(alignments / "bam.receipt.json", dict(
        sha256=digest(bam), index_sha256=digest(str(bam) + ".bai")))
    with pysam.AlignmentFile(bam) as handle:
        assert handle.get_reference_length("chr3") == 198295559

    build_reference(tmp_path, source)
    # Real FAM157A annotation existed in the source and was legitimately not
    # selected at this locus; the contig still belongs to the alignment.
    assert b'gene_name "FAM157A"' not in gzip.decompress((tmp_path / "reference/reference.gtf.gz").read_bytes())
    audit(tmp_path)
    report(tmp_path)
    outcomes = [json.loads((tmp_path / "outcomes" / (v["variant_id"] + ".json")).read_text()) for v in variants]
    assert [o["status"] for o in outcomes[:3]] == [
        "no_regional_annotation", "alignment_contig_unavailable", "outside_alignment_contig"]
    assert all(o["rna"] is None for o in outcomes[:3])
    assert outcomes[0]["input"] == fam
    if annotated_later:
        pinned = next(o for o in json.loads((DATA / "expected.json").read_text())
                      if o["input"]["variant_id"] == ntf["variant_id"])
        # The regional reference has a new content identity; evidence and
        # scientific answers must remain identical under that new identity.
        assert {k: v for k, v in outcomes[-1]["rna"].items() if k not in ("variant", "reference_name")} == {
            k: v for k, v in pinned["rna"].items() if k not in ("variant", "reference_name")}
        assert outcomes[-1]["rna"]["num_alt_reads"] > 0
    before = {p.name: p.read_bytes() for p in (tmp_path / "outcomes").iterdir()}
    audit(tmp_path)
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "outcomes").iterdir()}
    text = (tmp_path / "README.md").read_text()
    assert "no_regional_annotation | unavailable" in text
    assert "RNA support is unknown" in text
    assert "Prediction stage has not been run" in text
