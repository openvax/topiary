"""Source-pinned expression and all-locus RNA recount, entirely offline."""

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from .osteosarc_overlay_helpers import ROOT
from .sid_data import sid_read


def test_overlay_fixture_checksums():
    for filename, expected in json.loads((ROOT / "manifest.json").read_text()).items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected


def test_tpm_and_transcript_versions_are_original_source_measurements():
    evidence = pd.read_csv(ROOT / "transcript-evidence.tsv", sep="\t")
    genes = pd.read_csv(ROOT / "source/t2.genes.results", sep="\t").set_index("gene_id")
    transcripts = pd.read_csv(ROOT / "source/t2.isoforms.results", sep="\t").set_index("transcript_id")
    assert len(evidence) == 63
    assert evidence.allele_key.nunique() == 20
    assert evidence.transcript_match.eq("exact_hgvsc_version").all()
    for row in evidence.itertuples(index=False):
        assert row.gene_tpm == genes.loc[row.gene_rsem_id, "TPM"]
        assert row.transcript_tpm == transcripts.loc[row.transcript_rsem_id, "TPM"]
        assert row.transcript_rsem_id == row.hgvsc.split(":")[0]
        assert pd.isna(row.gene_expression) and pd.isna(row.trna_depth)
    assert evidence.transcript_tpm.eq(0).any()  # measured zero, not missing
    assert evidence.transcript_tpm.notna().all()
    assert evidence.gene_tpm.ge(1).any() and evidence.gene_tpm.lt(1).any()


@pytest.mark.isovar
@pytest.mark.parametrize("zero_coverage", [False, True])
def test_recount_all_twenty_alleles_and_rebuild_expression_sidecar_offline(
    tmp_path, monkeypatch, zero_coverage,
):
    import pysam
    import requests
    from varcode import Variant
    from scripts.osteosarc_rna_overlay import build

    def no_network(*args, **kwargs):
        pytest.fail("The pinned RNA recount must not use the network")

    monkeypatch.setattr(requests.sessions.Session, "request", no_network)
    monkeypatch.setattr(Variant, "gene_names", property(no_network))
    shutil.copytree(ROOT / "source", tmp_path / "source")
    shutil.copyfile(ROOT / "acquisition.json", tmp_path / "acquisition.json")
    path = tmp_path / "source/t2-pvac-regions.bam"
    shutil.copyfile(sid_read("osteosarc_rna_overlay/source/t2-pvac-regions.bam"), path)
    if zero_coverage:
        with pysam.AlignmentFile(path) as bam:
            header = bam.header.to_dict()
        # Controlled empty alignment: real contigs, explicitly no observations.
        with pysam.AlignmentFile(path, "wb", header=header):
            pass
    # The acquisition receipt pins the bytes originally acquired; the
    # openvax-v1 export holds the same records, serialized differently.
    acquisition_path = tmp_path / "acquisition.json"
    acquisition = json.loads(acquisition_path.read_text())
    acquisition["alignment"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    acquisition_path.write_text(json.dumps(acquisition))
    pysam.index(str(path))
    build(tmp_path)
    if zero_coverage:
        observed = pd.read_csv(tmp_path / "transcript-evidence.tsv", sep="\t")
        assert observed.depth_reads.eq(0).all()
        assert observed.alt_reads.eq(0).all()
        assert observed.vaf.isna().all()
        assert observed.coverage_status.eq("zero_usable_reads").all()
        assert observed.gene_tpm.gt(0).all()  # TPM cannot substitute for allele support
        return
    for filename in ("allele-evidence.tsv", "transcript-evidence.tsv"):
        expected = pd.read_csv(ROOT / filename, sep="\t")
        actual = pd.read_csv(tmp_path / filename, sep="\t")
        pd.testing.assert_frame_equal(actual, expected)
    counts = pd.read_csv(tmp_path / "allele-evidence.tsv", sep="\t")
    assert counts.depth_reads.eq(counts.ref_reads + counts.alt_reads + counts.other_reads).all()
    np.testing.assert_allclose(counts.vaf, counts.alt_reads / counts.depth_reads, rtol=1e-12)


@pytest.mark.isovar
@pytest.mark.parametrize("chrom,pos,ref,alt,expected", [
    ("chr1", 23362762, "G", "C", (23, 4)),
    ("chr3", 15645183, "T", "G", (10, 10)),
    ("chr17", 51154443, "G", "A", (0, 22)),
])
def test_selected_snv_counts_against_independent_aligned_base_oracle(chrom, pos, ref, alt, expected):
    import pysam

    # Scan the small original BAM without an index and inspect CIGAR-aligned
    # bases directly. No Isovar classification/merging function is called.
    called = {ref: set(), alt: set()}
    with pysam.AlignmentFile(sid_read("osteosarc_rna_overlay/source/t2-pvac-regions.bam")) as bam:
        for read in bam.fetch(until_eof=True):
            if (read.reference_name != chrom or read.is_unmapped or read.is_secondary
                    or read.is_duplicate or read.mapping_quality < 20):
                continue
            for query_pos, reference_pos in read.get_aligned_pairs(matches_only=True):
                if reference_pos == pos - 1:
                    base = read.query_sequence[query_pos]
                    if base in called:
                        called[base].add((read.get_tag("RG") if read.has_tag("RG") else "",
                                          read.query_name, read.flag & 0xC0))
    assert not (called[ref] & called[alt])
    assert (len(called[ref]), len(called[alt])) == expected
    row = pd.read_csv(ROOT / "allele-evidence.tsv", sep="\t").set_index("allele_key").loc[
        f"{chrom}:{pos}:{ref}>{alt}"]
    assert (row.ref_reads, row.alt_reads) == expected
