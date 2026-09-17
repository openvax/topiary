"""Real, previously unpredicted alleles; no network or predicted oracle."""

import pytest


@pytest.mark.isovar
@pytest.mark.parametrize("gene,sample,default_counts,primary_counts", [
    ("GLIS3", "T1-ONT-dedup", (2, 0, 0), (2, 0, 0)),
    ("GLIS3", "T2-ONT-dedup", (1, 0, 1), (1, 0, 1)),
    ("GLIS3", "T1-short", (32, 9, 1), (31, 9, 1)),
    ("GLIS3", "T2-short", (0, 0, 0), (0, 0, 0)),
    ("KTN1", "T1-ONT-dedup", (637, 0, 4), (637, 0, 4)),
    ("KTN1", "T2-ONT-dedup", (1388, 10, 8), (1388, 10, 8)),
    ("KTN1", "T1-short", (843, 0, 0), (842, 0, 0)),
    ("KTN1", "T2-short", (302, 2, 1), (292, 3, 0)),
])
def test_original_indel_records_preserve_sample_and_placement_policy(
    additional_indel_rna, gene, sample, default_counts, primary_counts,
):
    import pysam
    from isovar import ReadCollector
    from isovar.read_identity import fragment_ids

    variants, bams, _ = additional_indel_rna
    for secondary, expected in ((True, default_counts), (False, primary_counts)):
        with pysam.AlignmentFile(bams[gene + "." + sample]) as bam:
            evidence = ReadCollector(use_secondary_alignments=secondary).read_evidence_for_variant(
                variants[gene], bam)
        counts = tuple(len(fragment_ids(getattr(evidence, category + "_reads")))
                       for category in ("ref", "alt", "other"))
        assert counts == expected
