"""Tests for topiary.sources — Ensembl and PirlyGenes sequence sources."""

from topiary.sources import (
    available_tissues,
    cta_sequences,
    ensembl_proteome,
    non_cta_sequences,
    sequences_from_gene_names,
    sequences_from_gene_ids,
    sequences_from_transcript_ids,
    tissue_expressed_gene_ids,
)


def test_sequences_from_gene_names():
    seqs = sequences_from_gene_names(["BRAF", "TP53"])
    assert len(seqs) == 2
    keys = list(seqs.keys())
    assert any("BRAF" in k for k in keys)
    assert any("TP53" in k for k in keys)
    # BRAF protein should be ~766 aa
    braf_seq = [v for k, v in seqs.items() if "BRAF" in k][0]
    assert len(braf_seq) > 700


def test_sequences_from_gene_names_unknown():
    seqs = sequences_from_gene_names(["BRAF", "NOTAREALGENE_XYZ"])
    assert len(seqs) == 1


def test_sequences_from_gene_ids():
    # BRAF = ENSG00000157764
    seqs = sequences_from_gene_ids(["ENSG00000157764"])
    assert len(seqs) == 1
    assert any("BRAF" in k for k in seqs.keys())


def test_sequences_from_transcript_ids():
    # BRAF-001 = ENST00000496384
    seqs = sequences_from_transcript_ids(["ENST00000496384"])
    assert len(seqs) == 1


def test_cta_sequences():
    seqs = cta_sequences()
    assert len(seqs) > 100  # ~257 CTA genes, some may lack protein
    # All should be non-empty protein sequences
    for seq in seqs.values():
        assert len(seq) > 0


def test_available_tissues():
    tissues = available_tissues()
    assert "heart_muscle" in tissues
    assert "lung" in tissues
    assert "testis" in tissues
    assert len(tissues) >= 40


def test_tissue_expressed_gene_ids():
    gene_ids = tissue_expressed_gene_ids(["testis"], min_ntpm=1.0)
    assert len(gene_ids) > 1000  # many genes expressed in testis


def test_tissue_expressed_gene_ids_strict():
    loose = tissue_expressed_gene_ids(["heart_muscle"], min_ntpm=1.0)
    strict = tissue_expressed_gene_ids(["heart_muscle"], min_ntpm=100.0)
    assert len(strict) < len(loose)
