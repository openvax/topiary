"""Tests for topiary.sources — Ensembl and PirlyGenes sequence sources."""

import pytest

from topiary import AMINO_ACIDS
from topiary.sources import (
    available_tissues,
    best_protein_transcript,
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


@pytest.mark.pirlygenes
def test_cta_sequences():
    seqs = cta_sequences()
    assert len(seqs) > 100  # ~257 CTA genes, some may lack protein
    # All should be non-empty protein sequences
    for seq in seqs.values():
        assert len(seq) > 0


@pytest.mark.pirlygenes
def test_available_tissues():
    tissues = available_tissues()
    assert "heart_muscle" in tissues
    assert "lung" in tissues
    assert "testis" in tissues
    assert len(tissues) >= 40


@pytest.mark.pirlygenes
def test_tissue_expressed_gene_ids():
    gene_ids = tissue_expressed_gene_ids(["testis"], min_ntpm=1.0)
    assert len(gene_ids) > 1000  # many genes expressed in testis


@pytest.mark.pirlygenes
def test_tissue_expressed_gene_ids_strict():
    loose = tissue_expressed_gene_ids(["heart_muscle"], min_ntpm=1.0)
    strict = tissue_expressed_gene_ids(["heart_muscle"], min_ntpm=100.0)
    assert len(strict) < len(loose)


def test_predictor_with_alleles_and_model_classes():
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor, Affinity

    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor],
        alleles=["A*03:01"],
        filter_by=Affinity <= 500,
    )
    assert len(predictor.models) == 1
    assert "HLA-A*03:01" in predictor.models[0].alleles


def test_predictor_filter_and_sort_by_separate():
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor, Affinity, Presentation

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A*03:01"]),
        filter_by=Affinity <= 500,
        sort_by=[Presentation.score, Affinity.score],
    )
    assert predictor.filter_by is not None
    assert len(predictor.sort_by) == 2


# What release=None actually selects (#382). The docs claimed "the latest
# installed GRCh38 release (93 for human)"; neither half was true, and the
# stale claim caused the CI cache failure fixed by #379/#381.


def test_default_human_release_is_the_one_pyensembl_exposes():
    import pyensembl

    from topiary import ensembl_genome

    # Not "the latest release whose data is installed", and not a number
    # pinned in topiary: whatever the installed pyensembl points at.
    assert ensembl_genome(None, "human") is pyensembl.ensembl_grch38
    # Any spelling pyensembl accepts names the same species.
    assert ensembl_genome(None, "homo_sapiens") is pyensembl.ensembl_grch38


def test_non_human_species_has_no_default_release():
    from topiary.sources import ensembl_genome

    with pytest.raises(ValueError, match="Must specify release"):
        ensembl_genome(None, "mouse")


def test_the_module_docstring_states_the_rule_it_follows():
    """The docstring every public function's `release` entry points at."""
    import topiary.sources as sources

    note = sources.__doc__
    assert "Which Ensembl release" in note
    assert "ensembl_grch38" in note
    # The two claims that were wrong, and must not come back.
    assert "latest installed" not in note
    assert "93" not in note
    for name in ("ensembl_proteome", "sequences_from_gene_names", "cta_sequences",
                 "non_cta_sequences", "tissue_expressed_sequences"):
        assert "release : int, optional" in getattr(sources, name).__doc__, name


# ---------------------------------------------------------------------------
# Representative transcript (#573)
#
# Picking the longest translation handed predictors proteins Ensembl itself
# marks as incomplete: three CTA genes translate to a sequence starting with
# X, which MHCflurry rejects, and `topiary --cta` ended there.
# ---------------------------------------------------------------------------


class FakeTranscript:
    """The transcript attributes best_protein_transcript reads."""

    def __init__(self, transcript_id, protein_sequence, complete=True,
                 biotype="protein_coding"):
        self.id = transcript_id
        self.protein_sequence = protein_sequence
        self.complete = complete
        self.biotype = biotype


class FakeGene:
    def __init__(self, *transcripts):
        self.transcripts = list(transcripts)


def test_best_protein_transcript_prefers_a_complete_translation():
    incomplete = FakeTranscript("ENST2", "X" + "A" * 200, complete=False)
    complete = FakeTranscript("ENST1", "M" + "A" * 100)

    assert best_protein_transcript(FakeGene(incomplete, complete)) is complete


def test_best_protein_transcript_skips_nonsense_mediated_decay():
    decay = FakeTranscript("ENST1", "M" + "A" * 200, biotype="nonsense_mediated_decay")
    coding = FakeTranscript("ENST2", "M" + "A" * 100)

    assert best_protein_transcript(FakeGene(decay, coding)) is coding


def test_best_protein_transcript_takes_the_longest_of_equally_good_ones():
    short = FakeTranscript("ENST1", "M" + "A" * 100)
    long = FakeTranscript("ENST2", "M" + "A" * 200)

    assert best_protein_transcript(FakeGene(short, long)) is long


def test_best_protein_transcript_breaks_ties_by_id():
    """Two transcripts of the same protein must not alternate between runs."""
    first = FakeTranscript("ENST1", "MAAA")
    second = FakeTranscript("ENST2", "MAAA")

    assert best_protein_transcript(FakeGene(second, first)) is first
    assert best_protein_transcript(FakeGene(first, second)) is first


def test_best_protein_transcript_falls_back_when_nothing_is_complete():
    """A GTF annotating no start codons still gets a representative."""
    short = FakeTranscript("ENST1", "MAAA", complete=False)
    long = FakeTranscript("ENST2", "MAAAAAA", complete=False)

    assert best_protein_transcript(FakeGene(short, long)) is long


def test_best_protein_transcript_without_any_translation():
    assert best_protein_transcript(FakeGene(FakeTranscript("ENST1", None))) is None


def test_reported_gene_sequences_use_standard_residues():
    """The three genes from #573, read from the installed Ensembl release."""
    sequences = sequences_from_gene_names(["IZUMO2", "C16orf90", "LLCFC1"])

    assert len(sequences) >= 3
    for name, sequence in sequences.items():
        assert set(sequence).issubset(AMINO_ACIDS), name
        assert sequence.startswith("M"), name
