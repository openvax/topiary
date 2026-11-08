"""Gene source selection prefers annotated starts, not just protein length."""

from types import SimpleNamespace

import pytest

from topiary import sources
from topiary import select_protein_transcript


def transcript(identifier, sequence, complete_start):
    return SimpleNamespace(
        id=identifier, protein_sequence=sequence,
        start_codon_complete=complete_start,
    )


@pytest.fixture
def gene_sources(monkeypatch):
    gene = SimpleNamespace(id="GENE_ID", name="GENE", transcripts=[])
    genome = SimpleNamespace(
        genes_by_name=lambda name: [gene],
        gene_by_id=lambda identifier: gene,
        transcript_by_id=lambda identifier: next(
            t for t in gene.transcripts if t.id == identifier),
        transcripts_by_name=lambda name: [
            t for t in gene.transcripts if t.id == name],
    )
    monkeypatch.setattr(sources, "_get_genome", lambda *args: genome)
    monkeypatch.setattr(sources, "cta_gene_ids", lambda: {gene.id})
    monkeypatch.setattr(sources, "tissue_expressed_gene_ids", lambda *a, **k: {gene.id})
    return gene


GENE_LOADERS = [
    lambda: sources.sequences_from_gene_names(["GENE"]),
    lambda: sources.sequences_from_gene_ids(["GENE_ID"]),
    sources.cta_sequences,
    lambda: sources.tissue_expressed_sequences(["testis"]),
]


@pytest.mark.parametrize("load", GENE_LOADERS, ids=["names", "ids", "cta", "tissue"])
@pytest.mark.parametrize("partial_prefix", ["X", "M"])
def test_all_gene_sources_prefer_complete_start(gene_sources, load, partial_prefix):
    gene_sources.transcripts = [
        transcript("partial", partial_prefix + "A" * 30, False),
        transcript("short", "MACDEFGHIK", True),
        transcript("complete", "MNPQRSTVWYACDEFGHIK", True),
        transcript("missing", None, True),
    ]
    assert load() == {"GENE|complete": "MNPQRSTVWYACDEFGHIK"}


@pytest.mark.parametrize("load", GENE_LOADERS, ids=["names", "ids", "cta", "tissue"])
def test_incomplete_only_falls_back_with_identity_warning(gene_sources, load, caplog):
    gene_sources.transcripts = [
        transcript("short", "XACDE", False),
        transcript("partial", "XMNPQRSTVWY", False),
    ]
    assert load() == {"GENE|partial": "XMNPQRSTVWY"}
    assert "partial" in caplog.text
    assert "complete annotated start codon" in caplog.text


@pytest.mark.parametrize("load", GENE_LOADERS, ids=["names", "ids", "cta", "tissue"])
def test_gene_without_protein_is_omitted(gene_sources, load):
    gene_sources.transcripts = [
        transcript("missing", None, True), transcript("empty", "", True)]
    assert load() == {}


@pytest.mark.parametrize("source", ["cta", "tissue"])
def test_empty_derived_gene_set_has_no_sequences(monkeypatch, source):
    monkeypatch.setattr(sources, "cta_gene_ids", lambda: set())
    monkeypatch.setattr(sources, "tissue_expressed_gene_ids", lambda *a, **k: set())
    load = sources.cta_sequences if source == "cta" else lambda: sources.tissue_expressed_sequences(["testis"])
    assert load() == {}


@pytest.mark.parametrize("load", [sources.sequences_from_gene_names, sources.sequences_from_gene_ids])
def test_explicit_empty_gene_list_remains_an_input_error(load):
    with pytest.raises(ValueError, match="non-empty list"):
        load([])


def test_explicit_transcript_requests_preserve_partial_sequence(gene_sources):
    partial = transcript("partial", "XACDEFGHIK", False)
    partial.gene = gene_sources
    gene_sources.transcripts = [partial, transcript("complete", "MACDEFGHIK", True)]
    assert sources.sequences_from_transcript_ids(["partial"]) == {
        "GENE|partial": "XACDEFGHIK"}
    assert sources.sequences_from_transcript_names(["partial"]) == {
        "GENE|partial": "XACDEFGHIK"}


def test_public_selector_preserves_original_object_and_tie_order():
    first = transcript("first", "MACDEFGHIK", True)
    second = transcript("second", "MNPQRSTVWY", True)
    partial = transcript("partial", "M" * 40, False)
    assert select_protein_transcript(iter([partial, first, second])) is first
    assert select_protein_transcript([second, first, partial]) is second
    assert select_protein_transcript([partial]) is partial


@pytest.mark.parametrize("transcripts", [[], [transcript("empty", "", True)],
                                        [transcript("missing", None, False)]])
def test_public_selector_without_proteins(transcripts):
    assert select_protein_transcript(transcripts) is None


def test_complete_start_does_not_claim_an_unambiguous_protein():
    # Annotation completeness and predictor residue support are separate facts.
    annotated = transcript("annotated", "MXACDEFGHIK", True)
    partial = transcript("partial", "MACDEFGHIKLMNPQRSTVWY", False)
    assert select_protein_transcript([partial, annotated]) is annotated


@pytest.mark.parametrize("source_flag", ["gene_names", "gene_ids", "cta"])
def test_cli_selection_and_prediction_use_selected_transcript(gene_sources, source_flag):
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor
    from topiary.cli.args import _get_direct_input

    gene_sources.transcripts = [
        transcript("partial", "X" + "A" * 40, False),
        transcript("complete", "MNPQRSTVWYACDEFGHIK", True),
    ]
    value = True if source_flag == "cta" else ["GENE" if source_flag == "gene_names" else "GENE_ID"]
    args = SimpleNamespace(**{source_flag: value})
    sequences, is_peptides = _get_direct_input(args)
    assert not is_peptides
    assert sequences == {"GENE|complete": "MNPQRSTVWYACDEFGHIK"}
    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[9])])
    result = predictor.predict_from_named_sequences(sequences)
    assert len(result) > 0
    assert set(result.source_sequence_name) == {"GENE|complete"}
    for row in result.itertuples():
        assert row.peptide == sequences[row.source_sequence_name][
            row.peptide_offset:row.peptide_offset + len(row.peptide)]
