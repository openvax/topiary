"""Tests for previously untested CLI helper functions."""

import os
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from topiary.cli.args import _apply_exclusion
from topiary.cli.protein_changes import genome_from_args, protein_change_effects_from_args
from topiary.cli.rna import (
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# _apply_exclusion
# ---------------------------------------------------------------------------


def _make_exclusion_args(**overrides):
    defaults = dict(
        exclude_fasta=None,
        exclude_ensembl=False,
        exclude_non_cta=False,
        exclude_tissues=None,
        exclude_mode="substring",
        ensembl_release=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_apply_exclusion_non_df_returns_input():
    result = _apply_exclusion([1, 2, 3], _make_exclusion_args())
    assert result == [1, 2, 3]


def test_apply_exclusion_empty_df_returns_empty():
    df = pd.DataFrame({"peptide": []})
    result = _apply_exclusion(df, _make_exclusion_args())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_apply_exclusion_no_flags_noop():
    df = pd.DataFrame({"peptide": ["SIINFEKL", "ELAGIGILT"]})
    result = _apply_exclusion(df, _make_exclusion_args())
    assert len(result) == 2


def test_apply_exclusion_with_fasta(tmp_path):
    fasta_path = tmp_path / "ref.fasta"
    fasta_path.write_text(">ref\nMASIINFEKLGGGLLL\n")
    df = pd.DataFrame({"peptide": ["SIINFEKL", "XXXXXXXX"]})
    args = _make_exclusion_args(exclude_fasta=[str(fasta_path)])
    result = _apply_exclusion(df, args)
    assert len(result) == 1
    assert result.iloc[0]["peptide"] == "XXXXXXXX"


def test_apply_exclusion_exact_mode(tmp_path):
    fasta_path = tmp_path / "ref.fasta"
    fasta_path.write_text(">ref\nSIINFEKL\n")
    df = pd.DataFrame({"peptide": ["SIINFEKL", "SIINFEKLX"]})
    args = _make_exclusion_args(
        exclude_fasta=[str(fasta_path)], exclude_mode="exact"
    )
    result = _apply_exclusion(df, args)
    # Exact mode: only "SIINFEKL" matches exactly
    assert "SIINFEKLX" in result["peptide"].values


# ---------------------------------------------------------------------------
# genome_from_args
# ---------------------------------------------------------------------------


def test_genome_from_args_default():
    from pyensembl import ensembl_grch38
    args = SimpleNamespace(genome=None)
    result = genome_from_args(args)
    assert result is ensembl_grch38


def test_genome_from_args_explicit():
    args = SimpleNamespace(genome="GRCh38")
    result = genome_from_args(args)
    assert "38" in str(result.release) or result.reference_name == "GRCh38"


# ---------------------------------------------------------------------------
# transcript choice behind --protein-change
# ---------------------------------------------------------------------------


def protein_change_transcripts(monkeypatch, *transcripts, change="M5K"):
    """Which transcript --protein-change picks out of the given candidates."""
    from topiary import sources

    gene = SimpleNamespace(id="GENE_ID", gene_id="GENE_ID", name="GENE",
                           gene_name="GENE", transcripts=list(transcripts))
    for candidate in transcripts:
        candidate.gene = gene
    genome = SimpleNamespace(
        gene_names=lambda: {"GENE"},
        genes_by_name=lambda name: [gene],
    )
    monkeypatch.setattr(sources, "ensembl_genome", lambda **kwargs: genome)
    args = SimpleNamespace(
        genome=None, species="human", ensembl_release=None,
        protein_change=[("GENE", change)],
    )
    return [effect.transcript for effect in protein_change_effects_from_args(args)]


def candidate_transcript(identifier, sequence, complete=True,
                         biotype="protein_coding"):
    return SimpleNamespace(
        id=identifier, name=identifier, transcript_id=identifier,
        protein_sequence=sequence, sequence="A" * (3 * len(sequence)),
        complete=complete, biotype=biotype, is_protein_coding=True,
    )


def test_protein_change_prefers_a_complete_translation_over_a_longer_one(monkeypatch):
    """A longer partial translation no longer outranks a complete one."""
    partial = candidate_transcript("ENST02", "XACDM" + "G" * 40, complete=False)
    complete = candidate_transcript("ENST01", "ACDEM")
    assert protein_change_transcripts(monkeypatch, partial, complete) == [complete]


def test_protein_change_takes_the_longest_of_equally_complete_translations(monkeypatch):
    short = candidate_transcript("ENST01", "ACDEM")
    long = candidate_transcript("ENST02", "ACDEM" + "G" * 40)
    assert protein_change_transcripts(monkeypatch, short, long) == [long]


def test_protein_change_breaks_ties_by_transcript_id(monkeypatch):
    first = candidate_transcript("ENST01", "ACDEM")
    second = candidate_transcript("ENST02", "ACDEM")
    assert protein_change_transcripts(monkeypatch, second, first) == [first]
    assert protein_change_transcripts(monkeypatch, first, second) == [first]


def test_protein_change_without_a_matching_transcript_yields_no_effect(monkeypatch):
    """The residue named by the change is not the one the transcript has."""
    assert protein_change_transcripts(
        monkeypatch, candidate_transcript("ENST01", "ACDEQ")) == []


# ---------------------------------------------------------------------------
# rna_gene_expression_dict_from_args
# ---------------------------------------------------------------------------


def _make_rna_args(**overrides):
    defaults = dict(
        rna_gene_fpkm_tracking_file=None,
        rna_transcript_fpkm_tracking_file=None,
        rna_transcript_fpkm_gtf_file=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_rna_gene_expression_none():
    result = rna_gene_expression_dict_from_args(_make_rna_args())
    assert result is None


def test_rna_gene_expression_with_file():
    path = os.path.join(DATA_DIR, "genes.fpkm_tracking")
    args = _make_rna_args(rna_gene_fpkm_tracking_file=path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = rna_gene_expression_dict_from_args(args)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rna_gene_expression_deprecation_warning(caplog):
    """A visible notice, not a DeprecationWarning Python hides (#322)."""
    path = os.path.join(DATA_DIR, "genes.fpkm_tracking")
    args = _make_rna_args(rna_gene_fpkm_tracking_file=path)
    with caplog.at_level("WARNING"):
        assert rna_gene_expression_dict_from_args(args)
    assert caplog.messages == [
        "--rna-gene-fpkm-tracking-file is deprecated; "
        "use --gene-expression instead."
    ]


# ---------------------------------------------------------------------------
# rna_transcript_expression_dict_from_args
# ---------------------------------------------------------------------------


def test_rna_transcript_expression_none():
    result = rna_transcript_expression_dict_from_args(_make_rna_args())
    assert result is None


def test_rna_transcript_expression_cufflinks():
    path = os.path.join(DATA_DIR, "isoforms.fpkm_tracking")
    args = _make_rna_args(rna_transcript_fpkm_tracking_file=path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = rna_transcript_expression_dict_from_args(args)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_rna_transcript_expression_gtf():
    path = os.path.join(DATA_DIR, "B16-StringTie-chr1-subset.gtf")
    args = _make_rna_args(rna_transcript_fpkm_gtf_file=path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = rna_transcript_expression_dict_from_args(args)
    assert isinstance(result, dict)
    assert len(result) > 0


@pytest.mark.parametrize("flag,replacement,fixture", [
    ("rna_transcript_fpkm_tracking_file", "--transcript-expression",
     "isoforms.fpkm_tracking"),
    ("rna_transcript_fpkm_gtf_file", "--transcript-expression",
     "B16-StringTie-chr1-subset.gtf"),
])
def test_rna_transcript_expression_cufflinks_deprecation(
    caplog, flag, replacement, fixture,
):
    """A visible notice, not a DeprecationWarning Python hides (#322)."""
    path = os.path.join(DATA_DIR, fixture)
    args = _make_rna_args(**{flag: path})
    with caplog.at_level("WARNING"):
        assert rna_transcript_expression_dict_from_args(args)
    named = "--" + flag.replace("_", "-")
    assert caplog.messages == [
        f"{named} is deprecated; use {replacement} instead."
    ]
