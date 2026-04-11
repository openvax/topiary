"""Tests for previously untested CLI helper functions."""

import os
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from topiary.cli.args import _apply_exclusion
from topiary.cli.protein_changes import best_transcript, genome_from_args
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
# best_transcript
# ---------------------------------------------------------------------------


def _mock_transcript(protein_len, seq_len, name):
    return SimpleNamespace(
        protein_sequence="M" * protein_len,
        sequence="A" * seq_len,
        name=name,
    )


def test_best_transcript_longest_protein_wins():
    t1 = _mock_transcript(100, 500, "TX-001")
    t2 = _mock_transcript(200, 400, "TX-002")
    assert best_transcript([t1, t2]).name == "TX-002"


def test_best_transcript_tiebreak_by_seq_length():
    t1 = _mock_transcript(100, 500, "TX-001")
    t2 = _mock_transcript(100, 600, "TX-002")
    assert best_transcript([t1, t2]).name == "TX-002"


def test_best_transcript_tiebreak_by_name():
    t1 = _mock_transcript(100, 500, "EGFR-002")
    t2 = _mock_transcript(100, 500, "EGFR-001")
    assert best_transcript([t1, t2]).name == "EGFR-001"


def test_best_transcript_single():
    t = _mock_transcript(100, 500, "TX-001")
    assert best_transcript([t]).name == "TX-001"


def test_best_transcript_empty_raises():
    with pytest.raises(AssertionError):
        best_transcript([])


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


def test_rna_gene_expression_deprecation_warning():
    path = os.path.join(DATA_DIR, "genes.fpkm_tracking")
    args = _make_rna_args(rna_gene_fpkm_tracking_file=path)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        rna_gene_expression_dict_from_args(args)


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


def test_rna_transcript_expression_cufflinks_deprecation():
    path = os.path.join(DATA_DIR, "isoforms.fpkm_tracking")
    args = _make_rna_args(rna_transcript_fpkm_tracking_file=path)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        rna_transcript_expression_dict_from_args(args)
