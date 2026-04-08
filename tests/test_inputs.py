"""Tests for topiary.inputs — CSV and FASTA parsing."""

import tempfile
import os

import pytest

from topiary.inputs import read_fasta, read_peptide_csv, read_sequence_csv


# ---------------------------------------------------------------------------
# Peptide CSV
# ---------------------------------------------------------------------------


def test_peptide_csv_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("peptide\nSIINFEKL\nELAGIGIL\n")
        f.flush()
        result = read_peptide_csv(f.name)
    os.unlink(f.name)
    assert result == {"SIINFEKL": "SIINFEKL", "ELAGIGIL": "ELAGIGIL"}


def test_peptide_csv_with_names():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,peptide\npep1,SIINFEKL\npep2,ELAGIGIL\n")
        f.flush()
        result = read_peptide_csv(f.name)
    os.unlink(f.name)
    assert result == {"pep1": "SIINFEKL", "pep2": "ELAGIGIL"}


def test_peptide_csv_missing_column():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sequence\nSIINFEKL\n")
        f.flush()
        with pytest.raises(ValueError, match="Missing required column"):
            read_peptide_csv(f.name)
    os.unlink(f.name)


# ---------------------------------------------------------------------------
# Sequence CSV
# ---------------------------------------------------------------------------


def test_sequence_csv_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sequence\nMASIINFEKLGGG\n")
        f.flush()
        result = read_sequence_csv(f.name)
    os.unlink(f.name)
    assert len(result) == 1
    assert list(result.values())[0] == "MASIINFEKLGGG"


def test_sequence_csv_with_names():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,sequence\nBRAF,MASIINFEKLGGG\nTP53,MRKKLLLQQQ\n")
        f.flush()
        result = read_sequence_csv(f.name)
    os.unlink(f.name)
    assert result == {"BRAF": "MASIINFEKLGGG", "TP53": "MRKKLLLQQQ"}


def test_sequence_csv_missing_column():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("peptide\nSIINFEKL\n")
        f.flush()
        with pytest.raises(ValueError, match="Missing required column"):
            read_sequence_csv(f.name)
    os.unlink(f.name)


# ---------------------------------------------------------------------------
# FASTA
# ---------------------------------------------------------------------------


def test_fasta_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">protein1\nMASIINFEKL\nGGGLLL\n>protein2\nMRKKLLL\n")
        f.flush()
        result = read_fasta(f.name)
    os.unlink(f.name)
    assert result == {
        "protein1": "MASIINFEKLGGGLLL",
        "protein2": "MRKKLLL",
    }


def test_fasta_with_description():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">sp|P12345|BRAF_HUMAN some description\nMASIINFEKL\n")
        f.flush()
        result = read_fasta(f.name)
    os.unlink(f.name)
    # Name is first whitespace-delimited token after >
    assert "sp|P12345|BRAF_HUMAN" in result


def test_fasta_empty():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write("")
        f.flush()
        result = read_fasta(f.name)
    os.unlink(f.name)
    assert result == {}


def test_fasta_blank_lines():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">prot\n\nMASII\n\nNFEKL\n\n")
        f.flush()
        result = read_fasta(f.name)
    os.unlink(f.name)
    assert result == {"prot": "MASIINFEKL"}


# ---------------------------------------------------------------------------
# Integration: predict from CSV/FASTA inputs
# ---------------------------------------------------------------------------


def test_predict_from_peptide_csv():
    """End-to-end: peptide CSV → predictions DataFrame."""
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,peptide\npep1,SIINFEKL\npep2,ELAGIGILT\n")
        f.flush()
        sequences = read_peptide_csv(f.name)
    os.unlink(f.name)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[8, 9]),
    )
    df = predictor.predict_from_named_sequences(sequences)
    assert len(df) > 0
    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "source_sequence_name" in df.columns


def test_predict_from_fasta():
    """End-to-end: FASTA → predictions DataFrame."""
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(">test_protein\nMASIINFEKLGGGLLLAAA\n")
        f.flush()
        sequences = read_fasta(f.name)
    os.unlink(f.name)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
    )
    df = predictor.predict_from_named_sequences(sequences)
    assert len(df) > 0
    assert "test_protein" in df["source_sequence_name"].values
