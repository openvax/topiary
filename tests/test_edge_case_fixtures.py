"""Edge-case tests: empty files, malformed CSVs, missing columns."""

import os
import tempfile

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import TopiaryPredictor
from topiary.inputs import read_fasta, read_peptide_csv, read_peptide_fasta, read_sequence_csv


def _tmpfile(content, suffix=".csv"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


def test_predict_from_empty_dict():
    predictor = TopiaryPredictor(models=RandomBindingPredictor, alleles=["A0201"])
    df = predictor.predict_from_named_sequences({})
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0


def test_fasta_empty_file():
    path = _tmpfile("", ".fasta")
    try:
        seqs = read_fasta(path)
        assert len(seqs) == 0
    finally:
        os.unlink(path)


def test_peptide_csv_header_only():
    path = _tmpfile("name,peptide\n")
    try:
        seqs = read_peptide_csv(path)
        assert len(seqs) == 0
    finally:
        os.unlink(path)


def test_sequence_csv_header_only():
    path = _tmpfile("name,sequence\n")
    try:
        seqs = read_sequence_csv(path)
        assert len(seqs) == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Malformed CSVs
# ---------------------------------------------------------------------------


def test_peptide_csv_wrong_column_name():
    path = _tmpfile("name,pep\npep1,SIINFEKL\n")
    try:
        with pytest.raises((ValueError, KeyError)):
            read_peptide_csv(path)
    finally:
        os.unlink(path)


def test_sequence_csv_wrong_column_name():
    path = _tmpfile("name,seq\nbraf,MASIINFEKL\n")
    try:
        with pytest.raises((ValueError, KeyError)):
            read_sequence_csv(path)
    finally:
        os.unlink(path)


def test_peptide_csv_extra_columns_ok():
    path = _tmpfile("name,peptide,extra\npep1,SIINFEKL,foo\n")
    try:
        seqs = read_peptide_csv(path)
        assert seqs == {"pep1": "SIINFEKL"}
    finally:
        os.unlink(path)


def test_sequence_csv_empty_sequence():
    """CSV with an empty sequence value should produce an empty or short entry."""
    path = _tmpfile("name,sequence\nbraf,\n")
    try:
        seqs = read_sequence_csv(path)
        # Should either have an empty string or skip it
        assert isinstance(seqs, dict)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# FASTA edge cases
# ---------------------------------------------------------------------------


def test_fasta_multiple_sequences():
    content = ">seq1\nAAAAAAA\n>seq2\nBBBBBBB\n>seq3\nCCCCCCC\n"
    path = _tmpfile(content, ".fasta")
    try:
        seqs = read_fasta(path)
        assert len(seqs) == 3
    finally:
        os.unlink(path)


def test_fasta_multiline_sequence():
    content = ">long\nAAAAA\nBBBBB\nCCCCC\n"
    path = _tmpfile(content, ".fasta")
    try:
        seqs = read_fasta(path)
        assert seqs["long"] == "AAAAABBBBBCCCCC"
    finally:
        os.unlink(path)


def test_peptide_fasta_empty():
    path = _tmpfile("", ".fasta")
    try:
        seqs = read_peptide_fasta(path)
        assert len(seqs) == 0
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Full pipeline with edge cases
# ---------------------------------------------------------------------------


def test_pipeline_single_peptide():
    """Single short peptide through the full pipeline."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["B3501"],
    )
    df = predictor.predict_from_named_sequences({"pep": "SIINFEKLAA"})
    assert len(df) > 0


def test_pipeline_very_long_sequence():
    """Long sequence shouldn't crash."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"long": "A" * 500})
    assert len(df) > 0
