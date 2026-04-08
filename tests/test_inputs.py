"""Tests for topiary.inputs — CSV, FASTA, regions, exclusion."""

import math
import os
import tempfile

import pytest

from topiary.inputs import (
    build_exclusion_set,
    exclude_self_peptides,
    read_fasta,
    read_peptide_csv,
    read_peptide_fasta,
    read_sequence_csv,
    slice_regions,
)


def _tmpfile(content, suffix=".csv"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.flush()
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Peptide CSV
# ---------------------------------------------------------------------------


def test_peptide_csv_basic():
    path = _tmpfile("peptide\nSIINFEKL\nELAGIGIL\n")
    result = read_peptide_csv(path)
    os.unlink(path)
    assert result == {"SIINFEKL": "SIINFEKL", "ELAGIGIL": "ELAGIGIL"}


def test_peptide_csv_with_names():
    path = _tmpfile("name,peptide\npep1,SIINFEKL\npep2,ELAGIGIL\n")
    result = read_peptide_csv(path)
    os.unlink(path)
    assert result == {"pep1": "SIINFEKL", "pep2": "ELAGIGIL"}


def test_peptide_csv_missing_column():
    path = _tmpfile("sequence\nSIINFEKL\n")
    with pytest.raises(ValueError, match="Missing required column"):
        read_peptide_csv(path)
    os.unlink(path)


# ---------------------------------------------------------------------------
# Sequence CSV
# ---------------------------------------------------------------------------


def test_sequence_csv_basic():
    path = _tmpfile("sequence\nMASIINFEKLGGG\n")
    result = read_sequence_csv(path)
    os.unlink(path)
    assert len(result) == 1
    assert list(result.values())[0] == "MASIINFEKLGGG"


def test_sequence_csv_with_names():
    path = _tmpfile("name,sequence\nBRAF,MASIINFEKLGGG\nTP53,MRKKLLLQQQ\n")
    result = read_sequence_csv(path)
    os.unlink(path)
    assert result == {"BRAF": "MASIINFEKLGGG", "TP53": "MRKKLLLQQQ"}


def test_sequence_csv_missing_column():
    path = _tmpfile("peptide\nSIINFEKL\n")
    with pytest.raises(ValueError, match="Missing required column"):
        read_sequence_csv(path)
    os.unlink(path)


# ---------------------------------------------------------------------------
# FASTA (protein sequences)
# ---------------------------------------------------------------------------


def test_fasta_basic():
    path = _tmpfile(">protein1\nMASIINFEKL\nGGGLLL\n>protein2\nMRKKLLL\n", ".fasta")
    result = read_fasta(path)
    os.unlink(path)
    assert result == {"protein1": "MASIINFEKLGGGLLL", "protein2": "MRKKLLL"}


def test_fasta_with_description():
    path = _tmpfile(">sp|P12345|BRAF_HUMAN some description\nMASIINFEKL\n", ".fasta")
    result = read_fasta(path)
    os.unlink(path)
    assert "sp|P12345|BRAF_HUMAN" in result


def test_fasta_empty():
    path = _tmpfile("", ".fasta")
    result = read_fasta(path)
    os.unlink(path)
    assert result == {}


def test_fasta_blank_lines():
    path = _tmpfile(">prot\n\nMASII\n\nNFEKL\n\n", ".fasta")
    result = read_fasta(path)
    os.unlink(path)
    assert result == {"prot": "MASIINFEKL"}


# ---------------------------------------------------------------------------
# Peptide FASTA (each entry is one peptide)
# ---------------------------------------------------------------------------


def test_peptide_fasta():
    path = _tmpfile(">pep1\nSIINFEKL\n>pep2\nELAGIGIL\n", ".fasta")
    result = read_peptide_fasta(path)
    os.unlink(path)
    assert result == {"pep1": "SIINFEKL", "pep2": "ELAGIGIL"}


# ---------------------------------------------------------------------------
# Region slicing
# ---------------------------------------------------------------------------


def test_slice_regions_single():
    seqs = {"spike": "A" * 100, "orf1a": "B" * 200}
    result = slice_regions(seqs, {"spike": [(10, 20)]})
    assert "spike:10-20" in result
    assert len(result["spike:10-20"]) == 10
    # orf1a has no regions specified → included in full
    assert "orf1a" in result
    assert len(result["orf1a"]) == 200


def test_slice_regions_multiple():
    seqs = {"nuc": "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    result = slice_regions(seqs, {"nuc": [(0, 5), (20, 26)]})
    assert "nuc:0-5" in result
    assert result["nuc:0-5"] == "ABCDE"
    assert "nuc:20-26" in result
    assert result["nuc:20-26"] == "UVWXYZ"


def test_slice_regions_exclude():
    seqs = {"keep": "AAA", "drop": "BBB"}
    result = slice_regions(seqs, {"drop": []})
    assert "keep" in result
    assert "drop" not in result


def test_slice_regions_no_regions_dict():
    seqs = {"a": "AAA", "b": "BBB"}
    result = slice_regions(seqs, {})
    assert result == seqs


# ---------------------------------------------------------------------------
# Exclusion set
# ---------------------------------------------------------------------------


def test_build_exclusion_set():
    seqs = {"prot1": "ABCDEFGH"}
    excl = build_exclusion_set(seqs, lengths=[3])
    assert "ABC" in excl
    assert "BCD" in excl
    assert "FGH" in excl
    assert len(excl) == 6  # 8 - 3 + 1 = 6 trimers


def test_build_exclusion_set_multiple_lengths():
    seqs = {"p": "ABCDE"}
    excl = build_exclusion_set(seqs, lengths=[2, 3])
    # 2-mers: AB, BC, CD, DE = 4
    # 3-mers: ABC, BCD, CDE = 3
    assert len(excl) == 7


def test_exclude_self_peptides():
    import pandas as pd
    df = pd.DataFrame({
        "peptide": ["SIINFEKL", "SELF_PEP", "ELAGIGIL"],
        "score": [0.9, 0.5, 0.1],
    })
    exclusion = {"SELF_PEP"}
    result = exclude_self_peptides(df, exclusion)
    assert len(result) == 2
    assert "SELF_PEP" not in result["peptide"].values


def test_exclude_empty_set():
    import pandas as pd
    df = pd.DataFrame({"peptide": ["AAA", "BBB"], "score": [1, 2]})
    result = exclude_self_peptides(df, set())
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration: predict from various inputs
# ---------------------------------------------------------------------------


def test_predict_from_peptide_csv():
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor

    path = _tmpfile("name,peptide\npep1,SIINFEKL\npep2,ELAGIGILT\n")
    sequences = read_peptide_csv(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[8, 9]),
    )
    df = predictor.predict_from_named_sequences(sequences)
    assert len(df) > 0
    assert "peptide" in df.columns
    assert "kind" in df.columns


def test_predict_from_fasta():
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor

    path = _tmpfile(">test_protein\nMASIINFEKLGGGLLLAAA\n", ".fasta")
    sequences = read_fasta(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[9]),
    )
    df = predictor.predict_from_named_sequences(sequences)
    assert len(df) > 0
    assert "test_protein" in df["source_sequence_name"].values


def test_predict_fasta_with_regions_and_exclusion():
    """End-to-end: FASTA → region slice → predict → exclude self."""
    from mhctools import RandomBindingPredictor
    from topiary import TopiaryPredictor

    path = _tmpfile(">prot\nAAAAAAAAASIINFEKLAAAAAAAAA\n", ".fasta")
    sequences = read_fasta(path)
    os.unlink(path)

    # Slice to the interesting region
    sliced = slice_regions(sequences, {"prot": [(9, 17)]})
    assert "prot:9-17" in sliced
    assert sliced["prot:9-17"] == "SIINFEKL"

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor(alleles=["A0201"], default_peptide_lengths=[8]),
    )
    df = predictor.predict_from_named_sequences(sliced)
    assert len(df) > 0

    # Exclude: pretend SIINFEKL is a self-peptide
    exclusion = {"SIINFEKL"}
    df_filtered = exclude_self_peptides(df, exclusion)
    assert "SIINFEKL" not in df_filtered["peptide"].values
