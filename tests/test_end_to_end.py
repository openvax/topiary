"""End-to-end tests that exercise the full pipeline from input to filtered output.

These use RandomBindingPredictor for speed but exercise every code path
that a real predictor would hit.
"""

import os
import tempfile

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import Affinity, Presentation, TopiaryPredictor
from topiary.inputs import exclude_by, read_fasta, read_peptide_csv, read_peptide_fasta, read_sequence_csv, slice_regions
from topiary.ranking import RankingStrategy, apply_ranking_strategy, parse_ranking


def _tmpfile(content, suffix=".csv"):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    f.write(content)
    f.close()
    return f.name


ALLELES = ["A0201", "B0702"]


# ---------------------------------------------------------------------------
# End-to-end: peptide CSV → predict → filter → rank → exclude
# ---------------------------------------------------------------------------


def test_peptide_csv_full_pipeline():
    path = _tmpfile("name,peptide\npep1,SIINFEKLA\npep2,ELAGIGILT\npep3,AAAAAKVAA\n")
    seqs = read_peptide_csv(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor, RandomBindingPredictor],
        alleles=ALLELES,
        filter=Affinity <= 500,
        rank_by=Affinity.score,
    )
    df = predictor.predict_from_named_sequences(seqs)

    # Verify output schema
    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "score" in df.columns
    assert "value" in df.columns
    assert "affinity" in df.columns
    assert "prediction_method_name" in df.columns
    assert "source_sequence_name" in df.columns
    assert "peptide_length" in df.columns
    assert "peptide_offset" in df.columns

    # Two models → results from both (same predictor name since both RandomBinding)
    assert len(df) > 0

    # Affinity column only populated for pMHC_affinity rows
    aff_rows = df[df.kind == "pMHC_affinity"]
    non_aff_rows = df[df.kind != "pMHC_affinity"]
    if len(aff_rows) > 0:
        assert not aff_rows["affinity"].isna().all()
    if len(non_aff_rows) > 0:
        assert non_aff_rows["affinity"].isna().all()


def test_fasta_full_pipeline():
    path = _tmpfile(">braf\nMASIINFEKLGGGLLLAAA\n>tp53\nMRKKLLLQQQRRREEEYYY\n", ".fasta")
    seqs = read_fasta(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=ALLELES,
    )
    df = predictor.predict_from_named_sequences(seqs)
    assert len(df) > 0
    assert set(df["source_sequence_name"].unique()) == {"braf", "tp53"}


def test_peptide_fasta_full_pipeline():
    path = _tmpfile(">pep1\nSIINFEKLA\n>pep2\nELAGIGILT\n", ".fasta")
    seqs = read_peptide_fasta(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences(seqs)
    assert len(df) > 0


def test_sequence_csv_full_pipeline():
    path = _tmpfile("name,sequence\nbraf,MASIINFEKLGGGLLL\n")
    seqs = read_sequence_csv(path)
    os.unlink(path)

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences(seqs)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# End-to-end: regions
# ---------------------------------------------------------------------------


def test_fasta_with_regions():
    path = _tmpfile(">spike\n" + "A" * 50 + "SIINFEKL" + "A" * 50 + "\n", ".fasta")
    seqs = read_fasta(path)
    os.unlink(path)

    sliced = slice_regions(seqs, {"spike": [(50, 58)]})
    assert "spike:50-58" in sliced
    assert sliced["spike:50-58"] == "SIINFEKL"

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences(sliced)
    assert all(df["source_sequence_name"] == "spike:50-58")


def test_regions_multiple_intervals():
    seqs = {"prot": "AAAAASIINFEKLBBBBBBBBELAGIGILTCCCCC"}
    sliced = slice_regions(seqs, {"prot": [(5, 13), (21, 30)]})
    assert len(sliced) == 2
    assert sliced["prot:5-13"] == "SIINFEKL"


# ---------------------------------------------------------------------------
# End-to-end: exclusion modes
# ---------------------------------------------------------------------------


def test_exclude_by_substring_full_pipeline():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"target": "MASIINFEKLGGGLLL"})
    n_before = len(df)

    # Reference contains the target protein → all peptides should be excluded
    ref = {"ref": "MASIINFEKLGGGLLL"}
    df_filtered = exclude_by(df, ref, mode="substring")
    assert len(df_filtered) < n_before


def test_exclude_by_exact_full_pipeline():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"target": "MASIINFEKLGGGLLL"})
    n_before = len(df)

    ref = {"ref": "MASIINFEKLGGGLLL"}
    df_exact = exclude_by(df, ref, mode="exact")
    df_substr = exclude_by(df, ref, mode="substring")

    # Substring should exclude at least as many as exact
    assert len(df_substr) <= len(df_exact)


def test_exclude_by_with_unrelated_ref():
    """Reference with no shared peptides → nothing excluded."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"target": "SIINFEKLX"})
    n_before = len(df)

    ref = {"unrelated": "WWWWWWWWWWWWWWWWWW"}
    df_filtered = exclude_by(df, ref, mode="substring")
    assert len(df_filtered) == n_before


# ---------------------------------------------------------------------------
# End-to-end: filter + rank_by combinations
# ---------------------------------------------------------------------------


def test_filter_reduces_output():
    predictor_unfiltered = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    predictor_filtered = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
        filter=Affinity <= 100,  # very strict
    )
    seqs = {"prot": "MASIINFEKLGGGLLLAAA"}
    df_all = predictor_unfiltered.predict_from_named_sequences(seqs)
    df_filt = predictor_filtered.predict_from_named_sequences(seqs)

    # Filtered should have fewer or equal rows
    # (RandomBindingPredictor is random so we can't guarantee fewer,
    # but the filter code path is exercised)
    assert len(df_filt) <= len(df_all)


def test_rank_by_exercises_sort_path():
    """Verify rank_by creates a ranking strategy with sort_by."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
        rank_by=Affinity.score,
    )
    assert predictor.ranking_strategy is not None
    assert len(predictor.ranking_strategy.sort_by) == 1
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGGLLLAAA"})
    assert len(df) > 0


def test_compound_filter_or():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
        filter=(Affinity <= 500) | (Affinity.score >= 0.0),  # very permissive
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) > 0


def test_compound_filter_and():
    """AND filter with contradictory criteria should drop everything."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
        # value <= 1 AND value >= 99999 — impossible for same row
        filter=(Affinity <= 1) & (Affinity.value >= 99999),
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) == 0


# ---------------------------------------------------------------------------
# End-to-end: backward compat
# ---------------------------------------------------------------------------


def test_ic50_cutoff_backward_compat():
    predictor = TopiaryPredictor(
        mhc_model=RandomBindingPredictor(alleles=["A0201"]),
        ic50_cutoff=500,
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    # ic50_cutoff should create a ranking strategy internally
    assert predictor.ranking_strategy is not None


def test_mhc_model_backward_compat():
    predictor = TopiaryPredictor(
        mhc_model=RandomBindingPredictor(alleles=["A0201"]),
    )
    assert predictor.mhc_model is predictor.models[0]
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) > 0


# ---------------------------------------------------------------------------
# End-to-end: parse_ranking string → apply
# ---------------------------------------------------------------------------


def test_parse_ranking_string_applied():
    strategy = parse_ranking("affinity <= 500")
    if isinstance(strategy, RankingStrategy):
        rs = strategy
    else:
        rs = RankingStrategy(filters=[strategy])

    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
        filter=rs,
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    # Just verify it runs without error
    assert isinstance(df, pd.DataFrame)
