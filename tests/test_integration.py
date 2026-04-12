"""Integration tests exercising the full topiary workflow end-to-end."""

import pytest
from mhctools import RandomBindingPredictor

from topiary import Affinity, TopiaryPredictor
from topiary.inputs import exclude_by
from topiary.sources import (
    sequences_from_gene_names,
    tissue_expressed_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLELES = ["A0301"]

# Small set of genes for fast tests
GENES = ["BRAF", "TP53"]


def _small_predictor(**kwargs):
    return TopiaryPredictor(
        models=RandomBindingPredictor,
        alleles=ALLELES,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: gene name → predict → filter → exclude
# ---------------------------------------------------------------------------


def test_gene_names_to_predictions():
    """Full workflow: gene names → sequences → predict → filter."""
    seqs = sequences_from_gene_names(GENES)
    assert len(seqs) == 2

    predictor = _small_predictor(filter_by=Affinity <= 500)
    df = predictor.predict_from_named_sequences(seqs)

    assert len(df) > 0
    assert "kind" in df.columns
    assert "peptide" in df.columns
    assert "source_sequence_name" in df.columns


# ---------------------------------------------------------------------------
# Test: first-principles tissue-restricted targets
# ---------------------------------------------------------------------------


def test_tissue_restricted_predict():
    """Can predict from tissue-restricted sequences."""
    pytest.importorskip("pirlygenes")
    seqs = tissue_expressed_sequences(["testis"], min_ntpm=100.0)
    small = dict(list(seqs.items())[:3])
    predictor = _small_predictor()
    df = predictor.predict_from_named_sequences(small)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Test: tissue exclusion workflow
# ---------------------------------------------------------------------------


def test_tissue_exclusion_reduces_predictions():
    """Excluding vital-organ peptides should reduce prediction count."""
    pytest.importorskip("pirlygenes")
    seqs = sequences_from_gene_names(["BRAF"])
    predictor = _small_predictor()
    df = predictor.predict_from_named_sequences(seqs)
    n_before = len(df)

    vital_seqs = tissue_expressed_sequences(["heart_muscle"], min_ntpm=1.0)
    df_filtered = exclude_by(df, vital_seqs, mode="substring")

    assert len(df_filtered) < n_before


# ---------------------------------------------------------------------------
# Test: full workflow from first principles
# ---------------------------------------------------------------------------


def test_first_principles_workflow():
    """
    First-principles CTA-like workflow:
      1. Predict from genes expressed in reproductive tissues
      2. Exclude peptides that appear in vital organ proteins
         (substring match — 8-mer from heart in 9-mer from testis → excluded)
    """
    pytest.importorskip("pirlygenes")
    # 1. Targets: genes expressed in reproductive tissues
    target_seqs = tissue_expressed_sequences(
        ["testis", "placenta", "ovary"], min_ntpm=1.0,
    )
    small_targets = dict(list(target_seqs.items())[:3])

    # 2. Vital organ proteome (for exclusion)
    vital_seqs = tissue_expressed_sequences(
        ["heart_muscle", "lung", "liver"], min_ntpm=10.0,
    )

    # 3. Predict with filter + rank
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor,
        alleles=["A0201"],
        filter_by=Affinity <= 500,
        sort_by=Affinity.score,
    )
    df = predictor.predict_from_named_sequences(small_targets)

    # 4. Exclude peptides found in vital organs (substring match)
    df = exclude_by(df, vital_seqs, mode="substring")

    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "affinity" in df.columns
