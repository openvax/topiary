"""Integration tests exercising the full topiary workflow end-to-end."""

import pytest
import warnings
from mhctools import RandomBindingPredictor

from topiary import Affinity, ProteinInputWarning, TopiaryPredictor
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


def _small_tissue_sample(sequences, include_unknown):
    ordered = sorted(sequences.items())
    clean = [
        (name, sequence) for name, sequence in ordered if "X" not in sequence.upper()
    ]
    unknown = [
        (name, sequence) for name, sequence in ordered if "X" in sequence.upper()
    ]
    # Exercise a real unknown-residue input when this reference contains one,
    # rather than depending on hash-seed-specific gene iteration order.
    return dict(clean[:2] + unknown[:1] if include_unknown and unknown else clean[:3])


def _predict_with_checked_x_policy(predictor, sequences):
    excluded = {name for name, sequence in sequences.items() if "X" in sequence.upper()}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ProteinInputWarning)
        frame = predictor.predict_from_named_sequences(sequences)
    assert [warning.category for warning in caught] == (
        [ProteinInputWarning] if excluded else []
    )
    assert not set(frame.source_sequence_name) & excluded
    assert not frame.peptide.str.contains("X", case=False).any()
    return frame


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


@pytest.mark.pirlygenes
@pytest.mark.parametrize(
    "include_unknown", [False, True], ids=["clean", "with-unknown"]
)
def test_tissue_restricted_predict(include_unknown):
    """Can predict from tissue-restricted sequences."""
    seqs = tissue_expressed_sequences(["testis"], min_ntpm=100.0)
    small = _small_tissue_sample(seqs, include_unknown)
    df = _predict_with_checked_x_policy(_small_predictor(), small)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Test: tissue exclusion workflow
# ---------------------------------------------------------------------------


@pytest.mark.pirlygenes
def test_tissue_exclusion_reduces_predictions():
    """Excluding vital-organ peptides should reduce prediction count."""
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


@pytest.mark.pirlygenes
@pytest.mark.parametrize(
    "include_unknown", [False, True], ids=["clean", "with-unknown"]
)
def test_first_principles_workflow(include_unknown):
    """
    First-principles CTA-like workflow:
      1. Predict from genes expressed in reproductive tissues
      2. Exclude peptides that appear in vital organ proteins
         (substring match — 8-mer from heart in 9-mer from testis → excluded)
    """
    # 1. Targets: genes expressed in reproductive tissues
    target_seqs = tissue_expressed_sequences(
        ["testis", "placenta", "ovary"],
        min_ntpm=1.0,
    )
    small_targets = _small_tissue_sample(target_seqs, include_unknown)

    # 2. Vital organ proteome (for exclusion)
    vital_seqs = tissue_expressed_sequences(
        ["heart_muscle", "lung", "liver"],
        min_ntpm=10.0,
    )

    # 3. Predict with filter + rank
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor,
        alleles=["A0201"],
        filter_by=Affinity <= 500,
        sort_by=Affinity.score,
    )
    df = _predict_with_checked_x_policy(predictor, small_targets)

    # 4. Exclude peptides found in vital organs (substring match)
    df = exclude_by(df, vital_seqs, mode="substring")

    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "affinity" in df.columns
