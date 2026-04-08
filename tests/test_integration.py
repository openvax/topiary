"""Integration tests exercising the full topiary workflow end-to-end."""

import os
import tempfile

from mhctools import RandomBindingPredictor

from topiary import Affinity, Presentation, TopiaryPredictor
from topiary.inputs import exclude_by, read_fasta, slice_regions
from topiary.sources import (
    available_tissues,
    cta_sequences,
    sequences_from_gene_names,
    tissue_expressed_gene_ids,
    tissue_expressed_sequences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLELES = ["A0201"]

# Small set of genes for fast tests
GENES = ["BRAF", "TP53"]


def _small_predictor(**kwargs):
    return TopiaryPredictor(
        models=RandomBindingPredictor,
        alleles=ALLELES,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Test: model classes + alleles
# ---------------------------------------------------------------------------


def test_model_class_with_alleles():
    """Pass model class + alleles, predictor constructs it."""
    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor, RandomBindingPredictor],
        alleles=ALLELES,
    )
    assert len(predictor.models) == 2
    for m in predictor.models:
        assert "HLA-A*02:01" in m.alleles


def test_single_model_class():
    """Single class (not in list) works."""
    predictor = TopiaryPredictor(models=RandomBindingPredictor, alleles=ALLELES)
    assert len(predictor.models) == 1


# ---------------------------------------------------------------------------
# Test: filter + rank_by separation
# ---------------------------------------------------------------------------


def test_filter_only():
    predictor = _small_predictor(filter=Affinity <= 500)
    assert predictor.ranking_strategy is not None
    assert len(predictor.ranking_strategy.filters) == 1
    assert predictor.ranking_strategy.sort_by == []


def test_rank_by_only():
    predictor = _small_predictor(rank_by=Presentation.score)
    assert predictor.ranking_strategy is not None
    assert predictor.ranking_strategy.filters == []
    assert len(predictor.ranking_strategy.sort_by) == 1


def test_filter_and_rank_by():
    predictor = _small_predictor(
        filter=(Affinity <= 500) | (Presentation.rank <= 2.0),
        rank_by=[Presentation.score, Affinity.score],
    )
    assert len(predictor.ranking_strategy.filters) == 2
    assert len(predictor.ranking_strategy.sort_by) == 2


# ---------------------------------------------------------------------------
# Test: gene name → predict → filter → exclude
# ---------------------------------------------------------------------------


def test_gene_names_to_predictions():
    """Full workflow: gene names → sequences → predict → filter."""
    seqs = sequences_from_gene_names(GENES)
    assert len(seqs) == 2

    predictor = _small_predictor(filter=Affinity <= 500)
    df = predictor.predict_from_named_sequences(seqs)

    assert len(df) > 0
    assert "kind" in df.columns
    assert "peptide" in df.columns
    assert "source_sequence_name" in df.columns


# ---------------------------------------------------------------------------
# Test: FASTA → regions → predict → exclude
# ---------------------------------------------------------------------------


def test_fasta_regions_exclude_workflow():
    """FASTA → region slice → predict → exclude via pandas."""
    # Write a tiny FASTA
    fasta_content = ">prot1\nMASIINFEKLGGGLLLAAABBB\n>prot2\nMRKKLLLQQQRRREEE\n"
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    f.write(fasta_content)
    f.close()

    seqs = read_fasta(f.name)
    os.unlink(f.name)

    # Slice to region of interest
    sliced = slice_regions(seqs, {"prot1": [(2, 10)]})
    assert "prot1:2-10" in sliced
    assert sliced["prot1:2-10"] == "SIINFEKL"
    assert "prot2" in sliced  # unsliced

    # Predict
    predictor = _small_predictor()
    df = predictor.predict_from_named_sequences(sliced)
    assert len(df) > 0

    # Exclude: reference contains SIINFEKL
    df_filtered = exclude_by(df, {"ref": "SIINFEKL"}, mode="substring")
    assert "SIINFEKL" not in df_filtered["peptide"].values


# ---------------------------------------------------------------------------
# Test: first-principles tissue-restricted targets
# ---------------------------------------------------------------------------


def test_tissue_restricted_predict():
    """Can predict from tissue-restricted sequences."""
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
        filter=Affinity <= 500,
        rank_by=Affinity.score,
    )
    df = predictor.predict_from_named_sequences(small_targets)

    # 4. Exclude peptides found in vital organs (substring match)
    df = exclude_by(df, vital_seqs, mode="substring")

    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "affinity" in df.columns
