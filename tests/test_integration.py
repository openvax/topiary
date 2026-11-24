"""Integration tests exercising the full topiary workflow end-to-end."""

import math
import warnings

import pandas as pd
import pytest
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


class _DeterministicAffinityPredictor:
    """Deterministic MHCflurry-shaped affinity scanner for workflow assertions."""

    predictor_name = "deterministic_affinity"
    predictor_version = "1.0"
    default_peptide_lengths = [9]

    def __init__(self, alleles, affinities_by_peptide, *, default_affinity=2000.0):
        self.alleles = list(alleles)
        self.affinities_by_peptide = dict(affinities_by_peptide)
        self.default_affinity = float(default_affinity)

    def kind_support(self):
        return {
            "pMHC_affinity": {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            }
        }

    def predict_proteins_dataframe(self, name_to_sequence, peptide_lengths=None):
        lengths = list(peptide_lengths or self.default_peptide_lengths)
        rows = []
        for source_name, sequence in name_to_sequence.items():
            for length in lengths:
                for offset in range(max(0, len(sequence) - length + 1)):
                    peptide = sequence[offset : offset + length]
                    ic50 = self.affinities_by_peptide.get(peptide, self.default_affinity)
                    score = max(0.0, 1.0 - math.log(max(ic50, 1.0)) / math.log(50000.0))
                    for allele in self.alleles:
                        rows.append({
                            "source_sequence_name": source_name,
                            "offset": offset,
                            "peptide": peptide,
                            "allele": allele,
                            "kind": "pMHC_affinity",
                            "value": ic50,
                            "affinity": ic50,
                            "score": score,
                            "percentile_rank": min(100.0, ic50 / 100.0),
                            "predictor_name": self.predictor_name,
                            "predictor_version": self.predictor_version,
                            "n_flank": "",
                            "c_flank": "",
                        })
        return pd.DataFrame(rows)


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
    First-principles CTA-like workflow (#603):
      1. Predict from real reproductive-tissue genes with deterministic scores
      2. Verify X-policy handling, filter_by removal, and sort_by ordering
      3. Exclude peptides with an 8-mer substring in vital-organ proteins
         while retaining non-overlapping controls in ranked order
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
    vital_8mers = {
        seq[i : i + 8]
        for seq in vital_seqs.values()
        for i in range(max(0, len(seq) - 7))
    }

    # Partition real 9-mers from the X-free target proteins in natural scan
    # order into vital-organ overlapping peptides and non-overlapping controls.
    overlapping = []
    controls = []
    seen = set()
    for seq in small_targets.values():
        if "X" in seq.upper():
            continue
        for i in range(max(0, len(seq) - 8)):
            pep = seq[i : i + 9]
            if pep in seen:
                continue
            seen.add(pep)
            if pep[:8] in vital_8mers or pep[1:] in vital_8mers:
                overlapping.append(pep)
            else:
                controls.append(pep)

    assert len(overlapping) >= 2, "Expected real vital-organ overlapping 9-mers"
    assert len(controls) >= 4, "Expected real non-overlapping control 9-mers"

    vital_pass_1, vital_pass_2 = overlapping[0], overlapping[1]
    # Assign controls in explicitly ascending score order (weakest earliest in
    # scan order, strongest latest) so sorting descending must reorder input.
    control_weak, control_mid, control_strong, control_fail = controls[:4]
    affinities = {
        control_weak: 400.0,
        control_mid: 200.0,
        control_strong: 20.0,
        vital_pass_1: 120.0,
        vital_pass_2: 300.0,
        control_fail: 1500.0,
    }
    deterministic_model = _DeterministicAffinityPredictor(
        alleles=["HLA-A*02:01"],
        affinities_by_peptide=affinities,
        default_affinity=2000.0,
    )

    # 3. Predict with filter + rank
    predictor = TopiaryPredictor(
        models=[deterministic_model],
        alleles=["HLA-A*02:01"],
        filter_by=Affinity <= 500,
        sort_by=Affinity.score,
    )
    df_predicted = _predict_with_checked_x_policy(predictor, small_targets)

    expected_ranked_before_exclusion = [
        control_strong,  # 20 nM
        vital_pass_1,    # 120 nM
        control_mid,     # 200 nM
        vital_pass_2,    # 300 nM
        control_weak,    # 400 nM
    ]
    assert len(df_predicted) > 0
    assert (df_predicted["affinity"] <= 500).all()
    assert control_fail not in set(df_predicted["peptide"])
    assert (
        df_predicted["peptide"].drop_duplicates().tolist()
        == expected_ranked_before_exclusion
    )
    assert df_predicted["score"].is_monotonic_decreasing

    # 4. Exclude peptides found in vital organs (substring match)
    df = exclude_by(df_predicted, vital_seqs, mode="substring")

    assert len(df) > 0
    assert len(df) < len(df_predicted)
    assert not ({vital_pass_1, vital_pass_2} & set(df["peptide"]))
    assert df["peptide"].drop_duplicates().tolist() == [
        control_strong,
        control_mid,
        control_weak,
    ]
    assert (df["affinity"] <= 500).all()
    assert df["score"].is_monotonic_decreasing
    assert "peptide" in df.columns
    assert "kind" in df.columns
    assert "affinity" in df.columns
