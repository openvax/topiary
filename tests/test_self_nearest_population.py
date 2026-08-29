"""Populating self_nearest_* including paired MHC binding (topiary #190).

`SelfProteome.nearest()` says *which* healthy peptide a candidate resembles.
Whether that peptide is presented by the same allele is a separate question,
and it's the one a cross-reactivity judgement turns on: a near-identical self
peptide the patient's MHC never presents is not the same risk as one it does.

That second half needs a prediction pass wired into the predictor's own model
and allele configuration, which is why it can't reasonably live downstream.
"""

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import SelfProteome, TopiaryPredictor, apply_filter, self_nearest
from topiary.ranking import evaluate_scores, parse

ALLELES = ["HLA-A*02:01", "HLA-B*07:02"]


def _proteome():
    """A healthy proteome holding SIINFEKLA, one substitution from our query."""
    return SelfProteome.from_peptides(
        {"SELF_PROT": "SIINFEKLLSIINFEKLA"}, peptide_lengths=[9],
    )


def _predict(predict_self_nearest, proteome=None, peptide="SIINFEKLK"):
    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(ALLELES)],
        self_proteome=_proteome() if proteome is None else proteome,
        predict_self_nearest=predict_self_nearest,
    )
    result = predictor.predict_from_named_peptides({"pep1": peptide})
    return result.df if hasattr(result, "df") else result


# ---------------------------------------------------------------------------
# Opt-in
# ---------------------------------------------------------------------------


def test_similarity_alone_is_the_default():
    df = _predict(predict_self_nearest=False)

    assert "self_nearest_peptide" in df.columns
    assert "self_nearest_value" not in df.columns


def test_the_flag_populates_the_binding_columns():
    df = _predict(predict_self_nearest=True)

    for column in ("self_nearest_value", "self_nearest_score",
                   "self_nearest_percentile_rank"):
        assert column in df.columns
        assert df[column].notna().all()


def test_without_a_proteome_the_columns_are_present_but_empty():
    """Nothing to compare against; the shape stays predictable."""
    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(ALLELES)], predict_self_nearest=True,
    )
    result = predictor.predict_from_named_peptides({"pep1": "SIINFEKLK"})
    df = result.df if hasattr(result, "df") else result

    assert df["self_nearest_value"].isna().all()


# ---------------------------------------------------------------------------
# What the numbers mean
# ---------------------------------------------------------------------------


def test_the_binding_is_the_self_peptides_not_the_candidates():
    df = _predict(predict_self_nearest=True)

    # Different peptides, so different predictions — the column would be
    # useless if it echoed the row's own value.
    assert (df["self_nearest_value"] != df["value"]).all()


def test_it_is_scored_at_each_row_s_own_allele():
    """The point of the pass: presentation is allele-specific."""
    df = _predict(predict_self_nearest=True)

    by_allele = dict(zip(df["allele"], df["self_nearest_value"]))
    assert set(by_allele) == set(ALLELES)
    assert by_allele[ALLELES[0]] != by_allele[ALLELES[1]]


def test_the_same_self_peptide_is_found_for_both_alleles():
    df = _predict(predict_self_nearest=True)

    assert df["self_nearest_peptide"].nunique() == 1
    assert df["self_nearest_edit_distance"].tolist() == [1, 1]


def test_a_peptide_length_the_model_cannot_score_is_skipped():
    """8-mers here; the model only handles what it declares."""
    proteome = SelfProteome.from_peptides(
        {"SELF_PROT": "SIINFEKL"}, peptide_lengths=[8],
    )
    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(ALLELES, default_peptide_lengths=[9])],
        self_proteome=proteome, predict_self_nearest=True,
    )
    result = predictor.predict_from_named_peptides({"pep1": "SIINFEKLK"})
    df = result.df if hasattr(result, "df") else result

    # No 9-mer in the reference, so nothing to score and nothing invented.
    assert df["self_nearest_value"].isna().all()


# ---------------------------------------------------------------------------
# It reaches the DSL, which is what a policy consumes
# ---------------------------------------------------------------------------


def test_the_scope_reads_the_populated_columns():
    df = _predict(predict_self_nearest=True)

    scores = evaluate_scores(df, self_nearest.Affinity.value)

    assert scores.notna().all()
    assert scores.tolist() == df["self_nearest_value"].tolist()


def test_a_cross_reactivity_exclusion_runs_end_to_end():
    """The shape of a safety rule, on a frame topiary populated itself."""
    df = _predict(predict_self_nearest=True)
    threshold = df["self_nearest_value"].min() + 1.0

    kept = apply_filter(df, self_nearest.Affinity.value >= threshold)

    # The allele whose self peptide binds strongly is excluded; the other
    # survives.
    assert len(kept) < len(df)
    assert (kept["self_nearest_value"] >= threshold).all()


def test_the_string_form_reaches_it_too():
    df = _predict(predict_self_nearest=True)

    assert evaluate_scores(
        df, parse("self_nearest.affinity.value"),
    ).notna().all()


@pytest.mark.parametrize("column", [
    "self_nearest_peptide", "self_nearest_edit_distance",
    "self_nearest_gene_id", "self_nearest_transcript_id",
])
def test_the_similarity_columns_are_still_there(column):
    df = _predict(predict_self_nearest=True)

    assert column in df.columns
    assert df[column].notna().all()


def test_predictions_are_not_duplicated_by_the_join():
    """A self peptide matching several rows must not multiply them."""
    before = len(_predict(predict_self_nearest=False))
    after = len(_predict(predict_self_nearest=True))

    assert before == after == len(ALLELES)
