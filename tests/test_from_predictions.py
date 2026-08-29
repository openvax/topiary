"""from_predictions(): building the long form without copying the schema.

A consumer holding ``mhctools.Prediction`` objects — a report reader, a cache,
anything that didn't run a TopiaryPredictor end to end — had to write topiary's
long form by hand. That is a copy of the schema topiary can't see and can't
migrate: a column added here silently never reaches it (topiary #194).
"""

import pandas as pd
import pytest
from mhctools.pred import Prediction

from topiary import from_predictions
from topiary.ranking import EvalContext, evaluate_scores, parse


def _prediction(kind="pMHC_affinity", allele="HLA-A*02:01", value=50.0,
                score=0.9, percentile_rank=0.5, peptide="SIINFEKLA", offset=3):
    return Prediction(
        kind=kind, peptide=peptide, allele=allele, value=value, score=score,
        percentile_rank=percentile_rank, predictor_name="mhcflurry",
        predictor_version="2.1.1", source_sequence_name="prot1", offset=offset,
    )


# ---------------------------------------------------------------------------
# The schema it produces
# ---------------------------------------------------------------------------


def test_it_speaks_topiarys_column_vocabulary():
    df = from_predictions([_prediction()])

    assert "peptide_offset" in df.columns and "offset" not in df.columns
    assert "prediction_method_name" in df.columns
    assert "predictor_name" not in df.columns


def test_derived_columns_are_filled():
    df = from_predictions([_prediction()])

    row = df.iloc[0]
    assert row["peptide_length"] == len("SIINFEKLA")
    assert row["affinity"] == 50.0
    assert row["peptide_offset"] == 3


def test_affinity_is_only_backfilled_for_affinity_rows():
    df = from_predictions([
        _prediction(),
        _prediction(kind="antigen_processing", allele="", value=None,
                    score=0.77, percentile_rank=None),
    ])

    by_kind = dict(zip(df["kind"], df["affinity"]))
    assert by_kind["pMHC_affinity"] == 50.0
    assert pd.isna(by_kind["antigen_processing"])


def test_the_result_evaluates_in_the_dsl():
    """The point: what comes out is usable without further massaging."""
    df = from_predictions([_prediction()])

    assert evaluate_scores(df, parse("affinity.value")).tolist() == [50.0]
    assert EvalContext(df).group_keys == [
        "source_sequence_name", "peptide", "peptide_offset", "allele",
    ]


def test_a_frame_in_mhctools_shape_is_accepted_too():
    """predict_dataframe() hands back rows, not Prediction objects."""
    raw = pd.DataFrame([_prediction().to_row()])

    df = from_predictions(raw)

    assert df.iloc[0]["prediction_method_name"] == "mhcflurry"
    assert df.iloc[0]["peptide_offset"] == 3


def test_sample_name_is_written_through():
    df = from_predictions([_prediction()], sample_name="tumor")

    assert df["sample_name"].tolist() == ["tumor"]


def test_an_empty_input_still_has_topiarys_shape():
    """A caller that got mhctools' names back would break on the frame."""
    df = from_predictions([])

    assert len(df) == 0
    for column in ("peptide_offset", "prediction_method_name",
                   "peptide_length", "affinity"):
        assert column in df.columns
    assert "offset" not in df.columns and "predictor_name" not in df.columns


# ---------------------------------------------------------------------------
# allele_set — the column a hand-written builder would have missed
# ---------------------------------------------------------------------------


def test_a_sequence_applies_to_every_row():
    df = from_predictions(
        [_prediction(kind="pMHC_presentation")],
        allele_set=["HLA-B*07:02", "HLA-A*02:01"],
    )

    # Sorted, so the same genotype compares equal however it was listed.
    assert df["allele_set"].tolist() == ["HLA-A*02:01,HLA-B*07:02"]


def test_a_mapping_applies_per_kind():
    """A mixed list: presentation scored per genotype, affinity per allele."""
    df = from_predictions(
        [_prediction(), _prediction(kind="pMHC_presentation")],
        allele_set={"pMHC_presentation": ["HLA-A*02:01", "HLA-B*07:02"]},
    )

    by_kind = dict(zip(df["kind"], df["allele_set"]))
    assert by_kind["pMHC_presentation"] == "HLA-A*02:01,HLA-B*07:02"
    assert by_kind["pMHC_affinity"] == ""


def test_the_set_makes_the_frame_self_describing():
    """What the column is for: it survives where kind_support cannot."""
    df = from_predictions(
        [_prediction(kind="pMHC_presentation", value=None, score=0.9)],
        allele_set={"pMHC_presentation": ["HLA-A*02:01", "HLA-B*07:02"]},
    )

    assert "allele_set" in EvalContext(df).group_keys


def test_no_allele_set_column_when_none_is_given():
    df = from_predictions([_prediction()])

    assert "allele_set" not in df.columns


# ---------------------------------------------------------------------------
# It agrees with what the predictor produces
# ---------------------------------------------------------------------------


def test_it_matches_the_predictors_own_normalization():
    """One definition of the long form, shared by both paths."""
    from mhctools import RandomBindingPredictor

    from topiary import TopiaryPredictor

    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(["HLA-A*02:01"])],
    )
    result = predictor.predict_from_named_peptides({"pep1": "SIINFEKLA"})
    predicted = result.df if hasattr(result, "df") else result

    rebuilt = from_predictions(
        pd.DataFrame([
            {"peptide": row["peptide"], "allele": row["allele"],
             "kind": row["kind"], "score": row["score"], "value": row["value"],
             "percentile_rank": row["percentile_rank"],
             "predictor_name": row["prediction_method_name"],
             "predictor_version": row["predictor_version"],
             "source_sequence_name": row["source_sequence_name"],
             "offset": row["peptide_offset"], "sample_name": "",
             "n_flank": "", "c_flank": "", "tcr": None}
            for _, row in predicted.iterrows()
        ])
    )

    shared = ["peptide", "allele", "kind", "peptide_offset", "peptide_length",
              "prediction_method_name", "affinity"]
    pd.testing.assert_frame_equal(
        rebuilt[shared].reset_index(drop=True),
        predicted[shared].reset_index(drop=True),
    )


@pytest.mark.parametrize("bad", [object(), 42])
def test_a_non_prediction_input_fails_loudly(bad):
    with pytest.raises((AttributeError, TypeError)):
        from_predictions([bad])
