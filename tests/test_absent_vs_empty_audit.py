"""Two more absent-vs-empty conflations (topiary #223).

Found by auditing after #214, #216 and #219 all turned out to be the same
defect: two code paths disagreeing about what "nothing" means. Both of these
turn a missing value into the literal text "nan" and then treat it as data.
"""

import pathlib
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from topiary import Affinity, evaluate_scores, read_pvacseq, resolve_default_versions

PVACSEQ = pathlib.Path(__file__).parent / "data" / "pvacseq" / "mhc_i_all_epitopes.tsv"


def _versioned(pairs):
    return pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=value,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan", predictor_version=version)
        for value, version in pairs
    ])


# ---------------------------------------------------------------------------
# A row with no version cannot be selected by one
# ---------------------------------------------------------------------------
#
# The ambiguity check and resolve_default_versions were taught in 5.31.0 that
# NaN names no version. The selection path still compared stringified values,
# so the two disagreed: one said the row names nothing, the other said it
# names "nan" — and let you address it that way.


@pytest.mark.parametrize("missing", [np.nan, None, "", "  ", "nan"],
                         ids=["nan", "none", "empty", "blank", "literal-nan"])
def test_a_missing_version_cannot_be_selected_as_nan(missing):
    df = _versioned([(75.0, "4.2"), (999.0, missing)])

    with pytest.raises(ValueError, match="predictor_version 'nan'"):
        evaluate_scores(df, Affinity["netmhcpan", "nan"].value)


def test_the_available_list_omits_missing_versions():
    """It already did — the bug was that the omitted one still matched."""
    df = _versioned([(75.0, "4.2"), (999.0, np.nan)])

    with pytest.raises(ValueError) as excinfo:
        evaluate_scores(df, Affinity["netmhcpan", "nan"].value)

    assert "Available: ['4.2']" in str(excinfo.value)


def test_a_real_version_still_selects():
    df = _versioned([(75.0, "4.2"), (999.0, np.nan)])

    scores = evaluate_scores(df, Affinity["netmhcpan", "4.2"].value)

    assert scores.dropna().unique().tolist() == [75.0]


def test_selection_tolerates_surrounding_whitespace():
    """The resolver strips, so selection must match on the same shape."""
    df = _versioned([(75.0, " 4.2 "), (999.0, np.nan)])

    scores = evaluate_scores(df, Affinity["netmhcpan", "4.2"].value)

    assert scores.dropna().unique().tolist() == [75.0]


def test_selection_and_the_resolver_agree():
    """The property the bug violated: one notion of "is this a version"."""
    df = _versioned([(75.0, "4.1b"), (120.0, "4.2"), (999.0, np.nan)])

    resolved = resolve_default_versions(df)
    scores = evaluate_scores(df, Affinity.value, default_versions=resolved)

    assert resolved == {("pMHC_affinity", "netmhcpan"): "4.2"}
    assert scores.dropna().unique().tolist() == [120.0]


# ---------------------------------------------------------------------------
# pVACseq must not fabricate a variant id
# ---------------------------------------------------------------------------
#
# The coordinate fallback concatenated stringified columns, so a blank field
# became the text "nan" inside the identifier: chr1-154590262-nan-A. Stable,
# well-formed-looking, and wrong — every row sharing that gap grouped under
# one fabricated id.


def _fallback_file(tmp_path, blank_column=None, blank_row=0):
    """The fixture without its Index column, so the coordinate id is used."""
    df = pd.read_csv(PVACSEQ, sep="\t").drop(columns=["Index"])
    if blank_column is not None:
        df.loc[df.index[blank_row], blank_column] = None
    path = tmp_path / "mhc_i_all_epitopes.tsv"
    df.to_csv(path, sep="\t", index=False)
    return path


def test_a_missing_coordinate_gives_a_null_id_not_a_fabricated_one(tmp_path):
    path = _fallback_file(tmp_path, blank_column="Reference")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = read_pvacseq(path)

    ids = result.df["variant"]
    assert ids.isna().any()
    assert not any("nan" in str(v).lower() for v in ids.dropna())


@pytest.mark.parametrize(
    "column", ["Chromosome", "Start", "Reference", "Variant"],
)
def test_any_missing_component_nulls_the_id(tmp_path, column):
    path = _fallback_file(tmp_path, blank_column=column)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = read_pvacseq(path)

    assert not any(
        "nan" in str(v).lower() for v in result.df["variant"].dropna()
    )


def test_the_omission_is_announced(tmp_path):
    """Silently dropping an id would be its own absence bug."""
    path = _fallback_file(tmp_path, blank_column="Reference")

    with pytest.warns(UserWarning, match="no coordinate variant id"):
        read_pvacseq(path)


def test_complete_coordinates_are_unchanged_and_silent(tmp_path):
    path = _fallback_file(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = read_pvacseq(path)

    ids = sorted(result.df["variant"].dropna().unique())
    assert ids[0].count("-") == 3
    assert result.df["variant"].notna().all()


def test_the_index_column_still_wins_when_present():
    """The fallback is only for files with no Index; that path is untouched."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = read_pvacseq(PVACSEQ)

    assert not any("-" == str(v)[3] for v in result.df["variant"].dropna()[:1])
    assert result.df["variant"].notna().all()
