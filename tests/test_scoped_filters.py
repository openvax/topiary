"""Scoped fields in filters (topiary #192).

`wt.`, `self_nearest.`, `shuffled.` and `self.` used to raise in a comparison,
on the theory that filtering on a comparator was a mistake. It isn't: "the
mutant binds and the wildtype doesn't" is the differential criterion for
selecting a neoepitope, and it is an exclusion, not a sort. The real hazard —
a comparator column the producer never wrote, which makes the filter drop
everything — is now reported instead of banned.
"""

import warnings

import pandas as pd
import pytest

from topiary import Affinity, apply_filter, apply_sort, self_nearest, wt
from topiary.ranking import parse


def _df():
    """Two candidates: one tumor-specific, one whose wildtype also binds."""
    return pd.DataFrame([
        dict(source_sequence_name="s", peptide="SPECIFIC", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=50.0,
             score=0.9, percentile_rank=0.5, prediction_method_name="netmhcpan",
             wt_value=9000.0, self_nearest_value=8000.0),
        dict(source_sequence_name="s", peptide="SHARED", peptide_offset=9,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=60.0,
             score=0.8, percentile_rank=0.6, prediction_method_name="netmhcpan",
             wt_value=40.0, self_nearest_value=30.0),
    ])


# ---------------------------------------------------------------------------
# The analyses the guard used to block
# ---------------------------------------------------------------------------


def test_differential_binding_filter():
    """Mutant binds, wildtype doesn't — the neoepitope criterion."""
    kept = apply_filter(
        _df(), (Affinity.value <= 500) & (wt.Affinity.value >= 1000),
    )

    assert kept["peptide"].tolist() == ["SPECIFIC"]


def test_self_similarity_exclusion():
    """The shape a cross-reactivity rule takes."""
    kept = apply_filter(
        _df(), (Affinity.value <= 500) & (self_nearest.Affinity.value >= 1000),
    )

    assert kept["peptide"].tolist() == ["SPECIFIC"]


def test_the_string_form_works_too():
    """These arrive from --filter-by and from config files."""
    kept = apply_filter(
        _df(), parse("affinity <= 500 & wt.affinity.value >= 1000"),
    )

    assert kept["peptide"].tolist() == ["SPECIFIC"]


def test_a_scoped_clause_can_stand_alone():
    kept = apply_filter(_df(), wt.Affinity.value >= 1000)

    assert kept["peptide"].tolist() == ["SPECIFIC"]


def test_scoped_fields_still_work_in_ranking():
    """What the old error told people to do instead — unchanged."""
    ordered = apply_sort(
        _df(), [Affinity.score - wt.Affinity.score],
    )

    assert ordered["peptide"].tolist() == ["SPECIFIC", "SHARED"]


# ---------------------------------------------------------------------------
# The hazard the ban was standing in for
# ---------------------------------------------------------------------------


def test_a_missing_comparator_column_warns_in_a_filter():
    """NaN in a filter empties the frame; that must not happen quietly."""
    df = _df().drop(columns=["wt_value"])

    with pytest.warns(UserWarning, match="does not have"):
        kept = apply_filter(df, wt.Affinity.value >= 1000)

    assert len(kept) == 0


def test_the_warning_names_the_column_and_the_scope():
    df = _df().drop(columns=["wt_value"])

    with pytest.warns(UserWarning) as caught:
        apply_filter(df, wt.Affinity.value >= 1000)

    message = str(caught[0].message)
    assert "wt_value" in message and "wt_*" in message


def test_no_warning_when_the_columns_are_there():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_filter(_df(), wt.Affinity.value >= 1000)


def test_no_warning_outside_a_filter():
    """NaN is a sensible answer when ranking; only a filter is silent."""
    df = _df().drop(columns=["wt_value"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_sort(df, [Affinity.score - wt.Affinity.score])


def test_unscoped_fields_are_unaffected():
    """A missing kind is an ordinary NaN, not a producer mistake."""
    df = _df()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(apply_filter(df, parse("stability.score >= 0.5"))) == 0
