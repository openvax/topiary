"""TopiaryPredictor and TopiaryResult forwarding context options (topiary #178).

`kind_support` is already forwarded on both paths. `default_methods` was not,
and it is the one that turns a working configuration into a hard error: a
predictor running two models that produce the same kind makes every
unqualified reference in `sort_by` ambiguous, and there was no way to resolve
it through the predictor.

Filtering hides this — `filter_context=True` auto-aggregates directional
comparisons across methods — so the failure surfaces on the sort, not the
filter that looks the same.
"""

import pandas as pd
import pytest

from mhctools import RandomBindingPredictor

from topiary import Affinity, TopiaryResult, TopiaryPredictor
from topiary.io import Metadata


def _two_method_frame():
    rows = []
    for peptide, base in (("SIINFEKLA", 80.0), ("KLQAAMAVL", 300.0)):
        for method, offset in (("netmhcpan", 0.0), ("mhcflurry", 45.0)):
            rows.append(dict(
                source_sequence_name="s", peptide=peptide, peptide_offset=0,
                allele="HLA-A*02:01", kind="pMHC_affinity",
                value=base + offset, score=0.5, percentile_rank=1.0,
                prediction_method_name=method, predictor_version="1",
            ))
    return pd.DataFrame(rows)


def _result():
    return TopiaryResult(_two_method_frame(), Metadata(form="long"))


def _predictor(**kwargs):
    """A predictor with a real model, so only filter/sort is under test."""
    return TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"], **kwargs,
    )


# ---------------------------------------------------------------------------
# TopiaryResult
# ---------------------------------------------------------------------------


def test_sort_by_is_ambiguous_without_default_methods():
    """Unchanged behavior — the safety error is the starting point."""
    with pytest.raises(ValueError, match="Ambiguous"):
        _result().sort_by([Affinity.value])


def test_sort_by_forwards_default_methods():
    sorted_result = _result().sort_by(
        [Affinity.value], default_methods={"pMHC_affinity": "mhcflurry"},
    )

    assert len(sorted_result.df) == 4


def test_sort_by_actually_sorts_on_the_named_method():
    """Not merely 'it stopped raising' — the chosen method decides the order."""
    df = _two_method_frame()
    # Make the two methods disagree on which peptide is best.
    df.loc[df.prediction_method_name == "mhcflurry", "value"] = [900.0, 10.0]
    result = TopiaryResult(df, Metadata(form="long"))

    by_netmhcpan = result.sort_by(
        [Affinity.value], default_methods={"pMHC_affinity": "netmhcpan"},
    ).df["peptide"].tolist()
    by_mhcflurry = result.sort_by(
        [Affinity.value], default_methods={"pMHC_affinity": "mhcflurry"},
    ).df["peptide"].tolist()

    assert by_netmhcpan[0] == "SIINFEKLA"
    assert by_mhcflurry[0] == "KLQAAMAVL"


def test_filter_by_forwards_default_methods():
    """Filtering auto-aggregates, so this pins the method rather than a raise."""
    kept = _result().filter_by(
        Affinity.value <= 200, default_methods={"pMHC_affinity": "netmhcpan"},
    ).df

    assert set(kept["peptide"]) == {"SIINFEKLA"}


def test_filter_by_forwards_group_keys():
    kept = _result().filter_by(Affinity.value <= 200, group_keys=["peptide"]).df

    assert not kept.empty


def test_sort_by_forwards_group_keys():
    ordered = _result().sort_by(
        [Affinity.value], group_keys=["peptide"],
        default_methods={"pMHC_affinity": "netmhcpan"},
    ).df

    assert ordered["peptide"].tolist()[0] == "SIINFEKLA"


# ---------------------------------------------------------------------------
# TopiaryPredictor
# ---------------------------------------------------------------------------


def test_the_predictor_accepts_default_methods():
    predictor = _predictor(
        sort_by=Affinity.value,
        default_methods={"pMHC_affinity": "mhcflurry"},
    )

    assert predictor.default_methods == {"pMHC_affinity": "mhcflurry"}


def test_the_predictor_forwards_default_methods_to_sort():
    """The gap #178 names: a two-model predictor could not sort at all."""
    predictor = _predictor(
        sort_by=[Affinity.value],
        default_methods={"pMHC_affinity": "mhcflurry"},
    )

    ordered = predictor._apply_filter(_two_method_frame())

    assert len(ordered) == 4


def test_without_default_methods_the_predictor_still_raises():
    predictor = _predictor(sort_by=[Affinity.value])

    with pytest.raises(ValueError, match="Ambiguous"):
        predictor._apply_filter(_two_method_frame())


def test_the_predictor_forwards_default_methods_to_filter():
    predictor = _predictor(
        filter_by=Affinity.value <= 200,
        default_methods={"pMHC_affinity": "netmhcpan"},
    )

    kept = predictor._apply_filter(_two_method_frame())

    assert set(kept["peptide"]) == {"SIINFEKLA"}


def test_default_methods_defaults_to_none():
    assert _predictor().default_methods is None
