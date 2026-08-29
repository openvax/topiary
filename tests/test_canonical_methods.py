"""Opt-in canonical method resolution (topiary #193).

An unqualified reference to a kind produced by several models raises, which is
right — silently choosing a model is not something topiary should do behind a
caller's back. But there was no supported way to say "pick the canonical one",
so every consumer wrote its own preference table, and two tools could disagree
about what canonical means with nothing surfacing the difference.
"""

import pandas as pd
import pytest

from topiary import (
    CANONICAL_METHOD_PREFERENCE,
    apply_filter,
    evaluate_scores,
    resolve_default_methods,
    validate_default_methods,
)
from topiary.ranking import parse


def _row(kind, method, value=100.0, score=0.5):
    return dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
                allele="HLA-A*02:01", kind=kind, value=value, score=score,
                percentile_rank=1.0, prediction_method_name=method)


def _two_model_df():
    return pd.DataFrame([
        _row("pMHC_affinity", "netmhcpan", value=50.0),
        _row("pMHC_affinity", "mhcflurry", value=900.0),
        _row("pMHC_stability", "netmhcstabpan", value=3.0),
    ])


# ---------------------------------------------------------------------------
# What it resolves
# ---------------------------------------------------------------------------


def test_only_kinds_with_a_real_choice_are_resolved():
    """A kind with one model needs no default; saying so would be noise."""
    resolved = resolve_default_methods(_two_model_df())

    assert resolved == {"pMHC_affinity": "mhcflurry"}


def test_the_result_unblocks_an_unqualified_reference():
    df = _two_model_df()

    with pytest.raises(ValueError, match="Ambiguous"):
        evaluate_scores(df, parse("affinity.value"))

    scores = evaluate_scores(
        df, parse("affinity.value"), default_methods=resolve_default_methods(df),
    )
    assert scores.tolist()[0] == 900.0


def test_a_filter_was_never_blocked_and_still_is_not():
    """Filters auto-aggregate across models; only sorting and scoring raise."""
    df = _two_model_df()

    assert len(apply_filter(df, parse("affinity <= 500"))) > 0


def test_resolution_is_deterministic_for_unlisted_models():
    df = pd.DataFrame([
        _row("pMHC_affinity", "zebra-predictor"),
        _row("pMHC_affinity", "aardvark-predictor"),
    ])

    assert resolve_default_methods(df) == {
        "pMHC_affinity": "aardvark-predictor",
    }


def test_a_listed_model_beats_an_unlisted_one():
    df = pd.DataFrame([
        _row("pMHC_affinity", "aardvark-predictor"),
        _row("pMHC_affinity", "netmhcpan"),
    ])

    assert resolve_default_methods(df) == {"pMHC_affinity": "netmhcpan"}


def test_the_preference_order_is_overridable():
    df = _two_model_df()

    resolved = resolve_default_methods(df, preference=("netmhcpan",))

    assert resolved == {"pMHC_affinity": "netmhcpan"}


def test_the_shipped_order_prefers_general_predictors():
    """Documented as a tie-break convention, not a quality ranking."""
    assert CANONICAL_METHOD_PREFERENCE.index("netmhcpan") < (
        CANONICAL_METHOD_PREFERENCE.index("netmhcstabpan")
    )


@pytest.mark.parametrize("df", [
    None,
    pd.DataFrame(),
    pd.DataFrame([{"peptide": "SIINFEKLA"}]),
])
def test_a_frame_with_nothing_to_resolve_gives_nothing(df):
    assert resolve_default_methods(df) == {}


# ---------------------------------------------------------------------------
# Validation: an entry that names something the frame doesn't have
# ---------------------------------------------------------------------------


def test_a_model_absent_from_the_frame_is_reported():
    """Otherwise it sits inert until the day the kind becomes ambiguous."""
    df = _two_model_df()

    with pytest.raises(ValueError, match="netmhcpan-typo"):
        validate_default_methods(df, {"pMHC_affinity": "netmhcpan-typo"})


def test_an_unknown_kind_is_reported():
    with pytest.raises(ValueError, match="not a known kind"):
        validate_default_methods(_two_model_df(), {"banana": "mhcflurry"})


def test_a_kind_absent_from_the_frame_is_not_an_error():
    """A pipeline-wide default may cover kinds this frame doesn't carry."""
    df = pd.DataFrame([_row("pMHC_affinity", "mhcflurry")])

    validate_default_methods(df, {"pMHC_stability": "netmhcstabpan"})


def test_short_names_and_aliases_are_accepted():
    df = _two_model_df()

    validate_default_methods(df, {"affinity": "mhcflurry"})


def test_what_resolve_returns_always_validates():
    df = _two_model_df()

    validate_default_methods(df, resolve_default_methods(df))
