"""Resolving an ambiguous predictor version (topiary #214).

`default_methods` answers "which model, when a kind has several". Nothing
answered "which version, when a model has several" — so once 5.28.2 started
keeping both versions of a multi-version LENS table rather than dropping one,
every unqualified reference on that frame raised with no configured way
through, and a consumer's default scoring expression could not run at all.

That is the same gap as the sort ambiguity fixed in 5.29.0, one level down:
raising is right, having no way to answer is not.
"""

import numpy as np
import pandas as pd
import pytest

from topiary import (
    Affinity,
    EvalContext,
    apply_filter,
    apply_sort,
    evaluate_scores,
    resolve_default_versions,
    validate_default_versions,
)
from topiary.ranking import parse


def _frame(versions=("4.1b", "4.2"), values=(75.0, 120.0), method="netmhcpan"):
    return pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=value,
             score=0.5, percentile_rank=1.0,
             prediction_method_name=method, predictor_version=version)
        for version, value in zip(versions, values)
    ])


# ---------------------------------------------------------------------------
# Unconfigured behavior is unchanged: raising is still the default
# ---------------------------------------------------------------------------


def test_an_unqualified_reference_still_raises():
    with pytest.raises(ValueError, match="more than one predictor version"):
        parse("affinity.value").eval(EvalContext(_frame()))


def test_the_error_points_at_default_versions():
    """It named only the bracket form, which is per-reference, not configured."""
    with pytest.raises(ValueError, match="default_versions="):
        evaluate_scores(_frame(), Affinity.value)


# ---------------------------------------------------------------------------
# A configured default resolves it, the way default_methods does one level up
# ---------------------------------------------------------------------------


def test_a_default_version_resolves_the_reference():
    scores = evaluate_scores(
        _frame(), Affinity.value,
        default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
    )

    assert scores.dropna().unique().tolist() == [120.0]


def test_the_other_version_is_selectable():
    """Not "it stopped raising" — the named version is the one that answers."""
    scores = evaluate_scores(
        _frame(), Affinity.value,
        default_versions={("pMHC_affinity", "netmhcpan"): "4.1b"},
    )

    assert scores.dropna().unique().tolist() == [75.0]


def test_a_short_kind_alias_works_as_a_key():
    scores = evaluate_scores(
        _frame(), Affinity.value,
        default_versions={("affinity", "netmhcpan"): "4.2"},
    )

    assert scores.dropna().unique().tolist() == [120.0]


def test_sorting_works_with_a_default_version():
    ordered = apply_sort(
        _frame(), [Affinity.value],
        default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
    )

    assert len(ordered) == 2


def test_filtering_works_with_a_default_version():
    kept = apply_filter(
        _frame(), Affinity.value <= 100,
        default_versions={("pMHC_affinity", "netmhcpan"): "4.1b"},
    )

    assert len(kept) == 2


def test_an_explicit_version_still_wins():
    """A reference that pins a version is not overridden by a default."""
    scores = evaluate_scores(
        _frame(), Affinity["netmhcpan", "4.1b"].value,
        default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
    )

    assert scores.dropna().unique().tolist() == [75.0]


def test_a_default_for_another_kind_does_not_apply():
    with pytest.raises(ValueError, match="more than one predictor version"):
        evaluate_scores(
            _frame(), Affinity.value,
            default_versions={("pMHC_stability", "netmhcpan"): "4.2"},
        )


# ---------------------------------------------------------------------------
# resolve_default_versions
# ---------------------------------------------------------------------------


def test_resolve_picks_the_newest():
    assert resolve_default_versions(_frame()) == {
        ("pMHC_affinity", "netmhcpan"): "4.2",
    }


def test_resolve_can_pick_the_oldest():
    """A pipeline pinned to an older validated model wants the opposite."""
    assert resolve_default_versions(_frame(), prefer="oldest") == {
        ("pMHC_affinity", "netmhcpan"): "4.1b",
    }


def test_resolve_orders_by_pep_440_not_by_string():
    """The whole point of PEP 440 here: 4.10 is newer than 4.9."""
    df = _frame(versions=("4.9", "4.10"))

    assert resolve_default_versions(df) == {
        ("pMHC_affinity", "netmhcpan"): "4.10",
    }


def test_resolve_omits_a_model_with_one_version():
    """It only speaks where there is a real choice."""
    assert resolve_default_versions(_frame().head(1)) == {}


def test_resolve_keys_each_model_separately():
    """A version means nothing across models, so each gets its own answer."""
    df = pd.concat([
        _frame(),
        _frame(versions=("2.0", "2.1"), values=(10.0, 20.0),
               method="mhcflurry"),
    ], ignore_index=True)

    assert resolve_default_versions(df) == {
        ("pMHC_affinity", "netmhcpan"): "4.2",
        ("pMHC_affinity", "mhcflurry"): "2.1",
    }


def test_resolve_is_deterministic_on_unparseable_versions():
    df = _frame(versions=("nightly-a", "nightly-b"))

    assert resolve_default_versions(df) == {
        ("pMHC_affinity", "netmhcpan"): "nightly-b",
    }


def test_a_real_release_beats_an_unparseable_one():
    """'newest' should mean a release, not whichever string sorted last."""
    df = _frame(versions=("zzz-nightly", "4.2"))

    assert resolve_default_versions(df) == {
        ("pMHC_affinity", "netmhcpan"): "4.2",
    }


def test_resolve_rejects_an_unknown_preference():
    with pytest.raises(ValueError, match="newest.*oldest"):
        resolve_default_versions(_frame(), prefer="best")


def test_resolve_handles_a_frame_without_versions():
    df = _frame().drop(columns=["predictor_version"])

    assert resolve_default_versions(df) == {}


def test_the_resolver_output_feeds_straight_back_in():
    """The intended loop: resolve, then evaluate."""
    df = _frame()

    scores = evaluate_scores(
        df, Affinity.value, default_versions=resolve_default_versions(df),
    )

    assert scores.dropna().unique().tolist() == [120.0]


# ---------------------------------------------------------------------------
# Validation and shape
# ---------------------------------------------------------------------------


def test_validate_catches_a_version_that_never_ran():
    with pytest.raises(ValueError, match="predictor_version '9.9'"):
        validate_default_versions(
            _frame(), {("pMHC_affinity", "netmhcpan"): "9.9"},
        )


def test_validate_accepts_a_real_version():
    validate_default_versions(
        _frame(), {("pMHC_affinity", "netmhcpan"): "4.2"},
    )


def test_a_bare_kind_key_is_refused():
    """A version is only meaningful within a model, so the pair is required."""
    with pytest.raises(TypeError, match=r"\(kind, model\) pairs"):
        EvalContext(_frame(), default_versions={"pMHC_affinity": "4.2"})


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="not a known kind"):
        EvalContext(_frame(), default_versions={("banana", "netmhcpan"): "4.2"})


def test_a_non_string_version_is_refused():
    with pytest.raises(TypeError, match="must be a version string"):
        EvalContext(
            _frame(), default_versions={("pMHC_affinity", "netmhcpan"): 4.2},
        )


def test_a_context_carries_default_versions_through_derive():
    ctx = EvalContext(
        _frame(), default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
    )

    derived = ctx.derive(filter_context=True)

    assert derived.default_versions == ctx.default_versions


def test_a_shared_context_applies_its_default_versions():
    df = _frame()
    ctx = EvalContext(
        df, default_versions={("pMHC_affinity", "netmhcpan"): "4.2"},
    )

    scores = evaluate_scores(df, Affinity.value, context=ctx)

    assert scores.dropna().unique().tolist() == [120.0]


# ---------------------------------------------------------------------------
# Unknown versions: an absent version is not a version
# ---------------------------------------------------------------------------
#
# Plenty of frames simply do not record predictor_version -- a reader that
# has no version column, a source that reports one for some rows and not
# others. Treating a missing value as a version called "nan" invents a
# second version, then raises an ambiguity error naming a version the
# caller cannot possibly pass. It also made resolve_default_versions and
# the DSL disagree: the resolver dropped the NaN and reported no choice
# to make, while the DSL raised, so feeding the resolver's own answer
# back in still crashed.


@pytest.mark.parametrize("versions", [
    (np.nan, np.nan),
    ("4.2", np.nan),
    ("4.2", None),
    ("4.2", ""),
    ("4.2", "   "),
    ("4.2", "nan"),
    ("", ""),
], ids=["all-nan", "partial-nan", "none", "empty", "whitespace",
        "literal-nan-string", "all-empty"])
def test_an_unknown_version_is_not_an_ambiguity(versions):
    scores = evaluate_scores(_frame(versions=versions), Affinity.value)

    assert scores.notna().any()


@pytest.mark.parametrize("versions", [
    (np.nan, np.nan), ("4.2", np.nan), ("4.2", ""), ("4.2", "nan"),
], ids=["all-nan", "partial-nan", "empty", "literal-nan-string"])
def test_the_resolver_reports_no_choice_for_unknown_versions(versions):
    assert resolve_default_versions(_frame(versions=versions)) == {}


def test_a_frame_with_no_version_column_evaluates():
    df = _frame().drop(columns=["predictor_version"])

    assert evaluate_scores(df, Affinity.value).notna().any()


@pytest.mark.parametrize("versions", [
    (np.nan, np.nan), ("4.2", np.nan), ("4.2", ""),
], ids=["all-nan", "partial-nan", "empty"])
def test_the_resolver_and_the_dsl_agree(versions):
    """The documented loop must not crash on a frame the resolver passed on."""
    df = _frame(versions=versions)

    scores = evaluate_scores(
        df, Affinity.value, default_versions=resolve_default_versions(df),
    )

    assert scores.notna().any()


def test_two_real_versions_still_raise_when_some_rows_have_none():
    """An unknown version does not mask a genuine disagreement."""
    df = _frame(versions=("4.1b", "4.2", np.nan), values=(75.0, 120.0, 99.0))

    with pytest.raises(ValueError, match="more than one predictor version"):
        evaluate_scores(df, Affinity.value)


def test_the_error_never_names_a_missing_version():
    """It told the caller to pass 'nan', which is not a version."""
    df = _frame(versions=("4.1b", "4.2", np.nan), values=(75.0, 120.0, 99.0))

    with pytest.raises(ValueError) as excinfo:
        evaluate_scores(df, Affinity.value)

    message = str(excinfo.value)
    assert "netmhcpan 4.1b" in message and "netmhcpan 4.2" in message
    assert "nan" not in message.lower().split("default_versions")[0]


def test_a_resolved_default_still_works_alongside_unknown_versions():
    df = _frame(versions=("4.1b", "4.2", np.nan), values=(75.0, 120.0, 99.0))

    scores = evaluate_scores(
        df, Affinity.value, default_versions=resolve_default_versions(df),
    )

    assert scores.dropna().unique().tolist() == [120.0]


def test_validate_does_not_offer_a_blank_as_available():
    df = _frame(versions=("4.2", ""))

    with pytest.raises(ValueError) as excinfo:
        validate_default_versions(df, {("pMHC_affinity", "netmhcpan"): "9.9"})

    assert "Available: ['4.2']" in str(excinfo.value)
