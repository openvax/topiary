"""Sharing one EvalContext across operations (topiary #179).

Each entry point used to build its own context, recomputing the grouping —
a `drop_duplicates` over the frame plus a row→group code array — every time.
The options were shareable since #176; the *work* was not.

What is shareable is bounded by a fact worth stating plainly: `apply_filter`
and `apply_sort` both return new frames, so a context cannot be threaded
down a filter→sort→score pipeline. It is reusable across operations on one
unchanged frame — several `evaluate_scores` calls for different score
columns, or a filter and a sort keyed the same way.
"""

import pandas as pd
import pytest

from topiary import (
    Affinity,
    EvalContext,
    apply_filter,
    apply_sort,
    evaluate_scores,
)


def _frame(n_peptides=6, methods=("netmhcpan",)):
    rows = []
    for p in range(n_peptides):
        for allele in ("HLA-A*02:01", "HLA-B*07:02"):
            for method in methods:
                rows.append(dict(
                    source_sequence_name=f"src{p % 2}",
                    peptide=f"PEPTIDE{p}A"[:9],
                    peptide_offset=p,
                    allele=allele,
                    kind="pMHC_affinity",
                    value=float(100 * p + len(allele)),
                    score=0.5,
                    percentile_rank=float(p),
                    prediction_method_name=method,
                    predictor_version="1",
                ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A shared context gives the same answers
# ---------------------------------------------------------------------------


def test_scores_match_a_fresh_context():
    df = _frame()
    ctx = EvalContext(df)

    separate = evaluate_scores(df, Affinity.value)
    shared = evaluate_scores(df, Affinity.value, context=ctx)

    pd.testing.assert_series_equal(separate, shared)


def test_sort_matches_a_fresh_context():
    df = _frame()
    ctx = EvalContext(df)

    separate = apply_sort(df, [Affinity.value])
    shared = apply_sort(df, [Affinity.value], context=ctx)

    pd.testing.assert_frame_equal(separate, shared)


def test_filter_matches_a_fresh_context():
    df = _frame()
    ctx = EvalContext(df)

    separate = apply_filter(df, Affinity.value <= 300)
    shared = apply_filter(df, Affinity.value <= 300, context=ctx)

    pd.testing.assert_frame_equal(separate, shared)


def test_the_grouping_is_computed_once():
    """The point of sharing: the caches carry over, they aren't rebuilt."""
    df = _frame()
    ctx = EvalContext(df)
    evaluate_scores(df, Affinity.value, context=ctx)

    index_before = ctx.group_index
    codes_before = ctx.row_group_codes()
    evaluate_scores(df, Affinity.score, context=ctx)

    assert ctx.group_index is index_before
    assert ctx.row_group_codes() is codes_before


# ---------------------------------------------------------------------------
# filter_context differs between filter and sort, and must not leak
# ---------------------------------------------------------------------------


def test_a_filter_does_not_flip_the_caller_s_context():
    """apply_filter needs filter_context=True; the caller's stays as it was."""
    df = _frame()
    ctx = EvalContext(df)

    apply_filter(df, Affinity.value <= 300, context=ctx)

    assert ctx.filter_context is False


def test_the_derived_filter_context_reuses_the_grouping():
    df = _frame()
    ctx = EvalContext(df)
    ctx.group_index  # populate the cache

    derived = ctx.derive(filter_context=True)

    assert derived.filter_context is True
    assert derived.group_index is ctx.group_index
    assert derived.key_frame is ctx.key_frame


def test_sorting_stays_strict_through_a_filter_context():
    """A context that filtered must not carry auto-aggregation into a sort."""
    df = _frame(methods=("netmhcpan", "mhcflurry"))
    ctx = EvalContext(df, filter_context=True)

    with pytest.raises(ValueError, match="Ambiguous"):
        apply_sort(df, [Affinity.value], context=ctx)


# ---------------------------------------------------------------------------
# Reshaping an option drops the caches rather than reusing a wrong one
# ---------------------------------------------------------------------------


def test_deriving_a_new_grouping_does_not_inherit_the_old_index():
    df = _frame()
    ctx = EvalContext(df)
    ctx.group_index

    derived = ctx.derive(group_keys=["peptide"])

    assert derived.group_keys == ["peptide"]
    assert derived.group_index is not ctx.group_index
    assert len(derived.group_index) < len(ctx.group_index)


def test_derive_rejects_an_unknown_option():
    with pytest.raises(TypeError, match="Unknown EvalContext option"):
        EvalContext(_frame()).derive(filter_contxt=True)


# ---------------------------------------------------------------------------
# The staleness hazard the identity check exists for
# ---------------------------------------------------------------------------


def test_a_context_from_another_frame_is_refused():
    """The whole hazard: a stale context maps rows to the wrong groups."""
    df = _frame()
    ctx = EvalContext(df)
    filtered = apply_filter(df, Affinity.value <= 300)

    with pytest.raises(ValueError, match="different DataFrame"):
        evaluate_scores(filtered, Affinity.value, context=ctx)


def test_the_message_says_filter_and_sort_return_new_frames():
    """Because that is why a pipeline cannot thread one context through."""
    df = _frame()
    ctx = EvalContext(df)

    with pytest.raises(ValueError, match="return new frames"):
        apply_sort(apply_sort(df, [Affinity.value]), [Affinity.value],
                   context=ctx)


def test_an_equal_but_distinct_frame_is_still_refused():
    """Identity, not equality — a copy has its own row order to account for."""
    df = _frame()
    ctx = EvalContext(df)

    with pytest.raises(ValueError, match="different DataFrame"):
        evaluate_scores(df.copy(), Affinity.value, context=ctx)


def test_context_and_options_together_are_refused():
    df = _frame()
    ctx = EvalContext(df)

    with pytest.raises(ValueError, match="not both"):
        apply_filter(df, Affinity.value <= 300, context=ctx,
                     group_keys=["peptide"])


def test_a_non_context_is_refused():
    df = _frame()

    with pytest.raises(TypeError, match="must be an EvalContext"):
        evaluate_scores(df, Affinity.value, context={"group_keys": ["peptide"]})


# ---------------------------------------------------------------------------
# A shared context honors the options it was built with
# ---------------------------------------------------------------------------


def test_default_methods_come_from_the_shared_context():
    df = _frame(methods=("netmhcpan", "mhcflurry"))
    ctx = EvalContext(df, default_methods={"pMHC_affinity": "mhcflurry"})

    scores = evaluate_scores(df, Affinity.value, context=ctx)

    assert scores.notna().all()


def test_explicit_group_keys_survive_into_a_shared_context():
    df = _frame()
    ctx = EvalContext(df, group_keys=["peptide"])

    scores = evaluate_scores(df, Affinity.value, context=ctx)
    fresh = evaluate_scores(df, Affinity.value, group_keys=["peptide"])

    pd.testing.assert_series_equal(scores, fresh)
