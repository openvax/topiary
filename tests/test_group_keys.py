"""Group identity for DSL evaluation: explicit ``group_keys`` and inference.

Covers the two halves of grouping:

* **Explicit** — ``group_keys=`` is accepted by ``apply_filter``,
  ``apply_sort`` and ``evaluate_scores`` alike, so a caller with a stable
  provenance identity can use one grouping everywhere (topiary #175).
* **Inferred** — what topiary picks when the caller says nothing.
"""

import numpy as np
import pandas as pd
import pytest

from topiary.ranking import (
    Affinity,
    EvalContext,
    apply_filter,
    apply_sort,
    evaluate_scores,
)
from topiary.ranking.nodes import _pick_group_keys

# Peptide, context and offset are identical across the two rows; only the
# provenance ID differs.  This is the vaxrank case from openvax/vaxrank#345:
# two distinct variants that happen to produce the same 9mer in the same
# flanking context.
PROVENANCE_ROWS = [
    dict(
        prediction_id="variant-1", source_sequence_name="ctx",
        peptide="SIINFEKL", peptide_offset=10, allele="HLA-A*02:01",
        n_flank="AAA", c_flank="CCC", kind="pMHC_affinity",
        score=0.9, value=50.0, percentile_rank=0.2,
        prediction_method_name="netmhcpan",
    ),
    dict(
        prediction_id="variant-2", source_sequence_name="ctx",
        peptide="SIINFEKL", peptide_offset=10, allele="HLA-A*02:01",
        n_flank="AAA", c_flank="CCC", kind="pMHC_affinity",
        score=0.1, value=5000.0, percentile_rank=20.0,
        prediction_method_name="netmhcpan",
    ),
]

PROVENANCE_GROUP_KEYS = [
    "prediction_id", "source_sequence_name", "peptide", "peptide_offset",
    "allele",
]


def _provenance_df():
    return pd.DataFrame(PROVENANCE_ROWS)


# ---------------------------------------------------------------------------
# Explicit group_keys on apply_filter (topiary #175)
# ---------------------------------------------------------------------------


def test_apply_filter_group_keys_keep_distinct_provenance_apart():
    """One variant's passing decision must not leak into another's."""
    df = _provenance_df()

    kept = apply_filter(
        df, Affinity.value <= 500, group_keys=PROVENANCE_GROUP_KEYS,
    )

    assert kept["prediction_id"].tolist() == ["variant-1"]


def test_apply_filter_without_group_keys_merges_identical_sequences():
    """The inferred, sequence-oriented grouping collapses the two rows.

    Not a bug — it is what the sequence-keyed default means, and it is
    exactly why an explicit identity has to be passable.
    """
    df = _provenance_df()

    kept = apply_filter(df, Affinity.value <= 500)

    # One group holding both rows: nanmin over the group passes, so the
    # 5000 nM row rides along on the 50 nM row's decision.
    assert kept["prediction_id"].tolist() == ["variant-1", "variant-2"]


def test_apply_filter_group_keys_match_direct_eval_context():
    """Filtering and direct EvalContext scoring share one group index."""
    df = _provenance_df()
    node = Affinity.value <= 500

    ctx = EvalContext(
        df, group_keys=PROVENANCE_GROUP_KEYS, filter_context=True,
    )
    manual = node.eval(ctx).reindex(ctx.group_index)
    passing = set(manual[manual.fillna(False).astype(bool)].index)
    manual_keep = df[ctx.row_group_tuples().isin(passing)].reset_index(drop=True)

    via_api = apply_filter(df, node, group_keys=PROVENANCE_GROUP_KEYS)

    assert list(ctx.group_index.names) == PROVENANCE_GROUP_KEYS
    pd.testing.assert_frame_equal(via_api, manual_keep)


def test_apply_filter_none_group_keys_is_inferred_grouping():
    """Omitting group_keys is identical to passing the inferred ones."""
    df = _provenance_df()
    node = Affinity.value <= 500

    implicit = apply_filter(df, node)
    explicit = apply_filter(df, node, group_keys=_pick_group_keys(df))

    pd.testing.assert_frame_equal(implicit, explicit)


# ---------------------------------------------------------------------------
# The same kwarg on apply_sort and evaluate_scores
# ---------------------------------------------------------------------------


def test_apply_sort_group_keys_sort_distinct_provenance_separately():
    df = _provenance_df()

    ordered = apply_sort(
        df, [Affinity.value], group_keys=PROVENANCE_GROUP_KEYS,
    )

    # Ascending by affinity value: the 50 nM variant first.
    assert ordered["prediction_id"].tolist() == ["variant-1", "variant-2"]

    reverse = apply_sort(
        df, [Affinity.value], sort_direction="desc",
        group_keys=PROVENANCE_GROUP_KEYS,
    )
    assert reverse["prediction_id"].tolist() == ["variant-2", "variant-1"]


def test_evaluate_scores_group_keys_score_each_provenance_row():
    df = _provenance_df()

    scores = evaluate_scores(
        df, Affinity.value, group_keys=PROVENANCE_GROUP_KEYS,
    )

    assert scores.tolist() == [50.0, 5000.0]


def test_evaluate_scores_and_apply_filter_agree_on_group_keys():
    """Scores and the filter decision line up row-for-row."""
    df = _provenance_df()

    scores = evaluate_scores(
        df, Affinity.value, group_keys=PROVENANCE_GROUP_KEYS,
    )
    kept = apply_filter(
        df, Affinity.value <= 500, group_keys=PROVENANCE_GROUP_KEYS,
    )

    expected = df.loc[scores <= 500, "prediction_id"].tolist()
    assert kept["prediction_id"].tolist() == expected


def test_evaluate_scores_accepts_default_methods():
    """The context trio is uniform: evaluate_scores takes default_methods too."""
    rows = []
    for method, value in (("netmhcpan", 50.0), ("mhcflurry", 900.0)):
        rows.append(dict(
            source_sequence_name="ctx", peptide="SIINFEKL", peptide_offset=10,
            allele="HLA-A*02:01", kind="pMHC_affinity",
            score=0.5, value=value, percentile_rank=1.0,
            prediction_method_name=method,
        ))
    df = pd.DataFrame(rows)

    # Ambiguous without a default: two methods produce the same kind.
    with pytest.raises(ValueError):
        evaluate_scores(df, Affinity.value)

    scores = evaluate_scores(
        df, Affinity.value, default_methods={"affinity": "mhcflurry"},
    )
    assert scores.tolist() == [900.0, 900.0]


@pytest.mark.parametrize("call", [
    lambda df, node: apply_filter(df, node, ["peptide"]),
    lambda df, node: apply_sort(df, [node], "auto", ["peptide"]),
    lambda df, node: evaluate_scores(df, node, ["peptide"]),
])
def test_context_options_are_keyword_only(call):
    """Context knobs are keyword-only so their order can keep growing."""
    df = _provenance_df()
    with pytest.raises(TypeError):
        call(df, Affinity.value)


# ---------------------------------------------------------------------------
# Validation of explicit group_keys
# ---------------------------------------------------------------------------


def test_group_keys_unknown_column_suggests_near_match():
    df = _provenance_df()
    with pytest.raises(ValueError, match="group_keys column 'peptid'"):
        apply_filter(df, Affinity.value <= 500, group_keys=["peptid"])


def test_group_keys_unknown_column_lists_columns_when_nothing_is_close():
    df = _provenance_df()
    with pytest.raises(ValueError, match="Available columns"):
        EvalContext(df, group_keys=["zzzzzz"])


def test_group_keys_empty_sequence_rejected():
    df = _provenance_df()
    with pytest.raises(ValueError, match="non-empty sequence"):
        EvalContext(df, group_keys=[])


def test_group_keys_duplicates_rejected():
    df = _provenance_df()
    with pytest.raises(ValueError, match="duplicate entries"):
        EvalContext(df, group_keys=["peptide", "allele", "peptide"])


def test_group_keys_validated_before_any_evaluation():
    """The error comes from constructing the context, not from a node."""
    df = _provenance_df()
    with pytest.raises(ValueError, match="group_keys column"):
        EvalContext(df, group_keys=["prediction_id", "not_a_column"])


def test_group_keys_accept_any_sequence():
    df = _provenance_df()
    ctx = EvalContext(df, group_keys=tuple(PROVENANCE_GROUP_KEYS))
    assert ctx.group_keys == PROVENANCE_GROUP_KEYS


# ---------------------------------------------------------------------------
# Inferred group keys: sample_name
# ---------------------------------------------------------------------------


def _sample_df(sample_names):
    rows = []
    for i, sample_name in enumerate(sample_names):
        rows.append(dict(
            sample_name=sample_name, source_sequence_name="ctx",
            peptide="SIINFEKL", peptide_offset=10, allele="HLA-A*02:01",
            kind="pMHC_affinity", score=0.9, value=50.0 * (i + 1),
            percentile_rank=0.2, prediction_method_name="netmhcpan",
        ))
    return pd.DataFrame(rows)


def test_blank_sample_name_is_not_inferred_as_a_group_key():
    """mhctools stamps sample_name='' on every single-sample row.

    A constant blank column carries no identity; grouping by it would
    widen every group tuple for nothing.
    """
    df = _sample_df(["", ""])

    assert "sample_name" not in EvalContext(df).group_keys


def test_whitespace_only_sample_name_is_not_inferred_as_a_group_key():
    df = _sample_df(["   ", ""])

    assert "sample_name" not in EvalContext(df).group_keys


def test_null_sample_name_is_not_inferred_as_a_group_key():
    df = _sample_df([np.nan, None])

    assert "sample_name" not in EvalContext(df).group_keys


def test_real_sample_names_are_inferred_as_a_group_key():
    df = _sample_df(["tumor", "normal"])

    ctx = EvalContext(df)

    assert ctx.group_keys[0] == "sample_name"
    assert len(ctx.group_index) == 2


def test_sample_name_kept_when_blanks_mix_with_real_names():
    """A blank alongside a real name *is* a distinguishing value."""
    df = _sample_df(["", "tumor"])

    ctx = EvalContext(df)

    assert ctx.group_keys[0] == "sample_name"
    assert len(ctx.group_index) == 2


def test_blank_sample_name_does_not_change_filter_results():
    df = _sample_df(["", ""])
    df.loc[1, "value"] = 5000.0

    kept = apply_filter(df, Affinity.value <= 500)

    # Both rows share the same peptide/offset/allele group either way;
    # the point is that group tuples stay 4-wide.
    assert list(EvalContext(df).group_index.names) == [
        "source_sequence_name", "peptide", "peptide_offset", "allele",
    ]
    assert len(kept) == 2


def test_blank_sample_name_still_usable_as_an_explicit_group_key():
    """Inference skips it; an explicit request still honors it."""
    df = _sample_df(["", ""])

    ctx = EvalContext(df, group_keys=["sample_name", "peptide", "allele"])

    assert ctx.group_keys[0] == "sample_name"
