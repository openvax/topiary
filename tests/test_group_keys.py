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


# ---------------------------------------------------------------------------
# Single group key: group_index must match what groupby produces
# ---------------------------------------------------------------------------


def test_single_group_key_index_is_flat():
    """A 1-level MultiIndex would reindex to all-NaN against a groupby."""
    df = _provenance_df()

    index = EvalContext(df, group_keys=["prediction_id"]).group_index

    assert not isinstance(index, pd.MultiIndex)
    assert index.name == "prediction_id"
    assert index.tolist() == ["variant-1", "variant-2"]


def test_single_group_key_empty_frame_index_is_flat():
    df = _provenance_df().iloc[0:0]

    index = EvalContext(df, group_keys=["prediction_id"]).group_index

    assert not isinstance(index, pd.MultiIndex)
    assert index.name == "prediction_id"
    assert len(index) == 0


def test_single_group_key_row_group_tuples_are_bare_values():
    df = _provenance_df()

    ctx = EvalContext(df, group_keys=["prediction_id"])

    assert ctx.row_group_tuples().tolist() == ["variant-1", "variant-2"]


def test_single_group_key_filters_scores_and_sorts():
    df = _provenance_df()

    kept = apply_filter(df, Affinity.value <= 500, group_keys=["prediction_id"])
    scores = evaluate_scores(df, Affinity.value, group_keys=["prediction_id"])
    ordered = apply_sort(df, [Affinity.value], group_keys=["prediction_id"])

    assert kept["prediction_id"].tolist() == ["variant-1"]
    assert scores.tolist() == [50.0, 5000.0]
    assert ordered["prediction_id"].tolist() == ["variant-1", "variant-2"]


def test_single_group_key_column_and_isin_nodes():
    """Column-backed nodes group the same way as Field nodes."""
    from topiary.ranking import Column

    df = _provenance_df()
    df["peptide_length"] = df["peptide"].str.len()

    lengths = evaluate_scores(
        df, Column("peptide_length"), group_keys=["prediction_id"],
    )
    kept = apply_filter(
        df, Column("prediction_id").isin(["variant-2"]),
        group_keys=["prediction_id"],
    )

    assert lengths.tolist() == [8.0, 8.0]
    assert kept["prediction_id"].tolist() == ["variant-2"]


# ---------------------------------------------------------------------------
# Validation reaches the early-return paths
# ---------------------------------------------------------------------------


def test_group_keys_validated_on_empty_frame():
    """A typo must not pass just because the frame has no rows."""
    empty = _provenance_df().iloc[0:0]

    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        apply_filter(empty, Affinity.value <= 500, group_keys=["nope"])
    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        evaluate_scores(empty, Affinity.value, group_keys=["nope"])
    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        apply_sort(empty, [Affinity.value], group_keys=["nope"])


def test_group_keys_validated_when_node_is_a_no_op():
    df = _provenance_df()

    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        apply_filter(df, None, group_keys=["nope"])
    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        apply_sort(df, [], group_keys=["nope"])


def test_group_keys_bare_string_names_the_mistake():
    df = _provenance_df()

    with pytest.raises(ValueError, match="not the string 'peptide'"):
        apply_filter(df, Affinity.value <= 500, group_keys="peptide")


def test_group_keys_non_string_entry_raises_value_error():
    """difflib must not be handed a non-string key."""
    df = _provenance_df()

    with pytest.raises(ValueError, match="group_keys column 123"):
        EvalContext(df, group_keys=["peptide", 123])
    with pytest.raises(ValueError, match="group_keys column None"):
        EvalContext(df, group_keys=["peptide", None])


def test_missing_column_error_survives_mixed_type_column_labels():
    df = _provenance_df()
    df[7] = 1

    with pytest.raises(ValueError, match="group_keys column 'nope'"):
        EvalContext(df, group_keys=["nope"])


# ---------------------------------------------------------------------------
# Inferred keys that aren't in the frame
# ---------------------------------------------------------------------------


def test_uninferable_frame_explains_itself():
    """Missing identity columns get an explanation, not a KeyError."""
    df = _provenance_df().drop(columns=["peptide_offset"])

    with pytest.raises(ValueError, match="Cannot infer group keys"):
        EvalContext(df)
    with pytest.raises(ValueError, match="group_keys="):
        apply_filter(df, Affinity.value <= 500)


def test_uninferable_frame_still_works_with_explicit_keys():
    df = _provenance_df().drop(columns=["peptide_offset"])

    kept = apply_filter(
        df, Affinity.value <= 500,
        group_keys=["prediction_id", "peptide", "allele"],
    )

    assert kept["prediction_id"].tolist() == ["variant-1"]


def test_empty_frame_without_identity_columns_is_not_rejected():
    """Nothing to group; inference stays quiet rather than erroring."""
    assert EvalContext(pd.DataFrame()).group_keys


# ---------------------------------------------------------------------------
# Null keys: None / NaN / pd.NA must name the same group
# ---------------------------------------------------------------------------


def _null_key_df(column, keys):
    """Three rows whose *column* holds the given (possibly null) keys."""
    rows = []
    for key, value, peptide in zip(keys, (50.0, 100.0, 150.0), "abc"):
        rows.append({
            column: key, "peptide": peptide, "peptide_offset": 0,
            "allele": "HLA-A*02:01", "kind": "pMHC_affinity", "score": 0.5,
            "value": value, "percentile_rank": 1.0,
            "prediction_method_name": "netmhcpan",
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("nulls", [
    (None, np.nan),
    (np.nan, None),
    (pd.NA, np.nan),
    (None, pd.NA),
])
def test_null_spellings_collapse_into_one_group(nulls):
    """groupby(dropna=False) merges them, so group_index must too.

    Otherwise the group index holds keys no node result can carry and
    every row in the extra group scores NaN.
    """
    df = _null_key_df("gid", [*nulls, "x"])

    ctx = EvalContext(df, group_keys=["gid"])

    assert len(ctx.group_index) == 2
    assert ctx.row_group_codes().tolist() == [0, 0, 1]


def test_null_keys_score_their_group_not_nan():
    df = _null_key_df("gid", [None, np.nan, "x"])

    scores = evaluate_scores(df, Affinity.value, group_keys=["gid"])

    # Both null rows are one group; nanmin over it is 50.
    assert scores.tolist() == [50.0, 50.0, 150.0]


def test_null_keys_are_not_dropped_by_filter():
    df = _null_key_df("gid", [None, np.nan, "x"])

    kept = apply_filter(df, Affinity.value <= 60, group_keys=["gid"])

    assert kept["peptide"].tolist() == ["a", "b"]


def test_inferred_keys_do_not_drop_rows_with_none_identity():
    """Regression: None in an identity column silently lost the row.

    `TopiaryPredictor` writes `source_sequence_name = None`, so this is
    reachable without passing group_keys at all.
    """
    df = _null_key_df("source_sequence_name", [None, np.nan, "x"])

    kept = apply_filter(df, Affinity.value <= 60)
    ordered = apply_sort(df, [Affinity.value])
    scores = evaluate_scores(df, Affinity.value)

    # Distinct peptides, so one group per row: only the 50 nM row passes.
    assert kept["peptide"].tolist() == ["a"]
    assert ordered["peptide"].tolist() == ["a", "b", "c"]
    assert scores.tolist() == [50.0, 100.0, 150.0]


def test_row_group_codes_index_group_index():
    df = _provenance_df()

    ctx = EvalContext(df, group_keys=PROVENANCE_GROUP_KEYS)
    codes = ctx.row_group_codes()

    assert codes.tolist() == [0, 1]
    assert [ctx.group_index[c] for c in codes] == list(ctx.group_index)


def test_row_group_codes_empty_frame():
    ctx = EvalContext(_provenance_df().iloc[0:0])

    assert ctx.row_group_codes().tolist() == []


# ---------------------------------------------------------------------------
# Remaining validation edges
# ---------------------------------------------------------------------------


def test_group_keys_shape_checked_even_without_a_frame():
    with pytest.raises(ValueError, match="not the string 'peptide'"):
        evaluate_scores(None, Affinity.value, group_keys="peptide")
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluate_scores(None, Affinity.value, group_keys=[])


def test_uninferable_empty_frame_with_columns_is_rejected():
    """An empty frame still has columns to be wrong about."""
    df = pd.DataFrame(columns=["peptide", "allele"])

    with pytest.raises(ValueError, match="Cannot infer group keys"):
        EvalContext(df)


def test_array_like_group_key_gets_a_readable_error():
    """Otherwise: 'truth value of an array is ambiguous', or unhashable."""
    df = _provenance_df()

    with pytest.raises(ValueError, match="must be column names, got a ndarray"):
        EvalContext(df, group_keys=[np.array(["peptide", "allele"])])


def test_close_matches_never_suggest_a_stringified_label():
    """An int label 7 must not be offered as the string '7'."""
    df = _provenance_df()
    df[7] = 1

    with pytest.raises(ValueError) as excinfo:
        EvalContext(df, group_keys=["7"])

    assert "Did you mean" not in str(excinfo.value)


def test_all_missing_columns_are_named_at_once():
    from topiary.ranking import Column

    df = _provenance_df()
    node = Column("zscore") >= 1

    with pytest.raises(ValueError, match="zscore"):
        apply_filter(df, node)

    node = (Column("zscore") >= 1) & (Column("vaff") >= 0.1)
    with pytest.raises(ValueError) as excinfo:
        apply_filter(df, node)
    message = str(excinfo.value)
    assert "zscore" in message and "vaff" in message
