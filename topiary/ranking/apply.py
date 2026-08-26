"""Top-level entry points: :func:`apply_filter` and :func:`apply_sort`."""

from __future__ import annotations

from functools import cmp_to_key

import numpy as np
import pandas as pd
from mhctools import Kind

from .nodes import (
    EvalContext,
    Field,
    _kind_matches,
    _missing_column_error,
)


def _check_boolean_like(values: pd.Series):
    """Error if *values* has non-boolean-like entries.

    Accepts: True, False, 0, 1, 0.0, 1.0, NaN.
    """
    if values.dtype == bool:
        return
    try:
        arr = values.astype(float).to_numpy()
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Filter expression produced non-numeric values ({exc}). "
            f"Filters must be boolean-valued; use <= or >= to compare."
        ) from exc
    mask = ~np.isnan(arr)
    non_bool = arr[mask]
    if len(non_bool) == 0:
        return
    bad = non_bool[(non_bool != 0.0) & (non_bool != 1.0)]
    if len(bad) > 0:
        raise ValueError(
            f"Filter expression produced non-boolean values like {bad[0]!r}. "
            f"Filters must evaluate to True/False (or 0/1). "
            f"Use <= or >= to produce a boolean comparison, e.g. "
            f"`affinity.score >= 0.5`."
        )


def _collect_column_names(node):
    """Walk a DSLNode tree and return all explicit column references.

    Composites contribute via ``DSLNode.child_nodes()``; leaves that
    reference a column do so by exposing a ``col_name: str`` attribute
    (``Column``, ``IsIn``, future column-bearing leaves).  Both axes
    are open: adding a new composite just needs ``child_nodes()``,
    adding a new column-referencing leaf just needs ``col_name``.
    """
    names = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        col_name = getattr(n, "col_name", None)
        if isinstance(col_name, str):
            names.add(col_name)
        stack.extend(n.child_nodes())
    return names


def _validate_columns(df, node):
    """Raise early if *node* references columns not in *df*."""
    needed = _collect_column_names(node)
    if not needed:
        return
    missing = needed - set(df.columns)
    if not missing:
        return
    raise _missing_column_error(min(missing), df.columns)


def _infer_sort_direction(node):
    """Natural sort direction for a node (asc = smaller is better)."""
    if isinstance(node, Field):
        if node.field == "percentile_rank":
            return "asc"
        if _kind_matches(node.kind, Kind.pMHC_affinity) and node.field == "value":
            return "asc"
    return "desc"


def _resolve_sort_direction(node, sort_direction):
    if sort_direction == "auto":
        return _infer_sort_direction(node)
    return sort_direction


def evaluate_scores(df, node, *, group_keys=None, default_methods=None,
                    kind_support=None, fill=np.nan):
    """Evaluate a DSL *node* against *df* and align the result to ``df.index``.

    ``DSLNode.eval`` returns a Series indexed by the peptide-allele
    ``EvalContext.group_index`` — one value per group, not one per row.
    This helper wraps the group→row mapping every consumer ends up
    writing by hand:

    1. Build an :class:`EvalContext`.
    2. Call ``node.eval(ctx)``.
    3. Map each of ``df``'s rows to its group's value via
       :meth:`EvalContext.row_group_tuples`.

    *fill* controls NaN behavior for rows whose group was not scored
    (e.g. the node's kind is absent for a group).  Default is ``NaN`` so
    callers pick semantics — ``.fillna(0.0)`` for additive scoring,
    ``.fillna(-inf)`` for ranking.

    *group_keys*, *default_methods* and *kind_support* are the shared
    context options: :func:`apply_filter`, :func:`apply_sort` and
    :func:`evaluate_scores` all accept them and forward them to
    :class:`EvalContext`, so one grouping and one method resolution can
    be shared across filtering, sorting and scoring.  See
    :class:`EvalContext` for what each one means.

    Returns a ``pd.Series`` with ``df.index`` and a numeric dtype.
    """
    if node is None:
        raise ValueError("evaluate_scores requires a DSL node")
    if df is None or df.empty:
        return pd.Series([], index=df.index if df is not None else None,
                         dtype=float)

    ctx = EvalContext(
        df, group_keys=group_keys, default_methods=default_methods,
        kind_support=kind_support,
    )
    scored = node.eval(ctx)
    row_tuples = ctx.row_group_tuples()
    aligned = row_tuples.map(scored.to_dict())
    aligned.index = df.index
    aligned = pd.to_numeric(aligned, errors="coerce")
    if not (isinstance(fill, float) and np.isnan(fill)):
        aligned = aligned.fillna(fill)
    aligned.name = None
    return aligned


def apply_filter(df, node, *, group_keys=None, default_methods=None,
                 kind_support=None):
    """Apply a boolean-valued DSL node to *df*.

    Keeps all rows for peptide-allele groups whose evaluated value is
    truthy.  ``None`` for *node* is a no-op.

    *group_keys*, *default_methods* and *kind_support* are the shared
    context options: :func:`apply_filter`, :func:`apply_sort` and
    :func:`evaluate_scores` all accept them and forward them to
    :class:`EvalContext`, so one grouping and one method resolution can
    be shared across filtering, sorting and scoring.  See
    :class:`EvalContext` for what each one means.
    """
    if node is None:
        return df
    if df.empty:
        return df.reset_index(drop=True)

    _validate_columns(df, node)
    ctx = EvalContext(
        df, group_keys=group_keys, filter_context=True,
        default_methods=default_methods, kind_support=kind_support,
    )
    # Reindex defensively so a misbehaving node (index mismatch) surfaces
    # as NaN → False rather than silently picking up rows from a
    # different MultiIndex alignment.
    values = node.eval(ctx).reindex(ctx.group_index)
    _check_boolean_like(values)
    mask = values.fillna(False).astype(bool)

    passing = set(mask[mask].index)
    row_keys = ctx.row_group_tuples()
    keep = row_keys.isin(passing)
    return df[keep].reset_index(drop=True)


def apply_sort(df, sort_nodes, sort_direction="auto", *, group_keys=None,
               default_methods=None, kind_support=None):
    """Sort groups by one or more DSL nodes (lexicographic fallthrough).

    *sort_nodes* is a list of DSLNode.  Each node's direction is inferred
    from its shape (percentile_rank → asc; affinity.value → asc; other →
    desc) when *sort_direction* is ``"auto"``; otherwise the string
    value is used for all nodes.  NaN values do not force an ordering —
    they fall through to the next tiebreaker.

    *group_keys*, *default_methods* and *kind_support* are the shared
    context options: :func:`apply_filter`, :func:`apply_sort` and
    :func:`evaluate_scores` all accept them and forward them to
    :class:`EvalContext`, so one grouping and one method resolution can
    be shared across filtering, sorting and scoring.  See
    :class:`EvalContext` for what each one means.
    """
    if not sort_nodes:
        return df
    if df.empty:
        return df.reset_index(drop=True)

    for node in sort_nodes:
        _validate_columns(df, node)

    ctx = EvalContext(df, group_keys=group_keys,
                      default_methods=default_methods,
                      kind_support=kind_support)
    n_groups = len(ctx.group_index)
    n_keys = len(sort_nodes)
    values_matrix = np.empty((n_groups, n_keys), dtype=float)
    for j, node in enumerate(sort_nodes):
        arr = node.eval(ctx).reindex(ctx.group_index).to_numpy(
            dtype=float, na_value=np.nan,
        )
        values_matrix[:, j] = arr
    directions = np.array(
        [_resolve_sort_direction(n, sort_direction) == "asc" for n in sort_nodes],
        dtype=bool,
    )

    def _cmp(i, j):
        for col in range(n_keys):
            a = values_matrix[i, col]
            b = values_matrix[j, col]
            if np.isnan(a) or np.isnan(b):
                continue
            if a < b:
                return -1 if directions[col] else 1
            if a > b:
                return 1 if directions[col] else -1
        return 0

    sorted_idx = sorted(range(n_groups), key=cmp_to_key(_cmp))
    sorted_keys = [ctx.group_index[i] for i in sorted_idx]

    key_pos = {k: i for i, k in enumerate(sorted_keys)}
    row_keys = ctx.row_group_tuples()
    positions = row_keys.map(key_pos)
    assert positions.notna().all(), "internal: row group tuple not found in sort keys"
    ordered = df.assign(_sort_pos=positions.values).sort_values(
        "_sort_pos", kind="mergesort",
    )
    return ordered.drop(columns=["_sort_pos"]).reset_index(drop=True)
