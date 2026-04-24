"""Top-level entry points: :func:`apply_filter` and :func:`apply_sort`."""

from __future__ import annotations

from difflib import get_close_matches
from functools import cmp_to_key

import numpy as np
import pandas as pd
from mhctools import Kind

from .nodes import (
    Column,
    EvalContext,
    Field,
    _kind_matches,
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
    """Walk a DSLNode tree and return all explicit Column references.

    Uses ``DSLNode.child_nodes()`` so adding a new node type doesn't
    require touching this walker — new composites just need to
    implement ``child_nodes()``.
    """
    names = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        if isinstance(n, Column):
            names.add(n.col_name)
        stack.extend(n.child_nodes())
    return names


def _validate_columns(df, node):
    """Raise early if *node* references columns not in *df*."""
    needed = _collect_column_names(node)
    if not needed:
        return
    available = set(df.columns)
    missing = needed - available
    if not missing:
        return
    for col_name in sorted(missing):
        close = get_close_matches(col_name, sorted(available), n=3, cutoff=0.6)
        msg = f"Column {col_name!r} not found in DataFrame."
        if close:
            msg += f" Did you mean: {close}?"
        else:
            msg += f" Available columns: {sorted(available)}"
        raise ValueError(msg)


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


def apply_filter(df, node, default_methods=None):
    """Apply a boolean-valued DSL node to *df*.

    Keeps all rows for peptide-allele groups whose evaluated value is
    truthy.  ``None`` for *node* is a no-op.

    *default_methods* is forwarded to :class:`EvalContext` — see its
    docstring for the per-kind default ``prediction_method_name`` kwarg
    used to resolve unqualified Field references on multi-method inputs.
    """
    if node is None:
        return df
    if df.empty:
        return df.reset_index(drop=True)

    _validate_columns(df, node)
    ctx = EvalContext(df, default_methods=default_methods)
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


def apply_sort(df, sort_nodes, sort_direction="auto", default_methods=None):
    """Sort groups by one or more DSL nodes (lexicographic fallthrough).

    *sort_nodes* is a list of DSLNode.  Each node's direction is inferred
    from its shape (percentile_rank → asc; affinity.value → asc; other →
    desc) when *sort_direction* is ``"auto"``; otherwise the string
    value is used for all nodes.  NaN values do not force an ordering —
    they fall through to the next tiebreaker.

    *default_methods* is forwarded to :class:`EvalContext` — see its
    docstring.
    """
    if not sort_nodes:
        return df
    if df.empty:
        return df.reset_index(drop=True)

    for node in sort_nodes:
        _validate_columns(df, node)

    ctx = EvalContext(df, default_methods=default_methods)
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
