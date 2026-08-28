"""Top-level entry points: :func:`apply_filter` and :func:`apply_sort`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from mhctools import Kind

from .nodes import (
    ALLELE_SET_COLUMN,
    BestAlleleField,
    EvalContext,
    Field,
    _kind_value,
    _peptide_keys,
    _kind_matches,
    _missing_column_error,
    _normalize_group_keys,
    _unwrap_peptide_view,
)


def _check_group_keys(df, group_keys):
    """Validate explicit *group_keys* on a path that returns early.

    The normal path validates inside :class:`EvalContext`, but an empty
    frame or a no-op node returns before a context is ever built — a
    typo must not pass silently just because a pipeline has degenerated
    to zero rows or dropped its expression.
    """
    if group_keys is None:
        return
    # With no frame there are no columns to check against, but the shape
    # of the argument is still wrong or right on its own terms.
    _normalize_group_keys(pd.DataFrame() if df is None else df, group_keys)


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
    missing = sorted(needed - set(df.columns))
    if not missing:
        return
    raise _missing_column_error(missing, df.columns)


def _blank(values):
    series = pd.Series(values)
    return (series.isna() | (series.astype(str).str.strip() == "")).to_numpy()


def _peptide_level_groups(ctx):
    """Boolean array: which groups are about the peptide, not one allele.

    Two shapes qualify.  An allele-free prediction — antigen processing,
    say — has nothing in its ``allele`` column.  A genotype-level one
    names an allele (the predictor's deconvolved best presenter) but
    carries an ``allele_set``, which is what it is actually about.
    Either way the group holds no evidence about a single allele.
    """
    peptide_level = _blank(ctx.group_index.get_level_values("allele"))
    if ALLELE_SET_COLUMN in ctx.group_keys:
        peptide_level |= ~_blank(
            ctx.group_index.get_level_values(ALLELE_SET_COLUMN)
        )
    return peptide_level


def _collect_kinds(node):
    """Every prediction kind a DSL tree reads."""
    kinds = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n is None:
            continue
        kind = getattr(n, "kind", None)
        if kind is not None:
            kinds.add(_kind_value(kind))
        stack.extend(n.child_nodes())
    return kinds


def _keep_allele_free_evidence(ctx, node, mask):
    """Let allele-free evidence ride along with its peptide.

    An allele-free prediction lands in a group of its own, and a filter
    on an allele-scoped kind has nothing to say about it — that group
    holds no rows of the kind being filtered on, so it evaluates to NaN,
    which pandas turns into False, which drops the row.  The evidence a
    later ``peptide_view()`` would have read is then simply gone, and
    the peptide scores as if the prediction had never been made.

    So an allele-free group holding none of the kinds the filter reads
    is kept whenever the filter kept at least one of that peptide's
    allele groups: it is peptide-level evidence and the peptide
    survived.  A group the filter *does* read — ``processing.score >=
    0.9`` against the processing row itself — keeps its own answer, and
    a peptide excluded entirely takes its evidence with it.
    """
    peptide_keys = _peptide_keys(ctx.group_keys)
    if "allele" not in ctx.group_keys or not peptide_keys:
        return mask
    if "kind" not in ctx.df.columns:
        return mask
    node_kinds = _collect_kinds(node)
    if not node_kinds:
        return mask

    # Groups holding at least one row of a kind this filter reads.
    read = np.zeros(len(ctx.group_index), dtype=bool)
    codes = ctx.row_group_codes()
    read[codes[ctx.df["kind"].isin(node_kinds).to_numpy()]] = True

    candidates = _peptide_level_groups(ctx) & ~read & ~mask
    if not candidates.any():
        return mask

    groups = ctx.group_index.to_frame(index=False)
    groups["_kept"] = mask
    peptide_kept = groups.groupby(
        peptide_keys, sort=False, dropna=False,
    )["_kept"].transform("any").to_numpy()
    return mask | (candidates & peptide_kept)


def _infer_sort_direction(node):
    """Natural sort direction for a node (asc = smaller is better).

    Reads through the wrappers that reduce a field to one value per
    peptide — they change *which* row is read, never whether small or
    large is better, so ``peptide_view(Affinity.value)`` and
    ``Affinity.best_value`` sort ascending like the bare field.
    """
    node = _unwrap_peptide_view(node)
    if isinstance(node, (Field, BestAlleleField)):
        if node.field == "percentile_rank":
            return "asc"
        if _kind_matches(node.kind, Kind.pMHC_affinity) and node.field == "value":
            return "asc"
    return "desc"


def _neutral_ranked_key(values, ascending):
    """Rank one sort key, placing missing values in the middle.

    Comparing raw values pairwise and skipping a key when either side is
    missing is not an ordering: "equal" stops being transitive, so the
    result depends on the order the rows arrived in, and a worse group
    can outrank a better one.  Ranking fixes that — every group gets a
    definite position — while keeping the property that made the skip
    attractive: a group with no value for this key neither gains nor
    loses by it, sitting at the average rank of the groups that do have
    one, so the remaining keys decide.

    Returned smallest-sorts-first, whichever direction the key runs in.
    """
    present = ~np.isnan(values)
    ranked = np.zeros(len(values), dtype=float)
    if not present.any():
        # Nothing to rank: the key can't distinguish anyone.
        return ranked
    oriented = values if ascending else -values
    ranks = pd.Series(oriented[present]).rank(method="average").to_numpy()
    ranked[present] = ranks
    ranked[~present] = ranks.mean()
    return ranked


def _resolve_sort_direction(node, sort_direction):
    if sort_direction == "auto":
        return _infer_sort_direction(node)
    return sort_direction


def evaluate_scores(df, node, *, group_keys=None, default_methods=None,
                    kind_support=None, alleles=None, fill=np.nan):
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

    *group_keys*, *default_methods*, *kind_support* and *alleles* are
    the shared context options, forwarded to :class:`EvalContext` — see
    its docstring.

    Returns a ``pd.Series`` with ``df.index`` and a numeric dtype.
    """
    if node is None:
        raise ValueError("evaluate_scores requires a DSL node")
    if df is None or df.empty:
        _check_group_keys(df, group_keys)
        return pd.Series([], index=df.index if df is not None else None,
                         dtype=float)


    ctx = EvalContext(
        df, group_keys=group_keys, default_methods=default_methods,
        kind_support=kind_support, alleles=alleles,
    )
    scored = node.eval(ctx).reindex(ctx.group_index)
    aligned = pd.Series(
        scored.to_numpy()[ctx.row_group_codes()], index=df.index,
    )
    aligned = pd.to_numeric(aligned, errors="coerce")
    if not (isinstance(fill, float) and np.isnan(fill)):
        aligned = aligned.fillna(fill)
    aligned.name = None
    return aligned


def apply_filter(df, node, *, group_keys=None, default_methods=None,
                 kind_support=None, alleles=None):
    """Apply a boolean-valued DSL node to *df*.

    Keeps all rows for peptide-allele groups whose evaluated value is
    truthy.  ``None`` for *node* is a no-op.

    *group_keys*, *default_methods*, *kind_support* and *alleles* are
    the shared context options, forwarded to :class:`EvalContext` — see
    its docstring.
    """
    if node is None or df.empty:
        _check_group_keys(df, group_keys)
        return df if node is None else df.reset_index(drop=True)

    _validate_columns(df, node)
    ctx = EvalContext(
        df, group_keys=group_keys, filter_context=True,
        default_methods=default_methods, kind_support=kind_support,
        alleles=alleles,
    )
    # Reindex defensively so a misbehaving node (index mismatch) surfaces
    # as NaN → False rather than silently picking up rows from a
    # different MultiIndex alignment.
    values = node.eval(ctx).reindex(ctx.group_index)
    _check_boolean_like(values)
    mask = values.fillna(False).astype(bool).to_numpy()
    mask = _keep_allele_free_evidence(ctx, node, mask)

    keep = mask[ctx.row_group_codes()]
    return df[keep].reset_index(drop=True)


def apply_sort(df, sort_nodes, sort_direction="auto", *, group_keys=None,
               default_methods=None, kind_support=None, alleles=None):
    """Sort groups by one or more DSL nodes (lexicographic fallthrough).

    *sort_nodes* is a list of DSLNode.  Each node's direction is inferred
    from its shape (percentile_rank → asc; affinity.value → asc; other →
    desc) when *sort_direction* is ``"auto"``; otherwise the string
    value is used for all nodes.

    A group with no value for a key is neutral on it: it takes the
    average rank of the groups that do have one, so the key neither
    promotes nor penalizes it and the remaining keys decide.  Ties keep
    the order the groups appear in the frame.

    *group_keys*, *default_methods*, *kind_support* and *alleles* are
    the shared context options, forwarded to :class:`EvalContext` — see
    its docstring.
    """
    if not sort_nodes or df.empty:
        _check_group_keys(df, group_keys)
        return df if not sort_nodes else df.reset_index(drop=True)

    for node in sort_nodes:
        _validate_columns(df, node)

    ctx = EvalContext(df, group_keys=group_keys,
                      default_methods=default_methods,
                      kind_support=kind_support, alleles=alleles)
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

    lex_keys = [
        _neutral_ranked_key(values_matrix[:, col], ascending=directions[col])
        for col in range(n_keys)
    ]
    # np.lexsort reads its last argument as the primary key.
    sorted_idx = np.lexsort(tuple(reversed(lex_keys)))
    rank_of_group = np.empty(n_groups, dtype=int)
    rank_of_group[sorted_idx] = np.arange(n_groups)

    positions = rank_of_group[ctx.row_group_codes()]
    ordered = df.assign(_sort_pos=positions).sort_values(
        "_sort_pos", kind="mergesort",
    )
    return ordered.drop(columns=["_sort_pos"]).reset_index(drop=True)
