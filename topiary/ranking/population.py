"""Report-wide transforms over evaluated occurrence values, never raw rows."""

import math

import numpy as np
import pandas as pd

from .nodes import Column, DSLNode, _as_node, is_stated, prediction_field_values


def _one_sample(ctx):
    for name in ('sample_name', 'candidate_sample', 'patient_id'):
        if name in ctx.df and ctx.df[name].map(is_stated).any():
            if ctx.df.loc[ctx.df[name].map(is_stated), name].nunique() > 1:
                raise ValueError('Population transforms require one sample per evaluation partition')


class PopulationMaximum(DSLNode):
    """Broadcast the maximum finite occurrence value within one partition.

    Missing/nonfinite observations do not contribute. An empty or entirely
    missing population yields missing values. This is not a row aggregation:
    repeated predictions have already been resolved by the inner expression.
    Evaluate distinct reports separately; mixed named samples raise.
    """

    def __init__(self, inner):
        self.inner = _as_node(inner)

    def eval(self, ctx):
        _one_sample(ctx)
        values = self.inner.eval(ctx).astype(float)
        finite = values[np.isfinite(values)]
        return pd.Series(finite.max() if len(finite) else np.nan, index=ctx.group_index)

    def child_nodes(self):
        return [self.inner]

    def __repr__(self):
        return f'({self.inner!r}).population_max()'

    def to_ast_string(self):
        return f'PopulationMaximum({self.inner.to_ast_string()})'


class DenseRank(DSLNode):
    """Dense ranks of finite occurrence values in one report, starting at 1.

    Parameters
    ----------
    inner : DSLNode
        Numeric occurrence expression.
    ascending : bool or {0, 1}
        True places smaller values first; False places larger values first.
    missing_bottom : bool or {0, 1}
        False preserves missing/nonfinite scores. True assigns them the single
        rank after all finite values, matching pVACseq's dense bottom ranks.
    """

    def __init__(self, inner, ascending=True, missing_bottom=False):
        if ascending not in (0, 1) or missing_bottom not in (0, 1):
            raise ValueError('Dense rank direction and missing_bottom must be 0 or 1')
        self.inner = _as_node(inner)
        self.ascending = bool(ascending)
        self.missing_bottom = bool(missing_bottom)

    def eval(self, ctx):
        _one_sample(ctx)
        values = self.inner.eval(ctx).astype(float)
        values = values.where(np.isfinite(values))
        return values.rank(method='dense', ascending=self.ascending,
                           na_option='bottom' if self.missing_bottom else 'keep')

    def child_nodes(self):
        return [self.inner]

    def __repr__(self):
        return f'({self.inner!r}).dense_rank({int(self.ascending)}, {int(self.missing_bottom)})'

    def to_ast_string(self):
        return f'DenseRank({self.inner.to_ast_string()}, {self.ascending}, {self.missing_bottom})'


class FillMissing(DSLNode):
    """Explicitly replace missing numeric evidence, including an absent column.

    Contradictory evidence and ambiguous prediction models still raise. Only a
    missing arbitrary Column is treated as all missing; model lookup errors
    are never hidden. Nonfinite values remain nonfinite for domain auditing.
    """

    def __init__(self, inner, value):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Missing fill must be a finite numeric constant')
        self.inner, self.value = _as_node(inner), float(value)

    def eval(self, ctx):
        if isinstance(self.inner, Column) and self.inner.col_name not in ctx.df:
            return pd.Series(self.value, index=ctx.group_index)
        return self.inner.eval(ctx).fillna(self.value)

    def child_nodes(self):
        return [self.inner]

    def __repr__(self):
        return f'({self.inner!r}).fillna({self.value!r})'

    def to_ast_string(self):
        return f'FillMissing({self.inner.to_ast_string()}, {self.value!r})'


class OrdinalRank(DSLNode):
    """Dense lexicographic ranks over numeric expressions and text columns.

    Keys sort ascending in the supplied order; negate a numeric key for the
    opposite direction. Missing values sort last. Equal key tuples share a
    rank, so input ordering never resolves ties. Column keys can carry either
    numeric values or strings, but conflicting values within an occurrence
    raise. No values or source rows are modified.
    """

    def __init__(self, *keys):
        if not keys:
            raise ValueError('ordinal_rank requires at least one key')
        self.keys = tuple(_as_node(key) for key in keys)

    def eval(self, ctx):
        _one_sample(ctx)
        columns = []
        for node in self.keys:
            if isinstance(node, Column):
                if node.col_name not in ctx.df:
                    raise ValueError(f'Missing ordinal rank column {node.col_name!r}')
                stated = ctx.df.loc[ctx.df[node.col_name].map(is_stated), node.col_name]
                if len(stated) and not all(isinstance(value, str) for value in stated):
                    series = prediction_field_values(ctx.df, node.col_name,
                        group_keys=ctx.group_keys, errors='raise').reindex(ctx.group_index)
                    numeric = series.astype(float)
                    columns.append(numeric.map(lambda value: (0, value) if np.isfinite(value) else (1, 0.)))
                    continue
                def consistent(values):
                    known = [value for value in values if is_stated(value)]
                    unique = pd.unique(pd.Series(known, dtype=object))
                    if len(unique) > 1:
                        raise ValueError(f'Conflicting ordinal rank evidence in {node.col_name!r}')
                    return unique[0] if len(unique) else None
                series = ctx.df.groupby(ctx.group_keys, sort=False, dropna=False)[node.col_name].agg(consistent)
                series = series.reindex(ctx.group_index)
            else:
                series = node.eval(ctx).reindex(ctx.group_index)
            known = series[series.map(is_stated)]
            if len(known) and all(isinstance(value, str) for value in known):
                columns.append(series.map(lambda value: (0, value) if is_stated(value) else (1, '')))
            else:
                numeric = pd.to_numeric(series, errors='raise').astype(float)
                columns.append(numeric.map(lambda value: (0, value) if np.isfinite(value) else (1, 0.)))
        tuples = list(zip(*(column.tolist() for column in columns)))
        ranks = {key: i + 1. for i, key in enumerate(sorted(set(tuples)))}
        return pd.Series([ranks[key] for key in tuples], index=ctx.group_index, dtype=float)

    def child_nodes(self):
        return list(self.keys)

    def __repr__(self):
        return 'ordinal_rank(' + ', '.join(repr(key) for key in self.keys) + ')'

    def to_ast_string(self):
        return 'OrdinalRank(' + ', '.join(key.to_ast_string() for key in self.keys) + ')'


def ordinal_rank(*keys):
    """Return an ascending lexicographic occurrence-rank DSL node.

    Parameters
    ----------
    *keys : DSLNode
        Ordered numeric expressions or Column nodes carrying text or numbers.

    Returns
    -------
    OrdinalRank
        A portable rank expression. Use ``1 / ordinal_rank(...)`` for a
        higher-is-better score. Missing keys sort last; equal keys stay tied.
    """
    return OrdinalRank(*keys)
