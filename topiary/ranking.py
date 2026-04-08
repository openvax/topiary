"""
Filtering and ranking of epitope predictions across prediction kinds.

Express filter/rank heuristics via operator overloading on ``K``::

    from topiary import K

    # IC50 < 500 nM
    K.pMHC_affinity.value <= 500

    # OR: keep if affinity OR presentation passes
    (K.pMHC_affinity.value <= 500) | (K.pMHC_presentation.rank <= 2.0)

    # AND: must pass both
    (K.pMHC_affinity.value <= 500) & (K.pMHC_presentation.rank <= 2.0)

    # With ranking priority
    ((K.pMHC_affinity.value <= 500) | (K.pMHC_presentation.rank <= 2.0)).rank_by(
        K.pMHC_presentation.score, K.pMHC_affinity.score
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from mhctools import Kind


# ---------------------------------------------------------------------------
# Expression-building primitives
# ---------------------------------------------------------------------------


class _Field:
    """A reference to one column of one prediction kind. Supports comparison
    operators that produce :class:`EpitopeFilter` instances."""

    __slots__ = ("kind", "field")

    def __init__(self, kind: Kind, field: str):
        self.kind = kind
        self.field = field

    def __le__(self, threshold):
        """``value <= 500`` or ``rank <= 2.0``"""
        if self.field == "value":
            return EpitopeFilter(kind=self.kind, max_value=threshold)
        if self.field == "percentile_rank":
            return EpitopeFilter(kind=self.kind, max_percentile_rank=threshold)
        if self.field == "score":
            # "score <= X" is unusual (higher is better) but supported
            return EpitopeFilter(kind=self.kind, max_score=threshold)
        raise ValueError(f"Cannot apply <= to field {self.field!r}")

    def __ge__(self, threshold):
        """``score >= 0.5``"""
        if self.field == "score":
            return EpitopeFilter(kind=self.kind, min_score=threshold)
        raise ValueError(f"Cannot apply >= to field {self.field!r}")

    def __lt__(self, threshold):
        return self.__le__(threshold)

    def __gt__(self, threshold):
        return self.__ge__(threshold)


class KindAccessor:
    """Proxy for a prediction :class:`Kind` that provides typed field access.

    Typically used via the :data:`K` proxy rather than constructed directly::

        K.pMHC_affinity.value   # -> _Field for IC50 / value
        K.pMHC_affinity.rank    # -> _Field for percentile_rank
        K.pMHC_affinity.score   # -> _Field for score
    """

    __slots__ = ("kind",)

    def __init__(self, kind: Kind):
        self.kind = kind

    @property
    def value(self) -> _Field:
        """Kind-specific value (e.g. IC50 nM for affinity)."""
        return _Field(self.kind, "value")

    @property
    def rank(self) -> _Field:
        """Percentile rank (lower is better)."""
        return _Field(self.kind, "percentile_rank")

    @property
    def score(self) -> _Field:
        """Continuous score (higher is better)."""
        return _Field(self.kind, "score")


class _KindProxy:
    """Attribute proxy that maps mhctools Kind names to KindAccessors.

    Usage::

        from topiary import K
        K.pMHC_affinity.value <= 500
        K.pMHC_presentation.rank <= 2.0
    """

    def __getattr__(self, name: str) -> KindAccessor:
        try:
            return KindAccessor(Kind[name])
        except KeyError:
            raise AttributeError(
                f"No prediction kind {name!r}. "
                f"Available: {[k.name for k in Kind]}"
            ) from None


K = _KindProxy()


# ---------------------------------------------------------------------------
# Filter / strategy dataclasses with operator support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpitopeFilter:
    """A filter criterion on one prediction kind.

    All specified thresholds must be satisfied (AND within a single filter).
    Combine filters with ``|`` (OR) or ``&`` (AND) to build a
    :class:`RankingStrategy`.
    """

    kind: Kind
    max_value: Optional[float] = None
    max_percentile_rank: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None

    # -- combinators --

    def __or__(self, other):
        return _combine(self, other, require_all=False)

    def __and__(self, other):
        return _combine(self, other, require_all=True)

    def rank_by(self, *fields: _Field) -> RankingStrategy:
        """Attach a sort priority to this filter."""
        return RankingStrategy(
            filters=[self],
            sort_by=[(f.kind, f.field) for f in fields],
        )


@dataclass
class RankingStrategy:
    """Composite filter + ranking specification.

    Built by combining :class:`EpitopeFilter` instances with ``|`` / ``&``,
    or constructed directly.
    """

    filters: list = field(default_factory=list)
    require_all: bool = False
    sort_by: list = field(default_factory=list)

    # -- combinators --

    def __or__(self, other):
        return _combine(self, other, require_all=False)

    def __and__(self, other):
        return _combine(self, other, require_all=True)

    def rank_by(self, *fields: _Field) -> RankingStrategy:
        """Set (or replace) the sort priority."""
        return RankingStrategy(
            filters=list(self.filters),
            require_all=self.require_all,
            sort_by=[(f.kind, f.field) for f in fields],
        )


def _combine(left, right, require_all):
    """Merge two filters / strategies into one RankingStrategy."""
    def _as_filters(obj):
        if isinstance(obj, EpitopeFilter):
            return [obj]
        return list(obj.filters)

    return RankingStrategy(
        filters=_as_filters(left) + _as_filters(right),
        require_all=require_all,
    )


# ---------------------------------------------------------------------------
# Convenience constructors (still available for explicit use)
# ---------------------------------------------------------------------------


def affinity_filter(ic50_cutoff=None, percentile_cutoff=None, min_score=None):
    """Classic IC50 / percentile-rank filter on pMHC_affinity rows."""
    return EpitopeFilter(
        kind=Kind.pMHC_affinity,
        max_value=ic50_cutoff,
        max_percentile_rank=percentile_cutoff,
        min_score=min_score,
    )


def presentation_filter(max_rank=None, min_score=None):
    """Filter on pMHC_presentation (eluted-ligand) scores."""
    return EpitopeFilter(
        kind=Kind.pMHC_presentation,
        max_percentile_rank=max_rank,
        min_score=min_score,
    )


# ---------------------------------------------------------------------------
# Core apply logic
# ---------------------------------------------------------------------------

_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


def _pick_group_keys(df):
    if "variant" in df.columns:
        return _GROUP_KEYS_VARIANT
    return _GROUP_KEYS


def _row_passes_filter(row, filt):
    if row["kind"] != filt.kind.value:
        return False
    if filt.max_value is not None:
        v = row.get("value")
        if v is None or (isinstance(v, float) and np.isnan(v)) or v > filt.max_value:
            return False
    if filt.max_percentile_rank is not None:
        r = row.get("percentile_rank")
        if r is None or (isinstance(r, float) and np.isnan(r)) or r > filt.max_percentile_rank:
            return False
    if filt.min_score is not None:
        s = row.get("score")
        if s is None or (isinstance(s, float) and np.isnan(s)) or s < filt.min_score:
            return False
    if filt.max_score is not None:
        s = row.get("score")
        if s is None or (isinstance(s, float) and np.isnan(s)) or s > filt.max_score:
            return False
    return True


def _group_passes(group_df, strategy):
    if not strategy.filters:
        return True

    results = []
    for filt in strategy.filters:
        kind_rows = group_df[group_df["kind"] == filt.kind.value]
        if kind_rows.empty:
            results.append(False)
            continue
        passed = any(
            _row_passes_filter(row, filt) for _, row in kind_rows.iterrows()
        )
        results.append(passed)

    if strategy.require_all:
        return all(results)
    return any(results)


def _sort_key_for_group(group_df, sort_by):
    for kind, col in sort_by:
        kind_rows = group_df[group_df["kind"] == kind.value]
        if kind_rows.empty:
            continue
        vals = kind_rows[col].dropna()
        if not vals.empty:
            return vals.iloc[0]
    return float("-inf")


def apply_ranking_strategy(df, strategy):
    """Apply a :class:`RankingStrategy` to a predictions DataFrame.

    Returns a filtered (and optionally sorted) copy, keeping all kind rows
    for peptide-allele groups that pass the filters.
    """
    if df.empty:
        return df

    group_keys = _pick_group_keys(df)
    grouped = df.groupby(group_keys, sort=False)

    if strategy.filters:
        keep_mask = pd.Series(False, index=df.index)
        for _key, group_df in grouped:
            if _group_passes(group_df, strategy):
                keep_mask.loc[group_df.index] = True
        df = df[keep_mask]

    if strategy.sort_by and not df.empty:
        grouped = df.groupby(group_keys, sort=False)
        sort_keys = {}
        for key, group_df in grouped:
            sort_keys[key] = _sort_key_for_group(group_df, strategy.sort_by)
        df = df.copy()
        df["_sort_key"] = df.set_index(group_keys).index.map(
            lambda k: sort_keys.get(k, float("-inf"))
        )
        df = df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])

    return df.reset_index(drop=True)
