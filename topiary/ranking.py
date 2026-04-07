"""
Filtering and ranking of epitope predictions across prediction kinds.

Supports heuristics like:
  - "Keep peptides where affinity IC50 < 500nM OR presentation rank < 2%"
  - "Rank by presentation score when available, fall back to affinity"
  - "Require both affinity < 500nM AND presentation rank < 2%"
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from mhctools import Kind


@dataclass
class EpitopeFilter:
    """A filter criterion applied to one kind of prediction.

    A row of the matching kind passes this filter if ALL specified
    thresholds are satisfied (i.e. within-filter fields are AND'd).
    """

    kind: Kind
    max_value: Optional[float] = None  # e.g. IC50 < 500 nM
    max_percentile_rank: Optional[float] = None  # e.g. rank < 2.0
    min_score: Optional[float] = None  # e.g. EL score > 0.5


@dataclass
class RankingStrategy:
    """How to filter and rank a DataFrame of multi-kind predictions.

    Parameters
    ----------
    filters : list of EpitopeFilter
        A peptide-allele group is kept if it passes ANY filter (OR logic).
        Set ``require_all=True`` for AND logic.

    require_all : bool
        If True, a group must pass ALL filters to survive.

    sort_by : list of (Kind, column_name) tuples
        Ranking priority. The first kind that has a non-null value for the
        column in a group determines the sort key. For example::

            sort_by=[(Kind.pMHC_presentation, "score"),
                     (Kind.pMHC_affinity, "score")]

        means: rank by presentation score if available, fall back to affinity.
    """

    filters: list = field(default_factory=list)
    require_all: bool = False
    sort_by: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def affinity_filter(
    ic50_cutoff=None, percentile_cutoff=None, min_score=None
):
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
# Core logic
# ---------------------------------------------------------------------------

# Columns that identify a unique peptide-allele observation.
_GROUP_KEYS = ["source_sequence_name", "peptide", "peptide_offset", "allele"]
# Fallback when source_sequence_name has been renamed to "variant"
_GROUP_KEYS_VARIANT = ["variant", "peptide", "peptide_offset", "allele"]


def _pick_group_keys(df):
    if "variant" in df.columns:
        return _GROUP_KEYS_VARIANT
    return _GROUP_KEYS


def _row_passes_filter(row, filt):
    """Check whether a single DataFrame row satisfies *filt*."""
    if row["kind"] != filt.kind.value:
        return False
    if filt.max_value is not None:
        v = row.get("value")
        if v is None or np.isnan(v) or v > filt.max_value:
            return False
    if filt.max_percentile_rank is not None:
        r = row.get("percentile_rank")
        if r is None or np.isnan(r) or r > filt.max_percentile_rank:
            return False
    if filt.min_score is not None:
        s = row.get("score")
        if s is None or np.isnan(s) or s < filt.min_score:
            return False
    return True


def _group_passes(group_df, strategy):
    """Return True if a peptide-allele group survives the filter set."""
    if not strategy.filters:
        return True

    results = []
    for filt in strategy.filters:
        kind_rows = group_df[group_df["kind"] == filt.kind.value]
        if kind_rows.empty:
            results.append(False)
            continue
        passed = any(
            _row_passes_filter(row, filt)
            for _, row in kind_rows.iterrows()
        )
        results.append(passed)

    if strategy.require_all:
        return all(results)
    return any(results)


def _sort_key_for_group(group_df, sort_by):
    """Return the sort key value for a group, using fallback priority."""
    for kind, col in sort_by:
        kind_rows = group_df[group_df["kind"] == kind.value]
        if kind_rows.empty:
            continue
        vals = kind_rows[col].dropna()
        if not vals.empty:
            return vals.iloc[0]
    return float("-inf")


def apply_ranking_strategy(df, strategy):
    """Apply a RankingStrategy to a predictions DataFrame.

    Returns a filtered (and optionally sorted) copy of the DataFrame,
    keeping all kind rows for groups that pass the filters.
    """
    if df.empty:
        return df

    group_keys = _pick_group_keys(df)
    grouped = df.groupby(group_keys, sort=False)

    # Filter
    if strategy.filters:
        keep_mask = pd.Series(False, index=df.index)
        for _key, group_df in grouped:
            if _group_passes(group_df, strategy):
                keep_mask.loc[group_df.index] = True
        df = df[keep_mask]

    # Sort
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
