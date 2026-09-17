"""Explicit, non-destructive annotation joins for existing predictions."""

from copy import deepcopy
import json
import re

import pandas as pd

from .result import TopiaryResult
from .ranking import is_stated


def join_annotations(result, annotations, *, on, prefix, provenance):
    """Add namespaced annotations without replacing predictions or evidence.

    Parameters
    ----------
    result : TopiaryResult
        Predictions in either long or wide form. The active form, row order,
        index, existing columns and pipeline metadata are preserved.
    annotations : pandas.DataFrame
        One row per exact join key. Duplicate or missing annotation keys are
        rejected, even when their values agree: choosing or aggregating an
        observation is the caller's scientific decision, not a join policy.
    on : str or sequence of str
        Shared column names identifying the same subject in both tables.
        Include sample/reference identity when combining different datasets.
        No identifier normalization or sample equivalence is inferred.
    prefix : str
        Identifier prepended to every non-key column, separated by ``_``.
        Existing output-column names are rejected, never overwritten.
    provenance : dict
        Nonempty JSON-serializable description of the annotation source,
        units, sample, reference and processing policy as applicable. Stored
        under ``extra['annotation_overlays']`` and retained by TSV round trips.

    Returns
    -------
    TopiaryResult
        A new result. Unmatched prediction keys (including missing keys)
        receive missing annotations, not zeros. Empty annotation tables add
        all-missing columns; empty predictions remain empty. This does not
        promote added fields to canonical evidence or change any filters.
    """
    if not isinstance(result, TopiaryResult):
        raise TypeError("result must be a TopiaryResult")
    if not isinstance(annotations, pd.DataFrame):
        raise TypeError("annotations must be a pandas DataFrame")
    keys = [on] if isinstance(on, str) else list(on)
    if (not keys or not all(isinstance(k, str) and k.strip() for k in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("on must contain distinct, nonempty join-column names")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("prefix must be a nonempty identifier starting with a letter")
    if not isinstance(provenance, dict) or not provenance:
        raise ValueError("provenance must be a nonempty JSON-serializable dict")
    json.dumps(provenance, allow_nan=False)
    frame = result.df
    for label, table in (("predictions", frame), ("annotations", annotations)):
        if not table.columns.is_unique:
            raise ValueError(f"{label} has duplicate column names")
        missing = set(keys) - set(table.columns)
        if missing:
            raise ValueError(f"{label} lacks join columns: {sorted(missing)}")
    if not annotations[keys].apply(lambda column: column.map(is_stated)).all().all():
        raise ValueError("annotations has missing join keys")
    if annotations.duplicated(keys).any():
        raise ValueError("annotations has duplicate join keys")
    renamed = {c: f"{prefix}_{c}" for c in annotations if c not in keys}
    if not renamed:
        raise ValueError("annotations has no non-key columns")
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("Annotation column names collide after prefixing")
    collisions = set(renamed.values()) & set(frame.columns)
    if collisions:
        raise ValueError(f"Annotation columns already exist: {sorted(collisions)}")
    added = annotations.rename(columns=renamed)
    # Empty keys can have incompatible inferred dtypes; no merge is needed.
    if added.empty or frame.empty:
        merged = frame.copy()
        for column in renamed.values():
            # Constructing a nonempty Series(dtype=bool) invents True values.
            # Reindex an empty source Series so every missing dtype stays null.
            merged[column] = added[column].iloc[:0].reindex(frame.index)
    else:
        merged = frame.merge(added, on=keys, how="left", sort=False, validate="many_to_one")
        merged.index = frame.index.copy()
    merged.attrs = deepcopy(frame.attrs)
    metadata = deepcopy(result.metadata)
    overlays = metadata.extra.setdefault("annotation_overlays", [])
    if not isinstance(overlays, list):
        raise ValueError("annotation_overlays metadata must be a list")
    overlays.append({"prefix": prefix, "on": keys, "provenance": deepcopy(provenance)})
    return TopiaryResult(merged, metadata=metadata,
                         filter_by_ast=result.filter_by_ast, sort_by_ast=result.sort_by_ast)
