"""TSV IO for :class:`ProteinFragment` collections.

Format: one row per fragment.  Scalar fields map to columns of the same
name. ``target_intervals``, ``field_provenance``, and ``annotations`` are
JSON-serialized into their own columns; empty mappings serialize as the
empty object ``{}``, absent target_intervals serialize as empty strings.

Missing columns on read fall back to field defaults. Evidence fields from
5.47 and earlier are migrated to their assay-scoped names. Other unknown
columns are rejected with a clear error (catches typos; use ``annotations``
for tool-specific extensions).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from .protein_fragment import ProteinFragment, _migrate_fragment_dict
from .ranking import is_stated

_COLUMNS = [field.name for field in dataclasses.fields(ProteinFragment)]

_COLUMN_SET = set(_COLUMNS)


def _fragment_to_row(f: ProteinFragment) -> dict:
    """Convert a :class:`ProteinFragment` to a flat dict suitable for
    a TSV row (lists / dicts JSON-encoded)."""
    row = {}
    for col, val in f.to_dict().items():
        if col == "target_intervals":
            row[col] = json.dumps([list(p) for p in val]) if val is not None else ""
        elif col in ("annotations", "field_provenance"):
            row[col] = _json_text(val or {})
        elif val is None:
            row[col] = ""
        else:
            row[col] = val
    return row


def _stringify_json_key(key) -> str:
    """Convert a JSON-allowed dict key to its JSON string representation."""
    if isinstance(key, str):
        return key if type(key) is str else str.__str__(key)
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(int(key))
    if isinstance(key, float):
        return json.dumps(float(key))
    if key is None:
        return "null"
    raise TypeError(
        f"keys must be str, int, float, bool or None, not {type(key).__name__}"
    )


def _stringify_json_keys(value, _seen: set[int] | None = None):
    """Recursively stringify dict keys using JSON's key-encoding rules."""
    if isinstance(value, dict):
        marker = id(value)
        seen = set() if _seen is None else _seen
        if marker in seen:
            raise ValueError("Circular reference detected")
        seen.add(marker)
        try:
            return {
                _stringify_json_key(k): _stringify_json_keys(v, seen)
                for k, v in value.items()
            }
        finally:
            seen.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        seen = set() if _seen is None else _seen
        if marker in seen:
            raise ValueError("Circular reference detected")
        seen.add(marker)
        try:
            return [_stringify_json_keys(v, seen) for v in value]
        finally:
            seen.remove(marker)
    return value


def _json_text(value) -> str:
    """JSON with keys sorted after converting every dict key to its JSON string.

    Sorting first fails on mixed key types (``{1: ..., "b": ...}``) that
    JSON itself stores without complaint.
    """
    return json.dumps(_stringify_json_keys(value), sort_keys=True)


def _row_to_fragment(row: dict) -> ProteinFragment:
    """Inverse of :func:`_fragment_to_row`."""
    row = _migrate_fragment_dict(row, _COLUMN_SET)
    unknown = set(row.keys()) - _COLUMN_SET
    if unknown:
        raise ValueError(
            f"Unknown fragment-TSV column(s): {sorted(unknown)}. "
            f"Use the annotations JSON column for tool-specific fields."
        )

    def _clean(col):
        v = row.get(col, "")
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, str) and not is_stated(v):
            return None
        return v

    values = {}
    for column in row:
        value = _clean(column)
        if column == "target_intervals":
            values[column] = (
                [tuple(pair) for pair in json.loads(value)]
                if value is not None else None
            )
        elif column in ("annotations", "field_provenance"):
            values[column] = json.loads(value) if value is not None else {}
        elif column in ("gene_expression", "transcript_expression"):
            values[column] = float(value) if value is not None else None
        elif column.startswith("n_rna_"):
            # Blank means unknown, not zero.
            values[column] = int(float(value)) if value is not None else None
        else:
            values[column] = str(value) if value is not None else None

    if not values.get("fragment_id"):
        raise ValueError("fragment TSV row is missing fragment_id")
    if values.get("sequence") is None:
        values["sequence"] = ""
    return ProteinFragment.from_dict(values)


def write_fragments(fragments: Iterable[ProteinFragment], path, sep: str = "\t") -> None:
    """Write fragments to a TSV (or custom-separator) file.

    Parameters
    ----------
    fragments : iterable of ProteinFragment
    path : str or Path
    sep : str
        Column separator (default tab).
    """
    rows = [_fragment_to_row(f) for f in fragments]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_csv(Path(path), sep=sep, index=False)


def read_fragments(path, sep: str = "\t") -> List[ProteinFragment]:
    """Read fragments from a TSV (or custom-separator) file.

    Missing columns fall back to field defaults.  Unknown columns raise.
    """
    df = pd.read_csv(Path(path), sep=sep, dtype=object, keep_default_na=False)
    return [_row_to_fragment(r) for r in df.to_dict(orient="records")]


def iter_fragments(path, sep: str = "\t") -> Iterator[ProteinFragment]:
    """Stream fragments from a file one at a time (for large inputs)."""
    for chunk in pd.read_csv(
        Path(path), sep=sep, dtype=object, keep_default_na=False, chunksize=1000,
    ):
        for record in chunk.to_dict(orient="records"):
            yield _row_to_fragment(record)
