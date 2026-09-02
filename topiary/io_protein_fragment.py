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
    for col in _COLUMNS:
        val = getattr(f, col)
        if col == "target_intervals":
            row[col] = json.dumps([list(p) for p in val]) if val is not None else ""
        elif col in ("annotations", "field_provenance"):
            row[col] = json.dumps(val or {}, sort_keys=True)
        elif val is None:
            row[col] = ""
        else:
            row[col] = val
    return row


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
