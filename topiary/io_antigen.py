"""TSV IO for :class:`AntigenFragment` collections.

Format: one row per fragment.  Scalar fields map to columns of the same
name.  ``target_intervals`` and ``annotations`` are JSON-serialized into
their own columns; empty annotations serialize as the empty object
``{}``, absent target_intervals serialize as empty strings.

Missing columns on read fall back to field defaults.  Unknown columns
are rejected with a clear error (catches typos; use ``annotations`` for
tool-specific extensions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from .antigen import AntigenFragment

_COLUMNS = [
    "fragment_id",
    "source_type",
    "sequence",
    "reference_sequence",
    "germline_sequence",
    "target_intervals",
    "variant",
    "effect",
    "effect_type",
    "gene",
    "gene_id",
    "transcript_id",
    "gene_expression",
    "transcript_expression",
    "annotations",
]

_COLUMN_SET = set(_COLUMNS)


def _fragment_to_row(f: AntigenFragment) -> dict:
    """Convert an :class:`AntigenFragment` to a flat dict suitable for
    a TSV row (lists / dicts JSON-encoded)."""
    row = {}
    for col in _COLUMNS:
        val = getattr(f, col)
        if col == "target_intervals":
            row[col] = json.dumps([list(p) for p in val]) if val is not None else ""
        elif col == "annotations":
            row[col] = json.dumps(val or {}, sort_keys=True)
        elif val is None:
            row[col] = ""
        else:
            row[col] = val
    return row


def _row_to_fragment(row: dict) -> AntigenFragment:
    """Inverse of :func:`_fragment_to_row`."""
    unknown = set(row.keys()) - _COLUMN_SET
    if unknown:
        raise ValueError(
            f"Unknown antigen-TSV column(s): {sorted(unknown)}. "
            f"Use the annotations JSON column for tool-specific fields."
        )

    def _clean(col):
        v = row.get(col, "")
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, str) and v == "":
            return None
        return v

    ti_raw = _clean("target_intervals")
    if ti_raw is None:
        target_intervals = None
    else:
        target_intervals = [tuple(p) for p in json.loads(ti_raw)]

    ann_raw = _clean("annotations")
    annotations = json.loads(ann_raw) if ann_raw else {}

    def _num(col):
        v = _clean(col)
        return float(v) if v is not None else None

    def _str(col):
        v = _clean(col)
        return str(v) if v is not None else None

    fragment_id = _str("fragment_id")
    if fragment_id is None:
        raise ValueError("antigen TSV row is missing fragment_id")

    return AntigenFragment(
        fragment_id=fragment_id,
        source_type=_str("source_type"),
        sequence=_str("sequence") or "",
        reference_sequence=_str("reference_sequence"),
        germline_sequence=_str("germline_sequence"),
        target_intervals=target_intervals,
        variant=_str("variant"),
        effect=_str("effect"),
        effect_type=_str("effect_type"),
        gene=_str("gene"),
        gene_id=_str("gene_id"),
        transcript_id=_str("transcript_id"),
        gene_expression=_num("gene_expression"),
        transcript_expression=_num("transcript_expression"),
        annotations=annotations,
    )


def write_antigens(fragments: Iterable[AntigenFragment], path, sep: str = "\t") -> None:
    """Write fragments to a TSV (or custom-separator) file.

    Parameters
    ----------
    fragments : iterable of AntigenFragment
    path : str or Path
    sep : str
        Column separator (default tab).
    """
    rows = [_fragment_to_row(f) for f in fragments]
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df.to_csv(Path(path), sep=sep, index=False)


def read_antigens(path, sep: str = "\t") -> List[AntigenFragment]:
    """Read fragments from a TSV (or custom-separator) file.

    Missing columns fall back to field defaults.  Unknown columns raise.
    """
    df = pd.read_csv(Path(path), sep=sep, dtype=object, keep_default_na=False)
    return [_row_to_fragment(r) for r in df.to_dict(orient="records")]


def iter_antigens(path, sep: str = "\t") -> Iterator[AntigenFragment]:
    """Stream fragments from a file one at a time (for large inputs)."""
    for chunk in pd.read_csv(
        Path(path), sep=sep, dtype=object, keep_default_na=False, chunksize=1000,
    ):
        for record in chunk.to_dict(orient="records"):
            yield _row_to_fragment(record)
