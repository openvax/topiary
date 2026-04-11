"""Wide/long DataFrame conversion for Topiary prediction DataFrames.

Long form: one row per (peptide, allele, model, kind).
Wide form: one row per (peptide, allele, source), prediction columns become
``{model}_{kind}_{field}`` (e.g. ``netmhcpan_affinity_value``).
"""

import warnings

import numpy as np
import pandas as pd

from .ranking import _iter_known_kinds, _kind_name, _kind_short_name, _KIND_ALIASES

# Columns that are prediction-specific and get pivoted in wide form.
PREDICTION_COLUMNS = frozenset({
    "kind", "score", "value", "percentile_rank",
    "prediction_method_name", "predictor_version", "affinity",
})

# Wide-form field suffixes.
WIDE_FIELDS = frozenset({"value", "score", "rank"})

# Long column name → wide field suffix.
LONG_TO_WIDE_FIELD = {
    "value": "value",
    "score": "score",
    "percentile_rank": "rank",
}

# Wide field suffix → long column name.
WIDE_TO_LONG_FIELD = {v: k for k, v in LONG_TO_WIDE_FIELD.items()}


def _known_kind_short_names():
    """Return known kind short names sorted longest-first (for suffix matching)."""
    names = set()
    for kind in _iter_known_kinds():
        names.add(_kind_short_name(kind))
    return sorted(names, key=len, reverse=True)


def _kind_short_to_canonical(short_name):
    """Map a short kind name back to the canonical mhctools kind name string."""
    kind = _KIND_ALIASES.get(short_name)
    if kind is None:
        return short_name
    return _kind_name(kind)


def _parse_wide_column(col_name):
    """Parse a wide-form column name into (model_key, kind_short, field).

    Returns None if the column does not match the ``{model}_{kind}_{field}``
    pattern where kind is a known prediction kind and field is one of
    value/score/rank.
    """
    # Split off the rightmost segment as field candidate.
    parts = col_name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    prefix, field = parts
    if field not in WIDE_FIELDS:
        return None

    # Try matching against known kind short names, longest first, so
    # multi-underscore kinds like "antigen_processing" match before
    # shorter kinds.
    for kind_short in _known_kind_short_names():
        if prefix == kind_short:
            # Kind-only column with no model prefix — unusual but valid
            return (None, kind_short, field)
        if prefix.endswith("_" + kind_short):
            model_key = prefix[: -(len(kind_short) + 1)]
            if model_key:
                return (model_key, kind_short, field)

    return None


def detect_form(df):
    """Detect whether a DataFrame is in long or wide form.

    Returns ``"long"``, ``"wide"``, or ``"unknown"``.
    """
    if "kind" in df.columns:
        return "long"
    for col in df.columns:
        if _parse_wide_column(col) is not None:
            return "wide"
    return "unknown"


def to_wide(df):
    """Convert a long-form prediction DataFrame to wide form.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form DataFrame with a ``kind`` column.

    Returns
    -------
    pandas.DataFrame
        Wide-form DataFrame where prediction columns become
        ``{model}_{kind}_{field}`` columns.
    """
    if "kind" not in df.columns:
        raise ValueError(
            "DataFrame is not in long form: missing 'kind' column"
        )

    group_cols = [c for c in df.columns if c not in PREDICTION_COLUMNS]

    if df.empty:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    # Determine model keys.  Include version only on collision.
    version_collision = False
    if "prediction_method_name" in df.columns and "predictor_version" in df.columns:
        version_counts = (
            df.dropna(subset=["prediction_method_name"])
            .groupby("prediction_method_name")["predictor_version"]
            .nunique()
        )
        if (version_counts > 1).any():
            version_collision = True
            colliding = version_counts[version_counts > 1].index.tolist()
            warnings.warn(
                f"Multiple predictor versions for {colliding}; "
                "including version in wide column names",
                UserWarning,
                stacklevel=2,
            )

    work = df.copy()

    # Build model key per row.
    if "prediction_method_name" in work.columns:
        model_col = work["prediction_method_name"].fillna("unknown").astype(str)
    else:
        model_col = pd.Series("unknown", index=work.index)

    if version_collision and "predictor_version" in work.columns:
        version_col = work["predictor_version"].fillna("").astype(str)
        work["_model_key"] = model_col + "_" + version_col
    else:
        work["_model_key"] = model_col

    work["_kind_short"] = work["kind"].apply(
        lambda k: _kind_short_name(k) if hasattr(k, "name") or isinstance(k, str) else str(k)
    )

    # Build model→version metadata for .attrs.
    model_versions = {}
    if "prediction_method_name" in df.columns and "predictor_version" in df.columns:
        for method, version in (
            df.dropna(subset=["prediction_method_name"])
            .groupby("prediction_method_name")["predictor_version"]
            .first()
            .items()
        ):
            if pd.notna(version) and str(version):
                model_versions[str(method)] = str(version)

    # Melt each long field into wide column entries.
    records = []
    for long_field, wide_field in LONG_TO_WIDE_FIELD.items():
        if long_field not in work.columns:
            continue
        temp = work[group_cols + ["_model_key", "_kind_short", long_field]].copy()
        temp["_wide_col"] = (
            temp["_model_key"] + "_" + temp["_kind_short"] + "_" + wide_field
        )
        temp = temp.rename(columns={long_field: "_wide_val"})
        records.append(temp[group_cols + ["_wide_col", "_wide_val"]])

    if not records:
        return work[group_cols].drop_duplicates().reset_index(drop=True)

    melted = pd.concat(records, ignore_index=True)

    # Pivot: group keys as index, wide column names as columns.
    wide = melted.pivot_table(
        index=group_cols,
        columns="_wide_col",
        values="_wide_val",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    if model_versions:
        wide.attrs["topiary_models"] = model_versions

    return wide


def from_wide(df, metadata=None):
    """Convert a wide-form prediction DataFrame back to long form.

    Parameters
    ----------
    df : pandas.DataFrame
        Wide-form DataFrame with ``{model}_{kind}_{field}`` columns.
    metadata : topiary.io.Metadata, optional
        If provided, model versions are used to populate
        ``predictor_version``.

    Returns
    -------
    pandas.DataFrame
        Long-form DataFrame with ``kind``, ``score``, ``value``,
        ``percentile_rank``, ``prediction_method_name``, and
        ``predictor_version`` columns.
    """
    # Classify columns.
    pred_mapping = {}  # (model_key, kind_short) → {field: col_name}
    group_cols = []

    for col in df.columns:
        parsed = _parse_wide_column(col)
        if parsed is not None:
            model_key, kind_short, field = parsed
            key = (model_key, kind_short)
            pred_mapping.setdefault(key, {})[field] = col
        else:
            group_cols.append(col)

    if not pred_mapping:
        # No prediction columns found — return as-is with empty long columns.
        result = df.copy()
        for col in ["kind", "score", "value", "percentile_rank",
                     "prediction_method_name", "predictor_version", "affinity"]:
            if col not in result.columns:
                result[col] = np.nan
        return result

    # Build version lookup from metadata.
    version_lookup = {}
    if metadata is not None and hasattr(metadata, "models"):
        version_lookup = metadata.models or {}
    # Also check .attrs if available.
    if not version_lookup and hasattr(df, "attrs"):
        version_lookup = df.attrs.get("topiary_models", {})

    # For each group-key row, emit one long row per (model, kind).
    group_df = df[group_cols]
    long_rows = []

    for mk_kind, field_map in pred_mapping.items():
        model_key, kind_short = mk_kind
        canonical_kind = _kind_short_to_canonical(kind_short)

        chunk = group_df.copy()
        chunk["kind"] = canonical_kind
        chunk["prediction_method_name"] = model_key

        # Resolve version from metadata.
        version = version_lookup.get(model_key, np.nan)
        chunk["predictor_version"] = version

        for wide_field, long_col in WIDE_TO_LONG_FIELD.items():
            if wide_field in field_map:
                chunk[long_col] = df[field_map[wide_field]].values
            else:
                chunk[long_col] = np.nan

        long_rows.append(chunk)

    result = pd.concat(long_rows, ignore_index=True)

    # Reconstruct the affinity convenience column.
    is_affinity = result["kind"].apply(
        lambda k: _kind_name(k) if hasattr(k, "name") else str(k)
    ) == "pMHC_affinity"
    result["affinity"] = np.where(is_affinity, result["value"], np.nan)

    return result
