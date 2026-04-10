"""
Generic expression data loader with auto-detection for common formats.

Supports Salmon, Kallisto, RSEM, StringTie GTF, Cufflinks, and generic
TSV/CSV files. Returns a DataFrame with an ID column and one or more
numeric value columns.
"""

import logging
import os

import pandas as pd

from .common import infer_delimiter

logger = logging.getLogger(__name__)

# Default column mappings for auto-detected formats
_FORMAT_DEFAULTS = {
    "salmon": {"id_col": "Name", "val_col": "TPM"},
    "kallisto": {"id_col": "target_id", "val_col": "tpm"},
    "rsem_gene": {"id_col": "gene_id", "val_col": "TPM"},
    "rsem_transcript": {"id_col": "transcript_id", "val_col": "TPM"},
    "stringtie_gtf": {"id_col": "reference_id", "val_col": "FPKM"},
    "cufflinks": {"id_col": "tracking_id", "val_col": "FPKM"},
}


def detect_format(filepath):
    """Detect expression file format from filename and header.

    Returns a format key from _FORMAT_DEFAULTS, or None for generic files.
    """
    basename = os.path.basename(filepath).lower()

    if basename.endswith(".sf"):
        return "salmon"
    if basename.endswith(".gtf"):
        return "stringtie_gtf"
    if basename.endswith(".fpkm_tracking"):
        return "cufflinks"
    if basename.endswith(".genes.results"):
        return "rsem_gene"
    if basename.endswith(".isoforms.results"):
        return "rsem_transcript"

    # Check header for Kallisto
    if "abundance" in basename and basename.endswith(".tsv"):
        try:
            header = pd.read_csv(filepath, sep="\t", nrows=0).columns.tolist()
            if "target_id" in header and "tpm" in header:
                return "kallisto"
        except Exception:
            pass

    # Check header for Salmon (if not caught by .sf extension)
    try:
        header = pd.read_csv(filepath, sep="\t", nrows=0).columns.tolist()
        if "Name" in header and "TPM" in header and "NumReads" in header:
            return "salmon"
    except Exception:
        pass

    return None


def load_expression(filepath, id_col=None, val_cols=None):
    """Load expression data from a file.

    Parameters
    ----------
    filepath : str
        Path to expression file.
    id_col : str, optional
        Column name for IDs. Auto-detected if None.
    val_cols : str or list of str, optional
        Column name(s) for values. If None, auto-detect from format.
        If "*", load all numeric columns except the ID column.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: id_col (renamed to original name), plus
        one or more value columns.
    """
    fmt = detect_format(filepath)

    if fmt == "stringtie_gtf":
        return _load_gtf(filepath, id_col, val_cols)

    # Determine defaults from format
    if fmt and fmt in _FORMAT_DEFAULTS:
        defaults = _FORMAT_DEFAULTS[fmt]
        if id_col is None:
            id_col = defaults["id_col"]
        if val_cols is None:
            val_cols = defaults["val_col"]
        logger.info(
            "Detected %s format for %s (id=%s, value=%s)",
            fmt, filepath, id_col, val_cols,
        )
    elif id_col is None or val_cols is None:
        # Generic file — try to read header and pick columns
        return _load_generic(filepath, id_col, val_cols)

    # Read the file
    sep = _infer_sep(filepath)
    df = pd.read_csv(filepath, sep=sep, comment="#")

    # Normalize val_cols to a list
    if isinstance(val_cols, str):
        if val_cols == "*":
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            val_cols = [c for c in numeric_cols if c != id_col]
        else:
            val_cols = [val_cols]

    # Validate columns exist
    _check_columns(df, filepath, id_col, val_cols)

    keep = [id_col] + val_cols
    result = df[keep].copy()

    # Convert value columns to float
    for vc in val_cols:
        result[vc] = pd.to_numeric(result[vc], errors="coerce")

    # Drop rows with missing IDs
    result = result.dropna(subset=[id_col])

    logger.info(
        "Loaded %d rows from %s (%s)",
        len(result), filepath, ", ".join(val_cols),
    )
    return result


def _load_gtf(filepath, id_col=None, val_cols=None):
    """Load expression data from a StringTie GTF file."""
    try:
        import gtfparse
    except ImportError:
        raise ImportError(
            "gtfparse is required to read GTF expression files. "
            "Install with: pip install gtfparse"
        )

    df = gtfparse.read_gtf(filepath)
    # Filter to transcript features only
    if "feature" in df.columns:
        df = df[df["feature"] == "transcript"]

    id_col = id_col or "reference_id"
    if val_cols is None:
        # Prefer TPM if available, fall back to FPKM
        if "TPM" in df.columns:
            val_cols = ["TPM"]
        elif "FPKM" in df.columns:
            val_cols = ["FPKM"]
        else:
            raise ValueError(
                f"GTF file {filepath} has neither TPM nor FPKM columns"
            )
    elif isinstance(val_cols, str):
        val_cols = [val_cols] if val_cols != "*" else [
            c for c in df.select_dtypes(include="number").columns
            if c != id_col
        ]

    _check_columns(df, filepath, id_col, val_cols)

    result = df[[id_col] + val_cols].copy()
    for vc in val_cols:
        result[vc] = pd.to_numeric(result[vc], errors="coerce")
    result = result.dropna(subset=[id_col])

    logger.info("Loaded %d rows from GTF %s", len(result), filepath)
    return result


def _load_generic(filepath, id_col=None, val_cols=None):
    """Load a generic TSV/CSV, inferring columns if needed."""
    sep = _infer_sep(filepath)
    df = pd.read_csv(filepath, sep=sep, comment="#")

    if id_col is None:
        # Use first column as ID
        id_col = df.columns[0]
        logger.info("Using first column %r as ID for %s", id_col, filepath)

    if val_cols is None or val_cols == "*":
        # All numeric columns except ID
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        val_cols = [c for c in numeric_cols if c != id_col]
        if not val_cols:
            raise ValueError(
                f"No numeric columns found in {filepath} "
                f"(besides ID column {id_col!r})"
            )
        logger.info(
            "Auto-detected value columns for %s: %s", filepath, val_cols,
        )

    if isinstance(val_cols, str):
        val_cols = [val_cols]

    _check_columns(df, filepath, id_col, val_cols)

    result = df[[id_col] + val_cols].copy()
    for vc in val_cols:
        result[vc] = pd.to_numeric(result[vc], errors="coerce")
    result = result.dropna(subset=[id_col])

    logger.info("Loaded %d rows from %s", len(result), filepath)
    return result


def _infer_sep(filepath):
    """Infer delimiter, defaulting to tab."""
    try:
        sep = infer_delimiter(filepath)
        # Convert regex patterns to literal for pd.read_csv C engine
        if sep == r"\t":
            return "\t"
        if sep == r"\s+":
            return None  # whitespace — use python engine
        return sep
    except (ValueError, FileNotFoundError):
        return "\t"


def _check_columns(df, filepath, id_col, val_cols):
    """Validate that required columns exist in the DataFrame."""
    if id_col not in df.columns:
        available = sorted(df.columns.tolist())
        raise ValueError(
            f"ID column {id_col!r} not found in {filepath}. "
            f"Available: {available}"
        )
    if isinstance(val_cols, list):
        for vc in val_cols:
            if vc not in df.columns:
                available = sorted(df.columns.tolist())
                raise ValueError(
                    f"Value column {vc!r} not found in {filepath}. "
                    f"Available: {available}"
                )


def parse_expression_spec(spec):
    """Parse a CLI expression spec string.

    Formats:
        "file.tsv"                          → (None, "file.tsv", None, None)
        "gene_tpm:file.tsv"                 → ("gene_tpm", "file.tsv", None, None)
        "gene_tpm:file.tsv:Name:TPM"        → ("gene_tpm", "file.tsv", "Name", "TPM")
        "file.tsv:Name:TPM"                 → (None, "file.tsv", "Name", "TPM")

    Returns (name, filepath, id_col, val_cols).
    """
    parts = spec.split(":")

    # Handle the case where path might look like a single element
    # or where the first part is a name vs a filepath
    if len(parts) == 1:
        # Just a file path
        return None, parts[0], None, None

    if len(parts) == 2:
        # Could be name:file or file:id_col
        # Heuristic: if the first part looks like a file (has extension or /), treat as file
        if _looks_like_file(parts[0]):
            return None, parts[0], parts[1], None
        return parts[0], parts[1], None, None

    if len(parts) == 3:
        # Could be name:file:id_col or file:id_col:val_col
        if _looks_like_file(parts[0]):
            return None, parts[0], parts[1], parts[2]
        return parts[0], parts[1], parts[2], None

    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]

    raise ValueError(
        f"Invalid expression spec {spec!r}. "
        f"Expected: [name:]file[:id_col[:val_col]]"
    )


def _looks_like_file(s):
    """Heuristic: does this string look like a file path?"""
    return "/" in s or "\\" in s or "." in s


def load_expression_from_spec(spec, default_name=None):
    """Parse a spec string and load the expression data.

    Parameters
    ----------
    spec : str
        CLI expression spec (see parse_expression_spec).
    default_name : str, optional
        Default name prefix if not specified in spec.

    Returns
    -------
    (name, id_col, DataFrame)
        The name prefix, the original ID column name, and the loaded data.
    """
    name, filepath, id_col, val_cols = parse_expression_spec(spec)
    name = name or default_name

    df = load_expression(filepath, id_col=id_col, val_cols=val_cols)

    # The first column is the ID column
    actual_id_col = df.columns[0]

    return name, actual_id_col, df
