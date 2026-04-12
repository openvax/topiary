"""Read and write Topiary DataFrames with comment-block metadata.

File format
-----------
A topiary TSV/CSV file may begin with ``#key=value`` comment lines::

    #topiary_version=4.11.0
    #form=long
    #model:netmhcpan=4.1b
    #model:mhcflurry=2.1.1
    peptide\\tallele\\tkind\\t...

Standard tools (``pd.read_csv(comment="#")``) skip these lines and read
the data normally.  Topiary's ``read_tsv`` / ``read_csv`` additionally
parse the comment block into a :class:`Metadata` object.
"""

from collections import OrderedDict
from dataclasses import dataclass, field as dataclass_field
from io import StringIO
from pathlib import Path

import pandas as pd


@dataclass
class Metadata:
    """Comment-block metadata for a topiary file."""

    topiary_version: str = None
    form: str = None
    models: dict = dataclass_field(default_factory=OrderedDict)
    sources: list = dataclass_field(default_factory=list)
    filter_by: str = None
    sort_by: str = None
    extra: dict = dataclass_field(default_factory=OrderedDict)


# -- Comment block parsing / formatting ------------------------------------


def _parse_comment_block(lines):
    """Parse ``#key=value`` lines into a Metadata object.

    Returns (Metadata, n_comment_lines).
    """
    meta = Metadata()
    n = 0
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        n += 1
        content = stripped[1:]
        if "=" not in content:
            continue
        key, _, value = content.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "topiary_version":
            meta.topiary_version = value
        elif key == "form":
            meta.form = value
        elif key == "source":
            meta.sources.append(value)
        elif key == "filter_by":
            meta.filter_by = value
        elif key == "sort_by":
            meta.sort_by = value
        elif key.startswith("model:"):
            model_name = key[len("model:"):]
            meta.models[model_name] = value
        else:
            meta.extra[key] = value

    # Also parse bare #model:name lines (no =, version-less).
    # These were skipped by the "=" check above, so re-scan.
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        content = stripped[1:]
        if "=" not in content and content.startswith("model:"):
            model_name = content[len("model:"):]
            if model_name and model_name not in meta.models:
                meta.models[model_name] = ""

    return meta, n


def _format_comment_block(meta):
    """Format a Metadata object as ``#key=value`` lines."""
    lines = []
    if meta.topiary_version:
        lines.append(f"#topiary_version={meta.topiary_version}")
    if meta.form:
        lines.append(f"#form={meta.form}")
    for source in meta.sources:
        lines.append(f"#source={source}")
    for model_name, version in meta.models.items():
        if version:
            lines.append(f"#model:{model_name}={version}")
        else:
            lines.append(f"#model:{model_name}")
    if meta.filter_by:
        lines.append(f"#filter_by={meta.filter_by}")
    if meta.sort_by:
        lines.append(f"#sort_by={meta.sort_by}")
    for key, value in meta.extra.items():
        lines.append(f"#{key}={value}")
    return "\n".join(lines)


# -- Read ------------------------------------------------------------------


def _read_delimited(path, sep, tag=None):
    from .result import TopiaryResult

    path = Path(path)
    with open(path) as f:
        all_lines = f.readlines()

    meta, n_comment = _parse_comment_block(all_lines)

    data_text = "".join(all_lines[n_comment:])
    if not data_text.strip():
        df = pd.DataFrame()
    else:
        df = pd.read_csv(StringIO(data_text), sep=sep)

    # Record source (tag overrides filename).
    source_label = tag if tag is not None else path.name
    if source_label and source_label not in meta.sources:
        meta.sources.append(source_label)

    # Add a per-row source column if it's not already present.
    if "source" not in df.columns and len(df) > 0:
        df["source"] = source_label

    return TopiaryResult(df, meta)


def read_tsv(path, tag=None):
    """Read a topiary TSV file with comment-block metadata.

    Parameters
    ----------
    path : str or Path
    tag : str, optional
        Label for this file's rows.  Defaults to the filename.
        Used to populate the ``source`` column and Metadata.sources.

    Returns
    -------
    TopiaryResult
    """
    return _read_delimited(path, sep="\t", tag=tag)


def read_csv(path, tag=None):
    """Read a topiary CSV file with comment-block metadata.

    Parameters
    ----------
    path : str or Path
    tag : str, optional
        Label for this file's rows.  Defaults to the filename.

    Returns
    -------
    TopiaryResult
    """
    return _read_delimited(path, sep=",", tag=tag)


# -- Write -----------------------------------------------------------------


def _write_delimited(df, path, sep, metadata, index):
    from . import __version__
    from .wide import detect_form

    # Accept TopiaryResult too — pull out its df and metadata.
    # Use duck typing to avoid a circular import.
    if hasattr(df, "df") and hasattr(df, "metadata"):
        if metadata is None:
            metadata = df.metadata
        df = df.df

    path = Path(path)

    if metadata is None:
        metadata = Metadata()

    if not metadata.topiary_version:
        metadata.topiary_version = __version__

    if not metadata.form:
        metadata.form = detect_form(df)

    # Auto-extract model versions from long-form data.
    if (
        not metadata.models
        and "prediction_method_name" in df.columns
        and "predictor_version" in df.columns
    ):
        for method, version in (
            df.dropna(subset=["prediction_method_name"])
            .groupby("prediction_method_name")["predictor_version"]
            .first()
            .items()
        ):
            version_str = str(version).strip() if pd.notna(version) else ""
            if version_str:
                metadata.models[str(method)] = version_str
            else:
                # Record model even without version
                metadata.models[str(method)] = ""

    comment_block = _format_comment_block(metadata)

    with open(path, "w") as f:
        if comment_block:
            f.write(comment_block + "\n")
        df.to_csv(f, sep=sep, index=index)


def to_tsv(df, path, metadata=None, index=False):
    """Write a topiary DataFrame to TSV with comment-block metadata."""
    _write_delimited(df, path, sep="\t", metadata=metadata, index=index)


def to_csv(df, path, metadata=None, index=False):
    """Write a topiary DataFrame to CSV with comment-block metadata."""
    _write_delimited(df, path, sep=",", metadata=metadata, index=index)
