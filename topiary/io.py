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
        elif key.startswith("model:"):
            model_name = key[len("model:"):]
            meta.models[model_name] = value
        else:
            meta.extra[key] = value

    return meta, n


def _format_comment_block(meta):
    """Format a Metadata object as ``#key=value`` lines."""
    lines = []
    if meta.topiary_version:
        lines.append(f"#topiary_version={meta.topiary_version}")
    if meta.form:
        lines.append(f"#form={meta.form}")
    for model_name, version in meta.models.items():
        lines.append(f"#model:{model_name}={version}")
    for key, value in meta.extra.items():
        lines.append(f"#{key}={value}")
    return "\n".join(lines)


# -- Read ------------------------------------------------------------------


def _read_delimited(path, sep):
    path = Path(path)
    with open(path) as f:
        all_lines = f.readlines()

    meta, n_comment = _parse_comment_block(all_lines)

    data_text = "".join(all_lines[n_comment:])
    if not data_text.strip():
        return pd.DataFrame(), meta

    df = pd.read_csv(StringIO(data_text), sep=sep)
    return df, meta


def read_tsv(path):
    """Read a topiary TSV file with comment-block metadata.

    Returns (DataFrame, Metadata).
    """
    return _read_delimited(path, sep="\t")


def read_csv(path):
    """Read a topiary CSV file with comment-block metadata.

    Returns (DataFrame, Metadata).
    """
    return _read_delimited(path, sep=",")


# -- Write -----------------------------------------------------------------


def _write_delimited(df, path, sep, metadata, index):
    from . import __version__
    from .wide import detect_form

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
