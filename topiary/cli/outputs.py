# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common commandline arguments for output files
"""


import argparse
import html
import sys
from pathlib import Path

import pandas as pd

from ..io import _encode_json, _json_columns


# Six significant digits. A predictor reporting 6.296 was written as
# 6.296000000000002, and an affinity as 11927.161249112096 (#326); the
# digits past the sixth are noise from binary floating point.
_FLOAT_PRECISION = "%.6g"
_SOURCE_ORDER = "_topiary_source_order"
_PREVIEW_ROWS = 20
_PREVIEW_COLUMNS = (
    "peptide", "allele", "kind", "value", "value_unit", "score",
    "percentile_rank", "source_sequence_name",
)


def _single_character(value):
    """argparse type for --output-csv-sep: pandas writes only one-character separators.

    Checked while parsing, so a bad separator fails before predicting
    rather than after, leaving an empty file behind (#310).
    """
    if len(value) != 1:
        raise argparse.ArgumentTypeError(f"must be a single character, got {value!r}")
    return value


def _format_float(value):
    """Six significant digits, still recognizable as a float.

    ``"%.6g" % 2.0`` is ``"2"``, which pandas reads back as ``int64``, so
    writing the output and reloading it -- replaying a CLI file as a
    prediction cache, say -- would change a measurement column's dtype.
    Keeping the decimal point holds the dtype steady. Values needing an
    exponent already read back as floats.
    """
    text = _FLOAT_PRECISION % value
    return f"{text}.0" if text.lstrip("-").isdigit() else text


def _ordered_rows(df, args):
    """Put the rows in the order the output should present them.

    With ``--sort-by``, keep the order the ranking produced. Without it, a
    protein scan came out alphabetical by peptide, so a window at offset 16
    preceded one at offset 1 (#326). Rows now keep the order their source
    sequences first appeared and run in position order within each one.
    A peptide list, which has one row per source sequence, keeps its input
    order either way.
    """
    if getattr(args, "sort_by", None) or df.empty:
        return df
    if not {"source_sequence_name", "peptide_offset"} <= set(df.columns):
        return df
    first_seen = df.groupby("source_sequence_name", sort=False).ngroup()
    return (
        df.assign(**{_SOURCE_ORDER: first_seen})
        .sort_values([_SOURCE_ORDER, "peptide_offset"], kind="mergesort")
        .drop(columns=[_SOURCE_ORDER])
        .reset_index(drop=True)
    )


def _with_output_row(df, args):
    """Number the rows of a sorted result so their order survives re-sorting.

    Only with ``--sort-by``: without one the order is the scan's own
    positions, which ``peptide_offset`` already states.

    ``output_row`` is the row's 1-based position in this file, not a
    per-candidate rank. Sorting on a measurement puts the rows that have
    one first, so a peptide's other kinds sit further down rather than
    beside it; ``rank_candidates`` produces ``candidate_rank`` for a real
    per-candidate ranking.
    """
    if not getattr(args, "sort_by", None) or df.empty or "output_row" in df.columns:
        return df
    numbered = df.copy()
    numbered.insert(0, "output_row", range(1, len(numbered) + 1))
    return numbered


def _write_html(df, path):
    """Write a standalone page, not a bare ``<table>`` fragment (#18, #326).

    Missing cells are emptied to match the CSV writer, which renders both
    ``None`` and ``NaN`` as an empty cell; ``to_html`` printed the literal
    text ``None`` beside empty strings for the same missing value. Its
    ``na_rep`` does not reach ``None`` in an object column, so the cells
    are replaced here. A cell holding the *string* ``"None"`` is real text
    and is left alone.
    """
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else _format_float(value))
    display = display.astype(object).where(display.notna(), "")
    table = display.to_html(index=False)
    Path(path).write_text(
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(Path(path).stem)}</title>\n"
        "</head>\n<body>\n"
        f"{table}\n"
        "</body>\n</html>\n",
        encoding="utf-8",
    )


def add_output_args(arg_parser):
    output_group = arg_parser.add_argument_group(
        title="Output", description=(
            "How and where to write results. With no output path, show the "
            "first 20 prediction rows as a compact table."
        )
    )

    output_group.add_argument(
        "--output-csv", default=None,
        help="Path to output CSV file, or '-' for all rows on stdout",
    )

    output_group.add_argument(
        "--output-html", default=None, help="Path to output HTML file"
    )

    output_group.add_argument(
        "--output-csv-sep", default=",", type=_single_character,
        help="Separator for CSV file (one character)",
    )

    output_group.add_argument(
        "--subset-output-columns", nargs="*",
        help="Columns to include in the table preview and output files",
    )

    output_group.add_argument(
        "--rename-output-column",
        nargs=2,
        action="append",
        help=(
            "Rename original column (first parameter) to new" " name (second parameter)"
        ),
    )

    output_group.add_argument(
        "--print-columns",
        default=False,
        action="store_true",
        help="Print available output columns (to stderr when CSV uses stdout)",
    )

    return output_group


def write_outputs(
    df, args, print_df_before_filtering=False, print_df_after_filtering=False
):
    """Write exports or a compact preview of an already filtered result.

    Parameters
    ----------
    df : pandas.DataFrame
        Prediction rows in their final filtered and ranked order. An empty
        frame produces no preview rows; explicit exports are still written.
    args : argparse.Namespace
        Options from :func:`add_output_args`. A CSV path of ``'-'`` writes
        all rows to stdout. Without an output path, show up to 20 rows.
        Column selection and renaming apply to both exports and the preview.
    print_df_before_filtering, print_df_after_filtering : bool
        Legacy diagnostic prints before/after output-column selection.
        These go to stderr when stdout carries CSV, to keep it parseable.
    """
    csv_stdout = args.output_csv == "-"
    diagnostic_stream = sys.stderr if csv_stdout else sys.stdout
    if print_df_before_filtering:
        print(df, file=diagnostic_stream)

    df = _with_output_row(_ordered_rows(df, args), args)

    # An unknown column is an error, raised before anything is written: a
    # warning let a typo produce an index-only file and exit 0 (#326).
    all_columns = list(df.columns)
    if args.subset_output_columns:
        unknown = [c for c in args.subset_output_columns if c not in all_columns]
        if unknown:
            raise ValueError(
                f"--subset-output-columns: no column named {', '.join(map(repr, unknown))}; "
                f"available: {', '.join(all_columns)}")
        df = df.loc[:, args.subset_output_columns].copy()

    preview_columns = (
        list(df.columns) if args.subset_output_columns else
        [column for column in _PREVIEW_COLUMNS if column in df.columns]
    )
    if not preview_columns:
        preview_columns = list(df.columns)

    if args.rename_output_column:
        for old_name, new_name in args.rename_output_column:
            if old_name not in df.columns:
                reason = ("--subset-output-columns removed it"
                          if old_name in all_columns else "no such column")
                raise ValueError(
                    f"--rename-output-column: cannot rename {old_name!r}: {reason}; "
                    f"available: {', '.join(map(str, df.columns))}")
            df = df.rename(columns={old_name: new_name})
            preview_columns = [
                new_name if column == old_name else column
                for column in preview_columns
            ]

    if print_df_after_filtering:
        print(df, file=diagnostic_stream)

    if args.print_columns:
        print("Columns:", file=diagnostic_stream)
        for column in df.columns:
            print("-- %s" % column, file=diagnostic_stream)

    # Dicts and lists such as measurement_context would otherwise be written
    # as Python reprs, which only Python can read back. Missing cells stay
    # missing: these outputs carry no topiary encoding declaration.
    json_columns = _json_columns(df)
    if json_columns:
        df = df.copy()
        for column in json_columns:
            df[column] = df[column].map(
                lambda value, column=column: _encode_json(value, column=column)
                if isinstance(value, (dict, list)) else value)

    # Finish file outputs before streaming: a pipe consumer may stop early.
    if args.output_html:
        print("Saving %s..." % args.output_html, file=sys.stderr)
        _write_html(df, args.output_html)

    if args.output_csv:
        destination = sys.stdout if csv_stdout else args.output_csv
        if not csv_stdout:
            print("Saving %s..." % args.output_csv, file=sys.stderr)
        # index=False: the row number was written under the header "#", so
        # every reader that treats '#' as a comment -- including topiary's
        # own read_csv -- skipped the header and lost a row (#326).
        df.to_csv(destination, index=False, sep=args.output_csv_sep,
                  float_format=_format_float)

    if not args.output_csv and not args.output_html and len(df):
        print(df.head(_PREVIEW_ROWS).loc[:, preview_columns].to_string(index=False))
        if len(df) > _PREVIEW_ROWS:
            print(
                f"Showing {_PREVIEW_ROWS} of {len(df)} prediction rows "
                f"({len(df) - _PREVIEW_ROWS} omitted). "
                "Use --output-csv PATH or --output-csv - for all rows.",
                file=sys.stderr,
            )
