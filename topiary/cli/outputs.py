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


import logging
import sys


_PREVIEW_ROWS = 20
_PREVIEW_COLUMNS = (
    "peptide", "allele", "kind", "value", "value_unit", "score",
    "percentile_rank", "source_sequence_name",
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
        "--output-csv-sep", default=",", help="Separator for CSV file"
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

    if args.subset_output_columns:
        subset_columns = []
        for column in args.subset_output_columns:
            if column not in df.columns:
                logging.warning(
                    "Invalid column name '%s', available: %s"
                    % (column, list(df.columns))
                )
            else:
                subset_columns.append(column)
        df = df.loc[:, subset_columns].copy()

    preview_columns = (
        list(df.columns) if args.subset_output_columns else
        [column for column in _PREVIEW_COLUMNS if column in df.columns]
    )
    if not preview_columns:
        preview_columns = list(df.columns)

    if args.rename_output_column:
        for old_name, new_name in args.rename_output_column:
            if old_name not in df.columns:
                logging.warning(
                    "Can't rename column '%s' since it doesn't exist, available: %s"
                    % (old_name, list(df.columns))
                )
            else:
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

    # Finish file outputs before streaming: a pipe consumer may stop early.
    if args.output_html:
        print("Saving %s..." % args.output_html, file=sys.stderr)
        df.to_html(args.output_html, index=True)

    if args.output_csv:
        destination = sys.stdout if csv_stdout else args.output_csv
        if not csv_stdout:
            print("Saving %s..." % args.output_csv, file=sys.stderr)
        df.to_csv(destination, index=True, index_label="#", sep=args.output_csv_sep)

    if not args.output_csv and not args.output_html and len(df):
        print(df.head(_PREVIEW_ROWS).loc[:, preview_columns].to_string(index=False))
        if len(df) > _PREVIEW_ROWS:
            print(
                f"Showing {_PREVIEW_ROWS} of {len(df)} prediction rows "
                f"({len(df) - _PREVIEW_ROWS} omitted). "
                "Use --output-csv PATH or --output-csv - for all rows.",
                file=sys.stderr,
            )
