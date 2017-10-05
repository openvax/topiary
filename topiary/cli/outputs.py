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

from __future__ import print_function, division, absolute_import

import logging

def add_output_args(arg_parser):
    output_group = arg_parser.add_argument_group(
        title="Output",
        description="How and where to write results")

    output_group.add_argument(
        "--output-csv",
        default=None,
        help="Path to output CSV file")

    output_group.add_argument(
        "--output-html",
        default=None,
        help="Path to output HTML file")

    output_group.add_argument(
        "--output-csv-sep",
        default=",",
        help="Separator for CSV file")

    output_group.add_argument(
        "--subset-output-columns",
        nargs="*")

    output_group.add_argument(
        "--rename-output-column",
        nargs=2,
        action="append",
        help=(
            "Rename original column (first parameter) to new"
            " name (second parameter)"))

    output_group.add_argument(
        "--print-columns",
        default=False,
        action="store_true",
        help="Print columns before writing data to file(s)")

    return output_group

def write_outputs(
        df,
        args,
        print_df_before_filtering=False,
        print_df_after_filtering=False):
    if print_df_before_filtering:
        print(df)

    if args.subset_output_columns:
        subset_columns = []
        for column in args.subset_output_columns:
            if column not in df.columns:
                logging.warn(
                    "Invalid column name '%s', available: %s" % (
                        column, list(df.columns)))
            else:
                subset_columns.append(column)
        df = df[subset_columns]

    if args.rename_output_column:
        for (old_name, new_name) in args.rename_output_column:
            if old_name not in df.columns:
                logging.warn(
                    "Can't rename column '%s' since it doesn't exist, available: %s" % (
                        old_name, list(df.columns)))
            else:
                df.rename(columns={old_name: new_name}, inplace=True)

    if print_df_after_filtering:
        print(df)

    if args.print_columns:
        print("Columns:")
        for column in df.columns:
            print("-- %s" % column)

    if args.output_csv:
        print("Saving %s..." % args.output_csv)
        df.to_csv(
            args.output_csv,
            index=True,
            index_label="#",
            sep=args.output_csv_sep)

    if args.output_html:
        print("Saving %s..." % args.output_html)
        df.to_html(args.output_html, index=True)
