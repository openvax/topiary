# Copyright (c) 2016. Mount Sinai School of Medicine
#
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
        "--keep-output-columns",
        nargs="*")

    output_group.add_argument(
        "--rename-output-column",
        nargs=2,
        default=[],
        action="append")
    return output_group

def write_outputs(
        df,
        args,
        print_df_before_filtering=False,
        print_df_after_filtering=False):
    if print_df_before_filtering:
        print(df)

    if args.keep_output_columns is not None and len(args.keep_output_columns) > 0:
        for column in args.keep_output_columns:
            if column not in df.columns:
                raise ValueError("Invalid column name '%s', available: %s" % (
                    column, list(df.columns)))
        df = df[args.keep_output_columns]

    for (old_name, new_name) in args.rename_output_column:
        if old_name not in df.columns:
            raise ValueError(
                "Can't rename column '%s' since it doesn't exist, available: %s" % (
                    old_name, list(df.columns)))
        df = df.rename({old_name: new_name})

    if print_df_after_filtering:
        print(df)

    if args.output_csv:
        print("Saving %s..." % args.output_csv)
        df.to_csv(args.output_csv, index=True, index_label="#")

    if args.output_html:
        print("Saving %s..." % args.output_html)
        df.to_html(args.output_html, index=True)
