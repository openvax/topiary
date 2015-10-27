# Copyright (c) 2015. Mount Sinai School of Medicine
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

import re


def infer_delimiter(filename, comment_char="#", n_lines=3):
    """
    Given a file which contains data separated by one of the following:
        - commas
        - tabs
        - spaces
    Return the most likely separator by sniffing the first few lines
    of the file's contents.
    """
    lines = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith(comment_char):
                continue
            if len(lines) < n_lines:
                lines.append(line)
            else:
                break
    if len(lines) < n_lines:
        raise ValueError(
            "Not enough lines in %s to infer delimiter" % filename)
    candidate_delimiters = ["\t", ",", "\s+"]
    for candidate_delimiter in candidate_delimiters:
        counts = [len(re.split(candidate_delimiter, line)) for line in lines]
        first_line_count = counts[0]
        if all(c == first_line_count for c in counts) and first_line_count > 1:
            return candidate_delimiter
    raise ValueError("Could not determine delimiter for %s" % filename)


def check_required_columns(df, filename, required_columns):
    """
    Ensure that all required columns are present in the given dataframe,
    otherwise raise an exception.
    """
    available_columns = set(df.columns)
    for column_name in required_columns:
        if column_name not in available_columns:
            raise ValueError("FPKM tracking file %s missing column '%s'" % (
                filename,
                column_name))
