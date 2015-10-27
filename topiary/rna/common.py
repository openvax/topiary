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


def infer_separator(filename):
    """
    Given a file which contains data separated by one of the following:
        - commas
        - tabs
        - spaces
    Return the most likely separator by sniffing the first 1000 bytes
    of the file's contents.
    """
    with open(filename, "r") as f:
        # read first thousand bytes of the file which should contain at
        # least one instance of the field separator
        substring = f.read(1000)
        comma_counts = substring.count(",")
        tab_counts = substring.count("\t")
        if comma_counts > tab_counts:
            return ","
        elif tab_counts > 0:
            return "\t"
        elif " " in substring:
            return "\s+"
        else:
            raise ValueError(
                "Unable to infer field separator for %s" % filename)


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
