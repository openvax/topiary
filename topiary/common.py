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

from __future__ import print_function, division, absolute_import
import os

def parse_int_list(string):
    """
    Parses a string of numbers and ranges into a list of integers. Ranges
    are separated by "-" and inclusive of both the start and end number.

    Example:
        parse_int_list("8,10-12") == [8,10,11,12]
    """
    integers = []
    for substring in string.split(","):
        if "-" in substring:
            left, right = string.split("-")
            left_val = int(left.strip())
            right_val = int(right.strip())
            integers.extend(range(left_val, right_val + 1))
        else:
            integers.append(int(substring.strip()))
    return integers

def str2bool(value):
    return value.lower() in ('yes', 'true', 't', '1')

def env_var(name, converter=None, default=None):
    value = os.environ.get(name)
    if value is None or (isinstance(value, str) and len(value) == 0):
        return default
    elif converter:
        return converter(value)
    else:
        return value