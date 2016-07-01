
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
Commandline arguments related to epitope filtering
"""

from __future__ import print_function, division, absolute_import

def add_filter_args(arg_parser):
    filter_group = arg_parser.add_argument_group(
        title="Filtering Options",
        description="Criteria for removing epitopes from results")

    filter_group.add_argument(
        "--ic50-cutoff",
        help="Drop epitopes with predicted IC50 nM affinity above this value",
        default=None,
        type=float)

    filter_group.add_argument(
        "--percentile-cutoff",
        help="Drop epitopes with predicted IC50 percentile rank above this value",
        default=None,
        type=float)

    filter_group.add_argument(
        "--only-novel-epitopes",
        help="".join([
            "Drop epitopes which do not contain mutated residues or occur ",
            "in the self-ligandome."]),
        default=False,
        action="store_true")

    filter_group.add_argument(
        "--wildtype-ligandome-directory",
        help="".join([
            "Directory of 'self' ligand peptide sets, in files named ",
            "by allele (e.g. 'A0201'). Any predicted mutant epitope which ",
            "is in the files associated with the given alleles is treated as ",
            "wildtype (non-mutated)."]))
    return filter_group
