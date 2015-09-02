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
from collections import namedtuple

from mhctools import BindingPrediction

# an Epitope is like an mhctools BindingPrediction extended with a
# notion of whether the peptide contains mutant residues

epitope_fields = BindingPrediction._fields + (
    # full protein sequence from which epitopes were taken
    "full_protein_sequence",
    # offset into the protein sequence of the first amino acid in the epitope
    "protein_offset",
    # does the peptide sequence contain any mutated residues
    "contains_mutant_residues",
    # does this peptide occur elsewhere in the self ligandome for the
    # predicted allele that it binds to?
    "occurs_in_self_ligandome",
    # should we consider this as a mutant peptide?
    # Differs from 'contains_mutant_residues' in that it excludes
    # peptides that occur in the self-ligandome
    "novel_epitope",
)

EpitopePrediction = namedtuple("EpitopePrediction", epitope_fields)
