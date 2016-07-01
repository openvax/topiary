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

from __future__ import print_function, division, absolute_import
from collections import namedtuple

# ProteinFragment is the core subset of fields from MutantProteinFragment
# which can be repurposed for predicting self and viral epitopes
ProteinFragment = namedtuple("ProteinFragment", (
    "gene_name",
    "gene_id",
    "transcript_name",
    "transcript_id",
    "full_protein_length",
    # some or all of the amino acids in the protein which we'll be using
    # for epitope prediction
    "amino_acids",
    # where in the protein sequence did our prediction window start?
    # indices are a half-open interval
    "fragment_start_in_protein",
    "fragment_end_in_protein",
))
