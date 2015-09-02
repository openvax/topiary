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

ProteinSlice = namedtuple(
    "ProteinSlice",
    [
        "protein_sequence",
        "start_offset",
        "end_offset"
    ])

def protein_slices_around_mutations(effects, padding_around_mutation):
    """
    From each effect get a mutant ProteinSlice (which contains
    the full mutant protein sequence and the start/end offset in that sequence)
    around the mutation.
    """
    result = {}
    for effect in effects:
        seq = effect.mutant_protein_sequence
        # some effects will lack a mutant protein sequence since
        # they are either silent or unpredictable
        if seq:
            mutation_start = effect.aa_mutation_start_offset
            mutation_end = effect.aa_mutation_end_offset
            seq_start_offset = max(
                0,
                mutation_start - padding_around_mutation)
            seq_end_offset = min(
                len(seq),
                mutation_end + padding_around_mutation)
            result[effect] = ProteinSlice(
                seq,
                seq_start_offset,
                seq_end_offset)
    return result
