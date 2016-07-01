
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
import logging

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


# protein fragment containing a mutated amino acid sequence
MutantProteinFragment = namedtuple(
    "MutantProteinFragment",
    ProteinFragment._fields + (
        # genomic variant that caused a mutant protein to be produced
        "variant",
        # varcode Effect associated with the variant/transcript combination
        "effect",
        # half-open start/end indices of mutated amino acids in the fragment
        "mutation_start_in_fragment",
        "mutation_end_in_fragment",
        # half-open start/end indices mutated amino acids relative to the
        # full protein sequence
        "mutation_start_in_full_protein",
        "mutation_end_in_full_protein",
        # what was the reference sequence before the mutation?
        "reference_amino_acids"))


def varcode_effect_to_mutant_protein_fragment(
        effect,
        padding_around_mutation):
    """
    Returns MutantProteinFragment or None
    """
    original_protein_sequence = effect.transcript.protein_sequence

    # some effects will lack a mutant protein sequence since
    # they are either silent or unpredictable
    if original_protein_sequence is None:
        logging.info(
            ("Unable to create MutantProteinFragment for %s, "
             "original protein sequence missing") % (effect,))
        return

    mutant_protein_sequence = effect.mutant_protein_sequence
    if mutant_protein_sequence is None:
        logging.info(
            ("Unable to create MutantProteinFragment for %s, "
             "predicted mutant protein sequence missing") % (effect,))
        return
    mutation_start_in_protein = effect.aa_mutation_start_offset
    mutation_end_in_protein = effect.aa_mutation_end_offset
    fragment_start_in_protein = max(
        0,
        mutation_start_in_protein - padding_around_mutation)
    # some pseudogenes have stop codons in the reference sequence,
    # if we try to use them for epitope prediction we should trim
    # the sequence to not include the stop character '*'
    first_stop_codon_index = mutant_protein_sequence.find("*")
    if first_stop_codon_index < 0:
        first_stop_codon_index = len(mutant_protein_sequence)

    fragment_end_in_protein = min(
        first_stop_codon_index,
        mutation_end_in_protein + padding_around_mutation)
    subsequence = mutant_protein_sequence[fragment_start_in_protein:fragment_end_in_protein]
    mutation_start_in_fragment = max(
        0,
        mutation_start_in_protein - fragment_start_in_protein)
    mutation_end_in_fragment = max(
        0,
        min(len(subsequence), mutation_end_in_protein - fragment_end_in_protein))
    return MutantProteinFragment(
        gene_name=effect.gene_name,
        gene_id=effect.gene_id,
        transcript_name=effect.transcript_name,
        transcript_id=effect.transcript_id,
        amino_acids=subsequence,
        fragment_start_in_protein=fragment_start_in_protein,
        fragment_end_in_protein=fragment_end_in_protein,
        variant=effect.variant,
        effect=effect,
        mutation_start_in_fragment=mutation_start_in_fragment,
        mutation_end_in_fragment=mutation_end_in_fragment,
        mutation_start_in_full_protein=mutation_start_in_protein,
        mutation_end_in_full_protein=mutation_end_in_protein,
        reference_amino_acids=original_protein_sequence[
            fragment_start_in_protein:fragment_end_in_protein])
