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

from typechecks import require_integer

def protein_subsequences_around_mutations(effects, padding_around_mutation):
    """
    From each effect get a mutant protein sequence and pull out a subsequence
    around the mutation (based on the given padding). Returns a dictionary
    of subsequences and a dictionary of subsequence start offsets.
    """
    protein_subsequences = {}
    protein_subsequence_start_offsets = {}
    for effect in effects:
        protein_sequence = effect.mutant_protein_sequence
        # some effects will lack a mutant protein sequence since
        # they are either silent or unpredictable
        if protein_sequence:
            mutation_start = effect.aa_mutation_start_offset
            mutation_end = effect.aa_mutation_end_offset
            seq_start_offset = max(
                0,
                mutation_start - padding_around_mutation)
            seq_end_offset = min(
                len(protein_sequence),
                mutation_end + padding_around_mutation)
            subsequence = protein_sequence[seq_start_offset:seq_end_offset]
            print(effect, padding_around_mutation, subsequence, len(subsequence))
            protein_subsequences[effect] = subsequence
            protein_subsequence_start_offsets[effect] = seq_start_offset
    return protein_subsequences, protein_subsequence_start_offsets

def check_padding_around_mutation(given_padding, epitope_lengths):
    """
    If user doesn't provide any padding around the mutation we need
    to at least include enough of the surrounding non-mutated
    esidues to construct candidate epitopes of the specified lengths.
    """
    min_required_padding = max(epitope_lengths) - 1
    if not given_padding:
        return min_required_padding
    else:
        require_integer(given_padding, "Padding around mutation")
        if given_padding < min_required_padding:
            raise ValueError("Padding around mutation %d cannot "
                             "be less than %d for epitope lengths "
                             "%s" % (
                                given_padding,
                                min_required_padding,
                                epitope_lengths))
        return given_padding

def contains_mutant_residues(
        peptide_start_in_protein,
        peptide_length,
        mutation_start_in_protein,
        mutation_end_in_protein):
    peptide_end_in_protein = peptide_start_in_protein + peptide_length - 1
    return (
        peptide_start_in_protein < mutation_end_in_protein and
        peptide_end_in_protein >= mutation_start_in_protein
    )

def peptide_mutation_interval(
        peptide_start_in_protein,
        peptide_length,
        mutation_start_in_protein,
        mutation_end_in_protein):
    """
    Half-open interval of mutated residues in the peptide, determined
    from the mutation interval in the original protein sequence.

    Parameters
    ----------
    peptide_start_in_protein : int
        Position of the first peptide residue within the protein
        (starting from 0)

    peptide_length : int

    mutation_start_in_protein : int
        Position of the first mutated residue starting from 0. In the case of a
        deletion, the position where the first residue had been.

    mutation_end_in_protein : int
        Position of the last mutated residue in the mutant protein. In the case
        of a deletion, this is equal to the mutation_start_in_protein.
    )
    """
    if peptide_start_in_protein > mutation_end_in_protein:
        raise ValueError("Peptide starts after mutation")
    elif peptide_start_in_protein + peptide_length < mutation_start_in_protein:
        raise ValueError("Peptide ends before mutation")

    # need a half-open start/end interval
    peptide_mutation_start_offset = min(
        peptide_length,
        max(0, mutation_start_in_protein - peptide_start_in_protein))
    peptide_mutation_end_offset = min(
        peptide_length,
        max(0, mutation_end_in_protein - peptide_start_in_protein))
    return (peptide_mutation_start_offset, peptide_mutation_end_offset)
