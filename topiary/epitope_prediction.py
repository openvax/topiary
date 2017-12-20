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
from collections import defaultdict

from .sequence_helpers import (
    contains_mutant_residues,
    peptide_mutation_interval,
)

class EpitopePrediction(object):
    """
    Information about predicted epitopes from any protein, not restricted to
    mutated proteins. Fields are similar to mhctools.BindingPrediction but
    augmented with information about a source protein and the window within
    that protein that epitope predictions were drawn from.
    """
    def __init__(
            self,
            protein_id,
            protein_length,
            protein_subsequence,
            subsequence_start_in_protein,
            peptide,
            peptide_length,
            peptide_start_in_protein,
            peptide_start_in_subsequence,
            allele,
            # TODO: allow for multiple sources of prediction?
            # What if we want to have both stability and affinity measurements
            # for a single pMHC complex?
            value,
            percentile_rank,
            prediction_method_name):
        """
        Parameters
        ----------
        protein_id : str
            Ensembl ID of source protein

        protein_subsequence: str

        subsequence_start_in_protein : int
            Where in the protein sequence did our prediction window start?

        peptide_start_in_protein : int
            Offset of the peptide in the full protein

        peptide_start_in_subsequence : int
            Offset of the peptide in the subsequence

        allele : str
            MHC allele of binding prediction

        value : float
            Predicted IC50 affinity between peptide and allele

        percentile_rank : float

        prediction_method_name : str
            Name of peptide-MHC binding predictor
        """
        self.protein_id = protein_id
        self.protein_length = protein_length
        self.protein_subsequence = protein_subsequence
        self.subsequence_start_in_protein = subsequence_start_in_protein
        self.peptide = peptide
        self.peptide_length = peptide_length
        self.peptide_start_in_protein = peptide_start_in_protein
        self.peptide_start_in_subsequence = peptide_start_in_subsequence
        self.allele = allele
        # TODO: allow for multiple sources of prediction?
        # What if we want to have both stability and affinity measurements
        # for a single pMHC complex?
        self.value = value
        self.percentile_rank = percentile_rank
        self.prediction_method_name = prediction_method_name

class MutantEpitopePrediction(EpitopePrediction):
    """
    Epitopes arising from mutations (either cancer or germline)
    """
    def __init__(
            self,
            variant,
            effect,
            transcript_id,
            transcript_name,
            mutation_start_in_peptide,
            mutation_end_in_peptide,
            mutation_start_in_protein,
            mutation_end_in_protein,
            contains_mutant_residues,
            occurs_in_self_ligandome,
            novel_epitope,
            **kwargs):
        """
        variant : varcode.Variant
            Genomic variant that caused a mutant protein to be produced

        effect : varcode.Effect
            Effect prediction associated with the variant/transcript combination

        transcript_id : str
            Ensembl ID of transcript we're choosing to use for this variant

        transcript_name : str
            Name of transcript associated with transcript_id

        mutation_start_in_peptide : int
            Half-open interval of mutant residues within the full protein

        mutation_end_in_peptide : int

        mutation_start_in_protein : int

        mutation_end_in_protein : int

        contains_mutant_residues : bool
            Does the peptide sequence contain any mutated residues

        occurs_in_self_ligandome : bool
            Does this peptide occur elsewhere in the self ligandome for the
            predicted allele that it binds to?

        novel_epitope : bool
            Should we consider this as a mutant peptide?
            Differs from 'contains_mutant_residues' in that it excludes
            peptides that occur in the self-ligandome
        """
        EpitopePrediction.__init__(self, **kwargs)
        self.variant = variant
        self.effect = effect
        self.transcript_id = transcript_id
        self.transcript_name = transcript_name
        self.mutation_start_in_peptide = mutation_start_in_peptide
        self.mutation_end_in_peptide = mutation_end_in_peptide
        self.mutation_start_in_protein = mutation_start_in_protein
        self.mutation_end_in_protein = mutation_end_in_protein
        self.contains_mutant_residues = contains_mutant_residues
        self.occurs_in_self_ligandome = occurs_in_self_ligandome
        self.novel_epitope = novel_epitope


def build_epitope_collection_from_binding_predictions(
        binding_predictions,
        protein_subsequences,
        protein_subsequence_start_offsets,
        wildtype_ligandome_dict=None):
    """
    Given a collection of mhctools.BindingPrediction objects,
    add extra information to each entry and convert it to a
    topiary.EpitopePrediction, returning a collection of EpitopePrediction
    objects.

    Parameters
    ----------
    binding_predictions : collection of BindingPrediction objects

    protein_subsequences : dict
        Maps each varcode effect prediction to mutated region of protein

    mutant_protein_slices : dict
        Mapping from a varcode effect object to a ProteinSlice which contains
        the full mutant protein sequence, as well as the start/end offsets
        of the subsequence from which epitope predictions were made.

    wildtype_ligandome_dict : dict-like, optional
        Mapping from allele names to set of wildtype peptides predicted
        to bind to that allele. If any predicted mutant epitope is found
        in the peptide sets for the patient's alleles, it is marked as
        wildtype (non-mutant).
    """
    peptide_lengths = {
        binding_prediction.length
        for binding_prediction in binding_predictions}
    # map each peptide to a list of (effect, subsequence) pairs
    peptide_to_effect_dict = defaultdict(list)
    for effect, seq in protein_subsequences.items():
        for length in peptide_lengths:
            for i in range(len(seq) - length + 1):
                peptide = seq[i:i + length]
                peptide_to_effect_dict[peptide].append((effect, seq))

    epitope_predictions = []
    # TODO: group binding predictions by mutations
    # and have one object per mutation, which inherits

    for binding_prediction in binding_predictions:
        for effect, subsequence in peptide_to_effect_dict[binding_prediction.peptide]:
            full_protein_sequence = effect.mutant_protein_sequence
            subsequence_protein_offset = protein_subsequence_start_offsets[effect]
            peptide_start_in_protein = subsequence_protein_offset + binding_prediction.offset
            peptide = binding_prediction.peptide
            allele = binding_prediction.allele
            mutation_start_in_protein = effect.aa_mutation_start_offset
            mutation_end_in_protein = effect.aa_mutation_end_offset
            is_mutant = contains_mutant_residues(
                peptide_start_in_protein=peptide_start_in_protein,
                peptide_length=len(peptide),
                mutation_start_in_protein=mutation_start_in_protein,
                mutation_end_in_protein=mutation_end_in_protein)
            if is_mutant:
                mutation_start_in_peptide, mutation_end_in_peptide = peptide_mutation_interval(
                    peptide_start_in_protein=peptide_start_in_protein,
                    peptide_length=len(peptide),
                    mutation_start_in_protein=mutation_start_in_protein,
                    mutation_end_in_protein=mutation_end_in_protein)
            else:
                mutation_start_in_peptide = mutation_end_in_peptide = None
            # tag predicted epitopes as non-mutant if they occur in any of the
            # wildtype "self" binding peptide sets for the given alleles
            self_ligand = (
                wildtype_ligandome_dict is not None and
                peptide in wildtype_ligandome_dict[allele]
            )
            mutant_epitope_prediction = MutantEpitopePrediction(
                # TODO: check that all transcripts with coding sequences
                # have a protein ID
                protein_id=effect.transcript.protein_id,
                protein_length=len(full_protein_sequence),
                protein_subsequence=subsequence,
                subsequence_start_in_protein=subsequence_protein_offset,
                peptide=binding_prediction.peptide,
                peptide_length=len(binding_prediction.peptide),
                peptide_start_in_protein=peptide_start_in_protein,
                peptide_start_in_subsequence=binding_prediction.offset,
                mutation_start_in_peptide=mutation_start_in_peptide,
                mutation_end_in_peptide=mutation_end_in_peptide,
                mutation_start_in_protein=mutation_start_in_protein,
                mutation_end_in_protein=mutation_end_in_protein,
                allele=binding_prediction.allele,
                value=binding_prediction.value,
                percentile_rank=binding_prediction.percentile_rank,
                prediction_method_name=binding_prediction.prediction_method_name,
                variant=effect.variant,
                effect=effect,
                transcript_id=effect.transcript.id,
                transcript_name=effect.transcript.name,
                contains_mutant_residues=is_mutant,
                occurs_in_self_ligandome=self_ligand,
                novel_epitope=is_mutant and not self_ligand)
            epitope_predictions.append(mutant_epitope_prediction)
    return epitope_predictions
