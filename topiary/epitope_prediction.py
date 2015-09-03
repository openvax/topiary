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

from mhctools import EpitopeCollection

from .sequence_helpers import (
    contains_mutant_residues,
    peptide_mutation_interval,
)

# information about epitopes from any protein, not restricted to mutant
# proteins. Fields are similar to mhctools.BindingPrediction but augmented
# with information about a source protein and the window within that protein
# that epitope predictions were drawn from.
EpitopePrediction = namedtuple("EpitopePrediction",
    (
        # either an Ensembl ID or custom ID from transcriptome assembly
        "protein_id",
        "protein_length",
        # where in the protein sequence did our prediction window start?
        "protein_subsequence",
        "subsequence_start_in_protein",
        # peptide for which the binding prediction was made
        "peptide",
        "peptide_length",
        # offset of the peptide in the full protein
        "peptide_start_in_protein",
        # offset of the peptide in the subsequence we made predictions for
        "peptide_start_in_subsequence",
        "allele",
        # TODO: allow for multiple sources of prediction?
        # What if we want to have both stability and affinity measurements
        # for a single pMHC complex?
        "value",
        "measure",
        "percentile_rank",
        "prediction_method_name",
    ))

# epitopes arising from mutations (either cancer or germline)
MutantEpitopePrediction = namedtuple(
    "MutantEpitopePrediction",
    EpitopePrediction._fields + (
        # genomic variant that caused a mutant protein to be produced
        "variant",
        # varcode Effect associated with the variant/transcript combination
        "effect",
        # transcript we're choosing to use for this variant
        "transcript_id",
        "transcript_name",
        # half-open interval of mutant residues within the peptide sequence
        "mutation_start_in_peptide",
        "mutation_end_in_peptide",
        # half-open interval of mutant residues within the full protein
        "mutation_start_in_protein",
        "mutation_end_in_protein",
        # does the peptide sequence contain any mutated residues
        "contains_mutant_residues",
        # does this peptide occur elsewhere in the self ligandome for the
        # predicted allele that it binds to?
        "occurs_in_self_ligandome",
        # should we consider this as a mutant peptide?
        # Differs from 'contains_mutant_residues' in that it excludes
        # peptides that occur in the self-ligandome
        "novel_epitope",
    ))

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
        Assumes that their `source_sequence_key` field is a Varcode effect
        object (e.g. Substitution, FrameShift, etc...)

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
    epitope_predictions = []
    for binding_prediction in binding_predictions:
        effect = binding_prediction.source_sequence_key
        full_protein_sequence = effect.mutant_protein_sequence
        subsequence = protein_subsequences[effect]
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
            measure=binding_prediction.measure,
            percentile_rank=binding_prediction.percentile_rank,
            prediction_method_name=binding_prediction.prediction_method_name,
            variant=effect.variant,
            effect=effect,
            transcript_id=effect.transcript.id,
            transcript_name=effect.transcript.name,
            contains_mutant_residues=is_mutant,
            occurs_in_self_ligandome=self_ligand,
            novel_epitope=is_mutant and not self_ligand,
        )
        epitope_predictions.append(mutant_epitope_prediction)
    return EpitopeCollection(epitope_predictions)
