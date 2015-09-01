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
import logging

from typechecks import require_integer
from varcode import NonsilentCodingMutation
from mhctools import EpitopeCollection

from .epitope_prediction import EpitopePrediction
from .filters import (
    apply_filter,
    apply_epitope_filters,
    apply_effect_expression_filters,
    apply_variant_expression_filters,
)

def protein_sequences_and_offsets_around_mutations(
        effects,
        padding_around_mutation):
    """
    From each effect get the mutant protein sequence and the start/end
    offset in that sequence which contains all the mutant residues (along
    with any additional desired padding).
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
            result[effect] = (seq, seq_start_offset, seq_end_offset)
    return result

class MutantEpitopePredictor(object):
    def __init__(
            self,
            mhc_model,
            padding_around_mutation=None,
            ic50_cutoff=None,
            percentile_cutoff=None,
            wildtype_ligandome_dict=None,
            keep_wildtype_epitopes=False):
        """
        Parameters
        ----------
        mhc_model : object
            MHC Class I binding prediction model which expected to have a
            property 'epitope_lengths' that is a list of strings and
            method called 'predict' that takes a dictionary of sequences and
            returns an mhctools.EpitopeCollection.

        padding_around_mutation : int, optional
            Number of amino acids on each side of the mutation to include
            in candidate epitopes. If not given, then padding_around_mutation
            is set to one less than the maximum epitope length.

        ic50_cutoff : float, optional
            Drop any binding predictions with IC50 nM weaker than this cutoff.

        percentile_cutoff : float, optional
            Drop any binding predictions with percentile rank below this cutoff.

        keep_wildtype_epitopes : bool
            Keep epitopes which don't contain mutated residues
        """
        if padding_around_mutation is not None:
            require_integer(
                padding_around_mutation,
                "Padding around mutated residues")

        self.mhc_model = mhc_model

        # dictionary mapping tuples of HLA alleles to instances
        self.mhc_model_instances = {}

        self.epitope_lengths = mhc_model.epitope_lengths
        if min(self.epitope_lengths) < 1:
            raise ValueError(
                "Epitope lengths must be positive integers, got: %s" % (
                    min(self.epitope_lengths),))
        # If user doesn't provide any padding around the mutation we need
        # to at least include enough of the surrounding non-mutated
        # residues to construct candidate epitopes of the specified lengths
        min_required_padding = max(self.epitope_lengths) - 1
        if padding_around_mutation is None:
            padding_around_mutation = min_required_padding
        else:
            if padding_around_mutation < min_required_padding:
                raise ValueError("Padding around mutation cannot "
                                 "be less than %d for epitope lengths "
                                 "%s" % (min_required_padding,
                                         self.epitope_lengths))
        self.padding_around_mutation = padding_around_mutation
        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.wildtype_ligandome_dict = wildtype_ligandome_dict
        self.keep_wildtype_epitopes = keep_wildtype_epitopes

    def epitopes_from_mutation_effects(
            self,
            effects,
            gene_expression_dict=None,
            gene_expression_threshold=0.0,
            transcript_expression_dict=None,
            transcript_expression_threshold=0.0):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return a DataFrame of the predicted epitopes around each
        mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        gene_expression_dict : dict
            Dictionary mapping gene IDs to RNA expression estimates

        gene_expression_threshold : float
            If gene_expression_dict is given, only keep effects on genes
            expressed above this threshold.

        transcript_expression_dict : dict
            Dictionary mapping transcript IDs to RNA expression estimates

        transcript_expression_threshold : float
            If transcript_expression_dict is given, only keep effects on
            transcripts above this threshold.
        """

        # we only care about effects which impact the coding sequence of a
        # protein
        effects = apply_filter(
            lambda effect: isinstance(effect, NonsilentCodingMutation),
            effects,
            result_fn=effects.clone_with_new_elements,
            filter_name="Silent mutation")

        effects = apply_effect_expression_filters(
            effects,
            gene_expression_dict,
            gene_expression_threshold,
            transcript_expression_dict,
            transcript_expression_threshold)

        # group by variants, so that we end up with only one mutant
        # sequence per mutation
        variant_effect_groups = effects.groupby_variant()

        if len(variant_effect_groups) == 0:
            logging.warn("No candidates for MHC binding prediction")
            return EpitopeCollection([])

        if transcript_expression_dict:
            # if expression data is available, then for each variant
            # keep the effect annotation for the most abundant transcript
            top_effects = [
                variant_effects.top_expression_effect(
                    transcript_expression_dict)
                for variant_effects in variant_effect_groups.values()
            ]
        else:
            # if no transcript abundance data is available, then
            # for each variant keep the effect with the most significant
            # predicted effect on the protein sequence, along with using
            # transcript/CDS length as a tie-breaker for effects with the same
            # priority.
            top_effects = [
                variant_effects.top_priority_effect()
                for variant_effects in variant_effect_groups.values()
            ]

        # dictionary mapping varcode effect objects to subsequences
        # around each mutation, along with their start/end offsets in
        # the full protein sequence.
        mutant_sequence_and_offset_dict = protein_sequences_and_offsets_around_mutations(
            effects=top_effects,
            padding_around_mutation=self.padding_around_mutation)

        # dictionary mapping varcode effects to subsequences of each protein
        # containing mutant residues, used as argument to MHC binding
        # prediction model.
        mutant_subsequence_dict = {
            effect: source_sequence[start:end]
            for (effect, (source_sequence, start, end))
            in mutant_sequence_and_offset_dict.items()
        }

        # adjust offsets and source sequences of peptides in binding
        # predictions to reflect the longer source sequence they come from
        epitope_predictions = []

        for x in self.mhc_model.predict(mutant_subsequence_dict):
            effect = x.source_sequence_key
            # dict values are (source protein sequence, start pos, end pos)
            (source_sequence, source_start, _) = mutant_sequence_and_offset_dict[effect]

            # extracting the fields of the BindingPrediction to fix several
            # field values (see next several lines) and add extra field(s)
            # necessary to turn this into an EpitopePrediction
            fields = x.__dict__
            # update source_sequence to be the full mutant protein
            fields["source_sequence"] = source_sequence

            # shift the offset of each peptide to account for the fact that the
            # predictions were made over a sub-sequence of the full protein
            fields["offset"] = fields["offset"] + source_start

            peptide_start = x.offset
            peptide_end = x.offset + x.length - 1
            fields["contains_mutant_residues"] = (
                peptide_start < effect.aa_mutation_end_offset and
                peptide_end >= effect.aa_mutation_start_offset
            )
            # tag predicted epitopes as non-mutant if they occur in any of the
            # wildtype "self" binding peptide sets for the given alleles
            fields["occurs_in_self_ligandome"] = (
                self.wildtype_ligandome_dict is not None and
                x.peptide in self.wildtype_ligandome_dict[x.allele]
            )
            fields["mutant"] = (
                fields["contains_mutant_residues"] and
                not fields["occurs_in_self_ligandome"]
            )
            epitope_predictions.append(EpitopePrediction(**fields))

        return apply_epitope_filters(epitope_predictions)

    def epitopes_from_variants(
            self,
            variants,
            gene_expression_dict=None,
            gene_expression_threshold=0.0,
            transcript_expression_dict=None,
            transcript_expression_threshold=0.0,
            raise_on_variant_effect_error=True):
        """
        Parameters
        ----------
        variants : varcode.VariantCollection

        gene_expression_dict : dict
            Dictionary mapping gene IDs to RNA expression estimates

        gene_expression_threshold : float
            If gene_expression_dict is given, only keep effects on genes
            expressed above this threshold.

        transcript_expression_dict : dict
            Dictionary mapping transcript IDs to RNA expression estimates

        transcript_expression_threshold : float
            If transcript_expression_dict is given, only keep effects on
            transcripts above this threshold.

        raise_on_variant_effect_error : bool
        """
        # pre-filter variants by checking if any of the genes or
        # transcripts they overlap have sufficient expression.
        # I'm tolerating the redundancy of this code since it's much cheaper
        # to filter a variant *before* trying to predict its impact/effect
        # on the protein sequence.
        variants = apply_variant_expression_filters(
            variants,
            gene_expression_dict,
            gene_expression_threshold,
            transcript_expression_dict,
            transcript_expression_threshold)

        effects = variants.effects(
            raise_on_error=raise_on_variant_effect_error)

        return self.epitopes_from_mutation_effects(
            effects=effects,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=gene_expression_threshold,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=transcript_expression_threshold)
