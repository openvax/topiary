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
from mhctools import EpitopeCollection, BindingPrediction

from .epitope_helpers import extract_mutant_peptides, contains_mutation

def _apply_filter(
        filter_fn,
        collection,
        result_fn=None,
        filter_name="",
        collection_name=""):
    """
    Apply filter to effect collection and print number of dropped elements
    """
    n_before = len(collection)
    filtered = [x for x in collection if filter_fn(x)]
    n_after = len(filtered)
    if not collection_name:
        collection_name = collection.__class__.__name__
    logging.info(
        "%s filtering removed %d/%d entries of %s",
        filter_name,
        (n_before - n_after),
        n_before,
        collection_name)
    if result_fn:
        return result_fn(filtered)
    else:
        return filtered

def _dict_from_namedtuple(obj):
    return {
        field_name: getattr(obj, field_name)
        for field_name in obj._fields
    }

class MutantEpitopePredictor(object):
    def __init__(
            self,
            mhc_model,
            padding_around_mutation=0,
            ic50_cutoff=None,
            percentile_cutoff=None,
            drop_wildtype_epitopes=False):
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

        drop_wildtype_epitopes : bool
            Only keep epitopes which contain mutated residues
        """
        require_integer(
            padding_around_mutation, "Padding around mutated residues")

        self.mhc_model = mhc_model

        # dictionary mapping tuples of HLA alleles to instances
        self.mhc_model_instances = {}

        self.epitope_lengths = mhc_model.epitope_lengths
        if min(self.epitope_lengths) < 1:
            raise ValueError(
                "Epitope lengths must be positive integers, got: %s" % (
                    min(self.epitope_lengths),))
        # even if user doesn't want any padding around the mutation we need
        # to at least include enough of the surrounding non-mutated
        # residues to construct candidate epitopes of the specified lengths
        min_required_padding = max(self.epitope_lengths) - 1
        padding_around_mutation = max(
            padding_around_mutation, min_required_padding)
        self.padding_around_mutation = padding_around_mutation
        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.drop_wildtype_epitopes = drop_wildtype_epitopes

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

        mhc_alleles : list of strings
            MHC alleles

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
        # we only care about effects which impac the coding sequence of a
        # protein
        effects = _apply_filter(
            lambda effect: isinstance(effect, NonsilentCodingMutation),
            effects,
            result_fn=effects.clone_with_new_elements,
            filter_name="Silent mutation")
        if gene_expression_dict:
            effects = _apply_filter(
                lambda effect: (
                    gene_expression_dict.get(effect.gene_id, 0.0) >
                    gene_expression_threshold),
                effects,
                result_fn=effects.clone_with_new_elements,
                filter_name="Gene expression")
        if transcript_expression_dict:
            _apply_filter(
                lambda effect: (
                    transcript_expression_dict.get(effect.transcript_id, 0.0) >
                    transcript_expression_threshold
                ),
                effects,
                result_fn=effects.clone_with_new_elements,
                filter_name="Transcript expression")

        variant_effect_groups = effects.groupby_variant()

        if len(variant_effect_groups) == 0:
            logging.warn("No candidates for MHC binding prediction")
            return EpitopeCollection([])

        if transcript_expression_dict:
            # if expression data is available, then for each variant
            # keep the effect annotation for the most abundant variant
            def reduce_variant_effect_group(variant_effects):
                return variant_effects.top_expression_effect(
                    transcript_expression_dict)
        else:
            # if no transcript abundance data is available, then
            # for each variant keep the effect with the most significant
            # predicted effect on the protein sequence, along with using
            # transcript/CDS length as a tie-breaker for effects with the same
            # priority.
            def reduce_variant_effect_group(variant_effects):
                return variant_effects.top_priority_effect()

        final_effects = [
            reduce_variant_effect_group(variant_effects)
            for variant_effects in variant_effect_groups.values()
        ]

        peptide_interval_dict = extract_mutant_peptides(
            effects=final_effects,
            padding_around_mutation=self.padding_around_mutation)
        peptide_dict = {
            effect: source_sequence[start:end]
            for (effect, (source_sequence, start, end))
            in peptide_interval_dict.items()
        }
        # adjust offsets and source sequences of peptides in binding
        # predictions to reflect the longer source sequence they come from
        binding_predictions = []
        for x in self.mhc_model.predict(peptide_dict):
            effect = x.source_sequence_key
            (source_sequence, source_start, _) = peptide_interval_dict[effect]
            fields = _dict_from_namedtuple(x)
            fields["source_sequence"] = source_sequence
            fields["offset"] = fields["offset"] + source_start
            binding_predictions.append(BindingPrediction(**fields))

        # filter out low binders
        if self.ic50_cutoff:
            binding_predictions = _apply_filter(
                filter_fn=lambda x: x.value <= self.ic50_cutoff,
                collection=binding_predictions,
                filter_name="IC50 nM cutoff",
                collection_name="binding predictions",
            )
        if self.percentile_cutoff:
            binding_predictions = _apply_filter(
                filter_fn=lambda x: x.percentile_rank <= self.percentile_cutoff,
                collection=binding_predictions,
                filter_name="IC50 percentile rank cutoff",
                collection_name="binding predictions",
            )

        if self.drop_wildtype_epitopes:
            binding_predictions = _apply_filter(
                filter_fn=lambda x: contains_mutation(x, x.source_sequence_key),
                collection=binding_predictions,
                filter_name="Wildtype epitope",
                collection_name="binding predictions",
            )
        return EpitopeCollection(binding_predictions)

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

        if gene_expression_dict:
            variants = _apply_filter(
                lambda variant: any(
                    gene_expression_dict.get(transcript_id, 0.0) >
                    gene_expression_threshold
                    for transcript_id in variant.gene_ids
                ),
                result_fn=variants.clone_with_new_elements,
                filter_name="Gene expression")
        if transcript_expression_dict:
            variants = _apply_filter(
                lambda variant: any(
                    transcript_expression_dict.get(transcript_id, 0.0) >
                    transcript_expression_threshold
                    for transcript_id in variant.transcript_ids
                ),
                variants,
                result_fn=variants.clone_with_new_elements,
                filter_name="Transcript expression")

        effects = variants.effects(
            raise_on_error=raise_on_variant_effect_error)

        return self.epitopes_from_mutation_effects(
            effects=effects,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=gene_expression_threshold,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=transcript_expression_threshold)
