# Copyright (c) 2017. Mount Sinai School of Medicine
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

from .filters import (
    apply_epitope_filters,
    apply_effect_expression_filters,
    apply_variant_expression_filters,
    filter_silent_and_noncoding_effects,
)
from .sequence_helpers import (
    protein_subsequences_around_mutations,
    check_padding_around_mutation
)
from .epitope_prediction import (
    build_epitope_collection_from_binding_predictions,
)

class TopiaryPredictor(object):
    def __init__(
            self,
            mhc_model,
            padding_around_mutation=None,
            ic50_cutoff=None,
            percentile_cutoff=None,
            min_gene_expression=0.0,
            min_transcript_expression=0.0,
            only_novel_epitopes=False,
            wildtype_ligandome_dict=None,
            raise_on_error=True):
        """
        Parameters
        ----------
        mhc_model : mhctools.BasePredictor
            Any instance of a peptide-MHC binding affinity predictor

        padding_around_mutation : int
            How many residues surrounding a mutation to consider including in a
            candidate epitope. Default is the minimum size necessary for epitope
            length of the mhc model.

        gene_expression_threshold : float, optional
            If gene_expression_dict is given, only keep effects on genes
            expressed above this threshold.

        transcript_expression_threshold : float, optional
            If transcript_expression_dict is given, only keep effects on
            transcripts above this threshold.

        ic50_cutoff : float, optional
            Maximum predicted IC50 value for a peptide to be considered a binder.

        percentile_cutoff : float, optional
            Maximum percentile rank of IC50 values for a peptide to be considered
            a binder.

        only_novel_epitopes : bool, optional
            If True, then drop peptides which either don't contain a mutation or
            occur elsewhere in the self-ligandome.

        wildtype_ligandome_dict : dict-like, optional
            Mapping from allele names to set of wildtype peptides predicted
            to bind to that allele. If any predicted mutant epitope is found
            in the peptide sets for the patient's alleles, it is marked as
            wildtype (non-mutant).

        raise_on_error : bool
            Raise an exception if error is encountered or skip
            the variant or peptide which generated the error.
        """
        self.mhc_model = mhc_model
        self.padding_around_mutation = padding_around_mutation
        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.min_transcript_expression = min_transcript_expression
        self.min_gene_expression = min_gene_expression
        self.only_novel_epitopes = only_novel_epitopes
        self.wildtype_ligandome_dict = wildtype_ligandome_dict
        self.raise_on_error = raise_on_error

    def epitopes_from_mutation_effects(
            self,
            effects,
            transcript_expression_dict=None,
            gene_expression_dict=None):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return predicted epitopes around each mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        padding_around_mutation : int
            How many residues surrounding a mutation to consider including in a
            candidate epitope. Default is the minimum size necessary for epitope
            length of the mhc model.

        transcript_expression_dict : dict
            Dictionary mapping transcript IDs to RNA expression estimates. Used
            both for transcript expression filtering and for selecting the
            most abundant transcript for a particular variant. If omitted then
            transcript selection is done using priority of variant effects and
            transcript length.

        gene_expression_dict : dict, optional
            Dictionary mapping gene IDs to RNA expression estimates
        """
        padding_around_mutation = check_padding_around_mutation(
            given_padding=self.padding_around_mutation,
            epitope_lengths=self.mhc_model.default_peptide_lengths)

        # we only care about effects which impact the coding sequence of a
        # protein
        effects = filter_silent_and_noncoding_effects(effects)

        effects = apply_effect_expression_filters(
            effects,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=self.min_transcript_expression,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=self.min_gene_expression)

        # group by variants, so that we end up with only one mutant
        # sequence per mutation
        variant_effect_groups = effects.groupby_variant()

        if len(variant_effect_groups) == 0:
            logging.warn("No candidates for MHC binding prediction")
            return []

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

        # 1) dictionary mapping varcode effect objects to subsequences
        #    around each mutation
        # 2) dictionary mapping varcode effect to start offset of subsequence
        #    within the full mutant protein sequence
        protein_subsequences, protein_subsequence_offsets = \
            protein_subsequences_around_mutations(
                effects=top_effects,
                padding_around_mutation=padding_around_mutation)

        binding_predictions = self.mhc_model.predict(protein_subsequences)
        logging.info("MHC predictor returned %s peptide binding predictions" % (
            len(binding_predictions)))
        epitopes = build_epitope_collection_from_binding_predictions(
            binding_predictions=binding_predictions,
            protein_subsequences=protein_subsequences,
            protein_subsequence_start_offsets=protein_subsequence_offsets,
            wildtype_ligandome_dict=self.wildtype_ligandome_dict)
        return apply_epitope_filters(
            epitopes,
            ic50_cutoff=self.ic50_cutoff,
            percentile_cutoff=self.percentile_cutoff,
            only_novel_epitopes=self.only_novel_epitopes)

    def epitopes_from_variants(
            self,
            variants,
            transcript_expression_dict=None,
            gene_expression_dict=None):
        """
        Predict epitopes from a Variant collection, filtering options, and
        optional gene and transcript expression data.

        Parameters
        ----------
        variants : varcode.VariantCollection

        transcript_expression_dict : dict
            Maps from Ensembl transcript IDs to FPKM expression values.

        gene_expression_dict : dict, optional
            Maps from Ensembl gene IDs to FPKM expression values.
        """
        # pre-filter variants by checking if any of the genes or
        # transcripts they overlap have sufficient expression.
        # I'm tolerating the redundancy of this code since it's much cheaper
        # to filter a variant *before* trying to predict its impact/effect
        # on the protein sequence.
        variants = apply_variant_expression_filters(
            variants,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=self.min_transcript_expression,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=self.min_gene_expression)

        effects = variants.effects(raise_on_error=self.raise_on_error)

        return self.epitopes_from_mutation_effects(
            effects=effects,
            transcript_expression_dict=transcript_expression_dict,
            gene_expression_dict=gene_expression_dict)
