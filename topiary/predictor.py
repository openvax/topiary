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

from collections import OrderedDict


from .filters import (
    apply_effect_expression_filters,
    apply_variant_expression_filters,
    filter_silent_and_noncoding_effects,
)
from .sequence_helpers import (
    protein_subsequences_around_mutations,
    check_padding_around_mutation,
    contains_mutant_residues,
    peptide_mutation_interval,
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

        min_gene_expression : float, optional
            If gene expression values are provided, only keep effects on
            genes with expression above this threshold.

        min_transcript_expression : float, optional
            If transcript expression values are provided, only keep effects on
            transcripts with expression above this threshold.

        ic50_cutoff : float, optional
            Maximum predicted IC50 value for a peptide to be considered a binder.

        percentile_cutoff : float, optional
            Maximum percentile rank of IC50 values for a peptide to be considered
            a binder.

        only_novel_epitopes : bool, optional
            If True, then drop peptides which either don't contain a mutation.
            TODO: make this also check that peptide doesn't occur elsewhere in
            the reference ligandome

        raise_on_error : bool
            Raise an exception if error is encountered or skip
            the variant or peptide which generated the error.
        """
        self.mhc_model = mhc_model
        self.padding_around_mutation = check_padding_around_mutation(
            given_padding=padding_around_mutation,
            epitope_lengths=self.mhc_model.default_peptide_lengths)
        self.ic50_cutoff = ic50_cutoff
        self.percentile_cutoff = percentile_cutoff
        self.min_transcript_expression = min_transcript_expression
        self.min_gene_expression = min_gene_expression
        self.only_novel_epitopes = only_novel_epitopes
        self.raise_on_error = raise_on_error

    def predict_named_sequences(
            self, name_to_sequence_dict):
        """
        Parameters
        ----------
        name_to_sequence_dict : (str->str) dict
            Dictionary mapping sequence names to amino acid sequences

        Returns pandas.DataFrame with the following columns:
            - source_sequence_name
            - offset
            - peptide
            - allele
            - affinity
            - percentile_rank
            - prediction_method_name
            - length
        """
        return self.mhc_model.predict_subsequences_dataframe(name_to_sequence_dict)

    def predict_sequences(self, sequences):
        """
        Predict MHC ligands for sub-sequences of each input sequence.

        Parameters
        ----------
        sequences : list of str
            Multiple amino acid sequences (without any names or IDs)

        Returns DataFrame with the following fields:
            - source_sequence
            - offset
            - peptide
            - length
            - allele
            - affinity
            - percentile_rank
            - prediction_method_name
        """
        # make each sequence its own unique ID
        sequence_dict = {
            seq: seq
            for seq in sequences
        }
        df = self.predict_named_sequences(sequence_dict)
        return df.rename(columns={"source_sequence_name": "source_sequence"})

    def predict_mutation_effects(
            self,
            effects,
            transcript_expression_dict=None,
            gene_expression_dict=None):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return predicted epitopes around each mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        transcript_expression_dict : dict
            Dictionary mapping transcript IDs to RNA expression estimates. Used
            both for transcript expression filtering and for selecting the
            most abundant transcript for a particular variant. If omitted then
            transcript selection is done using priority of variant effects and
            transcript length.

        gene_expression_dict : dict, optional
            Dictionary mapping gene IDs to RNA expression estimates

        Returns DataFrame with the following columns:
            - variant
            - gene
            - transcript_id
            - transcript_name
            - effect
            - offset
            - peptide
            - length
            - allele
            - affinity
            - percentile_rank
            - prediction_method_name
            - contains_mutant_residues
            - mutation_start_in_peptide
            - mutation_end_in_peptide
        """

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
        effect_to_subsequence_dict, effect_to_offset_dict = \
            protein_subsequences_around_mutations(
                effects=top_effects,
                padding_around_mutation=self.padding_around_mutation)

        # since we know that each set of variant effects has been
        # reduced to a single 'top priority' effect, we can uniquely
        # identify each variant sequence by its original genomic variant
        variant_string_to_effect_dict = {
            effect.variant.short_description
            for effect in effect_to_subsequence_dict.keys()
        }
        variant_string_to_subsequence_dict = {
            effect.variant.short_description: seq
            for (effect, seq) in effect_to_subsequence_dict.items()
        }
        variant_string_to_offset_dict = {
            effect.variant.short_description: offset
            for (effect, offset) in effect_to_offset_dict.items()
        }
        df = self.predict_named_sequences(variant_string_to_subsequence_dict)
        logging.info("MHC predictor returned %d peptide binding predictions" % (
            len(df)))

        # since we used variant descrptions as the name of each sequence
        # let's rename that column to be more informative
        df = df.rename(columns={"source_sequence_name": "variant"})

        # adjust offset to be relative to start of protein, rather
        # than whatever subsequence we used for prediction
        def compute_peptide_offset_relative_to_protein(row):
            subsequence_offset = variant_string_to_offset_dict[row.variant]
            return row.offset + subsequence_offset

        df["offset"] = df.apply(compute_peptide_offset_relative_to_protein)

        if self.ic50_cutoff:
            df = df[df.affinity <= self.ic50_cutoff]
            logging.info("Kept %d predictions after filtering affinity <= %f" % (
                len(df), self.ic50_cutoff))

        if self.percentile_cutoff:
            df = df[df.percentile_rank <= self.percentile_cutoff]
            logging.info("Kept %d predictions after filtering percentile <= %f" % (
                len(df), self.percentile_rank))

        extra_columns = OrderedDict([
            ('gene', []),
            ('transcript_id', []),
            ('transcript_name', []),
            ('effect', []),
            ('contains_mutant_residues', []),
            ('mutation_start_in_peptide', []),
            ('mutation_end_in_peptide', []),
        ])

        for _, row in df.iterrows():
            variant_string = row.source_sequence_name
            effect = variant_string_to_effect_dict[variant_string]
            mutation_start_in_protein = effect.aa_mutation_start_offset
            mutation_end_in_protein = effect.aa_mutation_end_offset
            peptide_length = len(row.peptide)
            is_mutant = contains_mutant_residues(
                peptide_start_in_protein=row.offset,
                peptide_length=peptide_length,
                mutation_start_in_protein=mutation_start_in_protein,
                mutation_end_in_protein=mutation_end_in_protein)
            if is_mutant:
                mutation_start_in_peptide, mutation_end_in_peptide = peptide_mutation_interval(
                    peptide_start_in_protein=row.offset,
                    peptide_length=peptide_length,
                    mutation_start_in_protein=mutation_start_in_protein,
                    mutation_end_in_protein=mutation_end_in_protein)
            else:
                mutation_start_in_peptide = mutation_end_in_peptide = None
            # TODO: add extra boolean field
            #   novel = is_mutant | not_in_reference
            # Requires keeping a quick lookup structure for all peptides in
            # the reference proteome
            if is_mutant or not self.only_novel_epitopes:
                extra_columns["gene"].append(effect.variant.gene_name)
                extra_columns["transcript_id"].append(effect.transcript_id)
                extra_columns["transcript_name"].append(effect.transcript_name)
                extra_columns["effect"].append(effect.short_description)
                extra_columns["contains_mutant_residues"].append(is_mutant)
                extra_columns["mutation_start_in_peptide"].append(mutation_start_in_peptide)
                extra_columns["mutation_end_in_peptide"].append(mutation_end_in_peptide)
        for col, values in extra_columns.items():
            df[col] = values
        return df

    def predict_variants(
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

        Returns DataFrame with the following columns:
            - variant
            - gene
            - transcript_id
            - transcript_name
            - effect
            - offset
            - peptide
            - length
            - allele
            - affinity
            - percentile_rank
            - prediction_method_name
            - contains_mutant_residues
            - mutation_start_in_peptide
            - mutation_end_in_peptide
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

        return self.predict_mutation_effects(
            effects=effects,
            transcript_expression_dict=transcript_expression_dict,
            gene_expression_dict=gene_expression_dict)
