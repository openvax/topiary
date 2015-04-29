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

from typechecks import require_integer

def create_fasta_dict(effects, padding_around_mutation):
    fasta_dict = {}
    for effect in effects:
        # TODO: will mhctools take an object key?
        key = effect.variant
        seq = effect.mutant_protein_sequence
        # some effects will lack a mutant protein sequence since
        # they are either silent or unpredictable
        if seq:
            mutation_start = effect.aa_mutation_start_offset
            mutation_end = effect.aa_mutation_end_offset
            start = max(0, mutation_start - padding_around_mutation)
            end = min(len(seq), mutation_end + padding_around_mutation)
            print(effect, start, end)
            fasta_dict[key] = seq[start:end]
    return fasta_dict

class MutantEpitopePredictor(object):
    def __init__(
            self,
            mhc_model,
            padding_around_mutation=0):
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

    def epitopes_from_mutation_effects(
            self,
            effects,
            gene_expression_dict=None,
            gene_expression_threshold=0.0,
            transcript_expression_dict=None,
            transcript_expression_threshold=0.0):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return an EpitopeCollection of the predicted epitopes around each
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
        effects = effects.drop_silent_and_noncoding()
        if gene_expression_dict:
            effects = effects.filter_by_gene_expression(
                gene_expression_dict,
                gene_expression_threshold)
        if transcript_expression_dict:
            effects = effects.filter_by_transcript_expression(
                transcript_expression_dict,
                transcript_expression_threshold)

        variant_effect_groups = effects.groupby_variant()

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

        fasta_dict = create_fasta_dict(
            effects=final_effects,
            padding_around_mutation=self.padding_around_mutation)
        return self.mhc_model.predict(fasta_dict)

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
            variants = variants.filter_by_gene_expression(
                gene_expression_dict,
                gene_expression_threshold)
        if transcript_expression_dict:
            variants = variants.filter_by_transcript_expression(
                transcript_expression_dict,
                transcript_expression_threshold)

        effects = variants.effects(
            raise_on_error=raise_on_variant_effect_error)

        return self.epitopes_from_effects(
            effects=effects,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=gene_expression_threshold,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=transcript_expression_threshold)
