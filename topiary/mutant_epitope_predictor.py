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

from typechecks import require_integer, require_iterable_of

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
            mhc_model_class,
            epitope_lengths=[8, 9, 10, 11],
            padding_around_mutation=0):
        """
        Parameters
        ----------
        mhc_model_class : type
            Type of MHC Class I binding prediction model whose
            initializer must take a list of HLA allele names called 'alleles'
            and a list of integers called 'epitope_lengths'

        epitope_lengths : list of int

        padding_around_mutation : int, optional
            Number of amino acids on each side of the mutation to include
            in candidate epitopes. If not given, then padding_around_mutation
            is set to one less than the maximum epitope length.
        """
        require_iterable_of(epitope_lengths, int, "Epitope lengths")
        require_integer(
            padding_around_mutation, "Padding around mutated residues")

        self.mhc_model_class = mhc_model_class

        # dictionary mapping tuples of HLA alleles to instances
        self.mhc_model_instances = {}

        self.epitope_lengths = epitope_lengths
        if min(epitope_lengths) < 1:
            raise ValueError(
                "Epitope lengths must be positive integers, got: %s" % (
                    min(epitope_lengths),))
        # even if user doesn't want any padding around the mutation we need
        # to at least include enough of the surrounding non-mutated
        # residues to construct candidate epitopes of the specified lengths
        min_required_padding = max(epitope_lengths) - 1
        padding_around_mutation = max(
            padding_around_mutation, min_required_padding)
        self.padding_around_mutation = padding_around_mutation

    def model_for_alleles(self, mhc_alleles):
        if isinstance(mhc_alleles, str):
            mhc_alleles = (mhc_alleles,)
        else:
            mhc_alleles = tuple(mhc_alleles)
        if mhc_alleles not in self.mhc_model_instances:
            self.mhc_model_instances[mhc_alleles] = self.mhc_model_class(
                alleles=mhc_alleles,
                epitope_lengths=self.epitope_lengths)
        return self.mhc_model_instances[mhc_alleles]

    def predict(
            self,
            variant_collection,
            mhc_alleles,
            gene_expression_dict=None,
            min_gene_expresion_value=0.0,
            transcript_expression_dict=None,
            min_transcript_expresion_value=0.0,
            raise_on_variant_effect_error=True):
        """
        Parameters
        ----------
        variant_collection : varcode.VariantCollection

        mhc_alleles : list of strings

        gene_expression_dict : dict, optional
            Dictionary mapping Ensembl gene IDs to expression values.
            If not given, then don't filter variant effects by gene expression.

        min_gene_expresion_value : float, optional
            If gene_expression_dict is given, then we only keep effects who
            have gene expression estimates above this value.

        transcript_expression_dict : dict, optional
            Dictionary mapping Ensembl transcript IDs to expression values.
            If not given, then don't filter variant effects by transcript
            expression.

        min_transcript_expresion_value : float, optional
            If transcript_expression_dict is given,
            then we only keep effects who have transcript expression
            estimates above this value.

        raise_on_variant_effect_error : bool
        """
        mhc_model = self.model_for_alleles(mhc_alleles)

        effects = variant_collection.effects(
            raise_on_error=raise_on_variant_effect_error)
        print(effects)
        # filter by gene and transcript RNA abundance estimates
        if gene_expression_dict:
            effects = effects.filter_gene_expression(
                gene_expression_dict,
                min_gene_expresion_value)
        if transcript_expression_dict:
            effects = effects.filter_transcript_expression(
                transcript_expression_dict,
                min_transcript_expresion_value)

        # group effects by their variant and reduce each group
        # to a single effect by only keeping those with most significant
        # predicted effect on the protein sequence, along with using
        # transcript/CDS length as a tie-breaker for effects with the same
        # priority.
        variant_effects = effects.top_priority_effect_per_variant()
        fasta_dict = create_fasta_dict(
            effects=variant_effects.values(),
            padding_around_mutation=self.padding_around_mutation)
        return mhc_model.predict(fasta_dict)
