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

from collections import defaultdict

from typechecks import require_integer
import pandas as pd

class Predictor(object):
    def __init__(
            self,
            mhc_model_type,
            epitope_lengths=[8, 9, 10, 11],
            padding_around_mutation=None):
        """
        Parameters
        ----------
        mhc_model_type : type
            Type of MHC Class I binding prediction model whose
            initializer must take a list of HLA allele names called 'alleles'
            and a list of integers called 'epitope_lengths'

        epitope_lengths : list of int

        padding_around_mutation : int, optional
            Number of amino acids on each side of the mutation to include
            in candidate epitopes. If not given, then padding_around_mutation
            is set to one less than the maximum epitope length.
        """
        self.mhc_model_type = mhc_model_type
        self.epitope_lengths = epitope_lengths
        if min(epitope_lengths) < 1:
            raise ValueError(
                "Epitope lengths must be positive integers, got: %s" % (
                    min(epitope_lengths),))
        if padding_around_mutation is None:
            padding_around_mutation = max(epitope_lengths) - 1

        require_integer(padding_around_mutation, "padding around mutation")
        self.padding_around_mutation = padding_around_mutation

    def predict(
            self,
            variant_collection,
            hla_alleles,
            raise_on_variant_effect_error):
        mhc_model = self.mhc_model_type(
            alleles=hla_alleles,
            epitope_lengths=self.epitope_lengths)
        peptides_df = _create_dataframe(
            variant_collection=variant_collection,
            hla_alleles=hla_alleles,
            raise_on_error=raise_on_variant_effect_error)
        return mhc_model.predict(peptides_df)


def _create_dataframe(
        variant_collection,
        hla_alleles,
        predictor_type,
        raise_on_error):
    df_lists = defaultdict(list)
    for variant in variant_collection:
        for effect in variant.effects(raise_on_error=raise_on_error):
            try:
                # Skip over non-coding mutations (and unpredictable
                # coding mutations)
                mutation_start = effect.mutation_start
                mutation_end = effect.mutation_end
                source_sequence = effect.mutant_protein_sequence
            except Exception as e:
                if raise_on_error:
                    raise e
                continue
            df_lists['chr'].append(variant.contig)
            df_lists['pos'].append(variant.start)
            df_lists['ref'].append(variant.ref)
            df_lists['alt'].append(variant.alt)
            df_lists['MutationStart'].append(mutation_start)
            df_lists['MutationEnd'].append(mutation_end)
            df_lists['SourceSequence'].append(source_sequence)
            df_lists['GeneInfo'].append(None)
            df_lists['Gene'].append(effect.gene_name())
            df_lists['GeneMutationInfo'].append(variant.short_description())
            df_lists['PeptideMutationInfo'].append(effect.short_description())
            df_lists['TranscriptId'].append(effect.transcript.id)
    return pd.DataFrame(df_lists)
