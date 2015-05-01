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

"""
Helpers to convert between data representations
"""

from __future__ import print_function, division, absolute_import
from collections import defaultdict

import pandas as pd

def extract_mutant_peptides(effects, padding_around_mutation):
    """
    Map each effect onto a source sequence, and start/end of an interval
    from which the caller can construct a peptide subsequence containing
    mutated residues
    """
    result = {}
    for effect in effects:
        seq = effect.mutant_protein_sequence
        # some effects will lack a mutant protein sequence since
        # they are either silent or unpredictable
        if seq:
            mutation_start = effect.aa_mutation_start_offset
            mutation_end = effect.aa_mutation_end_offset
            start = max(0, mutation_start - padding_around_mutation)
            end = min(len(seq), mutation_end + padding_around_mutation)
            result[effect] = (seq, start, end)
    return result

def epitopes_to_dataframe(
        epitope_collection,
        gene_expression_dict=None,
        transcript_expression_dict=None):
    """
    An mhctools.EpitopeCollection creates a very noisy DataFrame representation
    (since it doesn't know e.g. that Topiary's sequence keys are variants).
    So, here's some specialized logic for making a DataFrame from a
    Variant-specific EpitopeCollection.

    Optional arguments for expression level dictionaries will cause the
    data frame to have columns 'gene_expression' and/or 'transcript_expression'.
    """
    column_dict = defaultdict(list)

    # list of column names and functions to extract the value for that column
    # from a single binding prediction.
    # Important for understanding this code: each element of epitope_collection
    # is an mhctools.BindingPrediction object whose `source_sequence_key` field
    # is being abused to store a Varcode.MutationEffect object (which acts as
    # a unique key for the longer amino acid sequences out of which we pulled
    # candidate epitopes)

    def get_effect(binding_prediction):
        return binding_prediction.source_sequence_key

    def get_variant(binding_prediction):
        return get_effect(binding_prediction).variant

    simple_column_extractors = [
        ("allele", lambda x: x.allele),
        ("peptide", lambda x: str(x.peptide)),
        ("length", lambda x: x.length),
        ("ic50", lambda x: x.value),
        ("percentile_rank", lambda x: x.percentile_rank),
        ("gene", lambda x: get_effect(x).gene_name),
        ("transcript", lambda x: get_effect(x).transcript_name),
        ("variant", lambda x: get_variant(x).short_description),
        ("effect", lambda x: get_effect(x).short_description),
        ("effect_type", lambda x: get_effect(x).__class__.__name__),
        ("prediction_method", lambda x: x.prediction_method_name),
        ("protein_sequence",
            lambda x: str(get_effect(x).mutant_protein_sequence)),
        ("protein_mutation_start",
            lambda x: get_effect(x).aa_mutation_start_offset),
        ("protein_mutation_end",
            lambda x: get_effect(x).aa_mutation_end_offset),
        ("peptide_offset_in_protein", lambda x: x.offset),
    ]
    if gene_expression_dict:
        key_fn_pair = (
            "gene_expression",
            lambda x: gene_expression_dict.get(get_effect(x).gene_id, 0.0)
        )
        simple_column_extractors.append(key_fn_pair)

    if transcript_expression_dict:
        key_fn_pair = (
            "transcript_expression",
            lambda x: transcript_expression_dict.get(
                get_effect(x).transcript_id, 0.0)
        )
        simple_column_extractors.append(key_fn_pair)

    for x in epitope_collection:
        for column_name, fn in simple_column_extractors:
            column_dict[column_name].append(fn(x))
        effect = x.source_sequence_key

        # half-open interval of mutated residues in the source sequence
        # the peptide was extracted from
        mut_start_in_source_seq = effect.aa_mutation_start_offset
        mut_end_in_source_seq = effect.aa_mutation_end_offset
        # TODO: write unit tests for all interval logic,
        # since it's easy to get wrong
        if (x.offset >= mut_end_in_source_seq or
                x.offset + x.length - 1 < mut_start_in_source_seq):
            peptide_mut_start = peptide_mut_end = None
            mutated = False
        else:
            # need a half-open start/end
            peptide_mut_start = min(
                x.length,
                max(0, mut_start_in_source_seq - x.offset))
            peptide_mut_end = min(
                x.length,
                max(0, mut_end_in_source_seq - x.offset))
            mutated = True
        column_dict['peptide_mutation_start'].append(peptide_mut_start)
        column_dict['peptide_mutation_end'].append(peptide_mut_end)
        column_dict['peptide_contains_mutation'].append(mutated)
    column_names = [pair[0] for pair in simple_column_extractors] + [
        'peptide_mutation_start',
        'peptide_mutation_end',
        'peptide_contains_mutation']
    return pd.DataFrame(
        column_dict,
        columns=column_names)

def epitopes_to_csv(epitope_collection, csv_path):
    df = epitopes_to_dataframe(epitope_collection)
    df.to_csv(csv_path, index=False)
