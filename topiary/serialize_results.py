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

import pandas as pd

def epitopes_to_dataframe(epitope_collection):
    """
    An mhctools.EpitopeCollection creates a very noisy DataFrame representation
    (since it doesn't know e.g. that Topiary's sequence keys are variants).
    So, here's some specialized logic for making a DataFrame from a
    Variant-specific EpitopeCollection
    """
    column_dict = defaultdict(list)

    # list of column names and functions to extract the value for that column
    # from a single binding prediction
    simple_column_extractors = [
        ("peptide", lambda x: x.peptide),
        ("allele", lambda x: x.allele),
        ("length", lambda x: x.length),
        ("ic50", lambda x: x.value),
        ("percentile_rank", lambda x: x.percentile_rank),
        ("prediction_method", lambda x: x.prediction_method_name),
        ("source_seq_offset", lambda x: x.offset),
        ("source_seq_mut_start", lambda x: x.aa_mutation_start_offset),
        ("source_seq_mut_end", lambda x: x.aa_mutation_end_offset)
    ]
    for x in epitope_collection:
        for column_name, fn in simple_column_extractors:
            column_dict[column_name].append(fn(x))
        effect = x.source_sequence_key
        variant = effect.variant
        # half-open interval of mutated residues in the source sequence
        # the peptide was extracted from
        mut_start_in_source_seq = effect.aa_mutation_start_offset
        mut_end_in_source_seq = effect.aa_mutation_end_offset
        column_dict['effect_type'] = effect.__class__.__name__
        column_dict['variant'] = variant.short_description
        mut_start = min(max(0, mut_start_in_source_seq - x.offset), x.length)
        mut_end = min(max(0, mut_end_in_source_seq), x.length)
        mutated = mut_start != mut_end
        column_dict['mut_start'] = mut_start
        column_dict['mut_end'] = mut_end
        column_dict['mutated'] = mutated
        return pd.DataFrame(column_dict)

def epitopes_to_csv(epitope_collection, csv_path):
    df = epitopes_to_dataframe(epitope_collection)
    df.to_csv(csv_path, index=False)
