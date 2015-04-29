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

def epitopes_to_dataframe(epitope_collection):
    """
    An mhctools.EpitopeCollection creates a very noisy DataFrame representation
    (since it doesn't know e.g. that Topiary's sequence keys are variants).
    So, here's some specialized logic for making a DataFrame from a
    Variant-specific EpitopeCollection
    """
    columns = defaultdict(list)
    for x in epitope_collection:
        columns['peptde'] = x.peptide
        columns['allele'] = x.allele
        columns['length'] = x.length
        columns['ic50'] = x.value
        columns['percentile_rank'] = x.percentile_rank
        columns['prediction_method'] = x.prediction_method_name
        effect = x.source_sequence_key
        variant = effect.variant
        mut_start = effect.aa_mutation_start_offset
        mut_end = effect.aa_mutation_end_offset
        columns['effect_type'] = effect.__class__.__name__
        columns['variant'] = variant.short_description

def epitopes_to_csv(epitope_collection, csv_path):
    df = epitopes_to_dataframe(epitope_collection)
    df.to_csv(csv_path)
