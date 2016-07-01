# Copyright (c) 2015-2016. Mount Sinai School of Medicine
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

import sys

from .. import epitopes_to_dataframe

from .rna import (
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .outputs import write_outputs
from .epitopes import predict_epitopes_from_args
from .arg_parser import make_arg_parser

def topiary(raw_args=None):
    """
    Script to generate epitope predictions from somatic cancer variants
    and (optionally) tumor RNA-seq data.

    Example usage:
        topiary \
            --mhc-predictor netmhcpan
            --mhc-alleles-file HLA.txt
            --vcf somatic.vcf
            --rna-gene-fpkm-file genes.fpkm_tracking
            --rna-transcript-fpkm-file isoforms.fpkm_tracking
            --filter-ic50 500
            --filter-percentile 2
            --output results.csv
    """
    if raw_args is None:
        raw_args = sys.args[1:]

    arg_parser = make_arg_parser()
    parsed_args = arg_parser.parse_args()
    print("Topiary commandline arguments:")
    print(parsed_args)

    epitopes = predict_epitopes_from_args(parsed_args)
    gene_expression_dict = rna_gene_expression_dict_from_args(parsed_args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(parsed_args)
    df = epitopes_to_dataframe(
        epitopes,
        gene_expression_dict=gene_expression_dict,
        transcript_expression_dict=transcript_expression_dict)
    write_outputs(df, parsed_args)
    print("Total count: %d" % len(df))
