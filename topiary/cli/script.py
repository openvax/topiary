# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from __future__ import print_function, division, absolute_import

import sys

from .args import arg_parser, predict_epitopes_from_args
from .rna import (
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args
)
from .outputs import write_outputs
from .. import epitopes_to_dataframe


def parse_args(args_list=None):
    if args_list is None:
        args_list = sys.argv[1:]
    return arg_parser.parse_args(args_list)

def main(args_list=None):
    """
    Script entry-point to predict neo-epitopes from genomic variants using
    Topiary.
    """
    args = parse_args(args_list)
    print("Topiary commandline arguments:")
    print(args)
    epitopes = predict_epitopes_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)
    df = epitopes_to_dataframe(
        epitopes,
        gene_expression_dict=gene_expression_dict,
        transcript_expression_dict=transcript_expression_dict)
    write_outputs(df, args)
    print("Total count: %d" % len(df))
