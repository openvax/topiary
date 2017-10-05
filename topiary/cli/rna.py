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
Common commandline arguments for filtering by gene/transcript expression
"""

from __future__ import print_function, division, absolute_import

from ..rna import (
    load_cufflinks_fpkm_dict,
    load_transcript_fpkm_dict_from_gtf
)

def add_rna_args(arg_parser):
    rna_group = arg_parser.add_argument_group(
        title="RNA-Seq",
        description="Transcript and gene abundance quantification")

    rna_group.add_argument(
        "--rna-transcript-fpkm-tracking-file",
        help="".join([
            "Cufflinks tracking file (FPKM estimates for Ensembl transcripts). ",
            "Used both for expression filtering and selecting the most abundant ",
            "transcript to use for determining a mutant protein sequence."]))

    rna_group.add_argument(
        "--rna-transcript-fpkm-gtf-file",
        help="".join([
            "GTF file containing FPKM estimates for Ensembl transcripts.",
            "Used both for expression filtering and selecting the most abundant ",
            "transcript to use for determining a mutant protein sequence."]))

    rna_group.add_argument(
        "--rna-min-transcript-expression",
        help="Minimum FPKM for transcript expression",
        default=0.0,
        type=float)

    rna_group.add_argument(
        "--rna-gene-fpkm-tracking-file",
        help="Cufflinks tracking file (FPKM estimates for Ensembl genes)",
        required=False)

    rna_group.add_argument(
        "--rna-min-gene-expression",
        help="Minimum FPKM for gene expression",
        default=0.0,
        type=float)

    return rna_group

def rna_gene_expression_dict_from_args(args):
    """
    Returns a dictionary mapping Ensembl gene IDs to FPKM expression values
    or None if neither Cufflinks tracking file nor StringTie GTF file specified
    in the commandline arguments.
    """
    if args.rna_gene_fpkm_tracking_file:
        return load_cufflinks_fpkm_dict(args.rna_gene_fpkm_tracking_file)
    else:
        return None

def rna_transcript_expression_dict_from_args(args):
    """
    Returns a dictionary mapping Ensembl transcript IDs to FPKM expression
    values or None if neither Cufflinks tracking file nor StringTie GTF file
    were specified.
    """
    if args.rna_transcript_fpkm_tracking_file:
        return load_cufflinks_fpkm_dict(args.rna_transcript_fpkm_tracking_file)
    elif args.rna_transcript_fpkm_gtf_file:
        return load_transcript_fpkm_dict_from_gtf(
            args.rna_transcript_fpkm_gtf_file)
    else:
        return None
