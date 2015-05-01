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
Common commandline arguments used by scripts
"""


from __future__ import print_function, division, absolute_import
import argparse
import logging

import mhctools
from mhctools.alleles import normalize_allele_name
import varcode

from .common import parse_int_list


arg_parser = argparse.ArgumentParser()

#
# Genomic Variants
#
variant_arg_group = arg_parser.add_argument_group(
    title="Variants",
    description="Genomic variant files")

variant_arg_group.add_argument("--vcf",
    default=[],
    action="append",
    help="Genomic variants in VCF format")

variant_arg_group.add_argument("--maf",
    default=[],
    action="append",
    help="Genomic variants in TCGA's MAF format",)

# TODO: add variant parsing from the commandline
#  variant_arg_group.add_argument("--variant",
#    default=[],
#    action="append",
#    help="Individual variant in a format such as chr1:3848C>G",)

def variant_collection_from_args(args):
    variant_collections = []
    for vcf_path in args.vcf:
        variant_collections.append(varcode.load_vcf(vcf_path))
    for maf_path in args.maf:
        variant_collections.append(varcode.load_maf(maf_path))

    if len(variant_collections) == 0:
        raise ValueError(
            "No variants loaded (use --maf or --vcf options)")
    elif len(variant_collections) == 1:
        return variant_collections[0]
    else:
        combined_variants = []
        for variant_collection in variant_collections:
            combined_variants.extend(list(variant_collection))
        return varcode.VariantCollection(combined_variants)

#
# Options passed to MHC binding predictor
#
mhc_options_arg_group = arg_parser.add_argument_group(
    title="MHC Prediction Options",
    description="MHC Binding Prediction Options")

mhc_options_arg_group.add_argument("--mhc-epitope-lengths",
    default=[8, 9, 10, 11],
    type=parse_int_list,
    help="Lengths of epitopes to consider for MHC binding prediction")

arg_parser.add_argument("--mhc-alleles-file",
    help="File with one HLA allele per line")

arg_parser.add_argument("--mhc-alleles",
    default="",
    help="Comma separated list of allele (default HLA-A*02:01)")

def mhc_alleles_from_args(args):
    alleles = [
        normalize_allele_name(allele.strip())
        for allele in args.mhc_alleles.split(",")
        if allele.strip()
    ]
    if args.mhc_alleles_file:
        with open(args.mhc_alleles_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    alleles.append(normalize_allele_name(line))
    if len(alleles) == 0:
        raise ValueError(
            "MHC alleles required (use --mhc-alleles or --mhc-alleles-file)")
    return alleles

#
# MHC Binding Prediction
#

mhc_predictor_arg_group = arg_parser.add_mutually_exclusive_group(
    required=True)

mhc_predictor_flags = [
    ("--mhc-pan",
            "Use local NetMHCcons binding predictor",
            mhctools.NetMHCpan),
    ("--mhc-cons",
            "Use local NetMHCcons binding predictor",
            mhctools.NetMHCcons),
    ("--mhc-random",
        "Random values instead for MHC binding prediction",
        mhctools.RandomBindingPredictor),
    ("--mhc-smm", "Use local SMM binding predictor", None),
    ("--mhc-smm-pmbec", "Use local SMM-PMBEC binding predictor", None),
    ("--mhc-pan-iedb",
            "Use NetMHCpan via IEDB's web API",
            mhctools.IedbNetMHCpan),
    ("--mhc-cons-iedb",
            "Use NetMHCpan via IEDB's web API",
            mhctools.IedbNetMHCcons),
    ("--mhc-smm-iedb",
            "Use SMM via IEDB's web API",
            mhctools.IedbSMM),
    ("--mhc-smm-pmbec-iedb",
            "Use SMM-PMBEC via IEDB's web API",
            mhctools.IedbSMM_PMBEC),
]

for flag, help_string, mhc_class in mhc_predictor_flags:
    mhc_predictor_arg_group.add_argument(flag,
        default=False,
        action="store_true",
        help=help_string)

def mhc_binding_predictor_from_args(args):
    mhc_class = None
    for (flag, _, curr_mhc_class) in mhc_predictor_flags:
        arg_name = flag[2:].replace("-", "_")
        if getattr(args, arg_name):
            mhc_class = curr_mhc_class
            break
    if mhc_class is None:
        raise ValueError("No MHC prediction method specified")
    alleles = mhc_alleles_from_args(args)
    epitope_lengths = args.epitope_lengths
    logging.info(
        ("Building MHC binding prediction %s"
         " for alleles %s"
         " and epitope lengths %s") % (
            mhc_class.__class__.__name__,
            alleles,
            epitope_lengths))
    return mhc_class(
        alleles=alleles,
        epitope_lengths=args.mhc_epitope_lengths)
#
# Mutated sequence options
#
arg_parser.add_argument(
    "--padding-around-mutation",
    default=0,
    help="How many extra amino acids to include on either side of a mutation",
    type=int)

arg_parser.add_argument(
    "--self-filter-directory",
    help="Directory with 'self' ligand peptide sets, in files named by allele")


###
# RNA-Seq data
###

rna_group = arg_parser.add_argument_group(
    title="RNA-Seq",
    description="Transcript and gene abundance quantification")

rna_group.add_argument(
    "--rna-gene-fpkm-file",
    help="Cufflinks tracking file (FPKM measurements for Ensembl genes)",
    required=False)

rna_group.add_argument(
    "--rna-min-gene-expression",
    help="Minimum FPKM for gene expression",
    default=0.0,
    type=float)

rna_group.add_argument(
    "--rna-remap-novel-genes-onto-ensembl",
    help=(
        "If a novel gene is fully contained by known Ensembl gene, then "
        "merge their expression values"),
    default=False,
    action="store_true")

rna_group.add_argument(
    "--rna-transcript-fpkm-file",
    help="Cufflinks tracking file (FPKM measurements for Ensembl transcripts)")

rna_group.add_argument(
    "--rna-min-transcript-expression",
    help="Minimum FPKM for transcript expression",
    default=0.0,
    type=float)

#
# Filtering of epitopes
#
filter_group = arg_parser.add_argument_group(
    title="Filtering Options",
    description="Criteria for removing epitopes from results")

filter_group.add_argument(
    "--ic50-cutoff",
    help="Drop epitopes with predicted IC50 nM affinity above this value",
    default=None,
    type=float)


filter_group.add_argument(
    "--percentile-cutoff",
    help="Drop epitopes with predicted IC50 percentile rank above this value",
    default=None,
    type=float)

#
# Misc
#
arg_parser.add_argument(
    "--skip-variant-errors",
    default=False,
    action="store_true",
    help="Skip variants which cause errors")

#
# Output
#

output_group = arg_parser.add_argument_group(
    title="Output",
    description="Write results to different formats")

output_group.add_argument(
    "--output-csv",
    default=None,
    help="Path to output CSV file")

output_group.add_argument(
    "--output-html",
    default=None,
    help="Path to output HTML file")
