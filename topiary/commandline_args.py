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
from pyensembl import genome_for_reference_name
import varcode

from .parsing_helpers import parse_int_list
from .rna import (
    load_cufflinks_fpkm_dict,
    load_transcript_fpkm_dict_from_gtf
)

arg_parser = argparse.ArgumentParser()

#
# Genomic Variants
#
variant_arg_group = arg_parser.add_argument_group(
    title="Variants",
    description="Genomic variant files")

variant_arg_group.add_argument(
    "--vcf",
    default=[],
    action="append",
    help="Genomic variants in VCF format")

variant_arg_group.add_argument(
    "--maf",
    default=[],
    action="append",
    help="Genomic variants in TCGA's MAF format",)


variant_arg_group.add_argument(
    "--variant",
    default=[],
    action="append",
    nargs=4,
    metavar=("CHR", "POS", "REF", "ALT"),
    help="Individual variant as 4 arguments giving chromsome, position, ref, "
    "and alt. Example: chr1 3848 C G. Use '.' to indicate empty alleles for "
    "insertions or deletions.")

variant_arg_group.add_argument(
    "--reference-name",
    type=str,
    help=(
        "What reference assembly your variant coordinates are using. "
        "Examples: 'hg19', 'GRCh38', or 'mm9'. "
        "This argument is ignored for MAF files, since each row includes "
        "the reference. "
        "For VCF files, this is used if specified, and otherwise is guessed from "
        "the header. For variants specfied on the commandline with --variant, "
        "this option is required."))

variant_arg_group.add_argument(
    "--json-variant-files",
    default=[],
    action="append",
    help="Path to Varcode.VariantCollection object serialized as a JSON file.")


def variant_collection_from_args(args):
    variant_collections = []

    if args.reference_name:
        genome = genome_for_reference_name(args.reference_name)
    else:
        # no genome specified, assume it can be inferred from the file(s)
        # we're loading
        genome = None

    for vcf_path in args.vcf:
        vcf_variants = varcode.load_vcf(vcf_path, genome=genome)
        variant_collections.append(vcf_variants)
    for maf_path in args.maf:
        maf_variants = varcode.load_maf(maf_path)
        variant_collections.append(maf_variants)

    if args.variant:
        if not genome:
            raise ValueError(
                "--reference-name must be specified when using --variant")

        variants = [
            varcode.Variant(
                chromosome,
                start=position,
                ref=ref,
                alt=alt,
                ensembl=genome)
            for (chromosome, position, ref, alt)
            in args.variant
        ]
        variant_collection = varcode.VariantCollection(variants)
        variant_collections.append(variant_collection)

    if len(variant_collections) == 0:
        raise ValueError(
            "No variants loaded (use --maf, --vcf, or --variant options)")

    for json_path in args.json_variant_files:
        with open(json_path, 'r') as f:
            json_string = f.read()
            variant_collections.append(
                varcode.VariantCollection.from_json(json_string))
    if len(variant_collections) == 0:
        raise ValueError(
            "No variants loaded (use --maf, --vcf, --json-variants options)")
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

mhc_options_arg_group.add_argument(
    "--mhc-epitope-lengths",
    default=[8, 9, 10, 11],
    type=parse_int_list,
    help="Lengths of epitopes to consider for MHC binding prediction")

arg_parser.add_argument(
    "--mhc-alleles-file",
    help="File with one HLA allele per line")

arg_parser.add_argument(
    "--mhc-alleles",
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

mhc_predictors = {
    "netmhc": mhctools.NetMHC,
    "netmhcpan": mhctools.NetMHCpan,
    "netmhciipan": mhctools.NetMHCIIpan,
    "netmhccons": mhctools.NetMHCcons,
    "random": mhctools.RandomBindingPredictor,
    # TODO implement SMM predictors in mhctools
    "smm": None,
    "smm-pmbec": None,
    # use NetMHCpan via IEDB's web API
    "netmhcpan-iedb": mhctools.IedbNetMHCpan,
    # use NetMHCcons via IEDB's web API
    "netmhccons-iedb": mhctools.IedbNetMHCcons,
    # use SMM via IEDB's web API
    "smm-iedb": mhctools.IedbSMM,
    # use SMM-PMBEC via IEDB's web API
    "smm-pmbec-iedb": mhctools.IedbSMM_PMBEC,
    # Class II MHC binding prediction using NetMHCIIpan via IEDB
    "netmhciipan-iedb": mhctools.IedbNetMHCIIpan,
}

mhc_predictor_arg_group = arg_parser.add_argument(
    "--mhc-predictor",
    choices=list(sorted(mhc_predictors.keys())),
    type=lambda s: s.lower().strip(),
    required=True)


def mhc_binding_predictor_from_args(args):
    mhc_class = mhc_predictors.get(args.mhc_predictor)
    if mhc_class is None:
        raise ValueError(
            "Invalid MHC prediction method: %s" % (args.mhc_predictor,))
    alleles = mhc_alleles_from_args(args)
    epitope_lengths = args.mhc_epitope_lengths
    logging.info(
        ("Building MHC binding prediction %s"
         " for alleles %s"
         " and epitope lengths %s") % (
            mhc_class.__class__.__name__,
            alleles,
            epitope_lengths))
    return mhc_class(
        alleles=alleles,
        epitope_lengths=epitope_lengths)
#
# Mutated sequence options
#
arg_parser.add_argument(
    "--padding-around-mutation",
    default=None,
    help="".join([
        "How many extra amino acids to include on either side of a mutation.",
        "Default is determined by epitope lengths but can be overridden to ",
        "predict wildtype epitopes in a larger context around a mutant residue.",
    ]),
    type=int)

###
# RNA-Seq data
###

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


def rna_gene_expression_dict_from_args(args):
    """
    Returns a dictionary mapping Ensembl gene IDs to FPKM expression values
    or None if neither Cufflinks tracking file nor StringTie GTF file specified
    in the commandline arguments.
    """
    if args.rna_gene_fpkm_tracking_file:
        return load_cufflinks_fpkm_dict(args.rna_gene_fpkm_file)
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


# TODO:
# --rna-read-evidence-bam flag for filtering by Varcode.read_evidence
# --rna-min-read-evidence minimum number of RNA reads containing variant allele
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

filter_group.add_argument(
    "--only-novel-epitopes",
    help="".join([
        "Drop epitopes which do not contain mutated residues or occur ",
        "in the self-ligandome."]),
    default=False,
    action="store_true")

filter_group.add_argument(
    "--wildtype-ligandome-directory",
    help="".join([
        "Directory of 'self' ligand peptide sets, in files named ",
        "by allele (e.g. 'A0201'). Any predicted mutant epitope which ",
        "is in the files associated with the given alleles is treated as ",
        "wildtype (non-mutated)."]))

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
