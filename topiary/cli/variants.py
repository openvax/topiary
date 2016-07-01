
# Copyright (c) 2016. Mount Sinai School of Medicine
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
Commandline arguments for loading genomic variants
"""

from __future__ import print_function, division, absolute_import

from pyensembl import genome_for_reference_name
from varcode import Variant, VariantCollection, load_vcf, load_maf

def add_variant_args(arg_parser):
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

    return variant_arg_group


def variant_collection_from_args(args):
    variant_collections = []

    if args.reference_name:
        genome = genome_for_reference_name(args.reference_name)
    else:
        # no genome specified, assume it can be inferred from the file(s)
        # we're loading
        genome = None

    for vcf_path in args.vcf:
        vcf_variants = load_vcf(vcf_path, genome=genome)
        variant_collections.append(vcf_variants)
    for maf_path in args.maf:
        maf_variants = load_maf(maf_path)
        variant_collections.append(maf_variants)

    if args.variant:
        if not genome:
            raise ValueError(
                "--reference-name must be specified when using --variant")

        variants = [
            Variant(
                chromosome,
                start=position,
                ref=ref,
                alt=alt,
                ensembl=genome)
            for (chromosome, position, ref, alt)
            in args.variant
        ]
        variant_collection = VariantCollection(variants)
        variant_collections.append(variant_collection)

    if len(variant_collections) == 0:
        raise ValueError(
            "No variants loaded (use --maf, --vcf, or --variant options)")

    for json_path in args.json_variant_files:
        with open(json_path, 'r') as f:
            json_string = f.read()
            variant_collections.append(
                VariantCollection.from_json(json_string))
    if len(variant_collections) == 0:
        raise ValueError(
            "No variants loaded (use --maf, --vcf, --json-variants options)")
    elif len(variant_collections) == 1:
        return variant_collections[0]
    else:
        combined_variants = []
        for variant_collection in variant_collections:
            combined_variants.extend(list(variant_collection))
        return VariantCollection(combined_variants)
