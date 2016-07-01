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

from ..lazy_ligandome_dict import LazyLigandomeDict
from ..epitope_prediction import predict_epitopes_from_variants

from .variants import variant_collection_from_args
from .mhc import mhc_binding_predictor_from_args
from .rna import (
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)

def predict_epitopes_from_args(args):
    """
    Returns an mhctools.EpitopeCollection of predictions based on the given
    commandline arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed commandline arguments for Topiary
    """
    mhc_model = mhc_binding_predictor_from_args(args)
    variants = variant_collection_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)
    if args.wildtype_ligandome_directory:
        wildtype_ligandome_dict = LazyLigandomeDict(
            args.wildtype_ligandome_directory)
    else:
        wildtype_ligandome_dict = None
    return predict_epitopes_from_variants(
        variants=variants,
        mhc_model=mhc_model,
        padding_around_mutation=args.padding_around_mutation,
        ic50_cutoff=args.ic50_cutoff,
        percentile_cutoff=args.percentile_cutoff,
        transcript_expression_dict=transcript_expression_dict,
        min_transcript_expression=args.rna_min_transcript_expression,
        gene_expression_dict=gene_expression_dict,
        min_gene_expression=args.rna_min_gene_expression,
        only_novel_epitopes=args.only_novel_epitopes,
        wildtype_ligandome_dict=wildtype_ligandome_dict,
        raise_on_variant_effect_error=not args.skip_variant_errors)
