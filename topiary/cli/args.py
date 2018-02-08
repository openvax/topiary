# Copyright (c) 2017. Mount Sinai School of Medicine
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

from argparse import ArgumentParser
from mhctools.cli import add_mhc_args, mhc_binding_predictor_from_args
from varcode.cli import add_variant_args, variant_collection_from_args

from .filtering import add_filter_args
from .rna import (
    add_rna_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .sequence import add_sequence_args
from .errors import add_error_args
from .outputs import add_output_args
from .protein_changes import add_protein_change_args
from ..predictor import TopiaryPredictor

def create_arg_parser(
        rna=True,
        mhc=True,
        variants=True,
        protein_changes=True,
        filters=True,
        sequence_options=True,
        error_options=True,
        output=True):
    arg_parser = ArgumentParser()
    if rna:
        add_rna_args(arg_parser)
    if mhc:
        add_mhc_args(arg_parser)
    if variants:
        add_variant_args(arg_parser)
    if protein_changes:
        add_protein_change_args(arg_parser)
    if filters:
        add_filter_args(arg_parser)
    if sequence_options:
        add_sequence_args(arg_parser)
    if error_options:
        add_error_args(arg_parser)
    if output:
        add_output_args(arg_parser)
    return arg_parser

# keeping global instance for backwards compatibility with existing code
arg_parser = create_arg_parser()

def predict_epitopes_from_args(args):
    """
    Returns an epitope collection from the given commandline arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed commandline arguments for Topiary
    """
    mhc_model = mhc_binding_predictor_from_args(args)
    variants = variant_collection_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)

    predictor = TopiaryPredictor(
        mhc_model=mhc_model,
        padding_around_mutation=args.padding_around_mutation,
        ic50_cutoff=args.ic50_cutoff,
        percentile_cutoff=args.percentile_cutoff,
        min_transcript_expression=args.rna_min_transcript_expression,
        min_gene_expression=args.rna_min_gene_expression,
        only_novel_epitopes=args.only_novel_epitopes,
        raise_on_error=not args.skip_variant_errors)
    return predictor.predict_from_variants(
        variants=variants,
        transcript_expression_dict=transcript_expression_dict,
        gene_expression_dict=gene_expression_dict)
