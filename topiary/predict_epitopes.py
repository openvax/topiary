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

from __future__ import print_function, division, absolute_import

from .commandline_args import (
    mhc_binding_predictor_from_args,
    variant_collection_from_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .mutant_epitope_predictor import MutantEpitopePredictor
from .lazy_ligandome_dict import LazyLigandomeDict

def predict_epitopes(
        variant_collection,
        mhc_model,
        padding_around_mutation,
        transcript_expression_dict,
        min_transcript_expression=0,
        gene_expression_dict=None,
        min_gene_expression=0,
        ic50_cutoff=500.0,
        percentile_cutoff=None,
        only_novel_epitopes=False,
        wildtype_ligandome_dict=None,
        raise_on_variant_effect_error=True):
    """
    Predict epitopes from a Variant collection, filtering options, and
    optional gene and transcript expression data.

    Parameters
    ----------
    variant_collection : varcode.VariantCollection

    mhc_model : mhctools.BasePredictor
        Any instance of a peptide-MHC binding affinity predictor

    padding_around_mutation : int
        How many residues surrounding a mutation to consider including in a
        candidate epitope.

    transcript_expression_dict : dict
        Maps from Ensembl transcript IDs to FPKM expression values.

    min_transcript_expression : float, optional
        Don't include epitopes from transcripts with FPKM values lower than this
        parameter.

    gene_expression_dict : dict, optional
        Maps from Ensembl gene IDs to FPKM expression values.

    min_gene_expression : float, optional
        Don't include epitopes from genes with FPKM values lower than this
        parameter.

    ic50_cutoff : float, optional
        Maximum predicted IC50 value for a peptide to be considered a binder.

    percentile_cutoff : float, optional
        Maximum percentile rank of IC50 values for a peptide to be considered
        a binder.

    only_novel_epitopes : bool, optional
        If True, then drop peptides which either don't contain a mutation or
        occur elsewhere in the self-ligandome.

    wildtype_ligandome_dict : dict-like, optional
        Mapping from allele names to set of wildtype peptides predicted
        to bind to that allele. If any predicted mutant epitope is found
        in the peptide sets for the patient's alleles, it is marked as
        wildtype (non-mutant).

    raise_on_variant_effect_error : bool, optional
        If False, then skip variants which raise exceptions during effect
        inference.
    """
    predictor = MutantEpitopePredictor(
        mhc_model=mhc_model,
        padding_around_mutation=padding_around_mutation,
        ic50_cutoff=ic50_cutoff,
        percentile_cutoff=percentile_cutoff,
        wildtype_ligandome_dict=wildtype_ligandome_dict,
        only_novel_epitopes=only_novel_epitopes)
    return predictor.epitopes_from_variants(
        variant_collection,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=min_gene_expression,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=min_transcript_expression,
        raise_on_variant_effect_error=raise_on_variant_effect_error)

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
    return predict_epitopes(
        variant_collection=variants,
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
