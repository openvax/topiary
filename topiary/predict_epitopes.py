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
import logging

from mhctools import EpitopeCollection

from .commandline_args import (
    mhc_binding_predictor_from_args,
    variant_collection_from_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .lazy_ligandome_dict import LazyLigandomeDict
from .filters import (
    apply_epitope_filters,
    apply_effect_expression_filters,
    apply_variant_expression_filters,
    filter_silent_and_noncoding_effects,
)
from .sequence_helpers import (
    protein_subsequences_around_mutations,
    check_padding_around_mutation
)
from .epitope_prediction import (
    build_epitope_collection_from_binding_predictions,
)

DEFAULT_IC50_CUTOFF = None
DEFAULT_PERCENTILE_CUTOFF = None

def predict_epitopes_from_mutation_effects(
            effects,
            mhc_model,
            padding_around_mutation=None,
            transcript_expression_dict=None,
            transcript_expression_threshold=0.0,
            gene_expression_dict=None,
            gene_expression_threshold=0.0,
            ic50_cutoff=DEFAULT_IC50_CUTOFF,
            percentile_cutoff=DEFAULT_PERCENTILE_CUTOFF,
            only_novel_epitopes=False,
            wildtype_ligandome_dict=None):
        """Given a Varcode.EffectCollection of predicted protein effects,
        return a DataFrame of the predicted epitopes around each
        mutation.

        Parameters
        ----------
        effects : Varcode.EffectCollection

        mhc_model : mhctools.BasePredictor
            Any instance of a peptide-MHC binding affinity predictor

        padding_around_mutation : int
            How many residues surrounding a mutation to consider including in a
            candidate epitope. Default is the minimum size necessary for epitope
            length of the mhc model.

        transcript_expression_dict : dict
            Dictionary mapping transcript IDs to RNA expression estimates. Used
            both for transcript expression filtering and for selecting the
            most abundant transcript for a particular variant. If omitted then
            transcript selection is done using priority of variant effects and
            transcript length.

        transcript_expression_threshold : float, optional
            If transcript_expression_dict is given, only keep effects on
            transcripts above this threshold.

        gene_expression_dict : dict, optional
            Dictionary mapping gene IDs to RNA expression estimates

        gene_expression_threshold : float, optional
            If gene_expression_dict is given, only keep effects on genes
            expressed above this threshold.

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
        """
        padding_around_mutation = check_padding_around_mutation(
            given_padding=padding_around_mutation,
            epitope_lengths=mhc_model.epitope_lengths)

        # we only care about effects which impact the coding sequence of a
        # protein
        effects = filter_silent_and_noncoding_effects(effects)

        effects = apply_effect_expression_filters(
            effects,
            transcript_expression_dict=transcript_expression_dict,
            transcript_expression_threshold=transcript_expression_threshold,
            gene_expression_dict=gene_expression_dict,
            gene_expression_threshold=gene_expression_threshold,
        )

        # group by variants, so that we end up with only one mutant
        # sequence per mutation
        variant_effect_groups = effects.groupby_variant()

        if len(variant_effect_groups) == 0:
            logging.warn("No candidates for MHC binding prediction")
            return EpitopeCollection([])

        if transcript_expression_dict:
            # if expression data is available, then for each variant
            # keep the effect annotation for the most abundant transcript
            top_effects = [
                variant_effects.top_expression_effect(
                    transcript_expression_dict)
                for variant_effects in variant_effect_groups.values()
            ]
        else:
            # if no transcript abundance data is available, then
            # for each variant keep the effect with the most significant
            # predicted effect on the protein sequence, along with using
            # transcript/CDS length as a tie-breaker for effects with the same
            # priority.
            top_effects = [
                variant_effects.top_priority_effect()
                for variant_effects in variant_effect_groups.values()
            ]

        # 1) dictionary mapping varcode effect objects to subsequences
        #    around each mutation
        # 2) dictionary mapping varcode effect to start offset of subsequence
        #    within the full mutant protein sequence
        protein_subsequences, protein_subsequence_offsets = \
            protein_subsequences_around_mutations(
                effects=top_effects,
                padding_around_mutation=padding_around_mutation)

        binding_predictions = mhc_model.predict(protein_subsequences)
        logging.info("MHC predictor returned %s peptide binding predictions" % (
            len(binding_predictions)))
        epitopes = build_epitope_collection_from_binding_predictions(
            binding_predictions=binding_predictions,
            protein_subsequences=protein_subsequences,
            protein_subsequence_start_offsets=protein_subsequence_offsets,
            wildtype_ligandome_dict=wildtype_ligandome_dict)
        return apply_epitope_filters(
            epitopes,
            ic50_cutoff=ic50_cutoff,
            percentile_cutoff=percentile_cutoff,
            only_novel_epitopes=only_novel_epitopes)

def predict_epitopes_from_variants(
        variants,
        mhc_model,
        padding_around_mutation=None,
        transcript_expression_dict=None,
        min_transcript_expression=0,
        gene_expression_dict=None,
        min_gene_expression=0,
        ic50_cutoff=DEFAULT_IC50_CUTOFF,
        percentile_cutoff=DEFAULT_PERCENTILE_CUTOFF,
        only_novel_epitopes=False,
        wildtype_ligandome_dict=None,
        raise_on_variant_effect_error=True):
    """
    Predict epitopes from a Variant collection, filtering options, and
    optional gene and transcript expression data.

    Parameters
    ----------
    variants : varcode.VariantCollection

    mhc_model : mhctools.BasePredictor
        Any instance of a peptide-MHC binding affinity predictor

    padding_around_mutation : int, optional
        How many residues surrounding a mutation to consider including in a
        candidate epitope. Default is the minimum size necessary for epitope
        length of the mhc model.

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
    # pre-filter variants by checking if any of the genes or
    # transcripts they overlap have sufficient expression.
    # I'm tolerating the redundancy of this code since it's much cheaper
    # to filter a variant *before* trying to predict its impact/effect
    # on the protein sequence.
    variants = apply_variant_expression_filters(
        variants,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=min_transcript_expression,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=min_gene_expression)

    effects = variants.effects(
        raise_on_error=raise_on_variant_effect_error)

    return predict_epitopes_from_mutation_effects(
        effects=effects,
        mhc_model=mhc_model,
        padding_around_mutation=padding_around_mutation,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=min_transcript_expression,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=min_gene_expression,
        ic50_cutoff=ic50_cutoff,
        percentile_cutoff=percentile_cutoff,
        only_novel_epitopes=only_novel_epitopes,
        wildtype_ligandome_dict=wildtype_ligandome_dict)

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
