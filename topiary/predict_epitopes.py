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

from typechecks import require_integer
from varcode import NonsilentCodingMutation
from mhctools import EpitopeCollection

from .commandline_args import (
    mhc_binding_predictor_from_args,
    variant_collection_from_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .lazy_ligandome_dict import LazyLigandomeDict
from .epitope_prediction import EpitopePrediction
from .filters import (
    apply_filter,
    apply_epitope_filters,
    apply_effect_expression_filters,
    apply_variant_expression_filters,
)
from .sequence_helpers import protein_slices_around_mutations

DEFAULT_IC50_CUTOFF = 500.0
DEFAULT_PERCENTILE_CUTOFF = 2.0

def check_padding_around_mutation(given_padding, epitope_lengths):
    """
    If user doesn't provide any padding around the mutation we need
    to at least include enough of the surrounding non-mutated
    esidues to construct candidate epitopes of the specified lengths
    """
    min_required_padding = max(epitope_lengths) - 1
    if not given_padding:
        return min_required_padding
    else:
        require_integer(given_padding, "Padding around mutation")
        if given_padding < min_required_padding:
            raise ValueError("Padding around mutation %d cannot "
                             "be less than %d for epitope lengths "
                             "%s" % (
                                given_padding,
                                min_required_padding,
                                epitope_lengths))
        return given_padding

def predict_epitopes_from_mutation_effects(
            effects,
            mhc_model,
            padding_around_mutation,
            transcript_expression_dict,
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
            candidate epitope.

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
        effects = apply_filter(
            lambda effect: isinstance(effect, NonsilentCodingMutation),
            effects,
            result_fn=effects.clone_with_new_elements,
            filter_name="Silent mutation")

        effects = apply_effect_expression_filters(
            effects,
            gene_expression_dict,
            gene_expression_threshold,
            transcript_expression_dict,
            transcript_expression_threshold)

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

        # dictionary mapping varcode effect objects to subsequences
        # around each mutation, along with their start/end offsets in
        # the full protein sequence.
        mutant_protein_slices = protein_slices_around_mutations(
            effects=top_effects,
            padding_around_mutation=padding_around_mutation)

        # dictionary mapping varcode effects to subsequences of each protein
        # containing mutant residues, used as argument to MHC binding
        # prediction model.
        mutant_subsequence_dict = {
            effect: protein_slice.protein_sequence[
                protein_slice.start_offset:protein_slice.end_offset]
            for (effect, protein_slice)
            in mutant_protein_slices.items()
        }

        # adjust offsets and source sequences of peptides in binding
        # predictions to reflect the longer source sequence they come from
        epitope_predictions = []

        for binding_prediction in mhc_model.predict(mutant_subsequence_dict):
            effect = binding_prediction.source_sequence_key
            # dict values are ProteinSlice(protein_sequence, start_offset, end_offset)
            protein_slice = mutant_protein_slices[effect]

            # extracting the fields of the BindingPrediction to add
            # extra field necessary to turn this into an EpitopePrediction
            fields = binding_prediction.__dict__

            # update source_sequence to be the full mutant protein
            fields["full_protein_sequence"] = protein_slice.protein_sequence

            # shift the offset of each peptide to account for the fact that the
            # predictions were made over a sub-sequence of the full protein
            fields["protein_offset"] = fields["offset"] + protein_slice.start_offset

            peptide_start = binding_prediction.offset
            peptide_end = binding_prediction.offset + binding_prediction.length - 1

            fields["contains_mutant_residues"] = (
                protein_slice.start_offset + peptide_start < effect.aa_mutation_end_offset and
                protein_slice.start_offset + peptide_end >= effect.aa_mutation_start_offset
            )
            # tag predicted epitopes as non-mutant if they occur in any of the
            # wildtype "self" binding peptide sets for the given alleles
            wildtype_peptides = wildtype_ligandome_dict[binding_prediction.allele]
            fields["occurs_in_self_ligandome"] = (
                wildtype_ligandome_dict is not None and
                binding_prediction.peptide in wildtype_peptides
            )
            fields["mutant"] = (
                fields["contains_mutant_residues"] and
                not fields["occurs_in_self_ligandome"]
            )
            epitope_predictions.append(EpitopePrediction(**fields))
        epitope_predictions = EpitopeCollection(epitope_predictions)

        logging.info("MHC predictor returned %s peptide binding predictions" % (
            len(epitope_predictions)))

        return apply_epitope_filters(
            epitope_predictions,
            ic50_cutoff=ic50_cutoff,
            percentile_cutoff=percentile_cutoff,
            only_novel_epitopes=only_novel_epitopes)

def predict_epitopes_from_variants(
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
    # pre-filter variants by checking if any of the genes or
    # transcripts they overlap have sufficient expression.
    # I'm tolerating the redundancy of this code since it's much cheaper
    # to filter a variant *before* trying to predict its impact/effect
    # on the protein sequence.
    variants = apply_variant_expression_filters(
        variant_collection,
        gene_expression_dict,
        min_gene_expression,
        transcript_expression_dict,
        min_transcript_expression)

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
