from .commandline_args import (
    mhc_binding_predictor_from_args,
    variant_collection_from_args,
    rna_gene_expression_dict_from_args,
    rna_transcript_expression_dict_from_args,
)
from .mutant_epitope_predictor import MutantEpitopePredictor

def predict_epitopes(
        variant_collection,
        mhc_model,
        padding_around_mutation,
        ic50_cutoff=500.0,
        percentile_cutoff=None,
        keep_wildtype_epitopes=False,
        gene_expression_dict=None,
        min_gene_expression=0,
        transcript_expression_dict=None,
        min_transcript_expression=0,
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

    ic50_cutoff : float, optional
        Maximum predicted IC50 value for a peptide to be considered a binder.

    percentile_cutoff : float, optional
        Maximum percentile rank of IC50 values for a peptide to be considered
        a binder.

    keep_wildtype_epitopes : bool, optional
        If True, then include peptides which do not contained mutated residues.

    gene_expression_dict : dict, optional
        Maps from Ensembl gene IDs to FPKM expression values.

    min_gene_expression : float, optional
        Don't include epitopes from genes with FPKM values lower than this
        parameter.

    transcript_expression_dict : dict, optional
        Maps from Ensembl transcript IDs to FPKM expression values.

    min_transcript_expression : float, optional
        Don't include epitopes from transcripts with FPKM values lower than this
        parameter.

    raise_on_variant_effect_error : bool, optional
        If False, then skip variants which raise exceptions during effect
        inference.
    """
    predictor = MutantEpitopePredictor(
        mhc_model=mhc_model,
        padding_around_mutation=padding_around_mutation,
        ic50_cutoff=ic50_cutoff,
        percentile_cutoff=percentile_cutoff,
        keep_wildtype_epitopes=keep_wildtype_epitopes)
    return predictor.epitopes_from_variants(
        variant_collection,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=min_gene_expression,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=min_transcript_expression,
        raise_on_variant_effect_error=raise_on_variant_effect_error)

def predict_epitopes_from_args(args):
    mhc_model = mhc_binding_predictor_from_args(args)
    variants = variant_collection_from_args(args)
    gene_expression_dict = rna_gene_expression_dict_from_args(args)
    transcript_expression_dict = rna_transcript_expression_dict_from_args(args)
    return predict_epitopes(
        variant_collection=variants,
        mhc_model=mhc_model,
        padding_around_mutation=args.padding_around_mutation,
        ic50_cutoff=args.ic50_cutoff,
        percentile_cutoff=args.percentile_cutoff,
        keep_wildtype_epitopes=args.keep_wildtype_epitopes,
        gene_expression_dict=gene_expression_dict,
        min_gene_expression=args.rna_min_gene_expression,
        transcript_expression_dict=transcript_expression_dict,
        min_transcript_expression=args.rna_min_transcript_expression,
        raise_on_variant_effect_error=not args.skip_variant_errors)
