
from topiary.filters import apply_variant_expression_filters

DEFAULT_FPKM = 1.0

def test_apply_variant_gene_expression_below_threshold(cancer_test_variants, gene_expression_dict):
    filtered = apply_variant_expression_filters(
        cancer_test_variants,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=2 * DEFAULT_FPKM,
        transcript_expression_dict=None,
        transcript_expression_threshold=None)
    assert len(filtered) == 0, \
        "All variants should have been filtered out but got: %s" % (filtered,)

def test_apply_variant_gene_expression_above_threshold(cancer_test_variants, gene_expression_dict):
    filtered = apply_variant_expression_filters(
        cancer_test_variants,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=0.5 * DEFAULT_FPKM,
        transcript_expression_dict=None,
        transcript_expression_threshold=None)
    assert len(filtered) == len(cancer_test_variants), \
        "Expected %s variants but got %s" % (len(cancer_test_variants), len(filtered))

def test_apply_variant_transcript_expression_below_threshold(cancer_test_variants, transcript_expression_dict):
    filtered = apply_variant_expression_filters(
        cancer_test_variants,
        gene_expression_dict=None,
        gene_expression_threshold=None,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=2 * DEFAULT_FPKM)
    assert len(filtered) == 0, \
        "All variants should have been filtered out but got: %s" % (filtered,)

def test_apply_variant_transcript_expression_above_threshold(cancer_test_variants, transcript_expression_dict):
    filtered = apply_variant_expression_filters(
        cancer_test_variants,
        gene_expression_dict=None,
        gene_expression_threshold=None,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=0.5 * DEFAULT_FPKM)
    assert len(filtered) == len(cancer_test_variants), \
        "Expected %s variants but got %s" % (len(cancer_test_variants), len(filtered))
