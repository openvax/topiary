from topiary.filters import apply_effect_expression_filters

DEFAULT_FPKM = 1.0


def test_apply_effect_gene_expression_below_threshold(cancer_test_effects, gene_expression_dict):
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=2 * DEFAULT_FPKM,
        transcript_expression_dict=None,
        transcript_expression_threshold=None,
    )
    assert (
        len(filtered) == 0
    ), "All variants should have been filtered out but got: %s" % (filtered,)


def test_apply_effect_gene_expression_above_threshold(cancer_test_effects, gene_expression_dict):
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=0.5 * DEFAULT_FPKM,
        transcript_expression_dict=None,
        transcript_expression_threshold=None,
    )
    assert len(filtered) == len(
        cancer_test_effects
    ), "Expected %s effects but got %s" % (len(cancer_test_effects), len(filtered))


def test_apply_effect_gene_expression_equal_threshold(cancer_test_effects, gene_expression_dict):
    # expect genes with expression at threshold to NOT get filtered
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=gene_expression_dict,
        gene_expression_threshold=DEFAULT_FPKM,
        transcript_expression_dict=None,
        transcript_expression_threshold=None,
    )
    assert len(filtered) == len(
        cancer_test_effects
    ), "Expected %s effects but got %s" % (len(cancer_test_effects), len(filtered))


def test_apply_effect_transcript_expression_below_threshold(cancer_test_effects, transcript_expression_dict):
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=None,
        gene_expression_threshold=None,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=2 * DEFAULT_FPKM,
    )
    assert (
        len(filtered) == 0
    ), "All effects should have been filtered out but got: %s" % (filtered,)


def test_apply_effect_transcript_expression_above_threshold(cancer_test_effects, transcript_expression_dict):
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=None,
        gene_expression_threshold=None,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=0.5 * DEFAULT_FPKM,
    )
    assert len(filtered) == len(
        cancer_test_effects
    ), "Expected %s effects but got %s" % (len(cancer_test_effects), len(filtered))


def test_apply_effect_transcript_expression_equal_threshold(cancer_test_effects, transcript_expression_dict):
    # expect transcripts with expression at threshold to NOT be filtered
    filtered = apply_effect_expression_filters(
        cancer_test_effects,
        gene_expression_dict=None,
        gene_expression_threshold=None,
        transcript_expression_dict=transcript_expression_dict,
        transcript_expression_threshold=DEFAULT_FPKM,
    )
    assert len(filtered) == len(
        cancer_test_effects
    ), "Expected %s effects but got %s" % (len(cancer_test_effects), len(filtered))
