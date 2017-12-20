from nose.tools import eq_
from mhctools import NetMHC
from topiary import epitopes_to_dataframe, TopiaryPredictor
from .data import cancer_test_variants

alleles = [
    'A02:01',
    'B*07:02',
    'HLA-C*07:02',
]

mhc_model = NetMHC(
    alleles=alleles,
    default_peptide_lengths=[8, 9, 10])

def test_epitopes_to_dataframe_length():
    predictor = TopiaryPredictor(
        mhc_model=mhc_model, only_novel_epitopes=False)
    epitopes = predictor.epitopes_from_variants(variants=cancer_test_variants,)
    df = epitopes_to_dataframe(epitopes)
    eq_(len(df), len(epitopes))

DEFAULT_FPKM = 1.0

def test_epitopes_to_dataframe_transcript_expression():
    predictor = TopiaryPredictor(
        mhc_model=mhc_model,
        only_novel_epitopes=False)
    epitopes = predictor.epitopes_from_variants(variants=cancer_test_variants)
    df = epitopes_to_dataframe(
        epitopes,
        transcript_expression_dict={
            transcript_id: DEFAULT_FPKM
            for variant in cancer_test_variants
            for transcript_id in variant.transcript_ids
        })
    assert "transcript_expression" in df.columns, \
        "transcript_expression missing from %s" % (df.columns,)
    assert(df["transcript_expression"] == DEFAULT_FPKM).all(), \
        "Invalid FPKM values in DataFrame transcript_expression column"

def test_epitopes_to_dataframe_gene_expression():
    predictor = TopiaryPredictor(
        mhc_model=mhc_model,
        only_novel_epitopes=False)

    epitopes = predictor.epitopes_from_variants(variants=cancer_test_variants)
    df = epitopes_to_dataframe(
        epitopes,
        gene_expression_dict={
            gene_id: DEFAULT_FPKM
            for variant in cancer_test_variants
            for gene_id in variant.gene_ids
        },)
    assert "gene_expression" in df.columns, \
        "gene_expression missing from %s" % (df.columns,)
    assert(df["gene_expression"] == DEFAULT_FPKM).all(), \
        "Invalid FPKM values in DataFrame gene_expression column"
