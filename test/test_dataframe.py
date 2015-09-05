from nose.tools import eq_
from mhctools import NetMHC
from topiary import epitopes_to_dataframe, predict_epitopes_from_variants
from .data import cancer_test_variants

alleles = [
    'A02:01',
    'B*07:02',
    'HLA-C*07:02',
]

epitope_lengths = [8, 9, 10]

mhc_model = NetMHC(
    alleles=alleles,
    epitope_lengths=epitope_lengths)

def test_epitopes_to_dataframe_length():
    epitopes = predict_epitopes_from_variants(
        variants=cancer_test_variants,
        mhc_model=mhc_model,
        transcript_expression_dict=None,
        only_novel_epitopes=False)
    df = epitopes_to_dataframe(epitopes)
    eq_(len(df), len(epitopes))

DEFAULT_FPKM = 1.0

def test_epitopes_to_dataframe_transcript_expression():
    epitopes = predict_epitopes_from_variants(
        variants=cancer_test_variants,
        mhc_model=mhc_model,
        only_novel_epitopes=False)
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
    epitopes = predict_epitopes_from_variants(
        variants=cancer_test_variants,
        mhc_model=mhc_model,
        only_novel_epitopes=False)
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

