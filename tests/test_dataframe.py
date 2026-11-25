import pandas as pd
import pytest
from mhctools import NetMHC
from topiary import TopiaryPredictor

pytestmark = pytest.mark.netmhc

alleles = [
    "A02:01",
    "B*07:02",
    "HLA-C*07:02",
]

DEFAULT_FPKM = 1.0


@pytest.fixture(scope="module")
def mhc_model():
    return NetMHC(alleles=alleles, default_peptide_lengths=[8, 9, 10])


def test_epitopes_to_dataframe_transcript_expression(mhc_model, cancer_test_variants):
    predictor = TopiaryPredictor(mhc_model=mhc_model, only_novel_epitopes=False)
    transcript_df = pd.DataFrame([
        {"transcript_id": transcript_id, "FPKM": DEFAULT_FPKM}
        for variant in cancer_test_variants
        for transcript_id in variant.transcript_ids
    ])
    df = predictor.predict_from_variants(
        variants=cancer_test_variants,
        expression_data={
            "gene": [],
            "transcript": [("transcript", "transcript_id", transcript_df)],
            "variant": [],
        },
    )

    assert (
        "transcript_fpkm" in df.columns
    ), "transcript_fpkm missing from %s" % (df.columns,)
    assert (
        df["transcript_fpkm"] == DEFAULT_FPKM
    ).all(), "Invalid FPKM values in DataFrame transcript_fpkm column"


def test_epitopes_to_dataframe_gene_expression(mhc_model, cancer_test_variants):
    predictor = TopiaryPredictor(mhc_model=mhc_model, only_novel_epitopes=False)
    gene_df = pd.DataFrame([
        {"gene_id": gene_id, "FPKM": DEFAULT_FPKM}
        for variant in cancer_test_variants
        for gene_id in variant.gene_ids
    ])
    df = predictor.predict_from_variants(
        variants=cancer_test_variants,
        expression_data={
            "gene": [("gene", "gene_id", gene_df)],
            "transcript": [],
            "variant": [],
        },
    )

    assert "gene_fpkm" in df.columns, "gene_fpkm missing from %s" % (
        df.columns,
    )
    assert (
        df["gene_fpkm"] == DEFAULT_FPKM
    ).all(), "Invalid FPKM values in DataFrame gene_fpkm column"
