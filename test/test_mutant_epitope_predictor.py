from mhctools import NetMHCpan
from nose.tools import eq_
from pyensembl import ensembl_grch37 as ensembl
from topiary import MutantEpitopePredictor
from varcode import Variant, VariantCollection

def _variant_collection():
    return VariantCollection([
        Variant(
          contig=10,
          start=100018900,
          ref='C',
          alt='T',
          ensembl=ensembl),
        Variant(
          contig=11,
          start=32861682,
          ref='G',
          alt='A',
          ensembl=ensembl)])

def _predictor(padding):
    return MutantEpitopePredictor(
      mhc_model_class=NetMHCpan,
      epitope_lengths=[9],
      padding_around_mutation=padding)

def _alleles():
    return [
      'A02:01',
      'a0204',
      'B*07:02',
      'HLA-B14:02',
      'HLA-C*07:02',
      'hla-c07:01'
    ]

def test_prediction_output():
    variant_collection = _variant_collection()
    predictor_without_padding = _predictor(padding=0)
    hla_alleles = _alleles()
    output_without_padding = predictor_without_padding.predict(
      variant_collection=variant_collection,
      mhc_alleles=hla_alleles)
    # one prediction for each variant * number of alleles
    eq_(len(output_without_padding.strong_binders(500.0)), 12)
    predictor_with_padding = _predictor(padding=0)
    output_with_padding = predictor_with_padding.predict(
      variant_collection=variant_collection,
      mhc_alleles=hla_alleles)
    eq_(len(output_with_padding), 636)
    df = output_with_padding.dataframe()
    binders = df[df['MHC_IC50'] < 500]
    eq_(len(binders), 12)
