from pyensembl import EnsemblRelease
from varcode import Variant, VariantCollection
from nose.tools import eq_

from utopia import predict


def test_basic():
    release = EnsemblRelease(75)
    variant_collection = VariantCollection([
        Variant(contig=10, start=100018900, ref='C', alt='T',
                ensembl=release),
        Variant(contig=11, start=32861682, ref='G', alt='A',
                ensembl=release)])
    output = predict(variant_collection,
                     hla_alleles=['A02:01', 'A02:04', 'B07:02',
                                  'B14:02', 'C07:02', 'C07:01'],
                     mutation_window_size=30)
    eq_(len(output), 636)
    binders = output[output['MHC_IC50'] < 500]
    eq_(len(binders), 12)
