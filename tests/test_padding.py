from nose.tools import eq_, assert_raises
from topiary import check_padding_around_mutation

def test_default_padding():
    # expect padding to be one less than the largest epitope length
    eq_(check_padding_around_mutation(None, [8, 9, 10]), 9)

def test_invalid_padding():
    # padding is insufficient for the epitope lengths given
    with assert_raises(ValueError):
        check_padding_around_mutation(2, [9])
