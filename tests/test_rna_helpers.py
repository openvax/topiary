import pandas as pd
from topiary.rna.cufflinks import parse_locus_column
from nose.tools import eq_


def test_parse_locus_column_with_chr():
    """
    test_parse_locus_column_with_chr: Test that 'chr' prefix from
    chromosome names gets correctly dropped
    """
    df = pd.DataFrame({"locus": ["chr1:10-20", "chrX:30-40"]})
    loci = df["locus"]
    chromosomes, starts, ends = parse_locus_column(loci)
    eq_(list(chromosomes), ["1", "X"])
    eq_(list(starts), [10, 30])
    eq_(list(ends), [20, 40])


def test_parse_locus_column_without_chr():
    """
    test_parse_locus_column_without_chr: Test that chromosome names can be
    parsed without 'chr' prefix
    """
    df = pd.DataFrame({"locus": ["1:10-20", "X:30-40"]})
    loci = df["locus"]
    chromosomes, starts, ends = parse_locus_column(loci)
    eq_(list(chromosomes), ["1", "X"])
    eq_(list(starts), [10, 30])
    eq_(list(ends), [20, 40])
