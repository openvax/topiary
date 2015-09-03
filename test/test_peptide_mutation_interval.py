from nose.tools import eq_, assert_raises
from topiary import peptide_mutation_interval

def test_peptide_mutation_interval_middle():
    start, end = peptide_mutation_interval(
        peptide_start_in_protein=10,
        peptide_length=9,
        mutation_start_in_protein=11,
        mutation_end_in_protein=12)
    eq_(start, 1)
    eq_(end, 2)


def test_peptide_mutation_interval_start():
    start, end = peptide_mutation_interval(
        peptide_start_in_protein=10,
        peptide_length=9,
        mutation_start_in_protein=7,
        mutation_end_in_protein=12)
    eq_(start, 0)
    eq_(end, 2)

def test_peptide_mutation_interval_end():
    start, end = peptide_mutation_interval(
        peptide_start_in_protein=10,
        peptide_length=9,
        mutation_start_in_protein=18,
        mutation_end_in_protein=20)
    eq_(start, 8)
    eq_(end, 9)

def test_peptide_mutation_interval_deletion():
    start, end = peptide_mutation_interval(
        peptide_start_in_protein=10,
        peptide_length=9,
        mutation_start_in_protein=15,
        mutation_end_in_protein=15)
    eq_(start, 5)
    eq_(end, 5)


def test_peptide_mutation_interval_no_overlap_before():
    with assert_raises(ValueError):
        peptide_mutation_interval(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=5,
            mutation_end_in_protein=6)

def test_peptide_mutation_interval_no_overlap_after():
    with assert_raises(ValueError):
        peptide_mutation_interval(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=25,
            mutation_end_in_protein=26)

