from nose.tools import eq_
from topiary import contains_mutant_residues

def test_contains_mutant_residues_before():
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=5,
            mutation_end_in_protein=6),
        False)


def test_contains_mutant_residues_after():
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=25,
            mutation_end_in_protein=26),
        False)

def test_contains_mutant_residues_inside():
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=12,
            mutation_end_in_protein=13),
        True)

def test_contains_mutant_residues_deletion_before_beginning():
    # peptide only contains the residue *after* the mutation
    # so it still looks like it's wildtype
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=10,
            mutation_end_in_protein=10),
        False)


def test_contains_mutant_residues_deletion_at_beginning():
    # peptide contains mutation before *and* after mutation so
    # it should count as having a mutant juxtaposition of residues
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=11,
            mutation_end_in_protein=11),
        True)

def test_contains_mutant_residues_deletion_after_end():
    # peptide only contains the residue *before* the mutation
    # so it still looks like it's wildtype
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=19,
            mutation_end_in_protein=19),
        False)

def test_contains_mutant_residues_deletion_at_end():
    # peptide contains mutation before *and* after mutation so
    # it should count as having a mutant juxtaposition of residues
    eq_(
        contains_mutant_residues(
            peptide_start_in_protein=10,
            peptide_length=9,
            mutation_start_in_protein=18,
            mutation_end_in_protein=18),
        True)
