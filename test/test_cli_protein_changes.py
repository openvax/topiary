from nose.tools import eq_
from topiary.cli.protein_changes import protein_change_effects_from_args
from topiary.cli.args import create_arg_parser

arg_parser = create_arg_parser(
    mhc=False,
    rna=False,
    output=False)

def test_protein_change_effects_from_args_substitutions():
    args = arg_parser.parse_args([
        "--protein-change", "EGFR", "T790M",
        "--genome", "grch37",
    ])

    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 1)
    effect = effects[0]
    eq_(effect.aa_ref, "T")
    eq_(effect.aa_mutation_start_offset, 789)
    eq_(effect.aa_alt, "M")

    transcript = effect.transcript
    eq_(transcript.name, "EGFR-001")

def test_protein_change_effects_from_args_malformed_missing_ref():

    args = arg_parser.parse_args([
        "--protein-change", "EGFR", "790M",
        "--genome", "grch37"])

    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 0)

def test_protein_change_effects_from_args_malformed_missing_alt():
    args = arg_parser.parse_args([
        "--protein-change", "EGFR", "T790",
        "--genome", "grch37"])
    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 0)

def test_protein_change_effects_from_args_multiple_effects():
    args = arg_parser.parse_args([
        "--protein-change", "EGFR", "T790M",
        "--protein-change", "KRAS", "G10D",
        "--genome", "grch37"])
    effects = protein_change_effects_from_args(args)
    print(effects)
    eq_(len(effects), 2)
