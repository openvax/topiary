from topiary.cli.protein_changes import protein_change_effects_from_args
from topiary.cli.args import create_arg_parser

from .common import eq_

arg_parser = create_arg_parser(mhc=False, rna=False, output=False)


def test_protein_change_effects_from_args_substitutions():
    args = arg_parser.parse_args(
        [
            "--protein-change",
            "EGFR",
            "T790M",
            "--genome",
            "grch37",
        ]
    )

    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 1)
    effect = effects[0]
    eq_(effect.aa_ref, "T")
    eq_(effect.aa_mutation_start_offset, 789)
    eq_(effect.aa_alt, "M")

    transcript = effect.transcript
    eq_(transcript.name, "EGFR-001")


def test_protein_change_effects_from_args_malformed_missing_ref():

    args = arg_parser.parse_args(
        ["--protein-change", "EGFR", "790M", "--genome", "grch37"]
    )

    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 0)


def test_protein_change_effects_from_args_malformed_missing_alt():
    args = arg_parser.parse_args(
        ["--protein-change", "EGFR", "T790", "--genome", "grch37"]
    )
    effects = protein_change_effects_from_args(args)
    eq_(len(effects), 0)


def test_protein_change_effects_from_args_multiple_effects():
    args = arg_parser.parse_args(
        [
            "--protein-change",
            "EGFR",
            "T790M",
            "--protein-change",
            "KRAS",
            "G10D",
            "--genome",
            "grch37",
        ]
    )
    effects = protein_change_effects_from_args(args)
    print(effects)
    eq_(len(effects), 2)


# --protein-change is now an input mode the CLI runs, not just a parser
# nothing called (#322).


def test_protein_change_predicts_peptides_tiling_the_substitution(tmp_path, capsys):
    """EGFR T790M, named at the protein level with no genomic coordinate."""
    import pandas as pd
    from topiary.cli.script import main

    out = tmp_path / "t790m.csv"
    main([
        "--mhc-predictor", "random", "--mhc-alleles", "HLA-A*02:01",
        "--mhc-peptide-lengths", "9",
        "--protein-change", "EGFR", "T790M", "--genome", "grch37",
        "--output-csv", str(out),
    ])
    capsys.readouterr()
    df = pd.read_csv(out)

    assert len(df) == 9                       # every 9-mer window over the residue
    assert set(df.gene) == {"EGFR"}
    assert set(df.effect) == {"p.T790M"}
    # The mutated residue walks back through the window as it slides, and
    # every peptide carries the substituted M at that position.
    assert df.mutation_start_in_peptide.tolist() == list(range(8, -1, -1))
    for peptide, position in zip(df.peptide, df.mutation_start_in_peptide):
        assert peptide[position] == "M"
    # No genomic variant was named, so the column stays empty rather than
    # carrying an invented coordinate.
    assert df.variant.isna().all()


def test_protein_change_is_refused_beside_a_genomic_input(capsys):
    import pytest

    from topiary.cli.script import main

    with pytest.raises(SystemExit) as exit_info:
        main([
            "--mhc-predictor", "random", "--mhc-alleles", "HLA-A*02:01",
            "--protein-change", "EGFR", "T790M",
            "--variant", "7", "55259515", "T", "G", "--genome", "grch37",
        ])
    # A malformed command line, so argparse's status and usage block.
    assert exit_info.value.code == 2
    message = capsys.readouterr().err
    assert "--protein-change is a separate input mode from --variant" in message
    assert "combine_sources" in message
