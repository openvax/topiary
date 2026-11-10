"""Species chooses source/reference data without rewriting presenting alleles."""

from argparse import Namespace
from types import SimpleNamespace

import pandas as pd
import pytest
from pyensembl import EnsemblRelease

from topiary.cli import args as cli
from topiary.cli.protein_changes import genome_from_args
from topiary.cli.references import exclusion_options, validate_source_options


HUMAN = "homo_sapiens"
MOUSE = "mus_musculus"
PEPTIDES = {HUMAN: "SIINFEKLA", MOUSE: "AAAAAAAAA"}
CONTROL = "VVVVVVVVV"


def prediction_args(*extra, alleles="HLA-A*02:01"):
    return cli.arg_parser.parse_args(
        [
            "--mhc-predictor",
            "random",
            "--mhc-alleles",
            alleles,
            "--mhc-peptide-lengths",
            "9",
            *extra,
        ]
    )


def install_species_sources(monkeypatch):
    calls = []

    def genes(names, *, species, release):
        calls.append(("source", species, release))
        return {"target": PEPTIDES[species], "control": CONTROL}

    def background(*, species, release):
        calls.append(("background", species, release))
        return {"shared_header": PEPTIDES[species]}

    monkeypatch.setattr(cli, "sequences_from_gene_names", genes)
    monkeypatch.setattr(cli, "ensembl_proteome", background)
    return calls


@pytest.mark.parametrize(
    "flag,loader",
    [
        ("--gene-names", "sequences_from_gene_names"),
        ("--gene-ids", "sequences_from_gene_ids"),
        ("--transcript-ids", "sequences_from_transcript_ids"),
        ("--ensembl-proteome", "ensembl_proteome"),
    ],
)
def test_all_ensembl_direct_sources_receive_species_and_release(
    monkeypatch, flag, loader
):
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return {"mouse": PEPTIDES[MOUSE]}

    monkeypatch.setattr(cli, loader, load)
    flags = [flag] if flag == "--ensembl-proteome" else [flag, "identifier"]
    args = prediction_args(*flags, "--species", "mouse", "--ensembl-release", "102")
    assert cli.predict_epitopes_from_args(args).peptide.tolist() == [PEPTIDES[MOUSE]]
    assert calls == [{"species": MOUSE, "release": 102}]


def test_cta_can_be_human_with_a_mouse_background(monkeypatch):
    calls = install_species_sources(monkeypatch)
    monkeypatch.setattr(cli, "cta_sequences", lambda release: {"cta": PEPTIDES[HUMAN]})
    args = prediction_args(
        "--cta",
        "--species",
        "Homo sapiens",
        "--exclude-ensembl",
        "--exclude-species",
        "mouse",
        "--exclude-ensembl-release",
        "102",
    )
    result = cli.predict_epitopes_from_args(args)
    assert result.peptide.tolist() == [PEPTIDES[HUMAN]]
    assert result.allele.tolist() == ["HLA-A*02:01"]
    assert calls == [("background", MOUSE, 102)]


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--gene-names", "Trp53", "--species", "mouse"], "--ensembl-release"),
        (
            ["--cta", "--species", "mouse", "--ensembl-release", "102"],
            "--cta.*human-only",
        ),
        (["--gene-names", "TP53", "--genome", "GRCm38:102"], "Direct sequence inputs"),
        (["--gene-names", "TP53", "--exclude-species", "mouse"], "require a built-in"),
    ],
)
def test_invalid_species_requests_fail_before_loading_a_model(
    monkeypatch, extra, message
):
    monkeypatch.setattr(
        cli, "sequences_from_gene_names", lambda *a, **k: {"p": CONTROL}
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A model must not be constructed for invalid source options")

    monkeypatch.setattr(cli, "predictors_from_args", forbidden)
    monkeypatch.setattr(cli, "cta_sequences", forbidden)
    with pytest.raises(ValueError, match=message):
        cli.predict_epitopes_from_args(prediction_args(*extra))


@pytest.mark.parametrize("flag", ["--exclude-non-cta", "--exclude-tissues"])
def test_human_only_backgrounds_check_reference_not_antigen_species(monkeypatch, flag):
    calls = install_species_sources(monkeypatch)
    reference_calls = []

    def human_reference(*args, **kwargs):
        reference_calls.append(kwargs)
        return {"normal": PEPTIDES[HUMAN]}

    monkeypatch.setattr(cli, "non_cta_sequences", human_reference)
    monkeypatch.setattr(cli, "tissue_expressed_sequences", human_reference)
    flags = [flag, "lung"] if flag == "--exclude-tissues" else [flag]
    args = prediction_args(
        "--gene-names",
        "Trp53",
        "--species",
        "mouse",
        "--ensembl-release",
        "102",
        *flags,
    )
    with pytest.raises(ValueError, match="human-only"):
        cli.predict_epitopes_from_args(args)
    assert reference_calls == []
    args.exclude_species = "human"
    result = cli.predict_epitopes_from_args(args)
    assert sorted(result.peptide) == sorted([PEPTIDES[MOUSE], CONTROL])
    assert reference_calls == [{"release": None}]
    assert calls == [("source", MOUSE, 102), ("source", MOUSE, 102)]


def test_variant_background_uses_actual_annotation_not_ignored_genome_flag(monkeypatch):
    calls = install_species_sources(monkeypatch)
    variants = [SimpleNamespace(genome=EnsemblRelease(102, "mouse"))]
    monkeypatch.setattr(cli, "variant_collection_from_args", lambda args: variants)

    def predict(self, variants, **kwargs):
        assert variants[0].genome.species.latin_name == MOUSE
        return self.predict_from_named_sequences(
            {"target": PEPTIDES[MOUSE], "control": CONTROL}
        )

    monkeypatch.setattr(cli.TopiaryPredictor, "predict_from_variants", predict)
    args = prediction_args(
        "--maf", "mouse.maf", "--genome", "GRCh38:115", "--exclude-ensembl"
    )
    result = cli.predict_epitopes_from_args(args)
    assert result.peptide.tolist() == [CONTROL]
    assert result.allele.tolist() == ["HLA-A*02:01"]
    assert calls == [("background", MOUSE, 102)]
    args.species = "human"
    with pytest.raises(ValueError, match="conflicts with input annotation species"):
        cli.predict_epitopes_from_args(args)
    args.species = "mouse"
    args.ensembl_release = 115
    with pytest.raises(ValueError, match="conflicts with input annotation release"):
        cli.predict_epitopes_from_args(args)


def test_explicit_background_does_not_borrow_the_sources_release():
    args = Namespace(
        species="human",
        ensembl_release=115,
        exclude_species="mouse",
        exclude_ensembl=True,
    )
    with pytest.raises(ValueError, match="--exclude-ensembl-release"):
        exclusion_options(args)
    args.exclude_ensembl_release = 102
    assert exclusion_options(args) == {"species": MOUSE, "release": 102}


def test_ambiguous_or_custom_annotations_need_an_explicit_background():
    human, mouse = EnsemblRelease(75, "human"), EnsemblRelease(102, "mouse")
    args = Namespace(exclude_ensembl=True)
    with pytest.raises(ValueError, match="multiple species"):
        exclusion_options(args, [human, mouse])
    with pytest.raises(ValueError, match="one numbered Ensembl release"):
        exclusion_options(args, [human, EnsemblRelease(115, "human")])
    args.exclude_ensembl_release = 75
    assert exclusion_options(args, [human, EnsemblRelease(115, "human")]) == {
        "species": HUMAN,
        "release": 75,
    }
    args.exclude_ensembl_release = None
    custom = SimpleNamespace(species=mouse.species, release=None)
    with pytest.raises(ValueError, match="one numbered Ensembl release"):
        exclusion_options(args, [custom])
    custom.release = "2026_04"
    with pytest.raises(ValueError, match="one numbered Ensembl release"):
        exclusion_options(args, [custom])
    args.exclude_species = "human"
    assert exclusion_options(args, [human, mouse]) == {
        "species": HUMAN,
        "release": None,
    }
    args.exclude_ensembl = False
    args.exclude_species = None
    assert exclusion_options(args, [human, mouse]) is None
    validate_source_options(args, [human, mouse])


def test_protein_change_source_checks_species_and_release():
    args = Namespace(genome="GRCm38:102", species="mouse", ensembl_release=102)
    genome = genome_from_args(args)
    assert genome.species.latin_name == MOUSE and genome.release == 102
    args.genome = None
    assert genome_from_args(args) == genome
    args.genome = "GRCh38:115"
    with pytest.raises(ValueError, match="conflicts with input annotation species"):
        genome_from_args(args)


def test_multiple_background_fastas_keep_colliding_headers(tmp_path):
    paths = []
    for species, peptide in PEPTIDES.items():
        path = tmp_path / (species + ".fasta")
        path.write_text(">shared_header\n" + peptide + "\n")
        paths.append(str(path))
    frame = pd.DataFrame({"peptide": [*PEPTIDES.values(), CONTROL]})
    args = Namespace(exclude_fasta=paths, exclude_mode="exact")
    result = cli._apply_exclusion(frame, args)
    assert result.peptide.tolist() == [CONTROL]
