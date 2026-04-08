import os
import tempfile

import pytest

from topiary.cli.args import arg_parser, predict_epitopes_from_args


def _tmpfile(content, suffix):
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False)
    handle.write(content)
    handle.close()
    return handle.name


def test_peptide_csv_predictions_are_scored_as_is():
    path = _tmpfile(
        "name,peptide\npep1,SIINFEKLA\npep2,ELAGIGILT\n",
        ".csv",
    )
    try:
        args = arg_parser.parse_args(
            [
                "--mhc-predictor",
                "random",
                "--mhc-alleles",
                "A0201",
                "--peptide-csv",
                path,
            ]
        )
        df = predict_epitopes_from_args(args)
    finally:
        os.unlink(path)

    assert sorted(df["peptide"].unique()) == ["ELAGIGILT", "SIINFEKLA"]
    assert sorted(df["source_sequence_name"].unique()) == ["pep1", "pep2"]
    assert set(df["peptide_offset"].unique()) == {0}


def test_direct_inputs_reject_variant_pipeline_args():
    path = _tmpfile(">pep1\nSIINFEKLA\n", ".fasta")
    try:
        args = arg_parser.parse_args(
            [
                "--mhc-predictor",
                "random",
                "--mhc-alleles",
                "A0201",
                "--fasta",
                path,
                "--variant",
                "1",
                "1",
                "A",
                "T",
                "--genome",
                "GRCh38",
            ]
        )
        with pytest.raises(
            ValueError,
            match="Direct sequence inputs can't be combined with variant/RNA inputs",
        ):
            predict_epitopes_from_args(args)
    finally:
        os.unlink(path)
