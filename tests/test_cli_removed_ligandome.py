"""An obsolete self-filter option must never silently admit unfiltered rows."""

import pytest

from topiary.cli import script


def test_obsolete_ligandome_option_fails_before_predictor_setup(monkeypatch, capsys, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Ignored self-filter option reached prediction")

    monkeypatch.setattr(script, "predict_epitopes_from_args", forbidden)
    peptides = tmp_path / "peptides.fa"
    peptides.write_text(">example\nSIINFEKL\n")
    with pytest.raises(SystemExit) as error:
        script.main(["--peptide-fasta", str(peptides),
                     "--wildtype-ligandome-directory", "self-ligands"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "unrecognized arguments: --wildtype-ligandome-directory self-ligands" in message
    assert "Traceback" not in message
