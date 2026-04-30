import pytest

from topiary.cli.script import main


def test_main_without_args_reports_cli_error(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "usage: topiary" in captured.err
    assert "topiary: error:" in captured.err
    assert "No prediction request specified" in captured.err
    assert "--mhc-predictor" in captured.err
    assert "No input specified" in captured.err
    assert "Traceback" not in captured.err
    assert "Namespace(" not in captured.out


def test_main_missing_mhc_source_reports_cli_error(tmp_path, capsys):
    peptide_csv = tmp_path / "peptides.csv"
    peptide_csv.write_text("peptide\nSLLQHLIGL\n")

    with pytest.raises(SystemExit) as exc_info:
        main(["--peptide-csv", str(peptide_csv)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "usage: topiary" in captured.err
    assert "--mhc-predictor" in captured.err
    assert "--mhc-cache-file / --mhc-cache-directory" in captured.err
    assert "Traceback" not in captured.err
    assert "Namespace(" not in captured.out
