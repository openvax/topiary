from topiary.cli.args import arg_parser
from topiary.cli.outputs import write_outputs
import tempfile
import warnings
import pandas as pd
import pytest
from pandas.errors import SettingWithCopyWarning

from .common import eq_


def test_write_outputs():

    with tempfile.NamedTemporaryFile(mode="r+", delete=False) as f:
        df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        args = arg_parser.parse_args(
            [
                "--output-csv",
                f.name,
                "--subset-output-columns",
                "x",
                "--rename-output-column",
                "x",
                "X",
                "--mhc-predictor",
                "random",
                "--mhc-alleles",
                "A0201",
            ]
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            write_outputs(
                df, args, print_df_before_filtering=True, print_df_after_filtering=True
            )

        print("File: %s" % f.name)
        df_from_file = pd.read_csv(f.name, index_col="#")

        df_expected = pd.DataFrame({"X": [1, 2, 3]})
        print(df_from_file)
        eq_(len(df_expected), len(df_from_file))
        assert (df_expected == df_from_file).all().all()
        assert not any(
            issubclass(warning.category, SettingWithCopyWarning)
            for warning in caught_warnings
        )


def test_preview_shows_selected_renamed_columns_and_reports_omitted_rows(capsys):
    df = pd.DataFrame({"peptide": [f"peptide_{i:02d}" for i in range(23)],
                       "annotation": list(range(23)), "hidden": "not selected"})
    args = arg_parser.parse_args([
        "--subset-output-columns", "annotation", "peptide",
        "--rename-output-column", "annotation", "source",
    ])
    write_outputs(df, args)
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[0].split() == ["source", "peptide"]
    assert len(lines) == 21  # one header plus twenty rows
    assert "peptide_19" in captured.out
    assert "peptide_20" not in captured.out
    assert "not selected" not in captured.out
    assert "20 of 23 prediction rows (3 omitted)" in captured.err
    assert "--output-csv -" in captured.err


def test_default_preview_retains_a_renamed_default_column(capsys):
    df = pd.DataFrame({"peptide": ["SIINFEKL"], "value": [10.0],
                       "value_unit": ["nM"], "internal_note": ["not displayed"]})
    write_outputs(df, arg_parser.parse_args(["--rename-output-column", "value", "ic50"]))
    output = capsys.readouterr().out
    assert output.splitlines()[0].split() == ["peptide", "ic50", "value_unit"]
    assert "nM" in output
    assert "not displayed" not in output


@pytest.mark.parametrize("flag,extension", [("--output-csv", "csv"), ("--output-html", "html")])
def test_explicit_file_output_does_not_also_print_a_preview(tmp_path, capsys, flag, extension):
    path = tmp_path / f"results.{extension}"
    write_outputs(pd.DataFrame({"peptide": ["SIINFEKL"]}),
                  arg_parser.parse_args([flag, str(path)]))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Saving" in captured.err
    assert "SIINFEKL" in path.read_text()


def test_csv_stdout_stays_clean_with_html_and_legacy_diagnostic_prints(tmp_path, capsys):
    html = tmp_path / "results.html"
    write_outputs(pd.DataFrame({"peptide": ["SIINFEKL"]}), arg_parser.parse_args([
        "--output-csv", "-", "--output-html", str(html), "--print-columns",
    ]), print_df_before_filtering=True, print_df_after_filtering=True)
    captured = capsys.readouterr()
    assert captured.out == "#,peptide\n0,SIINFEKL\n"
    assert "Columns:" in captured.err and "Saving" in captured.err
    assert "SIINFEKL" in html.read_text()
