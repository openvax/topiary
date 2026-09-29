from topiary.cli.args import arg_parser
from topiary.cli.outputs import write_outputs
import tempfile
import warnings
import pandas as pd
import pytest

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
        df_from_file = pd.read_csv(f.name)

        df_expected = pd.DataFrame({"X": [1, 2, 3]})
        print(df_from_file)
        eq_(len(df_expected), len(df_from_file))
        assert (df_expected == df_from_file).all().all()
        assignment_warnings = tuple(
            getattr(pd.errors, name) for name in ("SettingWithCopyWarning", "ChainedAssignmentError")
            if hasattr(pd.errors, name)
        )
        assert not any(
            issubclass(warning.category, assignment_warnings)
            for warning in caught_warnings
        )
        pd.testing.assert_frame_equal(df, pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]}))


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


def test_csv_and_html_outputs_write_dict_and_list_cells_as_json(tmp_path):
    import html
    import json

    context = {"estimate_type": "ml_predicted", "unit": "nM", "analyte": None}
    df = pd.DataFrame({
        "peptide": ["SIINFEKL", "SIINFEKLL", "SIINFEKLLL"],
        "measurement_context": [context, None, ["a", 1]],
    })
    csv_path, html_path = tmp_path / "results.csv", tmp_path / "results.html"
    write_outputs(df, arg_parser.parse_args([
        "--output-csv", str(csv_path), "--output-html", str(html_path),
    ]))
    written = pd.read_csv(csv_path)
    assert json.loads(written.measurement_context.iloc[0]) == context
    assert pd.isna(written.measurement_context.iloc[1])
    assert json.loads(written.measurement_context.iloc[2]) == ["a", 1]
    page = html.unescape(html_path.read_text())
    assert '"analyte":null' in page and "'analyte': None" not in page
    # The caller's frame keeps its mappings.
    assert df.measurement_context.iloc[0] is context


def test_csv_stdout_stays_clean_with_html_and_legacy_diagnostic_prints(tmp_path, capsys):
    html = tmp_path / "results.html"
    write_outputs(pd.DataFrame({"peptide": ["SIINFEKL"]}), arg_parser.parse_args([
        "--output-csv", "-", "--output-html", str(html), "--print-columns",
    ]), print_df_before_filtering=True, print_df_after_filtering=True)
    captured = capsys.readouterr()
    assert captured.out == "peptide\nSIINFEKL\n"
    assert "Columns:" in captured.err and "Saving" in captured.err
    assert "SIINFEKL" in html.read_text()


def test_renaming_a_column_the_subset_removed_says_so():
    df = pd.DataFrame({"peptide": ["SIINFEKL"], "value": [1.0]})
    args = arg_parser.parse_args([
        "--subset-output-columns", "peptide",
        "--rename-output-column", "value", "ic50",
    ])
    with pytest.raises(ValueError, match="--subset-output-columns removed it"):
        write_outputs(df, args)


def test_renaming_an_absent_column_is_an_error():
    df = pd.DataFrame({"peptide": ["SIINFEKL"]})
    args = arg_parser.parse_args(["--rename-output-column", "valu", "ic50"])
    with pytest.raises(ValueError, match="cannot rename 'valu': no such column"):
        write_outputs(df, args)


# Output files (#326): no index column, six significant digits, a real HTML
# page, and a row order a reader can follow.


def _scan_frame():
    """Two source sequences, windows deliberately out of position order."""
    return pd.DataFrame({
        "source_sequence_name": ["ova", "ova", "ova", "spike", "spike"],
        "peptide": ["AAAAAKAAA", "ASIINFEKL", "EKLAAAAAK", "LVSSQCVNL", "FVFLVLLPL"],
        "peptide_offset": [16, 1, 13, 4, 0],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


def test_csv_has_no_index_column(tmp_path):
    out = tmp_path / "results.csv"
    write_outputs(pd.DataFrame({"peptide": ["SIINFEKL"]}),
                  arg_parser.parse_args(["--output-csv", str(out)]))
    assert out.read_text().splitlines()[0] == "peptide"
    assert list(pd.read_csv(out).columns) == ["peptide"]


def test_csv_floats_keep_six_significant_digits_and_float_dtype(tmp_path):
    out = tmp_path / "results.csv"
    df = pd.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFTL", "NLVPMVATV"],
        # A long repr, a value whose noise is past the sixth digit, and a
        # whole number that must not read back as an int.
        "value": [11927.161249112096, 6.296000000000002, 2.0],
        "percentile_rank": [0.0011141304347574987, float("nan"), 2.0],
    })
    write_outputs(df, arg_parser.parse_args(["--output-csv", str(out)]))
    text = out.read_text()
    assert "11927.2" in text and "11927.161249112096" not in text
    assert "6.296," in text and "0.00111413" in text
    reloaded = pd.read_csv(out)
    assert reloaded.value.dtype == "float64"
    assert reloaded.percentile_rank.dtype == "float64"
    assert reloaded.percentile_rank.isna().sum() == 1


def test_html_is_a_standalone_page_with_consistent_missing_cells(tmp_path):
    out = tmp_path / "results.html"
    df = pd.DataFrame({
        "peptide": ["SIINFEKL", "GILGFVFTL"],
        "sample_name": ["", None],
        "measurement_context": [None, "{}"],
        "wt_peptide": [float("nan"), "LVVVGAGGV"],
        "note": ["None", "real text"],
    })
    write_outputs(df, arg_parser.parse_args(["--output-html", str(out)]))
    page = out.read_text()
    assert page.startswith("<!doctype html>")
    for required in ('<html lang="en">', '<meta charset="utf-8">',
                     "<title>results</title>", "</body>", "</html>"):
        assert required in page
    # Every missing value renders the same way, whatever its Python type,
    # and the column holding the literal text "None" keeps it.
    assert page.count("<td>None</td>") == 1
    assert page.count("<td></td>") == 4
    assert "NaN" not in page


def test_rows_of_a_scan_run_in_position_order(tmp_path):
    out = tmp_path / "scan.csv"
    write_outputs(_scan_frame(), arg_parser.parse_args(["--output-csv", str(out)]))
    written = pd.read_csv(out)
    # Source sequences keep the order they first appeared; within each one
    # the windows run in position order.
    assert written.source_sequence_name.tolist() == ["ova"] * 3 + ["spike"] * 2
    assert written.peptide_offset.tolist() == [1, 13, 16, 0, 4]


def test_a_peptide_list_keeps_its_input_order(tmp_path):
    out = tmp_path / "peptides.csv"
    peptides = ["SIINFEKL", "GILGFVFTL", "NLVPMVATV"]
    df = pd.DataFrame({
        "source_sequence_name": peptides, "peptide": peptides,
        "peptide_offset": [0, 0, 0],
    })
    write_outputs(df, arg_parser.parse_args(["--output-csv", str(out)]))
    assert pd.read_csv(out).peptide.tolist() == peptides


def test_output_row_numbers_a_sorted_result(tmp_path):
    out = tmp_path / "sorted.csv"
    args = arg_parser.parse_args(["--sort-by", "value", "--output-csv", str(out)])
    write_outputs(_scan_frame(), args)
    written = pd.read_csv(out)
    assert list(written.columns)[0] == "output_row"
    assert written.output_row.tolist() == [1, 2, 3, 4, 5]
    # The ranking decided the order, so the scan reordering stays out of it.
    assert written.peptide_offset.tolist() == [16, 1, 13, 4, 0]


def test_an_unsorted_result_has_no_output_row_column(tmp_path):
    out = tmp_path / "unsorted.csv"
    write_outputs(_scan_frame(), arg_parser.parse_args(["--output-csv", str(out)]))
    assert "output_row" not in pd.read_csv(out).columns


def test_output_row_can_be_requested_or_dropped_by_column_selection(tmp_path):
    out = tmp_path / "subset.csv"
    args = arg_parser.parse_args([
        "--sort-by", "value", "--output-csv", str(out),
        "--subset-output-columns", "output_row", "peptide",
    ])
    write_outputs(_scan_frame(), args)
    assert list(pd.read_csv(out).columns) == ["output_row", "peptide"]

    args = arg_parser.parse_args([
        "--sort-by", "value", "--output-csv", str(out),
        "--subset-output-columns", "peptide",
    ])
    write_outputs(_scan_frame(), args)
    assert list(pd.read_csv(out).columns) == ["peptide"]


def test_the_callers_frame_is_never_reordered_or_widened(tmp_path):
    df = _scan_frame()
    before = df.copy(deep=True)
    write_outputs(df, arg_parser.parse_args([
        "--sort-by", "value", "--output-csv", str(tmp_path / "out.csv")]))
    pd.testing.assert_frame_equal(df, before)
