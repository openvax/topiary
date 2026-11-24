import pathlib

import pytest
from mhcgnomes.errors import ParseError as AlleleParseError

from topiary import AlleleError
import topiary.cli.script as cli_script
from topiary.cli.script import main

_FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "netmhc_fixtures"
_HAS_FIXTURES = _FIXTURE_DIR.exists()


def _error_line(stderr):
    """The ``topiary: error:`` line, or a readable failure if absent.

    A bare ``next()`` raises ``StopIteration`` when the handler prints
    no error line at all -- exactly the regression worth reporting --
    and swallows the captured stderr that would explain it.
    """
    for line in stderr.splitlines():
        if line.startswith("topiary: error:"):
            return line
    raise AssertionError(
        f"no 'topiary: error:' line in stderr:\n{stderr}"
    )


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


@pytest.mark.skipif(not _HAS_FIXTURES, reason="NetMHC fixtures missing")
def test_main_reports_cached_predictor_miss_as_a_clean_cli_error(
    tmp_path, capsys,
):
    """A CachedPredictor coverage gap reaches the user as a clean CLI
    error, not a Python traceback (#296, #302, #304).

    ``CachedPredictor`` raises ``KeyError`` -- not ``ValueError`` -- for
    a peptide the cache doesn't cover and no fallback can resolve. Before
    this fix, main()'s top-level handler only caught
    ``(OSError, ValueError)``, so this specific failure was the one
    ``CachedPredictor`` error the CLI didn't give a clean message for.
    """
    fixture = _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
    # The fixture's cache covers SLLQHLIGL only; ask about a peptide it
    # was never predicted for, with no fallback configured.
    peptide_csv = tmp_path / "peptides.csv"
    peptide_csv.write_text("peptide\nGILGFVFTL\n")

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--peptide-csv", str(peptide_csv),
            "--mhc-cache-file", str(fixture),
            "--mhc-cache-format", "netmhcpan",
        ])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "usage: topiary" not in captured.err
    assert "topiary: error:" in captured.err
    assert "GILGFVFTL" in captured.err
    assert "no fallback set" in captured.err
    # Asserts the whole rendered line, not a quote spelling. Without
    # CachedPredictorCoverageError.__str__ this arrives wrapped in
    # quotes, and which quote character repr picks depends on the
    # message -- double, here, because the text contains single ones.
    # The earlier check looked for one spelling and so could never
    # fail. The cost of pinning the line exactly is that a reworded
    # coverage message breaks this test too; that is deliberate, since
    # the message is the contract being tested.
    error_line = _error_line(captured.err)
    assert error_line == "topiary: error: " + (
        "CachedPredictor: 1 peptide(s) missed and no fallback set.  "
        "Missed peptides: ['GILGFVFTL']."
    )
    assert "Traceback" not in captured.err
    assert "Namespace(" not in captured.out


def test_main_does_not_mask_an_unrelated_keyerror(monkeypatch):
    """The CLI catches CachedPredictorCoverageError specifically, not
    bare KeyError, so a genuine programming bug elsewhere in the same
    call graph -- an ordinary dict lookup that happens to raise
    KeyError -- still surfaces as a real exception instead of being
    silently reported as a clean, traceback-free CLI error alongside
    CachedPredictor's two intentional coverage-gap failures."""

    def _boom(args):
        raise KeyError("ENSG00000141510")

    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", _boom)

    with pytest.raises(KeyError, match="ENSG00000141510"):
        main(["--peptide-csv", "unused.csv"])


def test_main_reports_a_missing_input_file_readably(monkeypatch, capsys):
    """An OSError's message, not its errno.

    ``OSError.args`` is ``(errno, strerror)``, so unwrapping ``args[0]``
    turns a missing input file into the bare integer ``2``. The handler
    uses ``str(e)`` for every type it catches, which is correct for an
    OSError and, since ``CachedPredictorCoverageError`` formats itself,
    for that one too.

    Raised through a patched ``predict_epitopes_from_args`` rather than a
    real missing file: the thing under test is how the handler renders
    an OSError, and going through the real pipeline would make this
    depend on the NetMHC fixtures (the cache loads before the peptide
    CSV is read, so without them the error names the fixture, not this
    path, and the assertion below fails instead of skipping).
    """
    missing = "/nonexistent/definitely-missing.csv"

    def _raise_missing_file(args):
        raise FileNotFoundError(2, "No such file or directory", missing)

    monkeypatch.setattr(
        cli_script, "predict_epitopes_from_args", _raise_missing_file,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--peptide-csv", missing])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    error_line = _error_line(captured.err)
    assert "No such file or directory" in error_line
    assert missing in error_line
    assert "Traceback" not in captured.err


@pytest.mark.skipif(not _HAS_FIXTURES, reason="NetMHC fixtures missing")
def test_a_real_missing_peptide_csv_reaches_the_handler(tmp_path, capsys):
    """The seam the hermetic test above cannot cover.

    That one patches predict_epitopes_from_args, so it pins how the
    handler renders an OSError but not that a real missing file still
    produces one inside the try block. If the peptide-CSV loader began
    wrapping FileNotFoundError in another type, or read the file outside
    the handler, the hermetic test would still pass.
    """
    missing = tmp_path / "definitely-missing.csv"

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--peptide-csv", str(missing),
            "--mhc-cache-file", str(
                _FIXTURE_DIR / "netmhcpan_41_SLLQHLIGL_A0201.out"
            ),
            "--mhc-cache-format", "netmhcpan",
        ])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    error_line = _error_line(captured.err)
    assert "No such file or directory" in error_line
    assert str(missing) in error_line
    assert "Traceback" not in captured.err


def test_main_reports_predictor_setup_failure_readably(
    tmp_path, monkeypatch, capsys,
):
    """Predictor setup advice, through the real derivation path.

    ``--mhc-cache-format mhcflurry`` without an explicit predictor
    version makes ``from_mhcflurry`` derive one via
    ``mhcflurry_composite_version``, which raises
    ``PredictorSetupError`` when no model release is fetched. Its
    message names the command to run, and that was reaching CLI users as
    a stack trace.

    Driven by patching mhcflurry's own ``get_current_release`` rather
    than by raising a copy of the message from a stub, so a rewording of
    the real message, or ``cached_predictor_from_args`` starting to wrap
    the error, would fail this test rather than pass it.
    """
    pytest.importorskip("mhcflurry")
    import mhcflurry.downloads

    monkeypatch.setattr(
        mhcflurry.downloads, "get_current_release", lambda: None,
    )

    cache_csv = tmp_path / "preds.csv"
    cache_csv.write_text(
        "peptide,allele,mhcflurry_affinity\nSIINFEKL,HLA-A*02:01,120.5\n"
    )
    peptide_csv = tmp_path / "peptides.csv"
    peptide_csv.write_text("peptide\nSIINFEKL\n")

    with pytest.raises(SystemExit) as exc_info:
        main([
            "--peptide-csv", str(peptide_csv),
            "--mhc-cache-file", str(cache_csv),
            "--mhc-cache-format", "mhcflurry",
        ])

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    error_line = _error_line(captured.err)
    assert "mhcflurry-downloads fetch" in error_line
    assert "Traceback" not in captured.err


def test_main_does_not_mask_a_not_implemented_error(monkeypatch):
    """NotImplementedError subclasses RuntimeError, and must not be caught.

    An abstract method left unimplemented is a bug, not something the
    user can act on. Catching RuntimeError wholesale reported it as a
    clean CLI error -- and a blank one, since a bare
    NotImplementedError() stringifies to "".
    """

    def _abstract(args):
        raise NotImplementedError()

    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", _abstract)

    with pytest.raises(NotImplementedError):
        main(["--peptide-csv", "unused.csv"])


def test_main_never_prints_a_blank_error(monkeypatch, capsys):
    """A caught exception with no message still names something.

    str() is "" for an argument-less exception, so the handler would
    otherwise print "topiary: error: " and nothing else.
    """

    def _empty(args):
        raise ValueError()

    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", _empty)

    with pytest.raises(SystemExit):
        main(["--peptide-csv", "unused.csv"])

    assert _error_line(capsys.readouterr().err) == "topiary: error: ValueError"


# Exit status (#310, #324): a malformed command line keeps argparse's usage
# block and status 2; anything found after parsing exits 1 with one line.

_REQUEST = ["--mhc-predictor", "random", "--mhc-alleles", "HLA-A*02:01"]


def _run(argv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    return exc_info.value.code, capsys.readouterr().err


@pytest.mark.parametrize("error,expected", [
    (ImportError("pirlygenes is required for cancer-testis antigen gene lists"),
     "pirlygenes is required"),
    (AlleleParseError("Could not parse 'HLA-Q99:99'"),
     "could not parse MHC allele: Could not parse 'HLA-Q99:99'"),
    (AlleleError("'A2' is a serotype (HLA-A2), not an allele", "A2"),
     "topiary: error: 'A2' is a serotype (HLA-A2), not an allele"),
    (FileNotFoundError(2, "No such file or directory", "absent.csv"),
     "No such file or directory"),
])
def test_runtime_failures_exit_1_with_one_line(monkeypatch, capsys, error, expected):
    def fail(args):
        raise error
    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", fail)
    code, err = _run([*_REQUEST, "--peptide-csv", "peptides.csv"], capsys)
    assert code == 1
    assert expected in _error_line(err)
    assert "usage:" not in err and "Traceback" not in err


def _must_not_predict(args):
    pytest.fail("predicted before checking the paths")


@pytest.mark.parametrize("flag", ["--output-csv", "--output-html"])
def test_an_output_in_a_missing_directory_fails_before_predicting(
    monkeypatch, capsys, tmp_path, flag,
):
    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", _must_not_predict)
    target = tmp_path / "absent" / "results"
    code, err = _run([*_REQUEST, "--peptide-csv", "p.csv", flag, str(target)], capsys)
    assert code == 1
    assert "does not exist" in _error_line(err) and "usage:" not in err


def test_a_missing_predictor_path_fails_before_predicting(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", _must_not_predict)
    program = tmp_path / "nonexistent" / "netMHCpan"
    code, err = _run([*_REQUEST, "--peptide-csv", "p.csv",
                      "--mhc-predictor-path", str(program)], capsys)
    assert code == 1
    assert "--mhc-predictor-path" in _error_line(err)


def test_predictor_path_accepts_a_directory_or_a_program_on_path(tmp_path):
    from argparse import Namespace
    for program in (str(tmp_path), "python3"):
        cli_script._check_paths_before_predicting(Namespace(
            output_csv=None, output_html=None, mhc_predictor_path=program))


def test_a_missing_input_fails_before_predictors_are_built(monkeypatch, capsys, tmp_path):
    import topiary.cli.args as cli_args

    def must_not_build(args):
        pytest.fail("built predictors before reading the input")
    monkeypatch.setattr(cli_args, "predictors_from_args", must_not_build)
    code, err = _run([*_REQUEST, "--peptide-csv", str(tmp_path / "absent.csv")], capsys)
    assert code == 1
    assert "No such file" in _error_line(err)


def test_a_multi_character_csv_separator_is_a_command_line_error(capsys):
    code, err = _run([*_REQUEST, "--peptide-csv", "p.csv", "--output-csv-sep", "||"], capsys)
    assert code == 2
    assert "usage: topiary" in err and "single character" in _error_line(err)


def test_an_unknown_output_column_fails_without_writing(monkeypatch, capsys, tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"peptide": ["SIINFEKL"], "value": [1.0]})
    monkeypatch.setattr(cli_script, "predict_epitopes_from_args", lambda args: frame)
    out = tmp_path / "results.csv"
    code, err = _run([*_REQUEST, "--peptide-csv", "p.csv", "--output-csv", str(out),
                      "--subset-output-columns", "peptide", "ic50"], capsys)
    assert code == 1
    assert "'ic50'" in _error_line(err) and not out.exists()


def test_a_write_failure_after_predicting_exits_1(monkeypatch, capsys):
    import pandas as pd

    def fail(df, args):
        raise PermissionError(13, "Permission denied", "results.csv")
    monkeypatch.setattr(cli_script, "predict_epitopes_from_args",
                        lambda args: pd.DataFrame({"peptide": ["SIINFEKL"]}))
    monkeypatch.setattr(cli_script, "write_outputs", fail)
    code, err = _run([*_REQUEST, "--peptide-csv", "p.csv"], capsys)
    assert code == 1
    assert "Permission denied" in _error_line(err) and "Traceback" not in err


# The help text and the DSL must agree about how to reach the wt scope
# (#322): the help used to name wt_affinity, which parses as "affinity from
# a predictor called wt" and then fails at evaluation.


def _predict_wt_help():
    from topiary.cli.args import arg_parser
    for action in arg_parser._actions:
        if "--predict-wt" in action.option_strings:
            return action.help
    raise AssertionError("--predict-wt is not in the parser")


@pytest.mark.parametrize("expression,expected", [
    ("wt.affinity", "wt.affinity.value"),
    ("wt.affinity.rank", "wt.affinity.rank"),
    ("wt.affinity.score", "wt.affinity.score"),
    ("wt.presentation", "wt.presentation.value"),
])
def test_help_names_wt_forms_the_dsl_actually_reads(expression, expected):
    from topiary import parse

    assert expression in _predict_wt_help()
    assert parse(expression).to_expr_string() == expected


def test_help_warns_off_the_underscore_form_that_means_something_else():
    from topiary import parse

    help_text = _predict_wt_help()
    # Still parseable, which is why it needs saying: it resolves to a
    # predictor named "wt" rather than the wildtype scope.
    assert parse("wt_affinity").to_expr_string() == "affinity['wt'].value"
    assert "wt_affinity" in help_text and "fails" in help_text
    assert "wt.affinity" in help_text


# CLI discoverability (#574) and provenance sidecar preflight (#570).


def test_version_flag_prints_installed_version_and_exits_zero(capsys):
    import topiary

    with pytest.raises(SystemExit) as caught:
        cli_script.main(["--version"])
    assert caught.value.code == 0
    captured = capsys.readouterr()
    assert f"topiary {topiary.__version__}" in captured.out


def test_print_columns_alone_lists_static_columns_without_input(capsys):
    result = cli_script.main(["--print-columns"])
    assert result == 0
    captured = capsys.readouterr()
    assert "Columns:" in captured.out
    assert "-- peptide" in captured.out
    assert "-- allele" in captured.out
    assert "-- variant" in captured.out
    assert "-- varcode_candidate_selection" not in captured.out


def test_print_columns_alone_honors_optional_column_flags(capsys):
    result = cli_script.main([
        "--print-columns", "--sort-by", "affinity", "--predict-wt", "--include-provenance",
    ])
    assert result == 0
    out = capsys.readouterr().out
    assert "-- output_row" in out
    assert "-- wt_value" in out
    assert "-- varcode_candidate_selection" in out


def test_mhcflurry_class2_allele_fails_early_and_echoes_raw_allele(monkeypatch, capsys, tmp_path):
    import topiary.cli.args as cli_args

    def must_not_build(args):
        pytest.fail("built mhcflurry before rejecting class II allele")

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    monkeypatch.setattr(cli_args, "predictors_from_args", must_not_build)
    code, err = _run([
        "--mhc-predictor", "mhcflurry",
        "--mhc-alleles", "HLA-DRB1*01:01",
        "--peptide-csv", str(peptides),
    ], capsys)
    assert code == 1
    msg = _error_line(err)
    assert "mhcflurry predicts class I only" in msg
    assert "HLA-DRB1*01:01" in msg
    assert "HLA-DRA1*01:01-DRB1*01:01" not in msg


def test_mhcflurry_class2_allele_in_file_fails_early(monkeypatch, capsys, tmp_path):
    import topiary.cli.args as cli_args

    def must_not_build(args):
        pytest.fail("built mhcflurry before rejecting class II allele from file")

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    alleles = tmp_path / "alleles.txt"
    alleles.write_text("# patient alleles\nHLA-A*02:01\nHLA-DRB1*04:01\n", encoding="utf-8")
    monkeypatch.setattr(cli_args, "predictors_from_args", must_not_build)
    code, err = _run([
        "--mhc-predictor", "mhcflurry-affinity",
        "--mhc-alleles-file", str(alleles),
        "--peptide-csv", str(peptides),
    ], capsys)
    assert code == 1
    msg = _error_line(err)
    assert "mhcflurry-affinity predicts class I only" in msg
    assert "HLA-DRB1*04:01" in msg


def test_unknown_subset_output_columns_fails_before_predictors_are_built(monkeypatch, capsys, tmp_path):
    import topiary.cli.args as cli_args

    def must_not_build(args):
        pytest.fail("built predictors before validating --subset-output-columns")

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    monkeypatch.setattr(cli_args, "predictors_from_args", must_not_build)
    code, err = _run([
        *_REQUEST, "--peptide-csv", str(peptides),
        "--subset-output-columns", "peptide", "ic50_typo",
    ], capsys)
    assert code == 1
    assert "'ic50_typo'" in _error_line(err)


def test_unknown_rename_output_column_fails_before_predictors_are_built(monkeypatch, capsys, tmp_path):
    import topiary.cli.args as cli_args

    def must_not_build(args):
        pytest.fail("built predictors before validating --rename-output-column")

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    monkeypatch.setattr(cli_args, "predictors_from_args", must_not_build)
    code, err = _run([
        *_REQUEST, "--peptide-csv", str(peptides),
        "--rename-output-column", "nonexistent_col", "renamed",
    ], capsys)
    assert code == 1
    assert "'nonexistent_col'" in _error_line(err)


def test_skip_errors_flag_disables_raise_on_error(monkeypatch, tmp_path):
    import pandas as pd
    import topiary.cli.args as cli_args

    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    captured = {}

    class SpyPredictor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict_from_named_peptides(self, seqs):
            return pd.DataFrame({"peptide": list(seqs.values())})

    monkeypatch.setattr(cli_args, "TopiaryPredictor", SpyPredictor)
    args = cli_args.arg_parser.parse_args([
        "--mhc-predictor", "random", "--mhc-alleles", "HLA-A*02:01",
        "--peptide-csv", str(peptides), "--skip-errors",
    ])
    cli_args.predict_epitopes_from_args(args)
    assert captured["raise_on_error"] is False


def test_output_provenance_preflight_checks_missing_parent_dir(capsys, tmp_path):
    peptides = tmp_path / "peptides.csv"
    peptides.write_text("peptide\nSIINFEKL\n", encoding="utf-8")
    code, err = _run([
        *_REQUEST, "--peptide-csv", str(peptides),
        "--output-provenance", str(tmp_path / "no_such_dir" / "prov.json"),
    ], capsys)
    assert code == 1
    assert "--output-provenance" in _error_line(err)
