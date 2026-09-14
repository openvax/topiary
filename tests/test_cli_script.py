import pathlib

import pytest

import topiary.cli.script as cli_script
from topiary.cli.script import main

_FIXTURE_DIR = pathlib.Path(__file__).parent / "data" / "netmhc_fixtures"
_HAS_FIXTURES = _FIXTURE_DIR.exists()


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

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "usage: topiary" in captured.err
    assert "topiary: error:" in captured.err
    assert "GILGFVFTL" in captured.err
    assert "no fallback set" in captured.err
    # KeyError's str() reprs its argument, so without
    # CachedPredictorCoverageError.__str__ this arrives wrapped in
    # quotes. Which quote character depends on the message: repr picks
    # double quotes when the text itself contains single ones, as both
    # coverage messages do. Asserting on one spelling missed that and
    # could never fail -- assert the message starts unquoted instead.
    error_line = next(
        line for line in captured.err.splitlines()
        if line.startswith("topiary: error:")
    )
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
    -- which 5.56.0 did for every exception, to strip the repr quoting
    ``CachedPredictorCoverageError`` needed -- turns a missing input
    file into the bare integer ``2``. ``str()`` is correct for an
    OSError, and is what the handler uses for every type now.

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

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    error_line = next(
        line for line in captured.err.splitlines()
        if line.startswith("topiary: error:")
    )
    assert "No such file or directory" in error_line
    assert missing in error_line
    # The 5.56.0 regression this pins: the errno alone.
    assert error_line != "topiary: error: 2"
    assert "Traceback" not in captured.err


def test_main_reports_predictor_setup_failure_readably(monkeypatch, capsys):
    """RuntimeError carries actionable setup advice, not a bug report.

    ``mhcflurry_composite_version`` raises RuntimeError when mhcflurry is
    installed but has no model release fetched, and the message says to
    run ``mhcflurry-downloads fetch``. That reached CLI users as a stack
    trace, which buries the one thing they need to do.
    """
    advice = (
        "mhcflurry has no active model release.  Run "
        "`mhcflurry-downloads fetch` or pass predictor_version "
        "explicitly."
    )

    def _raise_setup_error(args):
        raise RuntimeError(advice)

    monkeypatch.setattr(
        cli_script, "predict_epitopes_from_args", _raise_setup_error,
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["--peptide-csv", "unused.csv"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert f"topiary: error: {advice}" in captured.err
    assert "mhcflurry-downloads fetch" in captured.err
    assert "Traceback" not in captured.err
