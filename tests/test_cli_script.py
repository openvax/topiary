import pathlib

import pytest

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
    # str(KeyError(...)) reprs its message with an extra quoted layer
    # ("'CachedPredictor: ...'" instead of "CachedPredictor: ..."); the
    # CLI error must read like the ValueError/OSError messages above it,
    # not like a KeyError repr.
    assert "\"'CachedPredictor" not in captured.err
    assert "Traceback" not in captured.err
    assert "Namespace(" not in captured.out
