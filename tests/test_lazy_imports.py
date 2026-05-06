import subprocess
import sys
import textwrap


def _run_import_check(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_import_topiary_does_not_import_varcode():
    _run_import_check(
        """
        import sys
        import topiary

        if any(k == "varcode" or k.startswith("varcode.") for k in sys.modules):
            raise SystemExit("import topiary imported varcode")
        """
    )


def test_cli_direct_parser_does_not_import_varcode():
    _run_import_check(
        """
        import sys
        from topiary.cli.args import arg_parser

        arg_parser.parse_args([
            "--mhc-predictor", "random",
            "--mhc-alleles", "A0201",
            "--peptide-csv", "peptides.csv",
        ])

        if any(k == "varcode" or k.startswith("varcode.") for k in sys.modules):
            raise SystemExit("direct CLI parser imported varcode")
        """
    )
