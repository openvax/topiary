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


def test_cli_variant_helpers_delegate_to_varcode():
    _run_import_check(
        """
        import topiary.cli.args as cli_args
        import varcode.cli

        if cli_args.add_variant_args is not varcode.cli.add_variant_args:
            raise SystemExit("Topiary is not delegating variant CLI args to Varcode")
        if cli_args.variant_collection_from_args is not varcode.cli.variant_collection_from_args:
            raise SystemExit("Topiary is not delegating variant loading to Varcode")
        """
    )
