"""Tests for CLI argcomplete tab completion, import-light startup, and help styling (#588)."""

import io
import os
from pathlib import Path
import pty
import re
import select
import subprocess
import sys
import textwrap

import pytest

from topiary.cli.args import arg_parser, create_arg_parser
from topiary.cli.help import (
    ANSI_BOLD,
    ANSI_BOLD_CYAN,
    ANSI_CYAN,
    ANSI_DIM,
    ANSI_RESET,
    ANSI_YELLOW,
    HelpArgumentParser,
    HelpFormatter,
    classify_status,
    color_enabled,
    style_help,
    style_status,
    style_table_header,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

HEAVY_MODULES = (
    "pandas",
    "numpy",
    "pyarrow",
    "torch",
    "mhcflurry",
    "pyensembl",
    "pysam",
    "scipy",
    "mhcgnomes",
    "isovar",
    "pirlygenes",
    "weirdo",
    "gtfparse",
)


class DummyStream(io.StringIO):
    """In-memory text stream with configurable TTY reporting."""

    def __init__(self, *, is_tty):
        super().__init__()
        self._is_tty = is_tty

    def isatty(self):
        return self._is_tty


def _run_argcomplete(tmp_path, comp_line, comp_point=None):
    """Execute ``topiary.cli.script.main`` under argcomplete's subprocess protocol."""
    out_file = tmp_path / "completions.txt"
    if comp_point is None:
        comp_point = len(comp_line)
    env = os.environ.copy()
    env.update(
        {
            "_ARGCOMPLETE": "1",
            "_ARGCOMPLETE_IFS": "\x0b",
            "_ARGCOMPLETE_STDOUT_FILENAME": str(out_file),
            "COMP_LINE": comp_line,
            "COMP_POINT": str(comp_point),
            "COMP_TYPE": "9",
        }
    )
    proc = subprocess.run(
        [sys.executable, "-c", "from topiary.cli.script import main; main()"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"argcomplete failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    if not out_file.exists():
        return []
    raw = out_file.read_text(encoding="utf-8")
    return [item for item in raw.split("\x0b") if item]


def _run_in_pty(cmd, env):
    """Run *cmd* attached to a pseudo-terminal and return decoded stdout/stderr."""
    master_fd, slave_fd = pty.openpty()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        os.close(slave_fd)
        slave_fd = -1
        chunks = []
        while True:
            ready, _, _ = select.select([master_fd], [], [], 5.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            try:
                data = os.read(master_fd, 4096)
            except OSError:
                break
            if not data:
                break
            chunks.append(data)
        proc.wait(timeout=15)
        return proc.returncode, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        if slave_fd >= 0:
            os.close(slave_fd)
        os.close(master_fd)


# ---------------------------------------------------------------------------
# Packaging & marker requirements
# ---------------------------------------------------------------------------


def test_python_argcomplete_ok_marker_within_first_1024_bytes():
    script_path = REPO_ROOT / "topiary" / "cli" / "script.py"
    head = script_path.read_bytes()[:1024]
    assert b"PYTHON_ARGCOMPLETE_OK" in head


def test_requirements_declares_argcomplete_floor_3():
    req_text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in req_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert any(re.match(r"^argcomplete\s*>=\s*3\b", line) for line in lines), (
        f"Expected argcomplete>=3 in requirements.txt, got: {lines}"
    )


# ---------------------------------------------------------------------------
# Subprocess argcomplete tab completion
# ---------------------------------------------------------------------------


def test_argcomplete_completes_mhc_flags(tmp_path):
    completions = _run_argcomplete(tmp_path, "topiary --mhc-")
    assert "--mhc-predictor" in completions
    assert "--mhc-alleles" in completions
    assert "--mhc-cache-file" in completions
    assert "--mhc-cache-format" in completions


def test_argcomplete_completes_output_and_filter_flags(tmp_path):
    output_completions = _run_argcomplete(tmp_path, "topiary --output-")
    assert "--output-csv" in output_completions
    assert "--output-html" in output_completions
    assert "--output-provenance" in output_completions

    filter_completions = _run_argcomplete(tmp_path, "topiary --filter-")
    assert "--filter-by" in filter_completions
    assert "--filter-logic" in filter_completions


@pytest.mark.parametrize(
    "comp_line,expected",
    [
        ("topiary --filter-logic ", {"any", "all"}),
        ("topiary --exclude-mode ", {"substring", "exact"}),
        ("topiary --sort-direction ", {"auto", "asc", "desc"}),
        ("topiary --protein-x-policy ", {"skip", "windows", "allow"}),
        (
            "topiary --mhc-cache-format ",
            {"topiary_output", "mhcflurry", "tsv", "netmhcpan"},
        ),
    ],
)
def test_argcomplete_completes_flag_choices(tmp_path, comp_line, expected):
    completions = set(_run_argcomplete(tmp_path, comp_line))
    assert expected <= completions, f"Missing {expected - completions} from {completions}"


# ---------------------------------------------------------------------------
# Import-light startup & lazy topiary/__init__.py
# ---------------------------------------------------------------------------


def test_subprocess_parser_and_script_import_no_heavy_modules():
    code = textwrap.dedent(
        f"""
        import sys
        import argcomplete
        import topiary.cli.script
        from topiary.cli.args import create_arg_parser

        parser = create_arg_parser()
        assert parser.prog == "topiary"

        heavy = {HEAVY_MODULES!r}
        loaded = [
            name for name in heavy
            if name in sys.modules or any(k.startswith(name + ".") for k in sys.modules)
        ]
        if loaded:
            raise SystemExit(f"Heavy modules imported during CLI setup: {{loaded}}")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_subprocess_import_topiary_is_lazy():
    code = textwrap.dedent(
        f"""
        import sys
        import topiary

        assert isinstance(topiary.__version__, str) and topiary.__version__
        assert "TopiaryPredictor" in topiary.__all__
        assert "TopiaryPredictor" in dir(topiary)
        assert set(topiary.__all__) == set(topiary.LAZY_EXPORTS)

        try:
            getattr(topiary, "nonexistent_topiary_symbol_xyz")
        except AttributeError:
            pass
        else:
            raise SystemExit("Expected AttributeError for unknown attribute")

        heavy = {HEAVY_MODULES!r} + ("varcode", "mhctools")
        loaded = [
            name for name in heavy
            if name in sys.modules or any(k.startswith(name + ".") for k in sys.modules)
        ]
        if loaded:
            raise SystemExit(f"import topiary eagerly imported: {{loaded}}")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_subprocess_help_version_and_print_columns_stay_import_light():
    for argv in (["--help"], ["--version"], ["--print-columns"]):
        code = textwrap.dedent(
            f"""
            import sys
            from topiary.cli.script import main

            try:
                main({argv!r})
            except SystemExit as exc:
                if exc.code not in (0, None):
                    raise

            heavy = {HEAVY_MODULES!r}
            loaded = [
                name for name in heavy
                if name in sys.modules or any(k.startswith(name + ".") for k in sys.modules)
            ]
            if loaded:
                raise SystemExit(f"Command {argv!r} imported heavy modules: {{loaded}}")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout


# ---------------------------------------------------------------------------
# Help styling & table helpers (topiary/cli/help.py)
# ---------------------------------------------------------------------------


def test_color_enabled_respects_tty_no_color_and_term_dumb(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    tty = DummyStream(is_tty=True)
    plain = DummyStream(is_tty=False)

    assert color_enabled(tty) is True
    assert color_enabled(plain) is False
    assert color_enabled(None) is False

    monkeypatch.setenv("NO_COLOR", "")
    assert color_enabled(tty) is True

    monkeypatch.setenv("NO_COLOR", "1")
    assert color_enabled(tty) is False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert color_enabled(tty) is False


def test_style_help_uses_single_cyan_hue_on_tty_and_plain_otherwise(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    sample = (
        "usage: topiary [-h] [--vcf VCF]\n"
        "\n"
        "Direct Inputs:\n"
        "  --vcf VCF  Input VCF file (default: None)\n"
        "  -h, --help show this help message\n"
    )
    tty = DummyStream(is_tty=True)
    plain = DummyStream(is_tty=False)

    assert style_help(sample, plain) == sample

    styled = style_help(sample, tty)
    assert f"{ANSI_BOLD}usage:{ANSI_RESET}" in styled
    assert f"{ANSI_BOLD_CYAN}Direct Inputs:{ANSI_RESET}\n" in styled
    assert f"{ANSI_CYAN}--vcf{ANSI_RESET}" in styled
    assert f"{ANSI_CYAN}-h{ANSI_RESET}" in styled
    assert f"{ANSI_CYAN}--help{ANSI_RESET}" in styled
    # Metavars, descriptions, and defaults stay plain.
    assert f"{ANSI_CYAN}VCF{ANSI_RESET}" not in styled
    assert "Input VCF file (default: None)" in styled
    # No bright ANSI colors (90-97) and no second hue.
    assert not re.search(r"\033\[(?:9[0-7]|3[12457]|4[0-7])m", styled)

    monkeypatch.setenv("NO_COLOR", "1")
    assert style_help(sample, tty) == sample

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert style_help(sample, tty) == sample


def test_style_table_header_rules(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    tty = DummyStream(is_tty=True)
    plain = DummyStream(is_tty=False)

    assert style_table_header("", tty) == ""
    assert style_table_header("peptide allele\nSIINFEKL A0201\n", plain) == (
        "peptide allele\nSIINFEKL A0201\n"
    )
    assert style_table_header("peptide allele\nSIINFEKL A0201\n", tty) == (
        f"{ANSI_BOLD_CYAN}peptide allele{ANSI_RESET}\nSIINFEKL A0201\n"
    )
    assert style_table_header("peptide allele", tty) == (
        f"{ANSI_BOLD_CYAN}peptide allele{ANSI_RESET}"
    )

    monkeypatch.setenv("NO_COLOR", "1")
    assert style_table_header("peptide allele\nSIINFEKL A0201\n", tty) == (
        "peptide allele\nSIINFEKL A0201\n"
    )


def test_classify_and_style_status_rules(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    tty = DummyStream(is_tty=True)
    plain = DummyStream(is_tty=False)

    for absent in ("-", "n/a", "absent", "None", "not applicable"):
        assert classify_status(absent) == "dim"
        assert style_status(absent, tty) == ANSI_DIM + absent + ANSI_RESET
        assert style_status(absent, plain) == absent

    for warning in ("missing", "stale", "error", "warning", "local dna.fa, missing"):
        assert classify_status(warning) == "warning"
        assert style_status(warning, tty) == ANSI_YELLOW + warning + ANSI_RESET
        assert style_status(warning, plain) == warning

    for normal in ("OK", "complete", "ready", "verified"):
        assert classify_status(normal) == "plain"
        assert style_status(normal, tty) == normal
        assert "\033[" not in style_status(normal, tty)

    assert style_status("custom", tty, level="dim") == ANSI_DIM + "custom" + ANSI_RESET
    assert style_status("custom", tty, level="warning") == ANSI_YELLOW + "custom" + ANSI_RESET
    assert style_status("missing", tty, level="plain") == "missing"

    with pytest.raises(ValueError, match="Unknown status level"):
        style_status("custom", tty, level="green")

    monkeypatch.setenv("NO_COLOR", "1")
    assert style_status("missing", tty) == "missing"
    assert style_status("-", tty) == "-"


def test_help_argument_parser_keeps_format_help_plain_and_styles_tty_print(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    assert isinstance(arg_parser, HelpArgumentParser)
    assert arg_parser.formatter_class is HelpFormatter

    # format_help() is always plain text (for docs generation and tests).
    plain_help = arg_parser.format_help()
    assert "\033[" not in plain_help
    assert "usage: topiary" in plain_help
    assert "--mhc-predictor" in plain_help

    # print_help() styles only on a color-capable TTY.
    tty_buf = DummyStream(is_tty=True)
    arg_parser.print_help(file=tty_buf)
    tty_out = tty_buf.getvalue()
    assert f"{ANSI_BOLD}usage:{ANSI_RESET}" in tty_out
    assert f"{ANSI_CYAN}--mhc-predictor{ANSI_RESET}" in tty_out
    assert f"{ANSI_BOLD_CYAN}Direct Inputs:{ANSI_RESET}" in tty_out

    non_tty_buf = DummyStream(is_tty=False)
    arg_parser.print_help(file=non_tty_buf)
    assert "\033[" not in non_tty_buf.getvalue()

    monkeypatch.setenv("NO_COLOR", "1")
    no_color_buf = DummyStream(is_tty=True)
    arg_parser.print_help(file=no_color_buf)
    assert "\033[" not in no_color_buf.getvalue()

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")
    dumb_buf = DummyStream(is_tty=True)
    arg_parser.print_help(file=dumb_buf)
    assert "\033[" not in dumb_buf.getvalue()


def test_help_argument_parser_disables_native_color_on_python_314(monkeypatch):
    import argparse

    seen_kwargs = {}

    def fake_init(self, *args, **kwargs):
        seen_kwargs.update(kwargs)

    monkeypatch.setattr(sys, "version_info", (3, 14, 0, "final", 0))
    monkeypatch.setattr(argparse.ArgumentParser, "__init__", fake_init)

    HelpArgumentParser(prog="topiary")
    assert seen_kwargs["color"] is False
    assert seen_kwargs["formatter_class"] is HelpFormatter


def test_help_formatter_strips_rst_literals():
    parser = HelpArgumentParser(prog="topiary")
    parser.add_argument("--demo", help="Uses ``foo`` and ``bar`` literals.")
    rendered = parser.format_help()
    assert "``" not in rendered
    assert "Uses foo and bar literals." in rendered


def test_pty_help_output_respects_tty_no_color_and_term_dumb():
    cmd = [sys.executable, "-c", "from topiary.cli.script import main; main(['--help'])"]

    # Piped stdout -> plain text.
    env_plain = os.environ.copy()
    env_plain.pop("NO_COLOR", None)
    env_plain["TERM"] = "xterm-256color"
    piped = subprocess.run(cmd, env=env_plain, capture_output=True, text=True, check=False)
    assert piped.returncode == 0
    assert "\033[" not in piped.stdout

    # Real TTY -> subtle cyan styling.
    rc, tty_out = _run_in_pty(cmd, env_plain)
    assert rc == 0
    assert f"{ANSI_BOLD}usage:{ANSI_RESET}" in tty_out
    assert f"{ANSI_CYAN}--mhc-predictor{ANSI_RESET}" in tty_out
    assert f"{ANSI_BOLD_CYAN}Direct Inputs:{ANSI_RESET}" in tty_out

    # Real TTY + NO_COLOR -> plain text.
    env_no_color = env_plain.copy()
    env_no_color["NO_COLOR"] = "1"
    rc, no_color_out = _run_in_pty(cmd, env_no_color)
    assert rc == 0
    assert "\033[" not in no_color_out

    # Real TTY + TERM=dumb -> plain text.
    env_dumb = env_plain.copy()
    env_dumb["TERM"] = "dumb"
    rc, dumb_out = _run_in_pty(cmd, env_dumb)
    assert rc == 0
    assert "\033[" not in dumb_out


def test_create_arg_parser_returns_help_argument_parser():
    parser = create_arg_parser(
        expression=False,
        mhc=False,
        variants=False,
        protein_changes=False,
    )
    assert isinstance(parser, HelpArgumentParser)
    assert "\033[" not in parser.format_help()


def test_topiary_init_type_checking_aliases_match_lazy_exports():
    """Keep static TYPE_CHECKING imports in exact 1:1 parity with LAZY_EXPORTS."""
    import ast
    import topiary

    init_path = REPO_ROOT / "topiary" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))

    type_checking_map = {}
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    mod = "." * stmt.level + (stmt.module or "")
                    for alias in stmt.names:
                        asname = alias.asname or alias.name
                        type_checking_map[asname] = (mod, alias.name)

    assert type_checking_map == topiary.LAZY_EXPORTS
    assert set(topiary.__all__) == set(topiary.LAZY_EXPORTS)

