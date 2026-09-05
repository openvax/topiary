"""Executable contracts for the lint and test shell gates."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$PYTHON_INVOCATION_LOG"
if [[ "${1:-}" == "-c" && "${2:-}" == "import xdist" ]]; then
    exit "${XDIST_IMPORT_STATUS:-0}"
fi
exit 0
""")
    path.chmod(0o755)
    return path


def _path_traps(tmp_path):
    trap_dir = tmp_path / "path-traps"
    trap_dir.mkdir()
    for name in ("python", "python3", "pytest", "ruff"):
        path = trap_dir / name
        path.write_text("""#!/bin/sh
printf '%s\\n' "$0 $*" >> "$PATH_TRAP_LOG"
exit 97
""")
        path.chmod(0o755)
    return trap_dir


def _run_gate(tmp_path, script_name, *args, env_updates=None):
    script = tmp_path / script_name
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(SOURCE_ROOT / script_name, script)
    shutil.copy2(
        SOURCE_ROOT / "scripts" / "resolve_python.sh",
        scripts / "resolve_python.sh",
    )
    env = os.environ.copy()
    env.update({
        "PATH": f"{_path_traps(tmp_path)}{os.pathsep}{env['PATH']}",
        "PATH_TRAP_LOG": str(tmp_path / "path-trap.log"),
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYTHON_INVOCATION_LOG": str(tmp_path / "python-invocations.log"),
        "TEST_SH_MAX": "1",
        "TEST_SH_MIN": "1",
    })
    if env_updates:
        for name, value in env_updates.items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
    result = subprocess.run(
        ["bash", str(script), *args],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    invocation_log = tmp_path / "python-invocations.log"
    invocations = (
        invocation_log.read_text().splitlines()
        if invocation_log.exists()
        else []
    )
    return result, invocations, tmp_path / "path-trap.log"


def test_lint_uses_selected_python_module_not_path_ruff(tmp_path):
    result, invocations, trap_log = _run_gate(tmp_path, "lint.sh")

    assert result.returncode == 0, result.stderr
    assert invocations == ["-m ruff check topiary/ tests/"]
    assert not trap_log.exists()


@pytest.mark.parametrize("script_name", ("lint.sh", "test.sh"))
@pytest.mark.parametrize("environment", ("active-venv", "repo-venv"))
def test_gate_resolves_virtualenv_python_when_python_is_unset(
        tmp_path, script_name, environment):
    if environment == "active-venv":
        venv = tmp_path / "active venv"
        env_updates = {"PYTHON": None, "VIRTUAL_ENV": str(venv)}
    else:
        venv = tmp_path / ".venv"
        env_updates = {"PYTHON": None, "VIRTUAL_ENV": None}
    selected = _fake_python(venv / "bin" / "python")

    result, invocations, trap_log = _run_gate(
        tmp_path, script_name, env_updates=env_updates,
    )

    assert result.returncode == 0, result.stderr
    expected = ["-m ruff check topiary/ tests/"]
    if script_name == "test.sh":
        expected = [
            "-c import xdist",
            "-m pytest -n 1 --cov=topiary/ --cov-report=term-missing tests",
        ]
        assert f"python={selected}" in result.stderr
    assert invocations == expected
    assert not trap_log.exists()


@pytest.mark.parametrize(
    "xdist_status, expected_pytest",
    (
        ("0", "-m pytest -n 1 --cov=topiary/ --cov-report=term-missing tests selected-test"),
        ("1", "-m pytest --cov=topiary/ --cov-report=term-missing tests selected-test"),
    ),
)
def test_test_gate_uses_same_python_for_xdist_probe_and_pytest(
        tmp_path, xdist_status, expected_pytest):
    result, invocations, trap_log = _run_gate(
        tmp_path,
        "test.sh",
        "selected-test",
        env_updates={"XDIST_IMPORT_STATUS": xdist_status},
    )

    assert result.returncode == 0, result.stderr
    assert invocations == ["-c import xdist", expected_pytest]
    assert not trap_log.exists()


@pytest.mark.parametrize("script_name", ("lint.sh", "test.sh"))
def test_gate_rejects_invalid_explicit_python_without_path_fallback(
        tmp_path, script_name):
    invalid = tmp_path / "missing-python"

    result, invocations, trap_log = _run_gate(
        tmp_path,
        script_name,
        env_updates={"PYTHON": str(invalid)},
    )

    assert result.returncode == 1
    assert f"Python interpreter not found or not executable: {invalid}" in result.stderr
    assert invocations == []
    assert not trap_log.exists()
