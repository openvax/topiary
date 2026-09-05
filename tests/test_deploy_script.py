"""Regression tests for deploy.sh release safety checks."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _run(command, cwd, env):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _git(repo, *args):
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


def _fake_python(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$PYTHON_INVOCATION_LOG"
if [[ "${1:-}" == "-c" ]]; then
    printf '%s\\n' "5.52.5"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    exit 1
elif [[ "${1:-}" == "-m" && "${2:-}" == "build" ]]; then
    mkdir -p dist
    touch dist/topiary-5.52.5-py3-none-any.whl
elif [[ "${1:-}" == "-m" && "${2:-}" == "twine" ]]; then
    exit 0
else
    exec "$REAL_PYTHON" "$@"
fi
""")
    path.chmod(0o755)
    return path


def _release_repo(tmp_path):
    repo = tmp_path / "release-repo"
    (repo / "topiary").mkdir(parents=True)
    (repo / "scripts").mkdir()
    for name in ("deploy.sh",):
        shutil.copy2(SOURCE_ROOT / name, repo / name)
    shutil.copy2(
        SOURCE_ROOT / "scripts" / "resolve_python.sh",
        repo / "scripts" / "resolve_python.sh",
    )
    (repo / "topiary" / "__init__.py").write_text('__version__ = "5.52.5"\n')
    (repo / ".gitignore").write_text(".venv/\nbuild/\ndist/\n")
    for name in ("lint.sh", "test.sh"):
        (repo / name).write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$PYTHON\" >> \"$GATE_PYTHON_LOG\"\n"
        )
        (repo / name).chmod(0o755)

    _git(repo, "init")
    _git(repo, "checkout", "-b", "master")
    _git(repo, "config", "user.name", "Test Release")
    _git(repo, "config", "user.email", "release@example.com")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Release fixture")
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", str(origin)],
        capture_output=True,
        text=True,
        check=True,
    )
    _git(repo, "remote", "add", "origin", str(origin))
    return repo


def _deploy(repo, env_updates):
    env = os.environ.copy()
    env.update({
        "GATE_PYTHON_LOG": str(repo.parent / "gate-python.log"),
        "PYTHON_INVOCATION_LOG": str(repo.parent / "python-invocations.log"),
        "REAL_PYTHON": sys.executable,
    })
    for name, value in env_updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return _run(["bash", "deploy.sh"], repo, env)


@pytest.mark.parametrize(
    "selection",
    ("explicit", "active-venv", "repo-venv", "path"),
)
def test_deploy_uses_one_interpreter_for_every_release_step(tmp_path, selection):
    repo = _release_repo(tmp_path)
    path_dir = tmp_path / "path-bin"
    path_python = _fake_python(path_dir / "python3")
    active_python = _fake_python(tmp_path / "active venv" / "bin" / "python")
    repo_python = _fake_python(repo / ".venv" / "bin" / "python")
    explicit_python = _fake_python(tmp_path / "explicit python")
    env = {
        "PYTHON": None,
        "VIRTUAL_ENV": None,
        "PATH": f"{path_dir}{os.pathsep}{os.environ['PATH']}",
    }
    expected = path_python
    if selection == "repo-venv":
        expected = repo_python
    elif selection == "active-venv":
        env["VIRTUAL_ENV"] = str(active_python.parents[1])
        expected = active_python
    elif selection == "explicit":
        env["PYTHON"] = str(explicit_python)
        env["VIRTUAL_ENV"] = str(active_python.parents[1])
        expected = explicit_python
    else:
        shutil.rmtree(repo / ".venv")

    result = _deploy(repo, env)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "gate-python.log").read_text().splitlines() == [
        str(expected), str(expected),
    ]
    invocations = (tmp_path / "python-invocations.log").read_text().splitlines()
    assert any(command.startswith("-c import topiary") for command in invocations)
    assert "-m pip index versions topiary" in invocations
    assert "-m build" in invocations
    assert "-m twine upload dist/topiary-5.52.5-py3-none-any.whl" in invocations
    assert "v5.52.5" in _git(repo, "tag", "--list").stdout


@pytest.mark.parametrize("kind", ("missing", "not-executable", "directory"))
def test_deploy_rejects_invalid_explicit_python_without_fallback(tmp_path, kind):
    repo = _release_repo(tmp_path)
    invalid = tmp_path / "invalid-python"
    if kind == "not-executable":
        invalid.write_text("#!/bin/sh\nexit 0\n")
    elif kind == "directory":
        invalid.mkdir()

    result = _deploy(repo, {"PYTHON": str(invalid)})

    assert result.returncode == 1
    assert f"Python interpreter not found or not executable: {invalid}" in result.stderr
    assert not (tmp_path / "gate-python.log").exists()


def test_deploy_script_uses_one_configured_python():
    """Every Python release tool is invoked as a selected-Python module."""
    script = Path("deploy.sh").read_text()
    assert "resolve_topiary_python" in script
    assert 'VERSION=$("${PYTHON}" -c' in script
    assert '"${PYTHON}" -m pip index versions topiary' in script
    assert '"${PYTHON}" -m build' in script
    assert '"${PYTHON}" -m twine upload dist/*' in script
    assert "rm -rf dist build" in script
