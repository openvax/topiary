"""Regression tests for deploy.sh release safety checks."""

import json
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
    printf '%s\\n' "${RELEASE_VERSION:-5.52.5}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "pip" ]]; then
    printf '%s\\n' "${PYPI_LISTING:-Available versions: 5.52.4}"
    exit "${PYPI_LOOKUP_STATUS:-0}"
elif [[ "${1:-}" == "-m" && "${2:-}" == "topiary.cli.release" ]]; then
    exec "$REAL_PYTHON" "$RELEASE_CHECK_DRIVER" "$@"
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
    for name in ("release.py", "cli/release.py"):
        source = SOURCE_ROOT / "topiary" / name
        destination = repo / "topiary" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source, destination)
    (repo / "topiary" / "cli" / "__init__.py").touch()
    # Exercise the real Python preflight and CLI in a subprocess; only the
    # HTTP boundary is replaced. No test may publish or contact PyPI.
    (repo / "pypi_response.py").write_text('''import io
import os
import runpy
import socket
import ssl
import sys
from urllib.error import HTTPError, URLError
from topiary import release

def response(request, timeout):
    with open(os.environ["PYPI_REQUEST_LOG"], "a") as log:
        log.write(request.full_url + "\\n")
    assert timeout == 10
    error = os.environ.get("PYPI_ERROR")
    if error == "timeout":
        raise TimeoutError("lookup timed out")
    if error == "dns":
        raise URLError(socket.gaierror("DNS lookup failed"))
    if error == "tls":
        raise URLError(ssl.SSLError("certificate verification failed"))
    status = int(os.environ.get("PYPI_STATUS", "404"))
    if status >= 400:
        raise HTTPError(request.full_url, status, "simulated PyPI failure", {}, None)
    body = os.environ.get("PYPI_BODY", "").encode()
    result = io.BytesIO(body)
    result.status = status
    return result

release.urlopen = response
sys.argv = sys.argv[2:]
runpy.run_module(sys.argv[0], run_name="__main__")
''')
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
        "RELEASE_CHECK_DRIVER": str(repo / "pypi_response.py"),
        "PYPI_REQUEST_LOG": str(repo.parent / "pypi-requests.log"),
    })
    for name, value in env_updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    return _run(["bash", "deploy.sh"], repo, env)


def _assert_release_stopped(repo):
    """Failure must precede every gate, build, upload and tag mutation."""
    assert not (repo.parent / "gate-python.log").exists()
    invocations = (repo.parent / "python-invocations.log").read_text()
    assert "-m build" not in invocations
    assert "-m twine" not in invocations
    assert not (repo / "dist").exists()
    assert not _git(repo, "tag", "--list").stdout
    assert not _git(repo, "ls-remote", "--tags", "origin").stdout


@pytest.mark.parametrize("error, diagnostic", [
    ("timeout", "timed out"), ("dns", "DNS lookup failed"),
    ("tls", "certificate verification failed"),
])
def test_preflight_stops_on_network_errors(tmp_path, error, diagnostic):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYPI_ERROR": error, "PYPI_LOOKUP_STATUS": "1",
    })
    assert result.returncode != 0
    assert diagnostic in result.stderr
    _assert_release_stopped(repo)


@pytest.mark.parametrize("status", [403, 429, 500, 503])
def test_preflight_stops_on_http_errors(tmp_path, status):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYPI_STATUS": str(status), "PYPI_LOOKUP_STATUS": "1",
    })
    assert result.returncode != 0
    assert str(status) in result.stderr
    _assert_release_stopped(repo)


@pytest.mark.parametrize("reported_version", ["5.52.5", "5.52.5.0", "v5.52.5"])
def test_preflight_refuses_already_published_version(tmp_path, reported_version):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYPI_STATUS": "200",
        "PYPI_BODY": json.dumps({"info": {"name": "topiary", "version": reported_version}}),
        "PYPI_LISTING": "Available versions: 5.52.5",
    })
    assert result.returncode != 0
    assert "topiary 5.52.5 already exists on PyPI" in result.stderr
    _assert_release_stopped(repo)


@pytest.mark.parametrize("nearby_version", ["5.52.50", "15.52.5", "5.52.5rc1", "5x52x5"])
def test_preflight_does_not_confuse_similar_versions(tmp_path, nearby_version):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYPI_STATUS": "404", "PYPI_LISTING": f"Available versions: {nearby_version}",
    })
    assert result.returncode == 0, result.stderr
    assert _git(repo, "tag", "--list").stdout.strip() == "v5.52.5"


@pytest.mark.parametrize("body", [
    "<html>upstream unavailable</html>", "null", "[]", "{}", '{"info": null}',
    '{"info": {"version": "5.52.5"}}',
    '{"info": {"name": "another-project", "version": "5.52.5"}}',
    '{"info": {"name": "topiary", "version": "5.52.50"}}',
    '{"info": {"name": "topiary", "version": "not-a-version"}}',
])
def test_preflight_stops_on_invalid_response(tmp_path, body):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "PYPI_STATUS": "200", "PYPI_BODY": body,
    })
    assert result.returncode != 0
    assert "Could not verify" in result.stderr
    _assert_release_stopped(repo)


def test_preflight_does_not_filter_yanked_or_incompatible_prereleases(tmp_path):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "RELEASE_VERSION": "5.52.5rc1", "PYPI_STATUS": "200",
        "PYPI_BODY": json.dumps({"info": {"name": "topiary", "version": "5.52.5rc1",
                                           "yanked": True, "requires_python": ">=100"},
                                 "urls": []}),
    })
    assert result.returncode != 0
    assert "5.52.5rc1 already exists on PyPI" in result.stderr
    _assert_release_stopped(repo)


def test_preflight_rejects_invalid_local_version_before_network(tmp_path):
    repo = _release_repo(tmp_path)
    result = _deploy(repo, {
        "PYTHON": str(_fake_python(tmp_path / "selected python")),
        "RELEASE_VERSION": "not-a-version",
    })
    assert result.returncode != 0
    assert "Invalid version" in result.stderr
    assert not (tmp_path / "pypi-requests.log").exists()
    _assert_release_stopped(repo)


@pytest.mark.parametrize("gate", ["branch", "worktree"])
def test_deploy_checks_repository_before_network(tmp_path, gate):
    repo = _release_repo(tmp_path)
    if gate == "branch":
        _git(repo, "checkout", "-b", "not-master")
    else:
        (repo / "uncommitted.txt").write_text("preserve me")
    result = _deploy(repo, {"PYTHON": str(_fake_python(tmp_path / "selected python"))})
    assert result.returncode != 0
    assert not (tmp_path / "pypi-requests.log").exists()
    _assert_release_stopped(repo)


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
    assert "-m topiary.cli.release topiary 5.52.5" in invocations
    assert (tmp_path / "pypi-requests.log").read_text().splitlines() == [
        "https://pypi.org/pypi/topiary/5.52.5/json",
    ]
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
    assert '"${PYTHON}" -m topiary.cli.release topiary "${VERSION}"' in script
    assert '"${PYTHON}" -m build' in script
    assert '"${PYTHON}" -m twine upload dist/*' in script
    assert "rm -rf dist build" in script
