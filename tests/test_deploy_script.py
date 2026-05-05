"""Regression tests for deploy.sh release safety checks."""

import re
from pathlib import Path


def test_deploy_script_uses_one_configured_python():
    """deploy.sh should not mix python3 and a different pip executable."""
    script = Path("deploy.sh").read_text()
    assert "PYTHON=${PYTHON:-python3}" in script
    assert 'VERSION=$("${PYTHON}" -c' in script
    assert '"${PYTHON}" -m pip index versions topiary' in script
    assert '"${PYTHON}" -m build' in script
    assert '"${PYTHON}" -m twine upload dist/*' in script
    assert "rm -rf dist build" in script

    for line in script.splitlines():
        command = line.strip()
        if (
            not command
            or command.startswith("#")
            or command.startswith("PYTHON=")
        ):
            continue
        assert re.search(r"(^|[;&|]\s*)python3(\s|$)", command) is None
        assert re.search(r"(^|[;&|]\s*)pip(\s|$)", command) is None
