#!/usr/bin/env bash

# Select one Python interpreter for Topiary's development and release scripts.
# Callers may set PYTHON explicitly; otherwise an active virtual environment,
# the repository virtual environment, and finally PATH are considered in that
# order. The exported absolute path is the contract shared by every gate.
resolve_topiary_python() {
    local repo_root
    local candidate
    local resolved_python

    repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
    if [[ -n "${PYTHON:-}" ]]; then
        candidate="${PYTHON}"
    elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
        candidate="${VIRTUAL_ENV}/bin/python"
    elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
        candidate="${repo_root}/.venv/bin/python"
    else
        candidate="python3"
    fi

    if ! resolved_python="$(command -v "${candidate}" 2>/dev/null)" || \
            [[ ! -f "${resolved_python}" || ! -x "${resolved_python}" ]]; then
        echo "Python interpreter not found or not executable: ${candidate}" >&2
        return 1
    fi
    if [[ "${resolved_python}" != /* ]]; then
        resolved_python="${PWD}/${resolved_python#./}"
    fi
    PYTHON="${resolved_python}"
    export PYTHON
}
