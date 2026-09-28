#!/usr/bin/env bash
# Install CI dependencies with a bounded retry for just-published releases.
#
# Usage: scripts/pip_install.sh <pip install arguments...>
#
# A sibling release (mhctools, isovar, vaxrank) can be on PyPI and still be
# missing from the index a runner resolves against for a few minutes, so pip
# reports "No matching distribution found" for an exact floor that is in fact
# published (#335). Every later attempt bypasses pip's HTTP cache so the index
# is fetched again.
#
# The requirements are passed through unchanged on every attempt: nothing is
# relaxed, and a genuine resolution conflict fails each attempt the same way,
# ending with the last error. PIP_INSTALL_ATTEMPTS (default 3) bounds the
# attempts and PIP_INSTALL_RETRY_DELAY (default 20) is the base wait in
# seconds, growing linearly.
set -uo pipefail

attempts="${PIP_INSTALL_ATTEMPTS:-3}"
delay="${PIP_INSTALL_RETRY_DELAY:-20}"
python="${PYTHON:-python}"

for (( attempt = 1; attempt <= attempts; attempt++ )); do
    if (( attempt == 1 )); then
        "$python" -m pip install "$@" && exit 0
    else
        "$python" -m pip install --no-cache-dir "$@" && exit 0
    fi
    if (( attempt < attempts )); then
        wait=$(( attempt * delay ))
        echo "pip install failed (attempt ${attempt}/${attempts}); retrying in ${wait}s with the index fetched again" >&2
        sleep "$wait"
    fi
done
echo "pip install failed after ${attempts} attempts" >&2
exit 1
