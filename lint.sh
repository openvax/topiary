#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"
resolve_topiary_python
unset SCRIPT_DIR

"${PYTHON}" -m ruff check topiary/ tests/

echo 'Passes ruff check'
