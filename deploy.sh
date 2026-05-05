#!/bin/bash
set -euo pipefail

# Deploy topiary to PyPI
# Usage: ./deploy.sh
# Override the interpreter with PYTHON=/path/to/python ./deploy.sh

PYTHON=${PYTHON:-python3}
PYTHON_BIN=$("${PYTHON}" -c "import sys; print(sys.executable)")
VERSION=$("${PYTHON}" -c "import topiary; print(topiary.__version__)")
echo "Deploying topiary v${VERSION}"
echo "Using Python: ${PYTHON_BIN}"

# Check we're on master
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "master" ]; then
    echo "ERROR: Must be on master branch (currently on ${BRANCH})"
    exit 1
fi

# Check working tree is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Working tree is not clean"
    git status --short
    exit 1
fi

# Check version isn't already on PyPI
if "${PYTHON}" -m pip index versions topiary 2>/dev/null | grep -q "${VERSION}"; then
    echo "ERROR: topiary ${VERSION} already exists on PyPI"
    exit 1
fi

# Lint
./lint.sh

# Tests
./test.sh

# Build
rm -rf dist build
"${PYTHON}" -m build
echo ""
echo "Uploading:"
ls -lh dist/

# Upload
"${PYTHON}" -m twine upload dist/*

# Tag and push
git tag "v${VERSION}"
git push --tags

echo ""
echo "Deployed topiary ${VERSION}: https://pypi.org/project/topiary/${VERSION}/"
