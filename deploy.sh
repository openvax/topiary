#!/usr/bin/env bash
set -euo pipefail

# Deploy topiary to PyPI
# Usage: ./deploy.sh
# Override the interpreter with PYTHON=/path/to/python ./deploy.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/resolve_python.sh"
resolve_topiary_python
unset SCRIPT_DIR

VERSION=$("${PYTHON}" -c "import topiary; print(topiary.__version__)")
RELEASE_TAG="v${VERSION}"
RELEASE_REF="refs/tags/${RELEASE_TAG}"
echo "Deploying topiary v${VERSION}"
echo "Using Python: ${PYTHON}"

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

# Refuse an ambiguous release and prove that the exact tag can be pushed before
# the irreversible PyPI upload. This cannot prevent a later network failure, but
# it catches unavailable push access and obvious ref conflicts while recovery is
# trivial.
if git show-ref --verify --quiet "${RELEASE_REF}"; then
    echo "ERROR: Local tag ${RELEASE_TAG} already exists"
    exit 1
fi
PUSH_URL=$(git remote get-url --push origin)
if ! REMOTE_TAG=$(git ls-remote --tags "${PUSH_URL}" "${RELEASE_REF}"); then
    echo "ERROR: Could not check remote tag ${RELEASE_TAG}"
    exit 1
fi
if [ -n "${REMOTE_TAG}" ]; then
    echo "ERROR: Remote tag ${RELEASE_TAG} already exists"
    exit 1
fi
echo "Checking push access for ${RELEASE_TAG}"
git push --dry-run origin "HEAD:${RELEASE_REF}"

# Stop on an existing release OR an unverifiable lookup, before any gates run.
"${PYTHON}" scripts/check_pypi_release.py topiary "${VERSION}"

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

# Tag and push only this release. Unrelated local tags must remain local.
git tag "${RELEASE_TAG}"
git push origin "${RELEASE_REF}:${RELEASE_REF}"

echo ""
echo "Deployed topiary ${VERSION}: https://pypi.org/project/topiary/${VERSION}/"
