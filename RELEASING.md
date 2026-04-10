# Releasing Topiary

## Steps

1. Bump the version in `topiary/__init__.py` and update `CHANGELOG.md` as part of your PR.
2. Merge the PR to master.
3. Wait for CI to pass on master.
4. Run `./deploy.sh` from master.

## What deploy.sh does

1. Verifies you're on master with a clean working tree
2. Checks the version isn't already published on PyPI
3. Runs `./lint.sh` (ruff)
4. Runs `./test.sh` (pytest with coverage)
5. Builds sdist and wheel via `python -m build`
6. Prompts for confirmation
7. Uploads to PyPI via twine
8. Creates a git tag (`v{version}`) and pushes it

## Prerequisites

```
pip install build twine ruff pytest pytest-cov
```

## Version scheme

We use [semver](https://semver.org/):
- **Major**: breaking API changes
- **Minor**: new features (backward compatible)
- **Patch**: bug fixes
