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
6. Uploads to PyPI via twine
7. Creates a git tag (`v{version}`) and pushes it

## Release environment

Use a dedicated environment so installs in sibling repositories cannot change
dependencies during validation or upload. From the Topiary checkout:

```bash
release_dir=$(mktemp -d "${TMPDIR:-/tmp}/topiary-release.XXXXXX")
python3 -m venv "$release_dir/venv"
export PYTHON="$release_dir/venv/bin/python"
"$PYTHON" -m pip install -e '.[isovar,pirlygenes]' build twine ruff pytest pytest-cov pytest-xdist
"$PYTHON" -m pip check
"$PYTHON" -m pip inspect > "$release_dir/environment.json"
mkdir "$release_dir/pytest"
export PYTEST_DEBUG_TEMPROOT="$release_dir/pytest"
./lint.sh
./test.sh
```

The environment report records installed versions and editable source paths.
Only Topiary is editable here; sibling packages come from published releases.
Keep `PYTHON` set to this interpreter when running `./deploy.sh` after merging.
All lint, test, build and upload steps use it. The separate pytest temporary
parent also prevents another repository's test cleanup from removing this
release's fixtures; do not set a shared `--basetemp` in `PYTEST_ADDOPTS`.

The full suite needs the Ensembl data and external tools described in CI, and
permission to bind localhost sockets for the HTTP download fixtures. A sandbox
that denies those sockets must grant access for the test run; do not skip the
tests or suppress their errors.

## Version scheme

We use [semver](https://semver.org/):
- **Major**: breaking API changes
- **Minor**: new features (backward compatible)
- **Patch**: bug fixes
