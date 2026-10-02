# Releasing Topiary

## Steps

1. Bump the version in `topiary/__init__.py` and update `CHANGELOG.md` as part of your PR.
2. Merge the PR to master.
3. Wait for CI to pass on master.
4. Run `./deploy.sh` from master.

## What deploy.sh does

1. Verifies you're on master with a clean working tree
2. Checks the version isn't already published on PyPI
   (`scripts/check_pypi_release.py`; fail-closed, so an unreachable or
   malformed PyPI response stops the release)
3. Runs `./lint.sh` (ruff)
4. Runs `./test.sh` (pytest with coverage)
5. Builds sdist and wheel via `python -m build`
6. Uploads to PyPI via twine
7. Creates a git tag (`v{version}`) and pushes it

## Development and release environment

Use a repository-local environment so installs in sibling repositories cannot
change dependencies during validation or upload. The scripts automatically use
`.venv` unless `PYTHON` or an active virtual environment selects another
interpreter. From the Topiary checkout:

```bash
python3 -m venv .venv
export PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install -e '.[isovar,pirlygenes]' build twine ruff pytest pytest-cov pytest-xdist
"$PYTHON" -m pip check
"$PYTHON" -m pip inspect > .venv/environment.json
./lint.sh
./test.sh
```

The environment report records installed versions and editable source paths.
Only Topiary is editable here; sibling packages come from published releases.
Keep `PYTHON` set to this interpreter when running `./deploy.sh` after merging.
For interactive use, run `source .venv/bin/activate`. All lint, test, build and
upload steps use this interpreter. `test.sh` gives every run its
own pytest temporary root, so another repository's test cleanup cannot remove
this release's fixtures; it is deleted after a passing run and kept after a
failing one. Set `PYTEST_DEBUG_TEMPROOT` yourself only to choose where it goes,
and do not set a shared `--basetemp` in `PYTEST_ADDOPTS`.

The full suite needs the Ensembl data and external tools described in CI, and
permission to bind localhost sockets for the HTTP download fixtures. A sandbox
that denies those sockets must grant access for the test run; do not skip the
tests or suppress their errors.

## Downstream verification

CI installs a pinned published Vaxrank release and runs the real candidate
scoring and vaccine-construction workflow. To repeat it locally:

```bash
"$PYTHON" -m pip install 'vaxrank==3.36.0'
"$PYTHON" -m pip check
"$PYTHON" -m pytest scripts/check_vaxrank_candidates.py -q
```

Update that pin when adopting a new downstream release; do not leave temporary
migration-branch pins after the corresponding release is published.

## Version scheme

We use [semver](https://semver.org/):
- **Major**: breaking API changes
- **Minor**: new features (backward compatible)
- **Patch**: bug fixes
