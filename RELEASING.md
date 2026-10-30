# Releasing Topiary

## Steps

1. Bump the version in `topiary/__init__.py` and update `CHANGELOG.md` as part of your PR.
2. Merge the PR to master.
3. Wait for CI to pass on master.
4. Run `./deploy.sh` from master.

## What deploy.sh does

1. Verifies you're on master with a clean working tree
2. Refuses an existing local or remote release tag, then verifies push access
   with a dry run of that exact tag ref
3. Checks the version isn't already published on PyPI
   (`scripts/check_pypi_release.py`; fail-closed, so an unreachable or
   malformed PyPI response stops the release)
4. Runs `./lint.sh` (ruff and the declared mypy scope)
5. Runs `./test.sh` (pytest with coverage)
6. Builds sdist and wheel via `python -m build`
7. Uploads to PyPI via twine
8. Creates the git tag (`v{version}`) and pushes only that exact tag ref

The dry run exercises the authenticated push path and catches unavailable
endpoints, common credential failures, and obvious ref conflicts before
publication. It cannot guarantee that the network, remote permissions, or
server-side rules will remain unchanged through the later real push.

## Recovering a tag after publication

If the PyPI upload succeeds but the final tag push fails, do not rerun
`deploy.sh`: the already-published-version guard will stop it. Preserve the
local `dist/` directory and recover conservatively:

1. Check the release's PyPI JSON response and compare the SHA-256 digest of
   every published wheel and source distribution with the corresponding local
   file in `dist/`.
2. Verify that clean local `master`, `origin/master`, and the intended merge
   commit are identical. Inspect the commit instead of assuming the checkout
   remained unchanged after the upload.
3. Verify that `v{version}` is absent remotely. Create it locally at the
   verified merge commit only if the failed deploy did not already create it.
4. Confirm `git rev-list -n1 v{version}` prints that exact merge commit, then
   push only the release ref:

   ```bash
   git push origin refs/tags/v{version}:refs/tags/v{version}
   ```

Do not force a conflicting remote tag or replace a published artifact. File an
issue with the artifact hashes, commit, tag state, and failed command if any
check is ambiguous.

## Development and release environment

Use a repository-local environment so installs in sibling repositories cannot
change dependencies during validation or upload. The scripts automatically use
`.venv` unless `PYTHON` or an active virtual environment selects another
interpreter. From the Topiary checkout:

```bash
python3 -m venv .venv
export PYTHON="$PWD/.venv/bin/python"
"$PYTHON" -m pip install -e . -r requirements-dev.txt
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

CI runs the real Vaxrank candidate scoring and vaccine-construction workflow.
During the coordinated Topiary 5.109 / Vaxrank 3.49 migration it installs the
exact Vaxrank migration commit: the previous published consumer caps Varcode
below 11. Use a separate consumer environment to repeat that check locally:

```bash
"$PYTHON" -m venv .venv/vaxrank-consumer
VAXRANK_PYTHON="$PWD/.venv/vaxrank-consumer/bin/python"
"$VAXRANK_PYTHON" -m pip install pytest 'vaxrank @ git+https://github.com/openvax/vaxrank.git@b5c1c9b8348072b005283f5e78096fed1e570300' -e .
"$VAXRANK_PYTHON" -m pip check
"$VAXRANK_PYTHON" -m pytest scripts/check_vaxrank_candidates.py -q
```

The separate consumer environment keeps this check independent of the release
environment's core integrations. Varcode 11.1 and Isovar 1.78 must already be
published; installation uses normal dependency resolution.

After Vaxrank 3.49 publishes, replace the source pin with its published version
in CI and this example ([Topiary #567](https://github.com/openvax/topiary/issues/567)).
Do not leave temporary migration-source pins after the corresponding release
is published.

CI also installs the normal wheel without extras and runs RNA assembly,
reference/CTA inputs, Weirdo ranking and typed cache replay outside the source
checkout. Isovar, PirlyGenes, OncoRef and Weirdo are core dependencies; their
test markers select workflows, not optional installations. The coordinated
data combination is PirlyGenes 6.1 with OncoRef 1.24.

The #520 sequence-utility audit found that Topiary's FASTA readers preserve
different record shapes (a named sequence mapping versus all provenance
records), while self matching and peptide properties have explicit I/L,
unknown-residue and mass conventions. Dependency consolidation does not
change those contracts or add an unused Biotite/scikit-bio backend.
The readers now share `iter_fasta_records` validation while retaining their
distinct duplicate-ID contracts: mapping readers keep the last record and
self-reference indexing retains every record
([Topiary #549](https://github.com/openvax/topiary/issues/549)).

## Version scheme

We use [semver](https://semver.org/):
- **Major**: breaking API changes
- **Minor**: new features (backward compatible)
- **Patch**: bug fixes
