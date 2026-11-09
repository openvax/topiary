# Stabilize tissue workflow validation and ship the X policy

## Spec

PR #600 merged the shared default-X exclusion and X-free-window override.
Its final local suite (6,353 tests), PR CI and master CI all passed. Deployment
then stopped before building/uploading: test_first_principles_workflow sampled
TRAC|ENST00000611116 from an unordered tissue-gene mapping and treated the
expected ProteinInputWarning as an error. The runtime exclusion worked.

Make both small real-tissue integration samples deterministic. When available,
include one original X-containing protein alongside clean proteins, and assert
the expected warning count and complete exclusion of those source names.
Keep other warnings strict. Preserve filter/ranking and vital-organ exclusion
checks. Do not change runtime biology, sources or model policy.

Every PR must bump the version. Ship the previously unpublished 5.111.8 changes
with this test fix as 5.111.9; consolidate the changelog under the shipped version.
Use the normal PR, full-validation, master-CI and clean-master deployment gates.

## Plan

- [x] File the integration-test regression (#601).
- [x] Stabilize both tissue samples and share explicit policy assertions.
- [ ] Run focused integration/policy tests, lint, docs, dependency and full checks.
- [ ] Merge after green PR CI, wait for master CI, run ./deploy.sh.
- [ ] Verify PyPI artifact hashes, release tag and installed-wheel behavior.
- [ ] Refresh next-work triage.

## Evidence

- Original feature PR: https://github.com/openvax/topiary/pull/600
- Deployment failure log: /tmp/topiary-5.111.8-deploy.log
- Traceback: tests/test_integration.py:123; ProteinInputWarning for TRAC.
- Failed deployment: 6,352 passed, one failed. No artifact upload or tag push.
- Do not infer a failing test from surviving temporary directories: those can
  belong to tests still running. The completed traceback identifies this failure.

Focused validation: 59 integration/policy tests passed; lint, strict docs and
pip check pass. Source and selected package metadata both report 5.111.9.
