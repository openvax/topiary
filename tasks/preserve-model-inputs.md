# Preserve model and cache protein inputs (#596)

## Spec

Remove the unconditional whole-protein alphabet gate from the shared scanner.
The configured predictor/cache receives the original mapping, including source
names, sequence text and flanking context. A model that accepts an input keeps
its predictions; a model that rejects it still raises its own error. Strict
cache coverage and explicit cache-miss reporting remain authoritative.

Remove the new, unmerged `scorable_proteins`, `scorable_peptide` and
`SCORABLE_AMINO_ACIDS` API: a standard alphabet is not a universal predictor
capability. The representative-transcript selector keeps its existing ranking
and uses the existing `AMINO_ACIDS` alphabet directly. No category-specific
scanner, synthetic sequence fragments or coordinate remapping is introduced.

This fixes #596. Generic partial-window recovery for models that reject a
sequence is outside this fix and remains tracked by #573. Correct the queued
PR's docs and tests so they do not promise that recovery or claim every model
rejects nonstandard residues.

## Plan

- [x] Add failing direct/wrapped predictor and complete-cache conformance cases,
  including mixed inputs, an X-leading singleton, original offsets and flanks.
- [x] Cover strict missing-cache behavior, explicit miss reports, model errors,
  fragments and wild-type prediction through the same shared path.
- [x] Remove the gate and misleading unreleased API; update docs/changelog.
- [ ] Verify focused regressions, lint, full tests and strict docs.
- [ ] Push the fix to #593, merge it into #595, reconcile metadata, and verify
  both PRs' CI. Preserve the existing release queue.

## Review

Before the fix, the direct predictor and complete cache return 12 occurrences
for XMNPQRSTVWYACDEFGHIK (11 have only standard residues). The Topiary wrapper
drops all 12 in a mixed batch and raises for that protein alone.

The regression battery failed in 15 cases before the fix (three preservation
cases passed). All 18 pass after removing the gate. The same failure also hid
an uncovered X-containing input from the strict cache and explicit miss report.
Older fragment/wild-type/exclusion fixtures containing nonstandard residues
were restored; the producer's accepted input must drive those tests.

Lint (Ruff and mypy), strict docs and pip check pass. The Ensembl 115 audit
confirms all 624 CTA transcript identities and sequences are unchanged.
The parent fix is pushed as 6ffa6b2. Its first full local run aborted when
the disk filled; after space recovered it is rerunning with pytest configured
to discard passing temporary fixtures immediately. This changes fixture
retention, not test selection. The integrated child branch passes all 22
focused regression and source-conformance cases, lint, strict docs and pip check.
Full suites and CI remain pending.
