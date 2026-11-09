# Protein X policy (#573)

## Revised spec after user clarification

The default skips whole protein scan inputs containing X (case insensitive),
warns with count/example names, continues with clean inputs, and returns a valid
empty table when all inputs are excluded. The requested override scores only
X-free windows of the original proteins. It never joins across an X, removes
residues to shift coordinates, or treats an internal X gap as a new terminus.
For flank-aware models, the required original N/C context must also be X-free.

Expose one `protein_x_policy`: `skip` (default), `windows` (requested clean-window
scan), and explicit `allow` (the original unfiltered producer/cache contract for
models supporting X). `allow` keeps producer errors strict; it is not the
recommended recovery mode. CLI `--protein-x-policy windows` is the requested
override. Replace the unreleased allow-X flag and error-message recovery helper.

Apply this once across CTA/gene/tissue/FASTA, variant/fragment and WT scans.
Retain names, original offsets, flank context and measurement identity. Candidate
rows survive missing WT windows with NaN WT predictions. Exact peptide APIs and
reference exclusion sets keep their existing semantics. Do not generalize to U
or another alphabet policy.

Use the model's declared scanner context to select windows before inference.
Batch native mhctools inputs through its aligned prediction API; reuse explicit
occurrence validation for legacy DataFrame-only producers. Extend native protein-cache queries with explicit selected windows, preserving
sparse-kind semantics. Use fallback and stored declarations to establish a
conservative window contract. Do not infer limits from observed flank lengths;
unknown/ambiguous context must be reported and conservatively exclude affected
X inputs, while clean inputs continue. Preserve strict cache misses for retained
windows and the existing explicit cache-miss reporting policy.

## Plan

- [x] Trace shared scan, cache, exact-occurrence and flank-declaration contracts.
- [x] Record the user's correction and revised input policy before edits.
- [x] Replace exception-message recovery with deterministic X-free selection.
- [x] Add shared public window selection/prediction helpers and Python/CLI policy.
- [x] Verify clean runs, internal/leading/trailing/consecutive X, short/all-X
      inputs, multiple lengths/models, asymmetric flanks, repeated peptides,
      original coordinates, cache replay/misses, sources, fragments and WT.
- [x] Preserve explicit `allow` producer conformance tests and all other residues.
- [x] Update docs/changelog/PR #600 and record the revised local validation.
- [ ] Pass lint, full suite, docs and pip check, then PR and master CI.
- [ ] Merge and deploy 5.111.8, verify published artifacts and behavior.

## Review and release status

PR #600 at 3c3a8b3 implemented default whole-protein exclusion and optional
producer-rejection recovery. That design is superseded by the user's request to
process only the non-X portions. Its 6,334-test pass is evidence for that earlier
head, not final validation of this revised design.

GitHub Actions/Checks has a confirmed outage:
https://www.githubstatus.com/incidents/tp8cghg7tlc4
The first CI attempt passed all 16 executed jobs, then failed with an internal
server error before dependent jobs were scheduled. A retry was already running
when the user clarified the override. No merge or publication has happened.
Release state/evidence: /tmp/topiary-573-release-state.json.

Revised implementation: 374 focused policy/cache/native-conformance tests pass.
Lint, strict docs, and pip check pass. Full ./test.sh is running; final validation
will be recorded against the new head before merge.

Full validation of 4bd3421: 6,350 passed, one inventory regression. A cache's
conservative selection bounds must not advertise the native producer's
`protein_flank_lengths` contract. Rename the cache method, leave model inventory
and row annotation unchanged, and rerun both focused and full checks.
