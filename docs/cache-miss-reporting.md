# Partial cache-backed prediction batches

Implementation specification for #304 and compatibility audit for #374.

The default remains strict: a missing peptide, context, kind or genotype raises
`CachedPredictorCoverageError`. Existing `raise_on_error` only controls variant
annotation and must not silently change prediction completeness.

An explicit `TopiaryPredictor(cache_miss_handler=...)` opts into retaining
successful model/input pairs. The handler receives a structured record for every
skipped pair, with input identity, model identity, prediction stage and original
error. Split only failed batches until the failing input is isolated. Skip the
entire sequence for that model, not just its missing window: this preserves
protein flanks and never substitutes an incompatible prediction. Other models
can still score that sequence. Whole-peptide, fragment, wildtype and self-nearest
prediction paths use the same policy; unexpected errors still propagate.

A public `predict_with_cache_miss_report` function supplies this shared policy.
The handler is the authoritative report across composed workflows; successful
rows remain ordinary prediction rows. Emit an explicit warning when any inputs
are skipped. Callers must retain the report alongside the partial table.

CLI `--cache-miss-report PATH` opts in, requires a real JSON file distinct from
inputs and prediction outputs, and writes a versioned completeness report before
writing the result table. Complete runs return 0; partial runs return 3, including
runs with zero successful predictions. CSV stdout remains parseable. The report
is written even if every cache lookup succeeds. No predictions are fabricated.

Tests must run strict and reporting modes on the same covered/uncovered inputs,
assert successful sequences/models survive, keep repeated peptide contexts
separate, preserve kind/genotype failure checks, exercise composed fragment/WT
and CLI workflows, and prove unrelated exceptions and handler failures still
raise. Fixture builders belong beside the tests.

Topiary 5.68.4 (#380) now delegates fixture resolution and generation to the
published Osteosarc 0.2.3 `fixture_assets` and `consumer_fixtures` APIs. Osteosarc
0.1.4 lacks these APIs and is no longer a compatible development baseline.
Use `osteosarc>=0.2.3,<0.3` to preserve compatible editable installs and accept
patch fixes within that API series. Verify shared fixtures offline,
built-distribution metadata, normal dependency resolution and an editable 0.2.3
installation. Shared source data and historical scientific expectations stay
pinned independently of the package version.
