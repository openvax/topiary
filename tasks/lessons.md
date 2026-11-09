# Lessons

## Name the repository when work crosses projects

The user had to ask whether the workspace was mhcgnomes after a requested
Vaxrank change shipped. When a request targets a sibling repository, state that
repository before starting and in the result. Keep the original checkout and
unrelated work intact; do not carry a sibling release queue into recommendations
for the workspace repository.

## Keep antigen categories as inputs to shared logic

The user explicitly requires CTA support to be part of the normal antigen
pipeline. Category wrappers should resolve membership and call shared public
loaders; transcript choice, prediction, filtering and ranking each need one
implementation. Verify this with a composed comparison using equivalent inputs
through multiple categories, including changed filter settings, rather than
only testing individual helpers. Preserve genuinely different input semantics,
such as an explicitly requested transcript or RNA-supported variant translation.

## Keep producer capabilities and alphabet checks separate

A standard-amino-acid alphabet does not establish what a configured model or
cache accepts. Test a shared wrapper against the producer directly, including
complete and incomplete caches, nonstandard residues, offsets, flanks and
wild-type baselines. Preserve accepted data and the producer's original errors.
Do not rewrite test fixtures to avoid residues a test producer already supports;
that can hide a regression in a newly added global filter.

## Separate an explicit input policy from model capabilities

The user explicitly chose whole-protein X rejection by default, with an option
for intentional use. Implement that policy centrally across antigen sources,
fragments and baselines. Keep preservation tests under the override instead of
removing X fixtures, and do not infer that all other nonstandard residues should
also be excluded. A rejection warning must identify excluded inputs; unrelated
producer errors and strict cache gaps must not disappear behind broad catches.

## An override must match the requested retained data

The user clarified that the X override should process the non-X portions, not
merely retry an unchanged X-containing protein and discard it on rejection.
Select valid original windows before inference, preserving coordinates and
required flank context. Never join across gaps or invent molecular termini.
Keep producer acceptance (`allow`) distinct from clean-window selection, and
remove exception-message matching when structural input selection solves the
reported failure.

## Integration samples must not depend on set iteration order

Sampling the first dictionary entries from a set-derived gene collection makes
real-data coverage depend on the Python hash seed. Sort test inputs and deliberately
include relevant edge cases when available. Assert expected policy warnings at
every affected prediction call; do not blanket-ignore them or rely on a clean
sample. Wait for the completed traceback before attributing a failure: a surviving
temporary directory may belong to an active test rather than the failed test.
