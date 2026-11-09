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
