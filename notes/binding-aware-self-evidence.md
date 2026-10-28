# Binding-aware self evidence (#412)

Expose one public function, `binding_aware_self_evidence`, and its
`SelfProteome.binding_evidence` delegate. Compose the existing all-origin
bounded search with supplied query/self predictions; do not run models or
select a reference implicitly. Require a stated method, version and endpoint;
select one measurement context and metric without pooling model ranks.

Return one row per query occurrence and allele. Retain every candidate and
origin in `self_nearest_candidates`, all nearest equally/stronger-binding
matches in `self_mimic_candidates`, and all strongest nearby matches in
`self_strongest_nearby_candidates`. Scalar fields are available only where
the tied candidates agree. Missing/ambiguous measurements, unknown origin
scope and incomplete searches cannot become a complete negative result.
Expose the two binding scopes through the existing DSL and typed TSV/CSV.

Extend the existing bounded search with an explicit Levenshtein radius for
substitutions/insertions/deletions. Record all requested reference lengths and
whether they were available. Keep the existing same-length Hamming default.

Do not encode an unvalidated universal anchor/TCR-facing position weighting.
Anchor substitutions can alter the bound peptide structure and TCR affinity
(https://www.nature.com/articles/s41467-023-37532-7). Sequence similarity and
same-allele prediction remain distinct from tissue presentation and TCR
recognition (https://pmc.ncbi.nlm.nih.gov/articles/PMC4725365/ and
https://www.nature.com/articles/s41541-023-00713-y).

Validate both public doors, tie/origin retention, explicit exclusions,
model/version/endpoint/context isolation, missing and conflicting evidence,
indel bounds, nonhuman alleles, DSL policy differences, typed IO and the full
suite. Bump the version and release only after green local and GitHub gates.
