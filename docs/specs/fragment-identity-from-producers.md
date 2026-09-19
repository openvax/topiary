# Fragment identity belongs to the producer

5.62.0 made prediction reject a repeated `fragment_id` with different content
(see `fragment-identity-collisions.md`). The rule was right, but it was enforced
only where fragments are consumed. The functions that create fragments still
produced colliding IDs and gave callers no way to avoid them. The documented
LENS path failed outright, and combining two alignments needed hand-built IDs.

Make every producer derive the ID from everything that distinguishes its
records. The frame path hashes its whole grouping key: sample, source,
variant, reported peptide and sequence. Absence is decided once per column
with `stated_values`, never per cell by truthiness. Sample-level observations
get one public labelling function, `fragments_for_sample`, which every
producer taking a `sample_name` routes through. The label namespaces the ID
and fills the prediction frame's `sample_name` column.

"Same content" means the same once stored by fragment IO, so a record and its
saved copy coalesce. Describe and adapt each Isovar result through one
extractor, with one filter disposition. Enforce the optional-dependency floor
at run time from Topiary's metadata, in the library and the test gate alike.

Parse audit inputs as typed tables before deciding anything: a malformed cell
becomes that entry's explicit status, while a missing column is a schema error.
Every trigger gets a synthetic test that fails on 5.62.0.

## Revision (5.64.0): a record is (sample, candidate)

5.63.0 wrote the sample label into the ID. Review showed three consequences.
Default cross-sample aggregation, keyed on the ID, stopped pooling. Label
sanitizing let different labels collide. And prediction could not tell a
cache's sample from the fragment's.

The ID now names the candidate and `ProteinFragment.sample_name` names the
observation; identity, equality and prediction rows use the pair. One ID must
mean one candidate in every sample (`CANDIDATE_FIELDS`), which lets prediction
scan it once. Prediction takes `sample_name` only from fragments.
