# Reject contradictory fragment identities

The all-variant audit found two GLIS3 observations with the same RNA-derived
fragment ID but different sample/policy annotations. Prediction converted the
list into two dictionaries, silently retaining only the last observation.

Keep identity canonical: identical JSON-serializable fragment records can be
coalesced, in input order; the same ID with different sequence, evidence,
provenance or annotations must raise before running models. Provide this as a
public `unique_fragments` function and use it in the shared prediction path.
Comparison uses normalized, key-sorted fragment JSON, not `ProteinFragment`
equality (which deliberately compares only IDs). Unserializable duplicate
records cannot be proved identical and must fail explicitly.

Drive the validator and prediction door through the same duplicate battery.
Separate observations in the audit get explicit sample/policy-scoped IDs;
test that both survive prediction, serialization and independent filtering.
