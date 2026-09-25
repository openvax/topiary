# Isovar comparison fixture

`isovar_hypotheses.json` is synthetic output from the released Isovar 1.32.0
`export_protein_hypotheses` API. It uses the public producer's
[`synthetic_result` fixture](https://github.com/openvax/isovar/blob/v1.32.0/tests/test_protein_hypotheses.py),
with `max_protein_sequences_per_variant=0`, sample `tumor-1`, and source
`synthetic-rna.bam`. No patient data or external reference download is used.

The fixture contains two synonymous cDNAs encoding MAQG, an alternative MAQD,
and a shorter AQG window with neither start nor stop completeness. Reads overlap
between hypotheses. The two main protein supports have 4 and 2 segments, but
their union is 5 segments (4 fragments). Each translation retains its own
support, frame context and observed edits. Rank is recorded without selection.

The permissive export is for comparison tests; it is not a change to Topiary's
default reconstruction, support thresholds, filtering or candidate selection.
