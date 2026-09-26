# Shared source and derived fixtures

See [the workflow and provenance guide](../../../docs/osteosarc-shared-data.md)
for source pins, scientific scope, acquisition, verification and regeneration.

Only the NTF3 case required by Topiary tests is kept. Its 15 reads are the
`openvax-v1` member `topiary/osteosarc_shared/vaccine-rna-v1/<case>.bam`, the
same records as Isovar's `vaccine-rna-v1` NTF3 file; the subset manifest
records the upstream manifest digest and still pins that file's bytes. Translation and synthetic prediction expectations are
separately versioned and are not new source data or biological binding claims.
