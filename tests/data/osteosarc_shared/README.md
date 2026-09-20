# Shared source and derived fixtures

See [the workflow and provenance guide](../../../docs/osteosarc-shared-data.md)
for source pins, scientific scope, acquisition, verification and regeneration.

Only the NTF3 case required by Topiary tests is bundled. Its BAM/index files
are byte-identical to Vaxrank PR #486's `vaccine-rna-v1` export; the subset
manifest records the upstream manifest digest. Translation and synthetic prediction expectations are
separately versioned and are not new source data or biological binding claims.
