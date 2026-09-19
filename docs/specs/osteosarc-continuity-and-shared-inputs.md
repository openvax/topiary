# Regional audit continuity and shared Osteosarc inputs

Scope: Topiary #348 and #349; use published Osteosarc 0.1.0 for acquisition
and shared cache access. Keep reconstruction and prediction in their existing
owners. Do not change historical alleles or biological expectations as a side
effect of acquisition migration; corrected MAP2 needs an explicit comparison.

## Required behavior

- A valid assembly contig without overlapping regional annotation has an
  explicit annotation outcome and real RNA evidence where collection succeeds.
  It cannot abort later inputs or be reported as zero support on exception.
  Invalid contigs and missing alignment/reference inputs remain distinguishable.
- Run reference construction, collection and reporting together in offline
  regressions, including an unannotated contig followed by an annotated locus.
  Preserve the FAM157A RefSeq/Ensembl discrepancy without inventing a transcript.
- Acquire original shared BAM/index assets through Osteosarc's public cache
  API, preserving the immutable vaccine-rna-v1 manifest, native alleles, source
  products, sizes and SHA-256 identities. No runtime dependency on Vaxrank.
- Use Osteosarc for audit source acquisition and indexed read extraction where
  its published APIs compose. Record reproducible blockers as Osteosarc issues.
  Ordinary tests are offline; acquisition is an explicit command.
- Version derived protein windows separately from source reads and prediction
  caches. Retain parent digests, transcript/gene/species, mutation offsets,
  reconstruction settings, and model/version/allele/peptide/context identities.
- Verify shared-root reuse, byte/record preservation, tamper rejection and a
  composed RNA-to-fragment-to-prediction/filter/rank workflow. Unresolved RNA
  paths produce no manufactured peptide.

## Validation and release

First reproduce the existing abort. Add focused regressions, run lint and the
full test suite, inspect the final diff, then require green GitHub CI before
merge. Bump the Topiary version in the PR and deploy from clean master, checking
both PyPI artifacts. Keep optional data tooling compatible with Topiary's base
Python support; Osteosarc itself requires Python 3.10+.

CI portability correction (#355): generated regional reference tests pin exact
decompressed GTF/FASTA records and source provenance; gzip headers and streams
can differ across platforms. Each run still verifies its own compressed hashes,
and acquired fixture assets retain their unchanged byte-level SHA-256 pins.
