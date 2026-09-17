# RNA alongside historical pVAC predictions

## Scope

Produce a new, reproducible RNA-enriched copy of all 21 archived final reports.
Do not edit the original reports, change predictions, claim to recover the
historical annotation step, or infer protein support from locus-level counts.

## Data contract

- Use the January 2025 UCLA resection RNA products corresponding to the tumor
  named by the pVAC run. Verify the alignment reference and record source
  URLs, checksums, acquisition details, sample mapping and software versions.
- Join RSEM TPM using explicit Ensembl gene/transcript identifiers. Preserve
  the source versioned IDs and label stable-ID-only matches; reject ambiguous
  matches rather than summing distinct transcript versions.
- Count the exact 20 genomic alleles using Isovar's public read collector and
  a declared placement/duplicate policy. State count units, denominator, and
  zero-coverage versus missing evidence. Verify coordinate conversion against
  reference sequence before counting.
- Add namespaced RNA columns and provenance to imported predictions. Preserve
  every existing column, row, null and prediction. No pooling across samples
  or technologies and no implicit replacement of canonical historical fields.
- Provide a compact per-variant/transcript evidence table and a coverage report
  alongside the full enriched Topiary files.

## Verification

Pin compact real-source fixtures. Test both pVAC report flavors, unchanged
prediction identity/values, ambiguous and missing join keys, missing versus
zero evidence, and save/reload plus a DSL filter whose result changes with
RNA thresholds. Run the full 21-file workflow and repository lint/tests before
shipping via green CI, merge and the normal deployment gate.
