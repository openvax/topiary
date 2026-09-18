# Explicit outcomes for every tracked osteosarc variant

## Scope and completion

Audit the union of the website's pinned 182 variant entries (including the 44
vaccine-nominated alleles), the 20
historical pVAC alleles, and the additional indel/rearrangement/deletion leads
in the supplied candidate report. Deduplicate by assembly and exact allele or
oriented junction, never by gene. Entries lacking a complete literal allele
remain explicit unresolved inputs. This is not an exhaustive reanalysis of
every caller's genome-wide output, and nomination is not biological validation.

Every tracked case must produce an accountable outcome: accepted RNA-derived
sequence and usable mutation-overlapping prediction inputs, a reconstructed
sequence rejected by named filters, no usable alternate evidence, insufficient
sequence support, or an explicitly unresolved coding consequence. Missing
acquisition/reference data and software failures are not biological negatives.
Do not weaken thresholds, guess frames, pool samples, manufacture prediction
scores, or relabel reference-derived peptides as RNA-reconstructed.

## Implementation sequence

1. Reconcile the pinned source inventories and existing tests. Reuse locally
   cached original alignments and reference subsets after checksum verification;
   acquire and cache only missing inputs. Preserve sample, reference, processing
   policy and original source identities.
2. Exercise all nominated alleles through current Isovar and Topiary, including
   serialization and mutation-overlapping peptide selection. Extend compact
   offline CI coverage beyond the existing six-locus and two-indel fixtures.
   Check translations against original reference/cDNA evidence, including
   additional RNA edits and the mitochondrial genetic code.
3. Expose per-variant outcomes without silently losing rejected/empty cases.
   Keep diagnostic reconstruction separate from default acceptance; test the
   public handoff paths together. Fix demonstrated defects in their owning repo.
4. Produce a source-linked local outcome report and genuine predictions for
   supported sequences with available, provenance-pinned predictors. Historical
   pVAC scores remain historical; new model outputs remain separately labelled.
5. Investigate remaining additional candidates using exact alleles and actual
   RNA paths. Preserve justified negative or unresolved results, their evidence,
   and the precise missing fact needed to resolve them.

## Verification and delivery

During verification, upstream Isovar 1.18.1 shipped a relevant insertion-boundary
fix (#296/#300): one-sided reads could be falsely called reference/other and
conflict with a true alternate template. Require that corrected optional extra,
replay unchanged original inputs in a separate archive, and explicitly document
every changed count/disposition. Preserve the earlier audit as historical, not
as current evidence. No non-RNA installation gains a new dependency.

Coverage assertions must fail if a source-inventory allele disappears. Tests
must distinguish zero support from unavailable data, test unchanged default
filters, and verify peptide sequences/intervals rather than merely nonempty
output. Network-free regression fixtures retain original read records and
checksums. Every code PR has a version bump, full lint/tests and green CI;
merge and publish using the normal clean-master deployment gate. Scientific
uncertainty is reported, not converted into a software pass condition.
