# Protein X policy (#573)

## Spec

Topiary skips entire protein scan inputs containing `X` (case-insensitive) by
default, before any predictor or cache sees them. One warning per scan stage
reports the count and a bounded plain-Python list of source names. Clean inputs
continue unchanged; an all-excluded batch returns the normal empty prediction
schema without invoking the producer. This deliberately excludes even clean
windows from those proteins, avoiding changes to offsets or flank context.

Apply the same public selection helper to named protein scans, all antigen
categories, variant/fragment scans, and wild-type protein baselines. A rejected
baseline leaves WT prediction values missing while retaining candidate rows.
The policy acts on the actual scan input (after explicit regions or variant
fragment extraction). Exact peptide/occurrence APIs and reference exclusion
sets retain their existing semantics. Do not generalize X to U or other residues.

Expose `allow_x_in_proteins=True` and `--allow-x-in-proteins` to pass original
proteins through for capable models and caches. Preserve sequence identity,
offsets, flanks, cached scores and cache-coverage policy. Override rejection
handling recognizes only mhctools' exact X-rejection signature and MHCflurry's
counted nonstandard-residue signature. Producers currently use bare ValueError
for both residue and unrelated model failures. Bisect failing batches and skip
only singleton inputs containing X that raise one of these recognized errors.
Successful batches pass through unchanged. Do not broadly suppress ValueErrors.
This is the assumed opt-in recovery policy after the optional clarification;
unknown custom-model validation errors and runtime failures remain strict.
The missing typed producer contract is tracked in openvax/mhctools#659.

## Plan

- [x] Trace common protein/fragment/WT paths and current upstream validation.
- [x] Record the default policy and distinguish it from model capabilities.
- [x] Add default and override conformance regressions, including mixed/all-X
      inputs, cache replay/misses, CTA/gene/tissue/FASTA, fragments and WT.
- [x] Implement the shared public selection helper, Python/CLI option and docs.
- [x] Settle override rejection handling without masking other failures.
- [x] Bump version and update changelog; review diff and source-category coverage.
- [ ] Pass focused tests, lint, full suite, strict docs and pip check.
- [ ] Open PR, verify CI, merge, wait master CI and deploy from clean master.
- [ ] Verify published artifacts and behavior, then inspect next open work.

## Review

The new policy suite passes 39 cases. Native MHCflurry 2.3.16 validation was
reproduced without model weights and recovered a mixed batch. Lint, strict docs
and pip check pass. The initial full run had 6,309 passes and six expected
policy-transition failures: five synthetic X fixtures now explicitly opt in;
the real tissue test verifies default exclusion warnings and clean predictions.
Final integrated/full verification and CI are pending.

Prior source-unification work shipped as 5.111.7 (PR #595) with all
6,293 local tests, PR/master CI, strict docs and dependency checks passing.
Its producer-input contract remains covered under the explicit X override.
