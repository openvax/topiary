# Explicit source and exclusion species contexts (#586)

## Spec

Preserve human CLI defaults while keeping antigen origin, normal/reference
backgrounds and presenting MHC independent. Human xenografts and HLA-transgenic
mouse inputs are intentional supported configurations; never infer source or
reference species from allele names. Pascolo et al. 1997 (PMID 9182675) establishes
HLA-A2.1-restricted responses in transgenic mice; no curated biological data or
allele parsing changes are needed.

Add --species for sequence-source selection and separate --exclude-species /
--exclude-ensembl-release overrides for built-in reference exclusions. Existing
--ensembl-release controls the source. Non-human built-ins require a numbered
release. Default exclusions inherit a single known source species/release;
explicit cross-species exclusions do not inherit the other species' release.
File-based reference inputs can supplement this background, preserving all
sequences even if different files reuse a header.

Use the actual loaded variant/effect annotations to establish source context;
validate explicit source species/release against those annotations, including
MAF/JSON inputs rather than trusting --genome alone. Do not reinterpret variant
coordinates. Ambiguous/custom source contexts require explicit reference options
when a built-in exclusion would otherwise guess. Preserve mixed-annotation runs
that do not request an ambiguous built-in background. Resolve protein-change
source options with the same checks. Reject ignored --genome on direct inputs.

Human-only CTA, non-CTA and tissue membership is checked in its own source or
reference context. Human CTA + mouse background is valid; mouse CTA or mouse
human-expression-background requests fail before model inference. MHC alleles
remain independently passed to the selected predictor.

Keep CLI configuration plumbing shared, use PyEnsembl normalization, and retain
existing public sequence loaders. Document defaults, cross-species examples,
independent background release selection, and the scope of sequence exclusion.
No host flag that merely relabels output, no automatic HLA/H-2 conversion, and no
claim that sequence matching alone models transplant immunology.

## Plan

- [x] Inspect issue, source/reference consumers and upstream annotation APIs.
- [x] Write spec and record the user's species-context and test-quality corrections.
- [x] Implement shared CLI resolution, input guards and lossless reference union.
- [x] Test default human, mouse, transgenic and xenograft outcomes, plus conflicts.
- [x] Add paired entry-point and composed workflow regressions with exact outputs.
- [x] Verify targeted mutations are caught; inspect the complete diff.
- [x] Update docs/changelog/version and refresh isolated editable metadata.
- [ ] Run lint, focused tests, docs, dependencies and the full suite.
- [ ] Open PR; merge after green CI; wait for master CI and run ./deploy.sh.
- [ ] Verify published artifacts/tag and an installed-wheel species workflow.
- [ ] Record review evidence and refresh next-work triage.

## Review

The focused suite passes 76 cases. Lint, strict documentation build and pip check
pass. Source and selected editable metadata both report 5.112.0 with the same 15
runtime requirements. The known #565 stale build metadata was archived before
validation; shared environments and source checkouts were preserved.

All three mutation probes fail at behavioral assertions: forcing human source
selection, ignoring the explicit background override, and restoring the released
FASTA overwrite behavior. The reference-header regression is recorded in #604.

Using installed reference data and actual MHCflurry weights, mouse Trp53 from
Ensembl 102 produced 382 peptides / 1,146 rows against HLA-A*02:01. All peptide
coordinates match the mouse sequence and all affinity values are finite and
positive. Default human TP53 produced 403 peptides / 1,209 rows. Exact-match exclusion against the 68,381-protein mouse reference retained 321
of those 403 human peptides. Independent direct substring searches verified
every retained/excluded peptide; cached native scores and HLA identity were
preserved. These receipts and the published-wheel checks will be retained with
the release.

The previous X-policy release is 5.111.9. Work uses the isolated Topiary checkout
and environment; no shared-environment package changes were made.
