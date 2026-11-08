# Shared antigen source pipeline

## Spec

CTA and tissue wrappers resolve gene membership and delegate to the public
`sequences_from_gene_ids` loader. Gene names, gene IDs, CTA genes and tissue
genes all use one representative-transcript selector. Resolved proteins enter
the ordinary predictor, DSL filtering/ranking and exclusion path; categories
do not introduce separate scoring or result formats.

PR #593 was revised during this work to fix the same transcript-selection bug.
This PR now follows it and uses its `best_protein_transcript(gene)` unchanged,
exporting that same function from `topiary`. No competing selector is added.
That inherited policy ranks complete protein-coding translations, then standard
amino-acid sequences, then length, with transcript-ID tie breaking. Source API
docs and CLI help must describe the inherited policy consistently.

Preserve exact transcript-ID/name requests and empty derived CTA/tissue sets.
Explicit empty gene lists remain input errors. Whole-proteome and non-CTA
exclusion sets enumerate proteins rather than choosing representatives.
Variant/RNA inputs preserve their supported translation and mutation coordinates.

## Plan

- [x] Reproduce the reported IZUMO2, C16orf90 and LLCFC1 choices on Ensembl 115.
- [x] Trace category membership, source resolution, prediction and exclusion.
- [x] Make CTA and tissue wrappers use the public gene-ID loader.
- [x] Reconcile #593 into this branch and remove the alternative selector.
- [x] Export the inherited public selector and update CLI/API documentation.
- [x] Cover every gene source, fallback, missing proteins and explicit inputs.
- [x] Compare equivalent CTA, gene-name, gene-ID and FASTA inputs through real
  protein scanning, filters, ranking and exact/shared-8-mer tissue exclusions.
- [ ] Run lint, full tests, strict docs and dependency checks on integrated code.
- [ ] Update PR #595 to follow #593 and verify CI on the integrated head.
- [ ] After the preceding release queue lands, merge and deploy from clean master.

## Review

The original complete-start-only implementation passed 6,256 local tests on
3c3a080. It selected ENST00000293405 (IZUMO2), ENST00000437192 (C16orf90), and
ENST00000409607 (primary-locus LLCFC1), preserving the separate LLCFC1 alternate
locus. That implementation is superseded here by #593's shared selector; its
four-protein CTA difference and fallback-warning semantics are not claims about
the integrated policy.

After integration, all 304 source and conformance tests pass, along with lint,
strict docs and dependency checks. The full suite is running. The conformance tests assert both
identical decisions across input routes and observable effects of changing
filter, sort and exclusion settings; exact source IDs and offsets are checked.

Version 5.111.7 follows #591/#592/#593 (5.111.4-5.111.6). This does not authorize
merging or renumbering the earlier PRs. Recheck their heads before publication.
