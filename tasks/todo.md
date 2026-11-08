# Complete-start transcript selection

- [x] Reproduce the three reported genes on Ensembl 115 and inspect the shared
  gene-name, gene-ID, CTA and tissue source path.
- [x] Check primary annotation semantics and PyEnsembl's start-codon API.
- [x] Record issue #594 and implement the source selection policy.
- [x] Prove the regression fails before the change and passes afterward.
- [ ] Exercise every gene source, explicit transcript inputs and prediction
  coordinates, including the three reported genes on Ensembl 115.
- [ ] Run lint, full tests, documentation build and review the diff.
- [ ] Open a versioned PR, verify CI, merge and deploy from clean master.

## Spec

Gene-derived representative proteins should prefer transcripts whose annotated
start codon spans three genomic positions, then choose the longest available
protein within that group. PyEnsembl's `start_codon_complete` implements the
annotation check and has existed since 2.3.9, below our dependency floor.
Do not infer annotation completeness from a leading M, remove X residues,
or transfer a protein between distinct genes sharing a symbol.

Put the decision in one public, documented `select_protein_transcript` function
exported from topiary. Empty/missing proteins do not compete. Exact ties retain
input order, preserving the old policy. With no complete-start alternative,
retain the longest available protein and warn with the gene/transcript identity
(the compatibility-preserving default while awaiting user preference).
No protein yields None.
The source loader remains the shared route for gene names, gene IDs, CTA genes
and tissue genes. Explicit transcript lookups remain exact requests.

Protein-change inputs first restrict candidates by a requested residue at a
specific coordinate and have separate nucleotide-length/name tie-breakers;
changing that mutation interpretation is outside this gene-source fix.
Whole-proteome and non-CTA exclusion sets enumerate all proteins, so this
representative-selection policy does not apply to them.

Source: https://www.gencodegenes.org/pages/tags.html defines cds_start_NF as
an unconfirmed coding-region start and describes preferring full-length over
partial transcripts for representative sets. This is a start-annotation
preference, not a claim of MANE/canonical status or OncoRef target qualification.

Ensembl 115 observed primary-locus choices, old -> complete-start alternative:
- IZUMO2 / ENSG00000161652: ENST00000600293 (234 aa, X...) ->
  ENST00000293405 (221 aa, M...).
- C16orf90 / ENSG00000215131: ENST00000399645 (191 aa, X...) ->
  ENST00000437192 (182 aa, M...).
- LLCFC1 / ENSG00000165131: ENST00000458732 (153 aa, X...) ->
  ENST00000409607 (122 aa, M...). Its alternate-locus gene ENSG00000275821
  remains distinct when resolving the shared LLCFC1 symbol.

## Review

Before implementation: 12 new regressions fail and 5 preservation cases pass.
After implementation: 30 source/regression tests pass; the expanded selector,
four-source conformance and CLI-to-prediction checks pass (27 targeted cases).
Lint (Ruff and mypy) and pip check pass in the isolated repository environment.

Version 5.111.7 provisionally follows the existing #591/#592/#593 queue
(5.111.4–5.111.6). Do not merge or renumber those unrelated PRs implicitly.
Recheck the queue before publication.
