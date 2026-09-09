# Original RNA for peptide-aware Topiary integration tests

The unmodified alignment, selection, manifest and reference files are copied
from Isovar 1.8.0, commit `7bee9bef690a0184a41a0d8397879915dd86fe37`:
https://github.com/openvax/isovar/tree/7bee9bef690a0184a41a0d8397879915dd86fe37/tests/data/osteosarc

These are public CC0 data from https://osteosarc.com/data/ (the original AWS
license entry is preserved in `source_registry.yaml`). The subset contains
482 bulk STAR T0 and 242 single-cell long-read ONT T1 alignment records at six
loci. Source URLs, original-record provenance and uncompressed checksums are
preserved in `manifest.json`. Records, qualities, names and tags are unchanged.
These are deliberately selected small regression fixtures, not full-region
VAF estimates, independent-molecule counts or matched technical replicates.

`protein_reference/` pins six original Ensembl release-87 GTF/cDNA/protein
records and their checksums. Tests verify them and index them in temporary
directories under a distinct partial-reference identity. The independent
oracle verifies wild-type translation, applies the exact genomic allele to
raw cDNA (respecting exon coordinates and strand), and uses NCBI code table 1.
It does not use Isovar/Varcode predictions as expected protein sequences.

Primary sequence conventions:

- https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1
- https://samtools.github.io/hts-specs/SAMv1.pdf
- https://osteosarc.com/variants/

The tests in `test_consumer_workflows.py` exercise the public Topiary API,
real Isovar read collection, assembly, translation and ranking, followed by
fragment save/reload, peptide prediction and DSL filtering. MHC scores are
deliberately synthetic: these tests establish sequence/evidence handoff,
not binding accuracy or historical vaccine selection (Vaxrank #414).

Settings comparisons preserve the independent per-base floor and relative
compatible-name budget. Short contexts remain short; no reference padding is
silently appended. No-alt results remain absent RNA fragments, not claims
that mutant expression is absent. Explicit reference fallback has its own
padding and never inherits measured RNA counts or reconstruction provenance.
Tests perform no downloads and do not import sibling repositories.
