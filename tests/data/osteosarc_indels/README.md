# GLIS3 / KTN1: exact-allele RNA reconstruction

GRCh38 anchored alleles from the user's 2026-09-17 candidate report:

- GLIS3: chr9:3856149 CTGATGTGG>C (8-nt deletion).
- KTN1: chr14:55627965 G>GTT (2-nt insertion).

Neither occurs in the archived pVACseq final reports. Missing predictions are
not evidence of absent RNA. These fixtures test reconstruction and prediction
inputs separately from those historical files.

## Original reads and independent reference

Eight gzipped SAMs retain complete original records (sequence, qualities,
CIGAR, flags and tags) overlapping the two loci, for T1/T2 deduplicated ONT
and oncoanalyser short-read RNA. No read is fabricated, trimmed, or selected
because it supports the alternate allele. Both secondary-placement policies
run against the same records; reference and other-allele reads are retained.
The manifest pins source URLs, original regional-BAM receipts/hashes,
zero-based half-open extraction intervals and SAM hashes. The parent audit is
[Isovar #291](https://github.com/openvax/isovar/pull/291), commit
db233f251b4dd965451db7cb3207162465c83c4a.

Two original Ensembl 87 transcript records provide a compact coding oracle:
GLIS3 ENST00000324333 and KTN1 ENST00000395308. This is not all-isoform coverage.
The helper verifies reference cDNA translation against original protein FASTA,
applies the exact genomic edit using GTF exon coordinates and strand, and
checks reconstructed sequence, changed interval and transcript. It does not
derive its expected protein from Varcode or Isovar.

Coordinate and translation sources:
[Ensembl GTF](https://www.ensembl.org/info/website/upload/gff.html),
[NCBI standard genetic code](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi#SG1),
[SAM segment/alignment semantics](https://samtools.github.io/hts-specs/SAMv1.pdf).

## Findings tested through Topiary's public interfaces

| Allele / RNA | Default ref / alt / other templates | Primary-only ref / alt / other | Reconstructed context |
| --- | --- | --- | --- |
| GLIS3 T1 ONT | 2 / 0 / 0 | 2 / 0 / 0 | none |
| GLIS3 T2 ONT | 1 / 0 / 1 | 1 / 0 / 1 | none |
| GLIS3 T1 short | 32 / 9 / 1 | 31 / 9 / 1 | 46 aa |
| GLIS3 T2 short | 0 / 0 / 0 | 0 / 0 / 0 | none |
| KTN1 T1 ONT | 636 / 0 / 1 | 636 / 0 / 1 | none |
| KTN1 T2 ONT | 1388 / 10 / 3 | 1388 / 10 / 3 | 29 aa |
| KTN1 T1 short | 843 / 0 / 0 | 842 / 0 / 0 | none |
| KTN1 T2 short | 301 / 3 / 0 | 290 / 3 / 0 | 18 aa |

Counts are RG/QNAME templates, not proven independent molecules. Zero alternate
support in a queried library does not establish absence in the tumor.
Isovar 1.18.1 corrects insertion-boundary assignments: one-sided reads cannot
prove the reference, and aligned bases across CIGAR D cannot prove an insertion.
This removes false reference/other evidence and recovers a falsely conflicted
KTN1 alternate template. Earlier 1.17 counts were 637/0/4 and 1388/10/8 for
T1/T2 ONT, and 302/2/1 (default) or 292/3/0 (primary-only) for T2 short RNA.
These are corrections on unchanged original SAMs, not newly acquired evidence.

Both assembly modes reconstruct the same independently checked contexts with
the unchanged absolute coverage floor of two and a 25-aa context objective.
The KTN1 frameshift terminates after five changed amino acids. Its 18-aa
short-read context is too short for a 25-mer, but supplies mutation-overlapping
9-mer prediction inputs. Tests exercise actual fragment serialization and
Topiary peptide selection; a random predictor tests transport, not MHC affinity.

**Reconstruction is not acceptance.** GLIS3 T1 short passes the normal filters.
With the corrected 1.18.1 counts, KTN1 T2 ONT and T2 short also pass (10 versus
3 and 3 versus 0 alternate/other templates respectively) under both placement
policies. Earlier rejections reflected the counting defect, not a biological
contradiction. Diagnostic and default-acceptance tests remain separate; the
full-index corpus also checks nine genuinely filtered reconstructions. No
default is weakened and no counts are backfilled into historical pVAC reports.

## Recreate the assets

Run the checked-in regeneration script with the #291 regional-BAM acquisition
directory and an original Ensembl 87 cache directory. It verifies each source
BAM hash before extraction and records full reference-source hashes. CI uses
only checked-in assets and creates private indexes in temporary directories;
collection and Isovar tests are also exercised with network access forbidden.
