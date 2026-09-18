# Every tracked osteosarc variant has an explicit outcome

The 2026-09-17 snapshot covers **all 182 entries** in the
[website index](https://osteosarc.com/variants/), not just its 44
vaccine-nominated entries. The union with the 20 historical pVAC alleles and
the additional exact indels adds ACSL6 and KTN1: **184 entries**. Distinct
FCGBP alleles remain distinct. This is not a new genome-wide variant-calling
analysis or a statement that nomination establishes tumor specificity.

## What ran

All 174 supplied literal GRCh38 alleles were checked against the genomic
reference and run through Isovar using original regional records from the
January-2025 UCLA T2 STAR BAM. Complete original Ensembl 87 transcript records
at these loci (751 transcripts) are retained as a compact reference subset.
Ten incomplete entries remain explicitly unavailable, not zero support.

The read policy excludes secondary and duplicate-flagged alignments, requires
MAPQ 20, excludes soft clips and merges overlapping paired templates. It does
not assert that the original STAR BAM was duplicate-marked. Read segments and
RG/QNAME templates are reported separately, neither as independent molecules.
Sample pairing follows public collection labels, not an independent genotype
fingerprint. Protein assembly requests context for a 25-aa objective (up to
49 aa); Isovar's result filters are unchanged.

| T2 outcome | Entries |
| --- | ---: |
| RNA protein reconstructed, passes recorded filters | 27 |
| RNA protein reconstructed, rejected by named filters | 9 |
| Usable reads but no alternate reads | 82 |
| No usable reads under this policy | 43 |
| Alternate evidence but no reconstructed protein | 12 |
| Noncoding predicted consequence, no reconstructed protein | 1 |
| Incomplete literal allele | 10 |

A negative in this T2 library is **not** absence across samples or technologies.
The 36 reconstructed sequences have independent checks against original
GTF/FASTA and observed RNA: strand, frame, amino acids, stop and edit interval.
The oracle allows additional RNA changes instead of substituting an isolated
reference-derived edit for the observed haplotype. MT-ND5 uses the vertebrate
mitochondrial code; this assumes mitochondrial origin and does not exclude
nuclear mitochondrial insertions (NUMTs). See the
[NCBI genetic codes](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi)
and [Ensembl GTF specification](https://www.ensembl.org/info/website/upload/gff.html).

The 27 passing observations produced **5,790 new mutation-overlapping class-I
affinity rows**, MHCflurry `2.2.1+release-2.2.0`, using the archived pVAC HLA
configuration and lengths 9–12. They are separately labelled predictions,
not measured presentation, tumor specificity, clinical eligibility or rewritten
historical pVAC results. Offline CI uses synthetic scores only for transport
and selection assertions; it does not claim those scores validate binding.

## The additional candidates are not forgotten

- **GTF3C5 and RNF213:** the T2 STAR pass reconstructs both deletions, but both
  fail the recorded alternate-to-other-template ratio filter. Diagnostic
  sequences remain available without entering the accepted prediction set.
- **GLIS3:** the pinned T1 short-read library reconstructs a 46-aa frameshift
  context and passes under both labelled placement policies. These are two
  analyses of one library, not independent supporting samples.
- **KTN1:** Isovar 1.18.1 corrects insertion-boundary assignments on the unchanged
  reads. T2 short RNA now has 3 alternate / 0 other templates and ONT has
  10 / 3, both passing under both placement policies. The earlier 2 / 1 and
  10 / 8 rejections reflected a counting bug. Protein sequences stay 18 and
  29 aa. The shorter context supplies 9–12mers, not a 25mer; no threshold is
  relaxed to make it pass.
- **GABBR1–SLC29A1 and OTUD7A–FMN1:** actual RNA paths are pinned in the
  [RNA path audit](osteosarc-rna-audit.md). Their coding frames remain unresolved;
  no protein is invented from a frame guessed at the breakpoint.
- **DLG5:** the deletion affects the canonical start. An internal-deletion
  peptide is not justified. The completed
  [sequence-resolved audit](https://github.com/openvax/isovar/pull/298) supports
  the DNA junction with original tumor alignments and assembled caller contigs;
  RNA junction and mutant coding frame remain unresolved. The simple nominal
  deletion/insertion fields do not reproduce the full observed haplotype.
- **AFF3 / KEAP1:** ordinary intron-spanning RNA does not discriminate their
  intronic deletion alleles from reference. Ensembl 87 coding-exon overlap is
  absent; KEAP1 overlaps a processed-transcript exon.
- **SPRED1, TGFBR2, TCF7L2:** remain unvalidated structural leads, with oriented
  allele, artifact or organoid-versus-patient provenance questions respectively.

The supplementary GLIS3/KTN1 run covers four libraries and two placement
policies per gene (16 labelled outcomes), producing 1,400 new affinity rows
for six passing observations. Different sample/policy observations have
different fragment IDs even when their sequences match. This exposed and fixed
[a silent overwrite](https://github.com/openvax/topiary/issues/345).

## What is missing for ten entries

The individual variant pages were also checked and cached, not just the VAF
table. Protein labels are not substitutes for a sequence-resolved allele.

| Entry | Source detail / remaining requirement |
| --- | --- |
| CABLES1 | `G>dup`, `NM_001100619:c.15_20dup`; resolve a versioned transcript and genomic duplication allele |
| GOLGA6L2 | `T>dup`, `NM_001304388:c.2158_2277dup`; resolve versioned reference and duplicated sequence |
| MUC3A | `A>dup`, `NM_005960:c.1352_1453dup`; resolve versioned reference and duplicated sequence |
| CCDC40 | `c.2851_2852ins(32)`; actual 32-nt inserted sequence is not supplied |
| DCHS2 | `c.4019-5_4075delins(60)`; actual 60-nt insertion and transcript accession clarification needed |
| GAPVD1 | `NM_015635:c.186-3_188delinsATTGG`; resolve versioned transcript and exact genomic deleted interval |
| USH2A | source `A>not_reported`, insertion label but p.Cys4728Phe; nucleotide annotation is incomplete/conflicting |
| COL3A1 | splice label without exact genomic allele |
| FAM157A | p.W70_Q71ins14 without exact genomic allele or inserted sequence |
| OTUD4 | p.A153del without exact genomic allele/transcript |

These are unresolved inputs, not demonstrated reconstruction failures or
evidence of biological absence. The original-run annotation-history question
[Topiary #339](https://github.com/openvax/topiary/issues/339) remains separate.

## Reproduce and inspect locally

`scripts/osteosarc_variant_audit.py` provides explicit `inventory`, `acquire`,
`reference`, `audit`, `diagnose`, `predict`, `additional`, `upstream` and `report` stages.
Run it from a checkout as `python -m scripts.osteosarc_variant_audit STAGE ROOT`;
the acquisition stage requires the cached original BAM index, reference
selection requires the original Ensembl 87 directory, and prediction requires
the archived class-I `inputs.yml`. Acquisition is opt-in and never a CI side
effect. `additional` reuses this checkout's original-read indel fixtures.
`upstream` pins the Isovar 1.18.1 structural/extended-RNA audit and compact
original-read evidence from its immutable release commit. That separate audit
queries 14 RNA products, preserving dependent processing families rather than
pooling them; it is not a 14-library re-run of every one of the 184 entries.

The current `.cache/osteosarc-all-variants-v181-2026-09-17/` contains source bytes and
SHA-256 receipts, regional BAM/index, references, per-entry outcomes, accepted
and filtered fragments, diagnostic stopping stages, genuine predictions and
the full source-linked `README.md`. The earlier 1.17.4 run is preserved separately;
current counts require the corrected optional Isovar floor of 1.18.1.
Historical pVAC files and the new RNA overlay
remain in their separate archives. The committed
`tests/data/osteosarc_all_variants/` corpus is about 7.7 MB and runs offline in
CI against both minimum and latest supported Isovar. Every literal allele
drives both diagnostic handoff doors and default acceptance; every accepted
fragment runs prediction → save/reload → filters that must change selection.
