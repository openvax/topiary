# Osteosarc RNA: indels and rearrangements

This is a source-pinned reconstruction audit, not a claim of peptide binding,
tumor specificity, clinical suitability, or coverage of every website variant.
Historical pVACseq reports and newer RNA evidence remain separate.

## Historical pVACseq annotation gap

The 21 archived final reports represent 20 genomic alleles. They lack RNA
depth, RNA VAF and gene expression already in the source. The retained April 27
variant input has 63 rows, all missing those annotations before prediction.
RNA sequencing exists; those quantities were not populated into this historical
prediction input. Recovering the original annotated VCF and checking sample
FORMAT tags and gene/transcript matching is tracked in
[Topiary #339](https://github.com/openvax/topiary/issues/339).
See [pVACseq coverage](pvacseq.md) for the offline selected-column tests.

## Exact small indels

All coordinates below are GRCh38 VCF anchors; counts are RG/QNAME templates,
not proven independent molecules. RNA products and samples are not pooled.

| Candidate | Exact allele | Evidence and status |
| --- | --- | --- |
| GTF3C5 | chr9:133057893 GGAGGAGGAGGAA>G | Present in historical pVAC reports and existing RNA regression fixtures. Additional default-policy T1/T2 ONT support is 117/128; short-read support 7/12. |
| RNF213 | chr17:80327830 ATAC>A | Present in historical pVAC reports. Additional T1/T2 ONT support 5/14; short-read support 42/11. |
| GLIS3 | chr9:3856149 CTGATGTGG>C | Absent from historical pVAC reports. Nine alternate T1 short-read templates; independently checked 46-aa RNA context. |
| KTN1 | chr14:55627965 G>GTT | Absent from historical pVAC reports. T2 ONT 10 alternate templates, 29-aa context; T2 short reads 2 by default or 3 under primary-only collection, 18-aa context. |

GLIS3 passes default result filters. KTN1 ONT fails the alternate-to-other
template ratio (10/8), as do default-policy KTN1 short reads (2/1).
Primary-only short-read collection passes (3/0). A reconstructed sequence is
therefore not automatically an accepted pipeline result. A short context can
supply mutation-overlapping 9-mers without being long enough for a 25-mer.
No default threshold has been relaxed.

## Actual rearrangement alignment paths

The extraction correction already shipped in
[Isovar #291 / 1.17.4](https://github.com/openvax/isovar/pull/291).
It uses **observed partner SAM records and their full CIGARs**. An SA tag
declares an alignment relationship; it is not a substitute for a missing
record, nor a reliable source of exact base-level mapping.
The [SAM specification](https://samtools.github.io/hts-specs/SAMv1.pdf)
defines primary/supplementary records, strand flags and clipping.

The follow-up matched every original record for **13 complete paths** against
the retained tagged-BAM regional acquisitions, verifying each acquisition hash.
The three selected input files are pinned byte-for-byte with their original
SAM/partner SAM records, source-query offsets and Ensembl 87 reference models.

| Rearrangement / sample | All complete paths | Distinct cell/UMI labels | Selected identical RNA window | Coding result |
| --- | ---: | ---: | --- | --- |
| GABBR1–SLC29A1 T1 | 2 | 2 | 120 nt; direct junction | unresolved frame |
| GABBR1–SLC29A1 T2 | 3 | 3 | 120 nt; direct junction | unresolved frame |
| OTUD7A–FMN1 T2 | 8 | 6 | 122 nt; AG insertion; 4 selected observations | unresolved frame |

The selected OTUD7A window contains **AG** in its displayed minus/plus
orientation. Do not substitute the DNA catalogue's CT spelling without
accounting for orientation and the observed RNA sequence. The four observations
supporting that exact selected window are not the eight total complete paths.
Cell/UMI labels are not independently validated molecule counts.

Both inputs return `unresolved_frame`, with reason
`no_exact_collinear_annotated_donor`, and **no translations** under the pinned
Ensembl 87 models. This means the observed RNA junction is not yet connected to
a supported coding frame; it does not mean the RNA junction is absent.
Do not guess a reading frame or translate all frames and label the result
RNA-supported. A next coding-reconstruction step needs a validated transcript
path and CDS/frame relationship.

Deduplicated ONT predecessors can retain SA declarations without the actual
partner records. Their lack of a complete observed path cannot establish
absence of fusion RNA. Tagged and deduplicated products are processing-related
and must not be counted as independent additional support.

## Offline checks and provenance

The [checked-in fixture manifest](https://github.com/openvax/topiary/blob/master/tests/data/osteosarc_rearrangements/manifest.json)
pins source-BAM receipts, 13 path identities and original-record hashes, and
three selected JSON input hashes. CI checks the **nine selected observations**:
paired-record identity, cell/UMI agreement, source sequence at the recorded
offset, junction insertion, evidence counts and continued absence of a coding
translation. This is not a rerun of all-source BAM extraction in CI.
Upstream [full-path tests](https://github.com/openvax/isovar/blob/db233f251b4dd965451db7cb3207162465c83c4a/tests/test_osteosarc_footprints.py)
recount the pinned original paths and validate their actual CIGAR-derived windows.

The local regeneration script verifies full source acquisitions before copying
inputs. No sequence is invented and no historical pVAC row is backfilled:

```sh
python tests/data/osteosarc_rearrangements/regenerate.py /path/to/2026-09-17_05-32-18-330588Z
```

## Larger leads remain separate

DLG5's event affects the canonical start region; its full DNA junction contains
additional sequence, so the nominal 79.5-kb deletion is insufficient to specify
the complete allele. The sequence-resolved follow-up is
[Isovar #292](https://github.com/openvax/isovar/issues/292).
AFF3's 128-bp and KEAP1's 211-bp deletions do not overlap coding-transcript exons
in the Ensembl 87 models checked. Ordinary RNA splice skips across those
intronic intervals do not distinguish deleted from wild-type alleles.
None should receive a guessed coding consequence from these footprints.
