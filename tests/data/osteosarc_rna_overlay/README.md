# January-2025 UCLA RNA alongside pVAC

Offline input for all 20 exact alleles in the 21 historical pVAC reports.

- `source/t2-pvac-regions.bam`: unchanged 684-KiB regional acquisition from the
  public original January UCLA STAR BAM, all records around all 20 loci.
  Its header records GRCh38/GENCODE v36 and the source RNA FASTQ names.
- `source/*.results`: unchanged original RSEM lines for 21 genes and 63
  transcripts in the retained pVAC input. The read-through NME1-NME2 gene
  explains why there are more genes than genomic alleles.
- `source/chr*.json`: original UCSC hg38 reference responses checking every
  VCF reference allele and the SNV/deletion coordinate conversion.
- `acquisition.json`: source URLs, full-source and selected-row hashes, row
  numbers, regional acquisition command and source reference checks.
- `allele-evidence.tsv`, `transcript-evidence.tsv`: pinned recount/join results;
  these are derived outputs, not an independent biological oracle.
- `manifest.json`: all fixture-file SHA256 hashes.

The full original expression files, 6.7-MB source BAM index and regional
BAM index remain in the local acquisition cache. CI rebuilds the regional
index in a temporary directory and recounts every allele offline. Selected
SNVs also have an independent CIGAR-aligned-base count check without Isovar.
Expression expectations come directly from the selected original RSEM lines.
Consumer tests run both report flavors, preserve old columns and missing values,
and verify that RNA filtering changes selection before and after save/reload.

Counts are sequenced segments, not independent molecules. The source BAM is
not claimed to be duplicate-marked. Counts support alleles at loci, not the
historical peptide sequences. The VAF denominator includes other alleles.
RSEM TPM is not allele-specific expression. See the public RNA overlay guide
for pairing, annotation versions and the complete processing policy.

Regenerate from a verified local cache:

```sh
python tests/data/osteosarc_rna_overlay/regenerate.py .cache/osteosarc-rna-overlay-2026-09-17
```
