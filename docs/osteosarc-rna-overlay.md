# RNA alongside the historical pVAC predictions

The new RNA layer is **not a repair of the original pVAC files**. It annotates
copies of their Topiary imports with January 2025 UCLA resection RNA evidence.
Prediction scores, historical missing values, original tiers and empty filtered
reports are unchanged. No binding predictions are rerun.

This implements [#342](https://github.com/openvax/topiary/issues/342).
The question of why the old annotation step left fields empty remains
[#339](https://github.com/openvax/topiary/issues/339).

## Inputs and interpretation

- The public January UCLA RSEM gene/isoform files provide **TPM estimates**,
  not counts of variant-supporting reads. Match Ensembl IDs explicitly; keep
  their versions and distinguish an exact HGVSc transcript-version match from
  a unique stable-ID-only match. Ambiguous stable IDs are rejected, not summed.
- The corresponding STAR genomic BAM names a GRCh38/GENCODE v36 reference.
  Retrieve its index and only windows around the 20 pVAC alleles. Verify each
  reference allele against GRCh38; do not confuse a pVAC Start with a VCF POS.
  This panel has SNVs and anchored deletions only.
- The source collection labels and dates establish the intended T2 pairing.
  This is not an independent DNA/RNA sample-fingerprint validation. Do not
  silently pool with T1, ONT or separately reprocessed alignments.
- Isovar counts reference, alternate and other **sequenced segments** under
  the recorded policy: MAPQ >= 20, secondary placements excluded, marked
  duplicates excluded, soft clips excluded, overlapping mates merged for
  allele assignment but segment identity retained for read counts. These are
  not independent molecule counts. The original STAR BAM is not claimed to
  have undergone duplicate marking.
- VAF is alternate / (reference + alternate + other). At zero usable depth,
  VAF is missing. Locus support does not prove that a particular transcript,
  assembled protein or peptide is produced. Gene/transcript TPM is not
  allele-specific expression.

The [RSEM output documentation](https://deweylab.github.io/RSEM/rsem-calculate-expression.html)
defines TPM and expected counts; the [pVACtools input documentation](https://pvactools.readthedocs.io/en/stable/pvacseq/input_file_prep.html)
describes the separate expression/coverage annotations. The study's coordinate
mapping follows the [pinned pVACtools 5.3.0 converter](https://github.com/griffithlab/pVACtools/blob/v5.3.0/pvactools/lib/input_file_converter.py),
with independent reference checks before counting.

## Reproduce with local caching

From this checkout, with `topiary[isovar]`, requests and samtools available:

```sh
python -m scripts.osteosarc_rna_overlay acquire --root .cache/osteosarc-rna-overlay-2026-09-17
python -m scripts.osteosarc_rna_overlay build --root .cache/osteosarc-rna-overlay-2026-09-17
python -m scripts.osteosarc_rna_overlay apply --root .cache/osteosarc-rna-overlay-2026-09-17 --archive .cache/osteosarc-pvacseq-2026-09-16
```

The archive is the separately retained 21-file historical import. Acquisition
retains complete expression files, source index, regional BAM/index, BAM header,
reference responses, URLs and SHA256 receipts. Reuse verifies hashes; generated
enriched reports are not overwritten. The output includes:

- `allele-evidence.tsv`: one row per exact genomic allele.
- `transcript-evidence.tsv`: the 63 original variant/transcript input rows plus
  new evidence, preserving the source's empty historical fields.
- `topiary/`: all 21 enriched imported reports in their original hierarchy.
- `acquisition.json`, `provenance.json`, `overlay-manifest.json`: acquisition,
  interpretation and verified save/reload receipts.

## Use the added evidence

```python
from topiary import read_tsv

result = read_tsv("path/to/enriched/report.tsv")
supported = result.filter_by("(rna_t2_alt_reads >= 3) & (rna_t2_gene_tpm >= 1)")
```

New columns use `rna_t2_`: `gene_tpm`, `transcript_tpm`, `alt_reads`, `ref_reads`,
`other_reads`, `depth_reads`, `vaf`, source IDs and matching/coverage status.
These do not replace canonical or `pvacseq_*` evidence. Original pVAC tiers
remain historical tiers; applying a new RNA filter does not rerun pVAC ranking.

For another validated source, the public `join_annotations` function performs
an exact many-to-one join on explicit keys, prefixes every added column, rejects
duplicates and collisions, and preserves provenance in `annotation_overlays`.
It does not infer equivalence between samples, references, genes or transcripts.

The recount composes Isovar's public coordinate-level collection and allele
classification functions, without downloading gene annotations. The separate
variant-level entry point's unnecessary annotation lookup for logging is
tracked in [Isovar #295](https://github.com/openvax/isovar/issues/295).
