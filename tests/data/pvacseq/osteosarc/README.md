# Offline osteosarc pVACseq regression corpus

Original, byte-preserved rows from all **21** final pVACseq reports published
under the public osteosarc dataset's `neoantigen_prediction/pvactools/` prefix.
The full archive was acquired and SHA256-verified on 2026-09-16. `manifest.json`
records original URLs, full-file digests and row counts, fixture digests,
one-based source data-row indexes, and individual row digests.

The full reports contain 131,209 rows across overlapping views. This subset
contains **454 rows** and all **20 genomic alleles represented in those reports**,
not every variant on osteosarc.com. All aggregate rows and the seven header-only
filtered reports are retained. All-epitopes selection keeps the aggregate's
exact best candidate, every genomic allele, HLA allele and variant type, and
examples of missing/present/zero metrics. No source value is synthesized.

GTF3C5 `chr9:133057893 GGAGGAGGAGGAA>G` and RNF213 `chr17:80327830 ATAC>A`
are present. GLIS3 `chr9:3856149 CTGATGTGG>C` and KTN1 `chr14:55627965 G>GTT`
are absent. Rearrangements and larger structural-deletion leads require
separate RNA reconstruction audits, not inference from these predictions.

## CI checks

- Identity, RNA/DNA evidence, expression, MT/WT affinity and rank, individual
  MHCflurry/NetMHCpan/NetMHCIIpan measurements, and native presentation,
  processing and immunogenicity values.
- Original-to-import and import-to-save/reload legs, including null masks,
  empty tables, metadata and mhcgnomes allele normalization. Numeric values use
  `rtol=atol=1e-12`: semantic preservation, not decimal-format identity.
- Aggregate versus all-epitopes agreement on each selected candidate, including
  all repeated April 25 NAV2 rows under distinct input indexes.
- A composed workflow where model choice and filtering threshold actually
  change selected peptide/allele groups, unchanged by save/reload.

Tests are offline and run no predictors or RNA reference downloads. They do
not establish predictor accuracy or candidate suitability. Unrecorded predictor
releases stay missing rather than being labeled as current.

## Why historical RNA annotations are missing

The retained April 27 **variant input** (63 rows, also pinned here) already has
literal `NA` for `trna_depth`, `trna_vaf`, `gene_expression` and
`transcript_expression`. All original final reports share that absence. This
is not a Topiary import loss, and does not mean RNA sequencing is unavailable.

The run's retained `log/inputs.yml` names pVACtools **5.3.0** and a VCF ending in
`.vep.and.expression.vcf.gz`. That release's
[converter](https://github.com/griffithlab/pVACtools/blob/v5.3.0/pvactools/lib/input_file_converter.py)
reads sample FORMAT tags `RDP` and `RAF` (or `RAD`/`RDP`) for RNA coverage/VAF,
and `GX`/`TX` matching gene/transcript identifiers for expression. It does not
calculate these from BAMs. The [official input instructions](https://pvactools.readthedocs.io/en/stable/pvacseq/input_file_prep.html)
describe upstream annotation.

The exact original VCF was absent from the public manifest inspected on
2026-09-17. Its filename does not establish that annotations were present,
correctly tagged, or assigned to the selected sample. Distinguishing these
causes requires that VCF and annotation logs. New T1/T2 evidence must remain
sample-specific, not be backfilled into a historical specimen by gene name.
Empty filtered reports do not establish which filters removed the rows.

## Regeneration

With the complete verified archive available, from the repository root:

```sh
python tests/data/pvacseq/osteosarc/regenerate.py /path/to/osteosarc-pvacseq-2026-09-16
```

The script verifies source hashes and extracts original bytes. It neither
imports Topiary nor uses its output to manufacture expected values.
