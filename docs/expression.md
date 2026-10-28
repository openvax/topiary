# RNA expression annotation

Gene-, transcript- and variant-level quantification can be joined onto variant
predictions. The loaded values become DataFrame columns, so they are available
to [filter and ranking expressions](ranking.md) like any other column.

```bash
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression salmon_quant.sf \
  --filter-by "ba <= 500 & gene_tpm >= 5" \
  --only-novel-epitopes \
  --output-csv results.csv
```

Expression flags only work with the mutation pipeline (`--vcf`, `--maf`,
`--variant`, `--json-variants`, or `--protein-change`). They are rejected for
direct sequence inputs (`--fasta`, `--peptide-csv`) because that output has no
`gene_id`, `transcript_id` or `variant` column to join on.

In a run mixing genomic variants and protein changes, gene and transcript files
can annotate both input forms. Variant files annotate only rows whose
`mutation_input` is `genomic_variant`; protein-change rows have a null `variant`
and null variant-expression values. This is absence of a coordinate, not zero
support. Use `mutation_input` when a filter must give the two input forms
different admission rules.

## How a file becomes columns

Each `--gene-expression`, `--transcript-expression` and `--variant-expression`
flag takes a spec string, `[name:]file[:id_col[:val_col]]`, and is handled in
three steps:

1. The file's format is auto-detected from its name and header (see below), or
   it is read as generic TSV/CSV.
2. Its columns are joined onto the prediction DataFrame by `gene_id`,
   `transcript_id` or `variant`, according to which flag named it. Those columns
   are present in variant-pipeline output.
3. Value columns are prefixed with the name (default: `gene`, `transcript` or
   `variant`) and lowercased.

### Auto-detected formats

| Extension / pattern | Format | Default ID column | Default value column |
|---------------------|--------|-------------------|---------------------|
| `.sf` | Salmon | `Name` | `TPM` |
| `abundance*.tsv` (with `target_id` + `tpm` header) | Kallisto | `target_id` | `tpm` |
| `.genes.results` | RSEM (gene) | `gene_id` | `TPM` |
| `.isoforms.results` | RSEM (transcript) | `transcript_id` | `TPM` |
| `.gtf` | StringTie | `reference_id` | `TPM` (or `FPKM`) |
| `.fpkm_tracking` | Cufflinks | `tracking_id` | `FPKM` |
| anything else | generic | first column | all numeric columns |

Override any of these with the full spec, `name:file:id_col:val_col`.

### Column naming

The name prefix and the original column name are joined as `{prefix}_{column}`,
both lowercased:

| Flag | File | Original column | Result column |
|------|------|----------------|---------------|
| `--gene-expression quant.sf` | Salmon | `TPM` | `gene_tpm` |
| `--gene-expression quant.sf` | Salmon | `NumReads` | `gene_numreads` |
| `--transcript-expression abundance.tsv` | Kallisto | `tpm` | `transcript_tpm` |
| `--variant-expression isovar.tsv` | generic | `num_alt_reads` | `variant_num_alt_reads` |
| `--gene-expression mygene:quant.sf` | Salmon | `TPM` | `mygene_tpm` |

### Transcript selection

When transcript-level expression is supplied, topiary uses it to pick the
highest-expressed transcript per variant instead of its default priority-based
selection.

## More CLI examples

```bash
# Transcript-level from Kallisto, used in the ranking
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --transcript-expression kallisto/abundance.tsv \
  --filter-by "ba <= 500" \
  --sort-by "affinity.descending_cdf(500, 200) + 0.1 * transcript_tpm.log1p()" \
  --output-csv results.csv

# Variant-level read support from isovar
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --variant-expression isovar_output.tsv \
  --filter-by "ba <= 500 & variant_num_alt_reads >= 3" \
  --output-csv results.csv

# All three levels at once
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression salmon_quant.sf \
  --transcript-expression kallisto/abundance.tsv \
  --variant-expression isovar_output.tsv \
  --filter-by "ba <= 500 & gene_tpm >= 5 & variant_num_alt_reads >= 3" \
  --output-csv results.csv

# Genomic variants plus a coordinate-free hotspot. The evidence threshold
# applies to genomic rows; the explicit first clause retains the hotspot.
topiary \
  --variant 7 55259515 T G \
  --protein-change EGFR T790M \
  --genome grch37 \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --variant-expression isovar.tsv \
  --filter-by "mutation_input == 'protein_change' | variant_num_alt_reads >= 3" \
  --output-csv results.csv

# Explicit spec instead of auto-detection
topiary \
  --vcf somatic.vcf \
  --mhc-predictor netmhcpan \
  --mhc-alleles HLA-A*02:01 \
  --gene-expression mygene:custom_quant.tsv:ensembl_id:tpm_value \
  --output-csv results.csv
```

## From Python

`load_expression_from_spec` applies the same detection, and
`predict_from_variants` takes the loaded tables keyed by level:

```python
import pandas as pd
from topiary import TopiaryPredictor, Affinity, Column
from topiary.rna import load_expression_from_spec
from mhctools import NetMHCpan
from varcode import load_vcf

gene_name, gene_id_col, gene_df = load_expression_from_spec(
    "salmon_quant.sf", default_name="gene"
)
# gene_df has columns ["Name", "TPM"]; gene_name = "gene", gene_id_col = "Name"

expression_data = {
    "gene": [(gene_name, gene_id_col, gene_df)],
    "transcript": [],
    "variant": [],
}

predictor = TopiaryPredictor(
    models=[NetMHCpan],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500) & (Column("gene_tpm") >= 5.0),
)

variants = load_vcf("somatic.vcf")
df = predictor.predict_from_variants(variants, expression_data=expression_data)
# df now has a "gene_tpm" column from the Salmon file
```

Readers that import another pipeline's report bring their own expression and
read-support columns under canonical names; see the
[consumer guide](consumer-guide.md#rna-and-dna-evidence) for that vocabulary.
