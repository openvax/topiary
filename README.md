# Topiary
Predict mutation-derived cancer T-cell epitopes from (1) somatic variants (2) tumor RNA expression data, and (3) patient HLA type.

## Example

```sh
./topiary \
  --vcf somatic.vcf \
  --mhc-pan \
  --mhc-alleles HLA-A*02:01,HLA-B*07:02 \
  --ic50-cutoff 500 \
  --percentile-cutoff 2.0 \
  --mhc-epitope-lengths 8-11 \
  --rna-gene-fpkm-file genes.fpkm_tracking \
  --rna-min-gene-expression 4.0 \
  --rna-transcript-fpkm-file isoforms.fpkm_tracking \
  --rna-min-transcript-expression 1.5 \
  --output-csv epitopes.csv \
  --output-html epitopes.html
```
