# Topiary
Predict mutation-derived cancer T-cell epitopes from (1) somatic variants (2) tumor RNA expression data, and (3) patient HLA type.

## Example

```sh
/topiary 
  --vcf somatic.vcf  
  --mhc-pan 
  --mhc-alleles-file patient_hla.txt 
  --filter-ic50 500 
  --filter-percentile 2.0 
  --mhc-epitope-lengths 8-11 
  --rna-gene-fpkm-file genes.fpkm_tracking 
  --rna-gene-expression-threshold 4.0
  --rna-transcript-fpkm-file isoforms.fpkm_tracking
  --rna-transcript-expression-threshold 1.5
  --output epitopes.csv
```
