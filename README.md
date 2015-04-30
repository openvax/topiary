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
  --rna-transcript-fpkm-file isoforms.fpkm_tracking
  --output epitopes.csv
```
