# Quickstart

## CLI

```bash
# From a FASTA of protein sequences
topiary --fasta proteins.fasta \
  --mhc-predictor netmhcpan --mhc-alleles A0201,B0702 \
  --ranking "affinity <= 500 | presentation.rank <= 2" \
  --output-csv results.csv

# From specific genes
topiary --gene-names BRAF TP53 EGFR \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --ranking "affinity <= 500"

# CTA proteins, excluding vital organ peptides
topiary --cta \
  --exclude-tissues heart_muscle lung liver \
  --mhc-predictor netmhcpan --mhc-alleles A0201

# From genomic variants (VCF)
topiary --vcf somatic.vcf \
  --mhc-predictor netmhcpan --mhc-alleles A0201 \
  --ranking "affinity <= 500" \
  --only-novel-epitopes
```

## Python API

### Basic prediction

```python
from topiary import TopiaryPredictor, Affinity
from mhctools import NetMHCpan

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["A0201"],
    filter=Affinity <= 500,
)
df = predictor.predict_from_sequences(["MASIINFEKLGGG"])
```

### Multiple models

```python
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["A0201", "A0301", "B0702"],
    filter=Affinity <= 500,
    rank_by=Presentation.score,
)
```

### Filter expressions

```python
from topiary import Affinity, Presentation, Stability

# Simple
Affinity <= 500
Affinity.rank <= 2.0
Presentation.score >= 0.5

# Combine with | (OR) and & (AND)
(Affinity <= 500) | (Presentation.rank <= 2.0)
(Affinity <= 500) & (Presentation.score >= 0.5)

# Three-way
(Affinity <= 500) | (Presentation.rank <= 2) | (Stability.score >= 0.5)
```

### Composite ranking with Gaussian normalization

```python
score = (
    0.5 * (1 - Affinity.value.norm(mean=500, std=200))
    + 0.5 * Presentation.score.norm(mean=0.5, std=0.3)
)

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["A0201"],
    filter=Affinity <= 500,
    rank_by=score,
)
```

### Transforms

```python
Affinity.value.clip(lo=1, hi=50000).log()  # log-transform IC50
abs(Affinity.value)                         # absolute value
Affinity.score ** 2                         # power
Affinity.value.sqrt()                       # square root
```

### Input formats

```python
from topiary.inputs import read_fasta, read_peptide_csv, read_peptide_fasta, read_sequence_csv, slice_regions

# Protein FASTA (scanned with sliding window)
seqs = read_fasta("proteins.fasta")

# Peptide FASTA (each entry predicted as-is)
seqs = read_peptide_fasta("peptides.fasta")

# CSV with 'peptide' column
seqs = read_peptide_csv("peptides.csv")

# CSV with 'sequence' column
seqs = read_sequence_csv("proteins.csv")

# Region slicing (half-open intervals)
sliced = slice_regions(seqs, {
    "spike": [(319, 541)],           # RBD only
    "nucleocapsid": [(0, 50), (350, 419)],
})
```

### Ensembl and tissue lookups

```python
from topiary.sources import (
    sequences_from_gene_names,
    tissue_expressed_sequences,
    cta_sequences,
    available_tissues,
)

# Specific genes
seqs = sequences_from_gene_names(["BRAF", "TP53", "EGFR"])

# Tissue-expressed genes
seqs = tissue_expressed_sequences(["testis"], min_ntpm=10.0)

# CTA genes (curated list from PirlyGenes)
seqs = cta_sequences()

# List available tissues
print(available_tissues())
```

### Exclusion

```python
from topiary.inputs import exclude_by
from topiary.sources import tissue_expressed_sequences

vital = tissue_expressed_sequences(["heart_muscle", "lung", "liver"])

# Substring mode: heart 8-mer inside CTA 9-mer → excluded
df = exclude_by(df, vital, mode="substring")

# Exact mode: only full peptide matches
df = exclude_by(df, vital, mode="exact")
```
