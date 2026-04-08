# Topiary

Predict cancer and viral epitopes from sequence data, with multi-model
support, composable ranking expressions, and tissue-aware exclusion.

## Installation

```bash
pip install topiary
```

For tissue-specific gene lists:

```bash
pip install pirlygenes
```

## Features

- **Multiple prediction models** — NetMHCpan, MHCflurry, NetMHCstabpan, etc.
- **Composable filter/rank expressions** — `Affinity <= 500`, `Presentation.rank <= 2.0`
- **Gaussian normalization** — `.norm(mean, std)` for composite scoring
- **Direct sequence inputs** — CSV, FASTA, gene names, Ensembl lookups
- **Tissue-aware exclusion** — exclude peptides from vital-organ proteomes
- **Variant-to-epitope pipeline** — VCF/MAF → protein effects → predictions

## Quick example

```python
from topiary import TopiaryPredictor, Affinity, Presentation
from topiary.sources import tissue_expressed_sequences
from topiary.inputs import exclude_by
from mhctools import NetMHCpan

# Predict from testis-expressed genes
targets = tissue_expressed_sequences(["testis", "placenta", "ovary"])

# Vital organ proteome for exclusion
vital = tissue_expressed_sequences(["heart_muscle", "lung", "liver"])

predictor = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["A0201", "A0301", "B0702"],
    filter=(Affinity <= 500) | (Presentation.rank <= 2.0),
    rank_by=Presentation.score,
)

df = predictor.predict_from_named_sequences(targets)
df = exclude_by(df, vital, mode="substring")
```
