# Ranking DSL

Topiary's ranking system uses composable expressions to filter and rank MHC binding predictions. Expressions are built with Python operators and evaluated lazily against groups of prediction rows.

## Prediction kinds

Each MHC prediction model produces one or more *kinds* of output. The built-in accessors are:

| Accessor | Kind | Description |
|----------|------|-------------|
| `Affinity` | `pMHC_affinity` | Binding affinity (IC50 nM) |
| `Presentation` | `pMHC_presentation` | Presentation score (EL) |
| `Stability` | `pMHC_stability` | pMHC complex stability |
| `Processing` | `antigen_processing` | Antigen processing score |

Each accessor has three fields:

| Field | Description | Example |
|-------|-------------|---------|
| `.value` | Raw value (e.g. IC50 nM) | `Affinity.value` |
| `.rank` | Percentile rank (lower = better) | `Affinity.rank` |
| `.score` | Normalized score (higher = better) | `Affinity.score` |

The default field is `.value`, so `Affinity <= 500` means `Affinity.value <= 500`.

## Filters

Create filters with comparison operators on fields:

```python
from topiary import Affinity, Presentation, Stability

# Simple filters
Affinity <= 500               # IC50 <= 500 nM
Affinity.rank <= 2.0          # percentile rank <= 2%
Presentation.score >= 0.5     # presentation score >= 0.5

# Combine with | (OR) and & (AND)
(Affinity <= 500) | (Presentation.rank <= 2.0)
(Affinity <= 500) & (Presentation.score >= 0.5)
```

## Transforms

Expressions support arithmetic and mathematical transforms:

```python
# Arithmetic
0.5 * Affinity.score + 0.5 * Presentation.score

# Gaussian CDF normalization -> maps to ~[0, 1]
Affinity.value.descending_cdf(mean=500, std=200)    # lower IC50 → higher output
Presentation.score.ascending_cdf(mean=0.5, std=0.3)  # higher score → higher output

# Logistic sigmoid (Vaxrank-compatible IC50 scoring)
# 1 / (1 + exp((x - midpoint) / width))
Affinity.logistic(midpoint=350, width=150)

# Other transforms
Affinity.value.clip(lo=1, hi=50000)    # clamp to range
Affinity.value.log()                    # natural log
Affinity.value.log10()                  # base-10 log
Affinity.value.sqrt()                   # square root
abs(Affinity.value)                     # absolute value
Affinity.value ** 2                     # power
```

## Multi-model disambiguation

When using multiple prediction models that produce the same kind (e.g. both NetMHCpan and MHCflurry produce `pMHC_affinity`), qualify with bracket syntax:

```python
from topiary import Affinity, Presentation

# Qualify by prediction method
Affinity["netmhcpan"] <= 500
Affinity["mhcflurry"].score
Presentation["mhcflurry"].rank <= 2

# Cross-model composite
score = (
    0.5 * Affinity["netmhcpan"].logistic(350, 150)
    + 0.5 * Affinity["mhcflurry"].logistic(350, 150)
)
```

**When only one model produces a kind, no bracket is needed.** `Affinity <= 500` works automatically. If you use it with multiple models producing the same kind, you get a clear error:

```
ValueError: Ambiguous: multiple models produce pMHC_affinity
(mhcflurry, netmhcpan). Use Affinity["modelname"] to disambiguate.
```

A typo in the method name also gives a helpful error:

```
ValueError: No pMHC_affinity predictions from method matching 'netmhcapn'.
Available: ['mhcflurry', 'netmhcpan']. Did you mean: ['netmhcpan']?
```

## Column() — arbitrary DataFrame columns

`Column()` reads any column from the predictions DataFrame, enabling peptide properties, variant metadata, or custom annotations as ranking signals:

```python
from topiary.ranking import Column

Column("cysteine_count")
Column("hydrophobicity") >= -0.5
Column("n_alt_reads").sqrt()

# In a composite score
score = (
    0.5 * Affinity.logistic(350, 150)
    - 0.2 * Column("cysteine_count")
    + 0.1 * Column("tcr_aromaticity")
)
```

If the column doesn't exist, you get a clear error with close-match suggestions:

```
ValueError: Column 'hydrophobicty' not found in DataFrame.
Did you mean: ['hydrophobicity']?
```

If the column contains non-numeric data:

```
TypeError: Column 'gene_name' contains non-numeric value 'BRAF' (str).
Only numeric columns can be used in ranking expressions.
```

## WT() — wildtype comparison

`WT()` wraps a kind accessor to read wildtype prediction columns (`wt_value`, `wt_score`, `wt_percentile_rank`). These columns are populated by `predict_column` after variant-derived predictions:

```python
from topiary import Affinity, WT

# Read WT binding values
WT(Affinity).value                        # wt_value column
WT(Affinity).score                        # wt_score column
WT(Affinity["netmhcpan"]).score           # qualified WT

# Differential binding (mutant vs wildtype)
Affinity.score - WT(Affinity).score

# Logistic differential
Affinity.logistic(350, 150) - WT(Affinity).logistic(350, 150)
```

!!! note
    `WT()` is for **ranking expressions only**, not filters. Use it in `rank_by`, not in `filter`. When WT columns don't exist (non-variant inputs), expressions evaluate to NaN.

## String form (CLI)

The `--ranking` flag and `--rank-by` flag accept string expressions:

| Python DSL | String form |
|---|---|
| `Affinity <= 500` | `affinity <= 500` or `ba <= 500` |
| `Affinity.rank <= 2` | `affinity.rank <= 2` |
| `Affinity.score >= 0.5` | `affinity.score >= 0.5` |
| `Affinity["netmhcpan"] <= 500` | `netmhcpan_affinity <= 500` or `netmhcpan_ba <= 500` |
| `Presentation["mhcflurry"].rank <= 2` | `mhcflurry_el.rank <= 2` |
| `Column("cysteine_count") <= 2` | `column(cysteine_count) <= 2` |
| `(A <= 500) \| (B.rank <= 2)` | `affinity <= 500 \| presentation.rank <= 2` |
| `(A <= 500) & (B.rank <= 2)` | `affinity <= 500 & presentation.rank <= 2` |

**Kind aliases:** `ba` / `aff` / `ic50` = Affinity, `el` = Presentation.

**Python-only** (no string form):

- Arithmetic: `0.5 * Affinity.score + 0.5 * Presentation.score`
- Transforms: `.logistic()`, `.ascending_cdf()`, `.descending_cdf()`, `.clip()`, `.hinge()`, `.log()`
- Aggregations: `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()`
- `WT()` expressions
- `Column()` in arithmetic (only `column(x) <= N` works in strings)

## Putting it together

```python
from topiary import (
    TopiaryPredictor, Affinity, Presentation, Column, WT,
)
from topiary.properties import add_peptide_properties
from mhctools import NetMHCpan, MHCflurry

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
    only_novel_epitopes=True,
)

# Predict
from varcode import load_vcf
df = predictor.predict_from_variants(load_vcf("somatic.vcf"))

# Add peptide properties
df = add_peptide_properties(df, groups=["manufacturability", "immunogenicity"])

# Composite score
score = (
    # Binding (average across models)
    0.25 * Affinity["netmhcpan"].logistic(350, 150)
    + 0.25 * Affinity["mhcflurry"].logistic(350, 150)
    # Presentation
    + 0.2 * Presentation["mhcflurry"].score
    # Manufacturability
    - 0.05 * Column("cysteine_count")
    - 0.05 * Column("instability_index").clip(lo=0, hi=100).ascending_cdf(50, 20)
    # Immunogenicity
    + 0.05 * Column("tcr_aromaticity")
)
```
