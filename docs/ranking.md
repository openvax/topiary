# Ranking DSL

Topiary's ranking system uses composable expressions — a single `DSLNode` tree — to filter and rank MHC binding predictions. Every expression evaluates vectorized across peptide-allele groups.

## Applying expressions

Two top-level entry points operate on a predictions DataFrame:

```python
from topiary import apply_filter, apply_sort, Affinity, Presentation

df = apply_filter(df, (Affinity <= 500) | (Presentation.rank <= 2.0))
df = apply_sort(df, [Presentation.score, Affinity.score])
```

`apply_filter` expects a boolean-valued expression (a `Comparison` or `BoolOp`). It errors if the evaluated Series contains values outside `{True, False, 0, 1, NaN}` — e.g. passing `Affinity.score` directly — pointing you at `<=` / `>=`.

`apply_sort` accepts a list of expressions as lexicographic tiebreakers. NaN values fall through to the next key instead of forcing an order.

`TopiaryPredictor(filter_by=..., sort_by=[...])` applies them automatically during prediction.

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

## Boolean-as-number composition

Comparisons return a boolean Series and still participate in arithmetic — `True` acts as 1, `False` as 0. This makes piecewise scoring natural:

```python
# Full score when strong binder, half score otherwise
(Affinity <= 500) * Affinity.score + (Affinity > 500) * 0.5 * Affinity.score

# Penalty for low expression
Affinity.score - 0.3 * (Column("gene_tpm") < 1)
```

Because the result is numeric (not boolean), don't pass it straight to `apply_filter` — wrap it with a comparison, e.g. `(... composite ...) >= 0.5`.

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

## wt. — wildtype comparison

The `wt.` scope prefix reads wildtype prediction columns (`wt_value`, `wt_score`, `wt_percentile_rank`). These columns are populated by `predict_column` after variant-derived predictions:

```python
from topiary import Affinity, wt

# Read WT binding values (Python API — capitalized kind names)
wt.Affinity.value                         # wt_value column
wt.Affinity.score                         # wt_score column
wt.Affinity["netmhcpan"].score            # qualified WT

# Differential binding (mutant vs wildtype)
Affinity.score - wt.Affinity.score

# Logistic differential
Affinity.logistic(350, 150) - wt.Affinity.logistic(350, 150)
```

The string DSL uses lowercase kind names:

```
wt.affinity.value
wt.affinity.score
wt.affinity["netmhcpan"].score
affinity.score - wt.affinity.score
```

!!! note
    `wt.` is for **sorting expressions only**, not filters. Use it in `sort_by`, not in `filter`. When WT columns don't exist (non-variant inputs), expressions evaluate to NaN.

## len and count() — peptide-level expressions

`len` reads the peptide length; `count('C')` counts amino acid occurrences in the peptide. Both compose with scope prefixes:

```python
# String DSL
len                           # peptide length
count('C')                    # cysteine count
wt.len                        # wildtype peptide length
wt.count('C')                 # wildtype cysteine count
count('C') - wt.count('C')   # gained/lost cysteines vs wildtype
count('KR') >= 2              # filter: at least 2 basic residues
```

## Method + version qualification

`Affinity["netmhcpan"]` filters to rows whose `prediction_method_name` contains the substring (case-insensitive). To disambiguate further, pass a tuple with an exact `predictor_version`:

```python
Affinity["netmhcpan", "4.1b"].value    # only NetMHCpan v4.1b rows
```

In the string DSL, both forms work:

```
affinity['netmhcpan'].value <= 500
affinity['netmhcpan', '4.1b'].value <= 500
```

## Parsing strings

A single `parse()` function takes a DSL string and returns a `DSLNode`:

```python
from topiary import parse, apply_filter

node = parse("affinity <= 500 | el.rank <= 2")
df = apply_filter(df, node)
```

`parse` handles the full grammar — arithmetic, comparisons, boolean combinators, transforms, aggregations, scoped fields.

## String form (CLI)

The `--filter-by` flag and `--sort-by` flag accept string expressions:

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

**All features work in both Python and CLI string form** (`--sort-by`):

- Arithmetic: `0.5 * affinity.score + 0.5 * presentation.score`
- Transforms: `.logistic()`, `.ascending_cdf()`, `.descending_cdf()`, `.clip()`, `.hinge()`, `.log()`
- Aggregations: `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()`
- `column(x)` in arithmetic, filters, and ranking
- Scope prefixes: `wt.affinity.score`, `wt.len`, `wt.count('C')`

`--sort-direction` controls whether lower or higher values rank first. The
default is `auto`: raw affinity values and percentile ranks sort ascending,
while all other sort expressions sort descending.

## Putting it together

```python
from topiary import (
    TopiaryPredictor, Affinity, Presentation, Column,
    apply_filter, apply_sort,
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

# Filter to plausible binders, then sort by composite score
df = apply_filter(df, (Affinity <= 500) | (Presentation.rank <= 2.0))
df = apply_sort(df, [score])
```
