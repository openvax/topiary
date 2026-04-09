# API Reference

## TopiaryPredictor

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | class, instance, or list | Predictor model(s). Classes require `alleles`. |
| `alleles` | list of str | HLA alleles. Used to construct model classes. |
| `filter` | EpitopeFilter or RankingStrategy | Which peptide-allele groups to keep. |
| `rank_by` | Expr or list of Expr | How to sort surviving groups. |
| `padding_around_mutation` | int | Residues around mutation for candidate epitopes. |
| `only_novel_epitopes` | bool | Drop peptides without mutated residues. |
| `min_gene_expression` | float | Minimum gene FPKM (variant inputs). |
| `min_transcript_expression` | float | Minimum transcript FPKM (variant inputs). |
| `raise_on_error` | bool | Raise on variant-effect errors vs. skip. |

### Prediction methods

| Method | Input | Behavior |
|--------|-------|----------|
| `predict_from_named_sequences(dict)` | `{name: sequence}` | Sliding-window scan |
| `predict_from_named_peptides(dict)` | `{name: peptide}` | Score as-is |
| `predict_from_sequences(list)` | `[sequence, ...]` | Sliding-window scan |
| `predict_from_variants(variants)` | VariantCollection | Full variant pipeline |
| `predict_from_mutation_effects(effects)` | EffectCollection | From pre-computed effects |

## Kind accessors

| Accessor | Kind | Default field |
|----------|------|---------------|
| `Affinity` | `pMHC_affinity` | `.value` (IC50 nM) |
| `Presentation` | `pMHC_presentation` | `.value` |
| `Stability` | `pMHC_stability` | `.value` |
| `Processing` | `antigen_processing` | `.value` |

### Fields

| Field | Column read | Description |
|-------|-------------|-------------|
| `.value` | `value` | Raw prediction value |
| `.rank` | `percentile_rank` | Percentile rank (lower = better) |
| `.score` | `score` | Normalized score (higher = better) |

### Multi-model bracket syntax

```python
Affinity["netmhcpan"].value       # filter to netmhcpan rows
Affinity["mhcflurry"].score       # filter to mhcflurry rows
```

Method matching is case-insensitive and substring-based. Errors with "Did you mean" on typos.

## Column

Reference any DataFrame column in expressions:

```python
Column("charge")                  # reads 'charge' column
Column("cysteine_count") <= 2     # creates ColumnFilter
```

Errors with close-match suggestions when column doesn't exist. Raises `TypeError` for non-numeric columns.

## ColumnFilter

Filter on arbitrary DataFrame columns. Created by `Column() <= N` or `parse_filter("column(name) <= N")`:

```python
ColumnFilter(col_name="cysteine_count", max_value=2)
ColumnFilter(col_name="hydrophobicity", min_value=-0.5)
```

Supports `|` (OR) and `&` (AND) combination with `EpitopeFilter`.

## WT

Wildtype comparison wrapper. Reads `wt_*` columns for ranking expressions:

```python
WT(Affinity).value                # reads wt_value
WT(Affinity).score                # reads wt_score
WT(Affinity["netmhcpan"]).score   # qualified WT
WT(Affinity)["netmhcpan"].score   # also works
```

For ranking expressions only (not filters). Returns NaN when WT columns absent.

## Expr transforms

| Method | Description |
|--------|-------------|
| `.left_cdf(mean, std)` | Gaussian CDF: higher input → higher output. Alias: `.norm()` |
| `.right_cdf(mean, std)` | 1-CDF: lower input → higher output (for IC50, rank) |
| `.logistic(midpoint, width)` | Logistic sigmoid: `1 / (1 + exp((x - midpoint) / width))` |
| `.clip(lo, hi)` | Clamp to range |
| `.hinge()` | `max(0, x)` — zeroes out negative values |
| `.log()` / `.log2()` / `.log10()` | Logarithm |
| `.log1p()` | `log(1 + x)`, accurate for small x |
| `.exp()` | Exponential |
| `.sqrt()` | Square root |
| `abs(expr)` | Absolute value |
| `expr ** n` | Power |
| `+`, `-`, `*`, `/` | Arithmetic between expressions and scalars |

## Aggregation functions

Combine multiple expressions, skipping NaN values:

| Function | Description |
|----------|-------------|
| `mean(a, b, ...)` | Arithmetic mean |
| `geomean(a, b, ...)` | Geometric mean (skips non-positive) |
| `minimum(a, b, ...)` | Minimum value |
| `maximum(a, b, ...)` | Maximum value |
| `median(a, b, ...)` | Median (mean of middle two for even count) |

```python
from topiary import mean, geomean, minimum

# Average binding across models
mean(Affinity["netmhcpan"].logistic(350, 150),
     Affinity["mhcflurry"].logistic(350, 150))

# Best binding across models
minimum(Affinity["netmhcpan"].value, Affinity["mhcflurry"].value)
```

## Filter expressions

| Expression | Creates |
|------------|---------|
| `Affinity <= 500` | `EpitopeFilter(kind=pMHC_affinity, max_value=500)` |
| `Affinity.rank <= 2` | `EpitopeFilter(kind=pMHC_affinity, max_percentile_rank=2)` |
| `Affinity.score >= 0.5` | `EpitopeFilter(kind=pMHC_affinity, min_score=0.5)` |
| `Column("x") <= 2` | `ColumnFilter(col_name="x", max_value=2)` |

Combine with `|` (OR) and `&` (AND). Chain with `.rank_by()`.

## String parsing

### Kind aliases

| Alias | Kind |
|-------|------|
| `affinity`, `ba`, `aff`, `ic50` | `pMHC_affinity` |
| `presentation`, `el` | `pMHC_presentation` |
| `stability` | `pMHC_stability` |
| `processing`, `antigen_processing` | `antigen_processing` |

### Tool-qualified kinds

Prefix with tool name and underscore: `netmhcpan_affinity`, `mhcflurry_el`, `netmhcpan_ba`.

### parse_filter examples

| String | Result |
|--------|--------|
| `"affinity <= 500"` | `EpitopeFilter(pMHC_affinity, max_value=500)` |
| `"netmhcpan_ba <= 500"` | `EpitopeFilter(pMHC_affinity, max_value=500, method="netmhcpan")` |
| `"mhcflurry_el.rank <= 2"` | `EpitopeFilter(pMHC_presentation, max_percentile_rank=2, method="mhcflurry")` |
| `"column(cysteine_count) <= 2"` | `ColumnFilter("cysteine_count", max_value=2)` |

### parse_ranking

Combines filters with `|` (OR) or `&` (AND):

```python
parse_ranking("affinity <= 500 | presentation.rank <= 2")
parse_ranking("netmhcpan_ba <= 500 & column(cysteine_count) <= 2")
```

Mixing `|` and `&` in one string is not supported; use the Python API for complex nesting.

## Input functions

| Function | Input | Returns |
|----------|-------|---------|
| `read_fasta(path)` | FASTA file | `{name: sequence}` |
| `read_peptide_fasta(path)` | FASTA of peptides | `{name: peptide}` |
| `read_peptide_csv(path)` | CSV with `peptide` col | `{name: peptide}` |
| `read_sequence_csv(path)` | CSV with `sequence` col | `{name: sequence}` |
| `slice_regions(seqs, regions)` | Sequences + intervals | `{name:start-end: subseq}` |
| `exclude_by(df, ref, mode)` | DataFrame + ref sequences | Filtered DataFrame |

## Source functions

| Function | Returns |
|----------|---------|
| `sequences_from_gene_names(names)` | `{GENE\|TRANSCRIPT: seq}` |
| `sequences_from_gene_ids(ids)` | `{GENE\|TRANSCRIPT: seq}` |
| `sequences_from_transcript_ids(ids)` | `{GENE\|TRANSCRIPT: seq}` |
| `sequences_from_transcript_names(names)` | `{GENE\|TRANSCRIPT: seq}` |
| `tissue_expressed_sequences(tissues)` | `{GENE\|TRANSCRIPT: seq}` |
| `tissue_expressed_gene_ids(tissues)` | `set` of Ensembl gene IDs |
| `cta_sequences()` | CTA protein sequences |
| `non_cta_sequences()` | Non-CTA protein sequences |
| `ensembl_proteome()` | All Ensembl proteins |
| `available_tissues()` | List of tissue names |

## Peptide properties

```python
from topiary.properties import add_peptide_properties, available_properties

# Add properties to predictions DataFrame
df = add_peptide_properties(df)                              # all
df = add_peptide_properties(df, groups=["core"])             # named group
df = add_peptide_properties(df, include=["charge"])          # specific
df = add_peptide_properties(df, peptide_column="wt_peptide", prefix="wt_")

# List available properties and their groups
available_properties()
```

Groups: `"core"`, `"manufacturability"`, `"immunogenicity"`. See [Peptide Properties](properties.md) for details.
