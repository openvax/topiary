# API Reference

## TopiaryPredictor

| Parameter | Type | Description |
|-----------|------|-------------|
| `models` | class, instance, or list | Predictor model(s). Classes require `alleles`. |
| `alleles` | list of str | HLA alleles. Used to construct model classes. |
| `filter_by` | DSLNode or str | Boolean filter expression (e.g. `Affinity <= 500` or `"affinity <= 500 \| el.rank <= 2"`). |
| `sort_by` | DSLNode or list of DSLNode | Sort expression(s). Lexicographic tiebreakers; NaN falls through. |
| `sort_direction` | "auto", "asc", or "desc" | Direction for sort keys (auto infers per-key). |
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
| `predict_from_fragments(fragments)` | `[ProteinFragment]` | Universal path — any origin; fragment-level metadata and `target_intervals` threaded through. |
| `predict_from_variants(variants)` | VariantCollection | Variant pipeline (builds `ProteinFragment`s internally and delegates). |
| `predict_from_mutation_effects(effects)` | EffectCollection | Same as `predict_from_variants` but starting from pre-computed effects. |

## CachedPredictor

Drop-in replacement for a live predictor that serves scores from a
pre-computed table. Pass as `models=cache` to `TopiaryPredictor`. See
[Cached Predictions](cached.md) for full detail.

| Loader | Source |
|--------|--------|
| `CachedPredictor.from_topiary_output(path)` | Parquet / TSV / CSV previously written from a topiary run. |
| `CachedPredictor.from_mhcflurry(path)` | mhcflurry-predict CSV output. `predictor_version` auto-composed from the installed mhcflurry when omitted. |
| `CachedPredictor.from_netmhcpan_stdout(path)` | NetMHCpan stdout capture (auto-detects 2.8 / 3 / 4 / 4.1).  Returns every kind present in the output — `-BA` runs surface both `pMHC_affinity` and `pMHC_presentation` rows per (peptide, allele). |
| `CachedPredictor.from_netmhc_stdout(path, version=...)` | Classic NetMHC stdout (3 / 4 / 4.1). |
| `CachedPredictor.from_netmhcpan_cons_stdout(path)` | NetMHCcons stdout. |
| `CachedPredictor.from_netmhciipan_stdout(path, version=...)` | NetMHCIIpan stdout (legacy / 4 / 4.3). |
| `CachedPredictor.from_netmhcstabpan_stdout(path)` | NetMHCstabpan stdout (pMHC stability). |
| `CachedPredictor.concat([caches], on_overlap=...)` | Merge shards (all must share name+version). `on_overlap`: `"raise"` / `"last"` / `"first"` / callable. |
| `CachedPredictor.from_directory(path, pattern="*", on_overlap=...)` | Glob a dir and concat every matching file. |
| `CachedPredictor.from_tsv(path, columns=..., prediction_method_name=..., predictor_version=...)` | Generic tab- or comma-delimited. |
| `CachedPredictor.from_dataframe(df, ...)` | In-memory DataFrame. |
| `CachedPredictor(fallback=live_predictor)` | Empty cache, lazy identity discovery — pure read-through over a live model. |

Constructor-level knobs:

| Parameter | Description |
|-----------|-------------|
| `fallback` | Live predictor to route misses through. Result is merged back into the cache. |
| `also_accept_versions` | Set of version strings treated as interchangeable with the cache's own version (opt-in equivalence for rc → final, etc.). |

Helper: `topiary.mhcflurry_composite_version()` returns
`"<package_version>+release-<model_release>"` for the locally-installed
mhcflurry. Automatically used by `from_mhcflurry` when no explicit
`predictor_version` is passed.

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
Column("cysteine_count") <= 2     # returns a Comparison (boolean DSL node)
```

Errors with close-match suggestions when column doesn't exist. Raises `TypeError` for non-numeric columns.

## DSL tree

Every DSL expression is a `DSLNode`. The tree is composed of:

| Leaf | Produces |
|------|----------|
| `Const(v)` | Constant scalar |
| `Column(name)` | Per-group first-row of that column |
| `Field(kind, field, method=None, version=None, scope="")` | Per-group first-row of `{scope}{field}` for rows matching `kind` (and `method` / `version` if given) |
| `Len(scope="")` | Peptide length |
| `Count(chars, scope="")` | Amino-acid character count |

| Composite | Purpose |
|-----------|---------|
| `BinOp(left, right, op)` | Elementwise `+`, `-`, `*`, `/`, `**` |
| `UnaryOp(inner, fn)` | `abs`, `log`, `log2`, `log10`, `log1p`, `exp`, `sqrt` |
| `NormExpr`, `SurvivalExpr`, `LogisticExpr`, `ClipExpr`, `AggExpr` | Gaussian CDF / survival, logistic, clip, aggregate |
| `Comparison(left, op, right)` | `<=`, `>=`, `<`, `>`, `==`, `!=` → boolean Series |
| `BoolOp(op, children)` | `&`, `\|`, `~` over boolean children |

Boolean and numeric nodes compose freely — `(Affinity <= 500) * Affinity.score` is valid.

## wt (scope prefix)

Wildtype scope prefix. Reads `wt_*` columns for ranking expressions:

```python
# Python API (capitalized kind names)
wt.Affinity.value                 # reads wt_value
wt.Affinity.score                 # reads wt_score
wt.Affinity["netmhcpan"].score    # qualified WT

# String DSL (lowercase kind names)
# wt.affinity.value
# wt.affinity.score
# wt.affinity["netmhcpan"].score
```

For ranking expressions only (not filters). Returns NaN when WT columns absent.

## len and count()

Peptide-level expressions that compose with scope prefixes:

```python
# String DSL
len                           # peptide length (reads peptide_length column)
count('C')                    # cysteine count (reads from peptide column)
wt.len                        # wildtype peptide length (reads wt_peptide_length)
wt.count('C')                 # wildtype cysteine count (reads from wt_peptide)
```

## Expr transforms

| Method | Description |
|--------|-------------|
| `.ascending_cdf(mean, std)` | Gaussian CDF: higher input → higher output. Alias: `.norm()` |
| `.descending_cdf(mean, std)` | 1-CDF: lower input → higher output (for IC50, rank) |
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
| `Affinity <= 500` | `Comparison(Field(pMHC_affinity, "value"), <=, 500)` |
| `Affinity.rank <= 2` | `Comparison(Field(pMHC_affinity, "percentile_rank"), <=, 2)` |
| `Affinity.score >= 0.5` | `Comparison(Field(pMHC_affinity, "score"), >=, 0.5)` |
| `Column("x") <= 2` | `Comparison(Column("x"), <=, 2)` |
| `(A) \| (B)` | `BoolOp(\|, [A, B])` |
| `(A) & (B)` | `BoolOp(&, [A, B])` |

Apply with `apply_filter(df, node)` and `apply_sort(df, [nodes])`.

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

### parse

The unified DSL parser. Returns a `DSLNode`.

| String | Result |
|--------|--------|
| `"affinity <= 500"` | `Comparison(Field(pMHC_affinity, "value"), <=, 500)` |
| `"netmhcpan_ba <= 500"` | `Comparison(Field(pMHC_affinity, "value", method="netmhcpan"), <=, 500)` |
| `"affinity['netmhcpan', '4.1b'].value <= 500"` | `Comparison(Field(..., method="netmhcpan", version="4.1b"), <=, 500)` |
| `"column(cysteine_count) <= 2"` | `Comparison(Column("cysteine_count"), <=, 2)` |
| `"a <= 500 \| b <= 2"` | `BoolOp(\|, [Comparison(a), Comparison(b)])` |
| `"a <= 500 & b <= 2"` | `BoolOp(&, [Comparison(a), Comparison(b)])` |

Mixing `|` and `&` follows standard precedence (`&` binds tighter than `|`); use parentheses for the other grouping.

## Input functions

| Function | Input | Returns |
|----------|-------|---------|
| `read_fasta(path)` | FASTA file | `{name: sequence}` |
| `read_peptide_fasta(path)` | FASTA of peptides | `{name: peptide}` |
| `read_peptide_csv(path)` | CSV with `peptide` col | `{name: peptide}` |
| `read_sequence_csv(path)` | CSV with `sequence` col | `{name: sequence}` |
| `read_tsv(path)` / `read_csv(path)` | Topiary-format table with comment-block metadata | `TopiaryResult` |
| `read_lens(path)` | LENS report (v1.4 / v1.5.1 / v1.9) | `TopiaryResult` (wide form) |
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
