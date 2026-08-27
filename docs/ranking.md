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

## Group identity

Expressions evaluate per *group*, not per row: one peptide-allele group can span several rows (one per predictor and kind). `apply_filter` keeps or drops whole groups, `apply_sort` orders groups, and `evaluate_scores` gives every row its group's score.

By default the group keys are inferred from the columns present:

| Columns present | Inferred group keys |
|---|---|
| `fragment_id` | `fragment_id, peptide, peptide_offset, allele` |
| `variant` | `variant, peptide, peptide_offset, allele` |
| neither | `source_sequence_name, peptide, peptide_offset, allele` |

A `sample_name` column is prepended when it holds real values. Blank or null-only `sample_name` columns are ignored — `mhctools` stamps `sample_name=""` on every row of a single-sample run, and a constant level carries no identity.

### Explicit group keys

Inferred keys are sequence-oriented, so two rows that share peptide, source sequence and offset land in the same group even when they came from different variants, transcripts or genes — one row's filter decision then applies to the other. When your frame carries a stable provenance identity, pass it explicitly:

```python
from topiary import apply_filter, apply_sort, evaluate_scores, Affinity

group_keys = ["prediction_id", "source_sequence_name", "peptide",
              "peptide_offset", "allele"]

kept = apply_filter(df, Affinity <= 500, group_keys=group_keys)
kept = apply_sort(kept, [Affinity.score], group_keys=group_keys)
scores = evaluate_scores(kept, Affinity.score, group_keys=group_keys)
```

`apply_filter`, `apply_sort` and `evaluate_scores` accept the same keyword-only context options — `group_keys`, `default_methods`, `kind_support` and `alleles` — so filtering, sorting and scoring can share one grouping and one method resolution.

One thing is deliberately *not* shared: on a frame where several methods produce the same kind, `apply_filter` auto-aggregates an unqualified reference across them (`nanmin` for `<`/`<=`, `nanmax` for `>`/`>=`), while `apply_sort` and `evaluate_scores` stay strict and raise on the ambiguity. Pass `default_methods={"affinity": "mhcflurry"}` (or qualify the reference, `Affinity["mhcflurry"]`) to get one answer from all three.

All three options are forwarded to `EvalContext`, which you can also build directly when you need the raw per-group Series:

```python
from topiary.ranking import EvalContext

ctx = EvalContext(df, group_keys=group_keys)
per_group = (Affinity <= 500).eval(ctx)   # indexed by ctx.group_index
```

Group key names are validated before anything is evaluated, so a typo fails immediately with a suggestion rather than deep inside a node — including on empty frames, where there is otherwise nothing to fail on.

With a single group key, `ctx.group_index` is a flat `Index` of bare values (matching `DataFrame.groupby` on one column); with several it is a `MultiIndex` of key tuples. Null keys are collapsed to one group per spelling — `None`, `NaN` and `pd.NA` in an identity column name the same group, as they do under `groupby(dropna=False)`.

To map a per-group result back onto rows, use `ctx.row_group_codes()`, which gives each row the position of its group in `group_index`. Looking group keys up by value is unreliable when a key is null: `NaN` never equals itself.

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

## peptide_view — one value per peptide

Expressions evaluate per (peptide, allele) group, but the *correct* per-peptide reduction depends on how the predictor treats alleles:

| `mhc_dependence` | Rows per peptide | Peptide-level value |
|---|---|---|
| `single_allele` (NetMHCpan, MHCflurry per-allele) | one per (peptide, allele) | best across the peptide's alleles |
| `haplotype` (MHCflurry presentation, haplotype mode) | one per peptide | that row, read directly |
| `none` (antigen processing) | one per peptide, no allele | that row, read directly |

`peptide_view()` picks the right one, so a mixed expression doesn't require you to track which kind needs `best_*` and which is allele-free:

```python
from topiary import peptide_view, Affinity, Processing

# "this peptide's processing score" and "this peptide's best affinity"
0.5 * peptide_view(Processing.score) + 0.5 * peptide_view(Affinity.score)
```

The allele-free case is the one that couldn't be written before. An antigen-processing row carries no allele, so it forms its own group, and a plain read leaves every per-allele group `NaN` — the row is in none of them. Since that reading has no useful meaning, a bare reference to an allele-free kind in an allele-keyed grouping is projected anyway, with a warning naming the explicit form:

```python
evaluate_scores(df, Processing.score)                # [0.77, 0.77, 0.77] + UserWarning
evaluate_scores(df, peptide_view(Processing.score))  # [0.77, 0.77, 0.77], silent
```

Per-allele kinds are left alone: a plain read there returns a real row, and choosing *which* row is a genuine decision that stays with `best_*` / `peptide_view`. Writing the wrapper explicitly is still the right thing — it silences the warning and says what the expression means:

That is why producers duplicated the processing row across the patient's alleles before handing topiary a frame. With `peptide_view` the frame keeps one canonical row and the value is broadcast at evaluation time:

```python
apply_filter(df, parse("affinity <= 500 & peptide_view(processing.score) >= 0.5"))
```

The allele-free row is still its own group. A filter on an allele-scoped kind has nothing to say about that group, so rather than dropping it — which would take the evidence out of the frame before the score expression reads it — topiary keeps it whenever the filter kept at least one of that peptide's allele groups. A filter that *does* read the allele-free kind (`peptide_view(processing.score) >= 0.9`) still decides it, and a peptide excluded entirely takes its evidence with it.

### Declaring the alleles to evaluate against

Groups come from the rows, so a peptide whose *only* evidence is allele-free has no per-allele group at all — and a consumer keyed by patient allele has nothing to read. The genotype isn't in the frame, so pass it:

```python
scores = evaluate_scores(
    df, peptide_view(Processing.score),
    group_keys=group_keys,
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
)
```

`alleles` adds one group per peptide per declared allele, giving `peptide_view` somewhere to broadcast into. The added groups hold no rows, so allele-scoped fields read `NaN` there — which is the truth: that allele has no prediction of its own. The frame itself is untouched; only the group index grows.

The mode comes from `EvalContext(kind_support=...)` — mhctools' per-(model, kind) metadata, available as `TopiaryPredictor.kind_support` — which is the only thing that can tell `haplotype` from `single_allele`, since both put a real allele on every row. Without it, a kind whose rows carry no allele is treated as allele-free and everything else as per-allele. Inconsistent data is an error, not a silent pick: a peptide-level kind with two different values for one peptide, or models that disagree about `mhc_dependence`, both raise.

Available in string form too, so it works in `--filter-by` / `--sort-by` and in config files: `peptide_view(processing.score)`. Sorting reads the direction through the wrapper, so `peptide_view(affinity.value)` ranks strong binders first under the default `--sort-direction auto`, exactly like the bare field.

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

Arithmetic and `<` / `<=` / `>` / `>=` comparisons require numeric values; using a non-numeric column in those contexts raises:

```
TypeError: Column 'gene_name' contains non-numeric value 'BRAF' (str).
Only numeric columns can be used in ranking expressions.
```

For categorical filtering on string columns (`mhc_class`, `gene`, `source`, ...) use the equality / membership methods described below.

### Equality / membership on any dtype

`Column("...").eq(value)`, `Column("...").ne(value)`, and `Column("...").isin(values)` produce an `IsIn` node that reads the column raw — no float cast — so string, boolean, and mixed-dtype columns work:

```python
from topiary import apply_filter, Affinity, Column

# Categorical equality
apply_filter(df, Column("mhc_class").eq("I"))
apply_filter(df, Column("gene").ne("HLA-A"))

# Membership
apply_filter(df, Column("mhc_class").isin(["I", "II"]))

# Compose with numeric clauses
apply_filter(
    df,
    (Affinity.value <= 500) & Column("mhc_class").eq("I"),
)

# Negate with ~
apply_filter(df, ~Column("source").isin(["control", "blacklist"]))
```

`DSLNode.__eq__` is intentionally not overridden — `Column("x") == "y"` still does Python identity equality and won't compose. Always use `.eq()` / `.ne()` / `.isin()`.

NaN handling matches pandas, not SQL: missing values evaluate to `False` for `.eq()` / `.isin()` and to `True` for `.ne()` / `~.eq()` (the inverse). To exclude NaN explicitly, compose with the source-of-truth column — e.g. `Column("mhc_class").ne("II") & Column("mhc_class").isin(["I", "II"])`.

The string parser accepts string literals on the right-hand side of `==` and `!=` (rejected with `<` / `<=` / `>` / `>=` since ordering on arbitrary strings isn't meaningful):

```python
from topiary import parse

parse('mhc_class == "I"')
parse('affinity.value <= 500 & mhc_class != "II"')
```

For the two most common categorical filters, `class_i` and `class_ii` are pre-built shortcuts:

```python
from topiary import apply_filter, Affinity, class_i, class_ii

apply_filter(df, class_i & (Affinity.value <= 500))
apply_filter(df, class_i | class_ii)   # both classes; identity filter
```

Both reference the `mhc_class` column. That column is present after `topiary.read_pvacseq()` (and any future loader that derives it); fresh `TopiaryPredictor` output doesn't ship a per-row `mhc_class` (class lives in `kind_support` at the model level), so derive it first if you need these on a freshly predicted DataFrame:

```python
from topiary import derive_mhc_class
df["mhc_class"] = derive_mhc_class(df["allele"])
```

## wt. — wildtype comparison

The `wt.` scope prefix reads wildtype prediction columns (`wt_value`, `wt_score`, `wt_percentile_rank`). For `predict_from_fragments()` and the variant-based APIs that build fragments internally, pass `predict_wt=True` to score each populated `wt_peptide` with the same MHC model(s):

```python
from topiary import Affinity, TopiaryPredictor, wt

predictor = TopiaryPredictor(
    models=[...],
    alleles=[...],
    predict_wt=True,
)
df = predictor.predict_from_variants(variants)

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

In the CLI, pass `--predict-wt` with variant-derived inputs before
using `wt.*` in `--sort-by`:

```bash
topiary ... --predict-wt --sort-by "affinity.score - wt.affinity.score"
```

!!! note
    `wt.` is for **sorting expressions only**, not filters. Use it in `sort_by`, not in `filter`. When WT columns don't exist, expressions evaluate to NaN. Rows without a length-compatible WT peptide also keep NaN WT prediction values.

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

In the string DSL, prefer these forms:

```
affinity[netmhcpan, 4.1b].value <= 500
affinity[netmhcpan, release-2.2.0].value <= 500
netmhcpan[4.1b]:affinity.value <= 500
netmhcpan-4.1b:affinity.value <= 500
wt.netmhcpan[4.1b]:ba.score
```

Use brackets for arbitrary version labels. The dash form is a compact
shortcut for numeric-leading versions; the required `:kind` suffix keeps
it distinct from ordinary subtraction.

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
| `Affinity["netmhcpan"] <= 500` | `netmhcpan:affinity <= 500` |
| `Affinity["netmhcpan", "4.1b"].score` | `netmhcpan[4.1b]:affinity.score` or `netmhcpan-4.1b:affinity.score` |
| `Presentation["mhcflurry"].rank <= 2` | `mhcflurry:el.rank <= 2` or `mhcflurry_el.rank <= 2` |
| `Column("cysteine_count") <= 2` | `column(cysteine_count) <= 2` |
| `(A <= 500) \| (B.rank <= 2)` | `affinity <= 500 \| presentation.rank <= 2` |
| `(A <= 500) & (B.rank <= 2)` | `affinity <= 500 & presentation.rank <= 2` |

**Recommended model-qualified syntax:**

| Meaning | Prefer |
|---|---|
| Method-specific kind | `netmhcpan:affinity.score` |
| Method-specific alias | `mhcflurry:ba.score`, `mhcflurry:el.rank` |
| Method + arbitrary version | `netmhcpan[release-2.2.0]:affinity.score` |
| Method + numeric-leading version | `netmhcpan-4.1b:affinity.score` |
| Wildtype scope | `wt.netmhcpan[4.1b]:ba.score` |

**Accepted aliases:** `ba` / `aff` / `ic50` = Affinity, `el` =
Presentation. Canonical serialization still uses bracket form
(`affinity[netmhcpan]`). The parser also accepts compatibility forms
such as `affinity:netmhcpan`, `netmhcpan.affinity`,
`affinity[netmhcpan]`, and `netmhcpan_ba`, but new docs and configs
should use the recommended forms above.

**All features work in both Python and CLI string form** (`--sort-by`):

- Arithmetic: `0.5 * affinity.score + 0.5 * presentation.score`
- Transforms: `.logistic()`, `.ascending_cdf()`, `.descending_cdf()`, `.clip()`, `.hinge()`, `.log()`
- Aggregations: `mean()`, `geomean()`, `minimum()`, `maximum()`, `median()`
- `column(x)` in arithmetic, filters, and ranking
- Scope prefixes: `wt.affinity.score`, `wt.len`, `wt.count('C')`

`--sort-direction` controls whether lower or higher values rank first. The
default is `auto`: raw affinity values and percentile ranks sort ascending,
while all other sort expressions sort descending.

## Combining Separate Predictor Runs

Run predictors together when that is convenient:

```python
from mhctools import NetMHCpan, MHCflurry
from topiary import TopiaryPredictor

combined = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
).predict_from_named_peptides(peptides)
```

When predictors need to run separately, use `combine_predictions` to
turn their complementary prediction rows back into the same long-form
shape:

```python
from mhctools import NetMHCpan, MHCflurry
from topiary import TopiaryPredictor, combine_predictions

netmhcpan_rows = TopiaryPredictor(
    models=NetMHCpan,
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
).predict_from_named_peptides(peptides)

mhcflurry_rows = TopiaryPredictor(
    models=MHCflurry,
    alleles=["HLA-A*02:01", "HLA-B*07:02"],
).predict_from_named_peptides(peptides)

combined = combine_predictions([netmhcpan_rows, mhcflurry_rows])
```

`TopiaryResult` owns the long/wide representation.  Loaders may naturally
produce wide results (for example LENS) or long results (for example pVACseq
and fresh predictor outputs), but callers can use `result.long_df`,
`result.wide_df`, `result.to_long()`, or `result.to_wide()` on demand. Topiary
merge functions normalize those forms internally instead of making callers
choose a representation before combining results.

You can also shard the same predictor over allele or peptide-length batches and
combine the shards.  Use `TopiaryPredictor(name=...)` when you want to keep
track of which batch produced each row:

```python
shards = []
for allele in ["HLA-A*02:01", "HLA-B*07:02"]:
    for length in [8, 9, 10, 11]:
        length_peptides = {
            name: peptide
            for name, peptide in peptides.items()
            if len(peptide) == length
        }
        shards.append(
            TopiaryPredictor(
                models=NetMHCpan,
                alleles=[allele],
                name=f"netmhcpan_{allele}_len{length}",
            ).predict_from_named_peptides(length_peptides)
        )

combined = combine_predictions(shards)
```

`prediction_method_name` is still the logical predictor name (`netmhcpan` in
the example above).  The optional `prediction_run_name` column is only
provenance for a particular run or shard.  That distinction lets distinct
NetMHCpan allele/length shards combine into one logical NetMHCpan result,
while overlapping shards with the same `(prediction_method_name, kind,
peptide, allele, sample/source context)` still fail as duplicates.
`to_wide()` drops `prediction_run_name` from the grouping keys, so a named
split run has the same wide shape as a single unsplit run.

The helper is intentionally strict. It rejects duplicate
`(prediction_method_name, kind, identity)` rows, and by default requires every
emitted `(prediction_method_name, kind)` group to cover the same peptide/allele
identity grid. This catches incomplete split runs before `to_wide()` can
produce half-populated rows. If you intentionally want a sparse union, pass
`coverage="partial"`; duplicate predictions are still rejected.

The combined result preserves the original rows: use each row's
`prediction_method_name`, `predictor_version`, `kind`, and value/rank columns
to inspect which predictor produced which quantity.  Use
`prediction_run_name` only to audit the batch that produced a row, not as a DSL
selector.

Allele aggregation remains part of the ranking DSL: for example,
`Affinity["netmhcpan"].best_value_allele` and
`Presentation["netmhcpan"].best_score_allele` report the allele associated
with the best BA or EL value across the combined allele grid.  For predictors
that emit one row per allele, such as NetMHCpan or MHCflurry in single-allele
mode, this is the best per-allele row after all shards are combined.  For
MHCflurry presentation in haplotype mode, MHCflurry itself sees the allele set
together and may emit one deconvolved best-allele row; combining independent
single-allele MHCflurry shards is therefore not the same calculation as a
direct haplotype-mode MHCflurry run.  If you intentionally combine haplotype
presentation rows with per-allele rows, use `coverage="partial"` because those
kinds have different identity grids by construction.  Processing-only
quantities that do not depend on allele should be read directly rather than
through `best_*`.

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
