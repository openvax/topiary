# Expression Semantics: Scopes, Prefixes, and Alternate Peptide Contexts

## The prediction row model

Every prediction row describes one measurement of one peptide in one allele context:

```
(peptide, allele, kind, method) → (value, score, percentile_rank)
```

- **peptide**: the amino acid sequence being scored
- **allele**: the HLA allele (e.g. `HLA-A*02:01`)
- **kind**: what was predicted (`pMHC_affinity`, `pMHC_presentation`, `pMHC_stability`, `antigen_processing`)
- **method**: which tool produced the prediction (`netmhcpan`, `mhcflurry`, etc.)

A single peptide-allele pair may have multiple rows — one per (kind, method) combination. These rows form a **group**, and all ranking/filtering expressions evaluate within a group.

### Group keys

```
(source_sequence_name | variant, peptide, peptide_offset, allele)
```

All rows sharing these keys are one group. An expression like `affinity.score` finds the `pMHC_affinity` row within the group and reads its `score` column.

## Implicit scope: the current peptide

Every expression is implicitly scoped to the **current row's peptide** and its predictions. When you write:

```
affinity.descending_cdf(500, 200)
```

you're asking: "what is this peptide's affinity IC50, normalized?" The peptide is never named — it's the one in the current group.

### Flanking context

The predictor also stores `n_flank` and `c_flank` — the amino acids flanking the peptide in its source protein. Some predictors (NetMHCpan 4.x) use these. They're part of the row but not currently exposed in the expression DSL.

## Contexts: alternate peptide scopes

A row can carry **inline references to other peptides and their predictions**. Each alternate peptide is a **context** — a named scope identified by a column prefix.

### Reserved context keywords

`wt`, `shuffled`, and `self` are reserved keywords in the expression DSL. They cannot be used as kind names, method names, or column references.

| Context | Prefix | Peptide column | Meaning |
|---------|--------|----------------|---------|
| *(default)* | *(none)* | `peptide` | The mutant/query peptide |
| `wt` | `wt_` | `wt_peptide` | Wildtype peptide at the same position |
| `shuffled` | `shuffled_` | `shuffled_peptide` | Randomly shuffled version of this peptide |
| `self` | `self_` | `self_peptide` | Best-matching peptide from the self proteome |

Each context produces a parallel set of columns with the same structure:

```
peptide     → value, score, percentile_rank, peptide_length, charge, hydrophobicity, ...
wt_peptide  → wt_value, wt_score, wt_percentile_rank, wt_peptide_length, wt_charge, ...
```

### The key insight

A context doesn't change what kind of data you're looking at — it changes **which peptide** you're looking at. `wt_score` is the same measurement as `score`, just applied to the wildtype peptide instead of the mutant one. The structure of each context's columns is identical to the default context.

## Expression DSL syntax

### Scope prefix syntax

A context keyword followed by `.` sets the scope for the rest of the expression term:

```
# Default scope (mutant peptide) — no prefix
affinity.score
affinity["netmhcpan"].descending_cdf(500, 200)

# Wildtype scope
wt.affinity.score
wt.affinity["netmhcpan"].descending_cdf(500, 200)

# Shuffled scope
shuffled.affinity.score

# Self-proteome scope
self.affinity.score
```

The context keyword is **always the first token** of an accessor. It cannot appear in the middle of an expression — `affinity.wt.score` is not valid. This makes parsing unambiguous: if the first identifier is a reserved context keyword and is followed by `.`, it's a scope prefix.

### How scope modifies field resolution

The scope prefix prepends `{prefix}_` to the column name that `Field.evaluate()` reads:

| Expression | Kind filter | Column read |
|------------|-------------|-------------|
| `affinity.value` | `pMHC_affinity` | `value` |
| `wt.affinity.value` | `pMHC_affinity` | `wt_value` |
| `shuffled.affinity.value` | `pMHC_affinity` | `shuffled_value` |
| `affinity.rank` | `pMHC_affinity` | `percentile_rank` |
| `wt.affinity.rank` | `pMHC_affinity` | `wt_percentile_rank` |

The kind and method filtering is unchanged — the context only affects which column is read from the matching row.

### Scope composes with everything

Transforms, arithmetic, method qualification — all work the same way within a scope:

```
wt.affinity.value.clip(1, 50000).log()
wt.affinity["netmhcpan"].score
0.5 * wt.affinity.score + 0.5 * wt.presentation.score
```

The scope selects the input; everything after that processes the output normally.

## Peptide-level expressions

### `len` — peptide length (precomputed)

`len` reads the `peptide_length` column (precomputed by the predictor as `df["peptide"].str.len()`). Under a scope prefix, it reads the corresponding prefixed column:

| Expression | Column read |
|------------|-------------|
| `len` | `peptide_length` |
| `wt.len` | `wt_peptide_length` |
| `shuffled.len` | `shuffled_peptide_length` |

`len` is a reserved keyword. It returns a numeric value and participates in arithmetic and transforms like any other expression:

```
len                       # peptide length
len >= 9                  # filter: keep 9-mers and longer
len - wt.len             # length difference (e.g. from indels)
```

### `count("X")` — amino acid count (dynamic)

`count("X")` counts occurrences of amino acid character(s) in the peptide string **at evaluation time**. It reads directly from the peptide column — no precomputed column required.

The argument is a string of one or more amino acid single-letter codes. Each character is counted independently and the counts are summed:

```
count("C")                # number of cysteines
count("KR")               # number of basic residues (K + R)
count("FYWH")             # number of aromatic residues
```

Under a scope prefix, `count` reads from the context's peptide column:

| Expression | Reads peptide from |
|------------|-------------------|
| `count("C")` | `peptide` |
| `wt.count("C")` | `wt_peptide` |
| `shuffled.count("C")` | `shuffled_peptide` |

Dynamic computation means `count` works even when no property columns have been precomputed. The trade-off: it's slightly slower than reading a precomputed column, but for single-column string operations this is negligible.

Usage in expressions:

```
count("C")                            # cysteine count
count("C") - wt.count("C")           # gained/lost cysteines vs wildtype
count("KR") >= 2                      # filter: at least 2 basic residues
0.5 * affinity.score - 0.1 * count("C")  # penalize cysteines
```

### Short aliases for kind accessors

These exist today and are unchanged:

| Alias | Expands to |
|-------|-----------|
| `ba`, `aff`, `ic50` | `affinity` (Kind.pMHC_affinity) |
| `el` | `presentation` (Kind.pMHC_presentation) |

Aliases compose with scope prefixes: `wt.ba.score` = `wt.affinity.score`.

Underscore-qualified kind names also compose: `wt.netmhcpan_ba.score` = `wt.affinity["netmhcpan"].score`.

## Comparison expressions

The most common use of alternate scopes is differential binding:

```
# How much better does the mutant bind vs wildtype?
affinity.score - wt.affinity.score

# Normalized differential
affinity.logistic(350, 150) - wt.affinity.logistic(350, 150)

# Is the mutant predicted stronger than a shuffled decoy?
affinity.score - shuffled.affinity.score

# Composite with differential weighting
0.5 * affinity.descending_cdf(500, 200) +
0.3 * (affinity.score - wt.affinity.score) +
0.2 * presentation.score

# Property-aware: penalize gained cysteines, reward shorter peptides
0.6 * affinity.descending_cdf(500, 200) +
0.2 * presentation.score -
0.1 * (count("C") - wt.count("C")).hinge() -
0.1 * (len - 9).hinge()
```

## Predictor-managed contexts

The predictor owns the mechanics of populating contexts. Each context is registered with the predictor and knows:
- Which peptide column to read from
- How to generate the alternate peptide (if applicable)
- Which prediction/property columns it has populated

### Context lifecycle

```python
predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01"],
)

# 1. Predict from variants — wt_peptide column is generated automatically
df = predictor.predict_from_variants(variants)

# 2. Populate context predictions
df = predictor.add_context(df, "wt")
# - Reads wt_peptide column
# - Runs each model on those peptides
# - Adds wt_value, wt_score, wt_percentile_rank (per kind x method)
# - Adds wt_peptide_length
# - Records "wt" in df.attrs["contexts"]

df = predictor.add_context(df, "shuffled")
# - Generates shuffled_peptide by randomly permuting each peptide
# - Runs each model on shuffled peptides
# - Adds shuffled_value, shuffled_score, shuffled_percentile_rank
# - Adds shuffled_peptide_length

df = predictor.add_context(df, "self", proteome=ensembl_proteome())
# - For each peptide, finds the closest match in the proteome
# - Adds self_peptide
# - Runs each model on self peptides
# - Adds self_value, self_score, self_percentile_rank
# - Adds self_peptide_length
```

### Context tracking

The DataFrame carries metadata about which contexts are populated:

```python
df.attrs["contexts"]  # {"wt", "shuffled", "self"}
```

This lets the ranking DSL:
1. **Validate** at strategy-build time that referenced contexts exist
2. **Warn** clearly instead of silently producing NaN columns
3. **List** available contexts for tab completion / help

### Properties within contexts

`add_peptide_properties` already supports the `peptide_column` and `prefix` parameters. Context-aware usage:

```python
# Compute properties for the default peptide
df = add_peptide_properties(df, groups=["core"])

# Compute properties for the WT peptide
df = add_peptide_properties(df, peptide_column="wt_peptide", prefix="wt_",
                            groups=["core"])
```

These could be folded into `add_context` or kept as a separate step — the predictor just needs to know which property columns exist so the DSL can validate `wt.hydrophobicity` references.

## NaN propagation when a context is missing

When a context's columns don't exist, field evaluation returns NaN. NaN propagates through arithmetic:

```
affinity.score - wt.affinity.score  →  NaN  (if wt_ columns absent)
```

This is correct — the expression can't be evaluated without the data. But the user experience should be:

1. **Build time**: If a ranking strategy references `wt.affinity.score` and the DataFrame doesn't have `wt_score`, emit a warning naming the missing context and the expression that needs it.
2. **Eval time**: Return NaN (not an error), so partial results are still usable.
3. **Output**: Rows with NaN sort keys sink to the bottom of rankings.

## Parser grammar (updated)

```
expr        := term (('+' | '-') term)*
term        := power (('*' | '/') power)*
power       := unary ('**' power)?
unary       := '-' unary | postfix
postfix     := atom ('.' IDENT call? | '[' STRING ']')*
atom        := NUMBER
             | '(' expr ')'
             | 'abs' '(' expr ')'
             | AGGREGATION '(' expr (',' expr)* ')'
             | 'count' '(' STRING ')'
             | 'column' '(' IDENT ')'
             | 'len'
             | CONTEXT '.' kind_ref          -- scoped accessor
             | kind_ref                       -- default scope
kind_ref    := IDENT ('[' STRING ']')? ('.' IDENT)?

CONTEXT     := 'wt' | 'shuffled' | 'self'   -- reserved keywords
AGGREGATION := 'mean' | 'geomean' | 'minimum' | 'maximum' | 'median'
```

When the parser sees `CONTEXT '.'`, it passes the prefix down to the `Field` constructor. The `Field` stores both the column prefix and the base field name, and prepends the prefix at evaluation time.

When the parser sees `CONTEXT '.' 'len'`, it produces a `Column` reading `{prefix}_peptide_length`.

When the parser sees `CONTEXT '.' 'count' '(' STRING ')'`, it produces a `Count` node reading from `{prefix}_peptide`.

## Internal representation

### Field with scope

```python
@dataclass
class Field(Expr):
    kind: Kind
    field: str                    # "value", "score", "percentile_rank"
    method: Optional[str] = None
    scope: str = ""               # "", "wt_", "shuffled_", "self_"

    def evaluate(self, group_df):
        kind_rows = group_df[group_df["kind"] == self.kind.value]
        # ... method filtering ...
        col_name = self.scope + self.field  # e.g. "wt_score"
        return float(kind_rows.iloc[0][col_name])
```

### Len expression

```python
class Len(Expr):
    scope: str = ""

    def evaluate(self, group_df):
        col = self.scope + "peptide_length"
        return float(group_df.iloc[0][col])
```

### Count expression

```python
class Count(Expr):
    chars: str          # e.g. "C", "KR"
    scope: str = ""

    def evaluate(self, group_df):
        peptide_col = (self.scope + "peptide") if self.scope else "peptide"
        peptide = group_df.iloc[0][peptide_col]
        return float(sum(peptide.count(c) for c in self.chars))
```

## Repr / round-trip format

Expressions with scopes repr as:

```
wt.affinity.score
wt.affinity['netmhcpan'].descending_cdf(500, 200)
shuffled.affinity.score
len
wt.len
count('C')
wt.count('KR')
affinity.score - wt.affinity.score
```

These are valid expression strings that `parse_expr` can re-parse, enabling round-tripping.

## Expression data as DSL fields

### The problem with the current RNA interface

Currently, RNA expression filtering is a separate pre-processing step outside the DSL:

```bash
# Old: separate flags, separate code path, Cufflinks-specific
--rna-gene-fpkm-tracking-file genes.fpkm_tracking
--rna-min-gene-expression 4.0
```

This is wrong for several reasons:
1. Expression filtering is disconnected from binding filters — they can't be combined in a single expression
2. The flag names are tool-specific (Cufflinks tracking files) when the data is just a `gene_id → number` mapping
3. Expression values can't participate in ranking (only in pre-filtering)
4. Different input sources provide different expression data, but the interface doesn't adapt

### Expression fields

Expression data should be first-class fields in the DSL, alongside binding affinity and presentation score. The data comes from user-provided files and is attached to rows by matching gene/transcript IDs.

**Gene/transcript expression** (available for all input types if user provides data):

| Field | Units | Source | CLI example |
|-------|-------|--------|-------------|
| `gene_tpm` | TPM | RNA-seq quantification | `--expression gene_tpm:quant.sf:Name:TPM` |
| `gene_fpkm` | FPKM | Legacy RNA-seq | `--expression gene_fpkm:genes.fpkm:id:FPKM` |
| `transcript_tpm` | TPM | Transcript-level | `--expression transcript_tpm:quant.sf:Name:TPM` |

**Variant-level RNA evidence** (from isovar or user-provided):

Isovar (`run_isovar(variants, alignment_file)`) provides per-variant RNA read evidence by assembling reads around each variant locus. Its output maps directly to DSL fields:

| DSL field | Isovar field | Units | What it measures |
|-----------|-------------|-------|------------------|
| `rna_alt_reads` | `num_alt_reads` | count | RNA reads supporting the variant allele |
| `rna_ref_reads` | `num_ref_reads` | count | RNA reads supporting the reference allele |
| `rna_total_reads` | `num_total_reads` | count | Total reads overlapping the locus |
| `rna_alt_fraction` | `fraction_alt_reads` | 0–1 | Fraction of reads supporting variant (RNA VAF) |
| `rna_other_reads` | `num_other_reads` | count | Reads supporting neither ref nor alt |
| `rna_alt_fragments` | `num_alt_fragments` | count | Deduplicated alt fragments |

Isovar also provides the assembled mutant protein sequence (`protein_sequence`), which becomes the peptide source for RNA-informed predictions — connecting back to issue #102 (mutant protein sequences from multiple sources).

**Variant-level DNA evidence** (from VCF annotations or user-provided):

| DSL field | Units | Source |
|-----------|-------|--------|
| `vaf` | 0–1 | Variant allele frequency (from VCF INFO/FORMAT) |
| `alt_reads` | count | DNA reads supporting alt allele |
| `ref_reads` | count | DNA reads supporting ref allele |
| `tumor_depth` | count | Total depth at variant locus |

**Single-cell** (from user-provided annotations):

| DSL field | Units | Source |
|-----------|-------|--------|
| `n_cells` | count | Cells expressing the gene/variant |
| `umis` | count | UMI counts |
| `cell_fraction` | 0–1 | Fraction of cells in cluster expressing |

### Isovar integration path

Isovar's `IsovarResult` objects carry rich variant-level evidence. The integration works in two directions:

**1. As a peptide source** (issue #102): Isovar assembles mutant protein sequences from RNA, providing an alternative to varcode's DNA-only prediction. The `protein_sequence` field becomes an input to `predict_from_named_sequences()`.

**2. As expression/evidence annotations**: Isovar's quantitative fields (`num_alt_reads`, `fraction_alt_reads`, `num_total_reads`, etc.) become columns in the prediction DataFrame, accessible via the expression DSL:

```bash
# Require RNA evidence for the variant
--ranking "ba <= 500 & rna_alt_reads >= 3 & rna_alt_fraction >= 0.01"

# Weight ranking by RNA support
--rank-by "0.5 * affinity.descending_cdf(500, 200) + 0.3 * presentation.score + 0.2 * rna_alt_fraction"
```

The predictor would load isovar results alongside or instead of variants:

```python
from isovar import run_isovar
from varcode import load_vcf

variants = load_vcf("somatic.vcf")
isovar_results = run_isovar(variants, "tumor_rna.bam")

predictor = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=["HLA-A*02:01"],
    filter_by=(Affinity <= 500),
)

# Option A: Use isovar's assembled protein sequences
df = predictor.predict_from_isovar(isovar_results)
# Peptides come from RNA assembly, evidence fields auto-populated

# Option B: Use varcode peptides but attach isovar evidence
df = predictor.predict_from_variants(variants)
df = predictor.annotate_from_isovar(df, isovar_results)
# Adds rna_alt_reads, rna_alt_fraction, etc. matched by variant
```

### Isovar filter thresholds as DSL expressions

Isovar has its own filter system (`filter_thresholds` in `run_isovar`). These map 1:1 to DSL filters:

| Isovar filter | DSL equivalent |
|---|---|
| `min_num_alt_reads: 3` | `rna_alt_reads >= 3` |
| `min_fraction_alt_reads: 0.005` | `rna_alt_fraction >= 0.005` |
| `max_num_ref_reads: 1e9` | `rna_ref_reads <= 1e9` |
| `min_ratio_alt_to_other_fragments: 3` | `column(ratio_alt_to_other_fragments) >= 3` |

By expressing isovar's filters in the DSL, users can combine RNA evidence thresholds with binding predictions in a single expression — instead of filtering in two separate stages.

### Loading expression data

Three flags, each baking in the join key:

```bash
--gene-expression name:file:id_column:value_column       # joins on gene_id
--transcript-expression name:file:id_column:value_column  # joins on transcript_id
--variant-expression name:file:id_column:value_column     # joins on variant
```

The join key is implicit in the flag — gene expression matches on `gene_id` in the DataFrame, transcript on `transcript_id`, variant on `variant`. This avoids ambiguity about which ID space the file's identifiers belong to.

**Simple default**: when only a file is given, use sensible defaults for common formats:

```bash
# Simple: just a file, auto-detect columns
--gene-expression salmon/quant.sf
--transcript-expression abundance.tsv

# With explicit column names
--gene-expression gene_tpm:salmon/quant.sf:Name:TPM
--transcript-expression transcript_tpm:kallisto/abundance.tsv:target_id:tpm
--gene-expression gene_fpkm:rsem.genes.results:gene_id:FPKM

# Multiple in one run
--gene-expression gene_tpm:salmon/quant.sf:Name:TPM \
--transcript-expression tx_tpm:salmon/quant.sf:Name:TPM

# Variant-level (from isovar output or any variant-keyed TSV)
--variant-expression rna_alt_reads:isovar_results.tsv:variant:num_alt_reads
--variant-expression rna_vaf:isovar_results.tsv:variant:fraction_alt_reads
```

Auto-detection for common formats:

| File pattern | Detected format | Default ID column | Default value column |
|---|---|---|---|
| `*.sf` (Salmon) | TSV | `Name` | `TPM` |
| `*abundance.tsv` (Kallisto) | TSV | `target_id` | `tpm` |
| `*.genes.results` (RSEM) | TSV | `gene_id` | `TPM` |
| `*.isoforms.results` (RSEM) | TSV | `transcript_id` | `TPM` |
| `*.gtf` (StringTie) | GTF | `reference_id` | `FPKM` |
| `*.fpkm_tracking` (Cufflinks) | TSV | `tracking_id` | `FPKM` |
| Other TSV/CSV | Inferred from headers | — | — |

When auto-detection can't determine columns, the flag requires the full `name:file:id_col:val_col` form.

### Unified filtering

Expression fields participate in the same DSL as binding predictions:

```bash
# Filter: good binder AND expressed
--ranking "ba <= 500 & gene_tpm >= 4"

# Filter: strong presentation OR highly expressed with moderate binding
--ranking "(el.score >= 0.9) | (gene_tpm >= 10 & ba <= 1000)"

# Rank by composite including expression
--rank-by "0.4 * affinity.descending_cdf(500, 200) + 0.3 * presentation.score + 0.2 * gene_tpm.log().ascending_cdf(2, 1) + 0.1 * vaf"
```

### Expression fields compose with everything

Transforms work on expression fields just like binding fields:

```
gene_tpm.log()                              # log-transform TPM
gene_tpm.ascending_cdf(10, 5)               # normalize to [0,1]
gene_tpm.clip(0, 100)                       # clamp range
vaf.hinge()                                 # zero out negative values
```

Expression fields compose with binding and peptide fields:

```
# Neoantigen priority score: binding × expression × novelty
0.4 * affinity.descending_cdf(500, 200)
+ 0.3 * presentation.score
+ 0.2 * gene_tpm.log().ascending_cdf(2, 1)
+ 0.1 * (affinity.score - wt.affinity.score)

# Single-cell: weight by cell fraction
affinity.logistic(350, 150) * cell_fraction

# Variant support: require RNA evidence
--ranking "ba <= 500 & rna_alt_reads >= 3"
```

### Implementation

Expression fields are just `column()` references with short aliases. The parser recognizes them as keywords and resolves them to `Column(name)`:

```
gene_tpm           →  Column("gene_tpm")
transcript_tpm     →  Column("transcript_tpm")
vaf                →  Column("vaf")
```

This means any user-provided column works — the aliases are just convenience for common ones. Users can always fall back to `column(my_custom_field)` for arbitrary data.

### What this replaces

| Old interface | New interface |
|---|---|
| `--rna-gene-fpkm-tracking-file FILE` | `--gene-expression FILE` or `--gene-expression gene_fpkm:FILE:tracking_id:FPKM` |
| `--rna-min-gene-expression 4.0` | `--ranking "gene_fpkm >= 4"` |
| `--rna-transcript-fpkm-tracking-file FILE` | `--transcript-expression FILE` |
| `--rna-transcript-fpkm-gtf-file FILE` | `--transcript-expression tx_fpkm:FILE:reference_id:FPKM` |
| `--rna-min-transcript-expression 1.5` | `--ranking "tx_fpkm >= 1.5"` |
| *(no variant-level support)* | `--variant-expression rna_alt_reads:FILE:variant:num_alt_reads` |
| Hard threshold, discard before prediction | Soft threshold in DSL, participates in ranking |

The old flags would remain as deprecated aliases during a transition period.

## Summary of decisions

| Decision | Choice |
|----------|--------|
| Context keywords | `wt`, `shuffled`, `self` — **reserved** in the parser |
| Syntax | `wt.affinity.score` — dot prefix, always first position |
| Flat underscore form | **Not supported** — use dot syntax |
| `WT()` wrapper | **Removed** — replaced by `wt.` prefix |
| `len` | Precomputed column (`peptide_length` / `wt_peptide_length`) |
| `count("X")` | Dynamic — reads peptide string at eval time |
| Context population | Predictor-managed via `add_context()` |
| Missing context | NaN at eval time, warning at build time |
| Backward compat for `WT()` | **None** — clean break |
| Expression loading | `--gene-expression`, `--transcript-expression`, `--variant-expression` (join key baked into flag) |
| Expression filtering | Unified with binding filters in `--ranking` |
| Expression aliases | `gene_tpm`, `transcript_tpm`, `vaf`, etc. → `Column(name)` |
| Simple default | `--gene-expression quant.sf` auto-detects columns for common formats |
