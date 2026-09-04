# Reading pVACseq results

[pVACtools](https://pvactools.readthedocs.io/) is one of the most widely used cancer-neoantigen prediction pipelines. Its `pvacseq` tool emits per-variant peptide × allele scores from an upstream MHC prediction stack (NetMHCpan, MHCflurry, MHCnuggets, etc.) and aggregates them into a final candidate list.

`topiary.read_pvacseq(path)` reads that output back into Topiary's long-form schema so you can:

- **Re-score** pVACseq candidates with Topiary's own MHC models (different version, different alleles, different ensemble).
- **Compare** pVACseq's scores side-by-side with fresh predictions in one DataFrame.
- **Re-filter and re-rank** using Topiary's DSL — including the [WT scope](ranking.md#wt-wildtype-comparison), [self-similarity](self_proteome.md) checks, [peptide properties](properties.md), and custom expressions.
- **Combine MHC-I and MHC-II** results into one ranking pass.

## Output flavors

pVACseq emits two TSV flavors per MHC class:

| File | Granularity | Per-algorithm scores |
|------|-------------|----------------------|
| `*.all_epitopes.tsv` | one row per (peptide, allele, length) candidate | each algorithm's MT/WT affinity, processing, presentation, and immunogenicity measurements |
| `*.all_epitopes.aggregated.tsv` | one row per variant; pVACseq picks the Best Peptide × Allele | aggregate affinity and any reported presentation ranks |

`read_pvacseq()` auto-detects which flavor you have from the column headers — same call site works for both:

```python
from topiary import read_pvacseq

r = read_pvacseq("HCC1395_TUMOR.MHC_I.all_epitopes.aggregated.tsv")
r = read_pvacseq("HCC1395_TUMOR.MHC_I.all_epitopes.tsv")
```

Returns a [`TopiaryResult`](api.md) with `r.df` (long-form DataFrame), `r.sources` (provenance), and `r.extra["pvacseq_format"]` (`"aggregated"` or `"all_epitopes"`).

## Schema mapping

The loader produces Topiary's standard long-form schema. Affinity,
presentation, processing, and immunogenicity measurements become separate
rows. WT companions populate the `wt_*` schema so the [`wt`
scope](ranking.md#wt-wildtype-comparison) works without further setup.

Affinity rows use the following mapping:

| Topiary column | Aggregated TSV source | all_epitopes TSV source |
|----------------|------------------------|--------------------------|
| `peptide` | `Best Peptide` | `MT Epitope Seq` |
| `allele` | `Allele` (mhcgnomes-normalized) | `HLA Allele` (mhcgnomes-normalized) |
| `value`, `affinity`, `score` | `IC50 MT` | `Median MT IC50 Score` (or `Best MT IC50 Score` when pVACseq was run with `--top-score-metric=Best`) |
| `percentile_rank` | `%ile MT` | `Median MT Percentile` (or `Best MT Percentile`) |
| `wt_value`, `wt_affinity`, `wt_score` | `IC50 WT` | `Median WT IC50 Score` (or `Corresponding WT IC50 Score`) |
| `wt_percentile_rank` | `%ile WT` | `Median WT Percentile` (or `Corresponding WT Percentile`) |
| `wt_peptide` | reconstructed from `Best Peptide` + `Pos` + `AA Change` for missense; NaN otherwise | `WT Epitope Seq` (always present) |
| `kind` | `"pMHC_affinity"` | same |
| `prediction_method_name` | `"pvacseq"` (synthesized) | same |
| `predictor_version` | `pd.NA` (pVACseq doesn't surface a method version per row) | same |

Presentation rows preserve both pVACtools' summary and its underlying EL
algorithms:

| Topiary row | pVACtools source | Method name |
|-------------|------------------|-------------|
| Aggregate presentation rank | `Pres %ile MT/WT` (aggregated) or `Median/Best MT Presentation Percentile` and its WT companion (`all_epitopes`) | `pvacseq` |
| MHCflurry presentation score and rank | `MHCflurryEL Presentation {MT,WT} {Score,Percentile}` | `mhcflurry` |
| NetMHCpan presentation score and rank | `NetMHCpanEL {MT,WT} Presentation Score` or plain `NetMHCpanEL {MT,WT} Score`, plus `{MT,WT} Percentile` | `netmhcpan` |
| NetMHCIIpan presentation score and rank | equivalent `NetMHCIIpanEL` columns | `netmhciipan` |
| BigMHC presentation score and rank | explicit presentation columns or plain `BigMHC_EL {MT,WT} Score`, plus percentiles | `bigmhc_el` |
| MHCflurry processing score and rank | `MHCflurryEL Processing {MT,WT} {Score,Percentile}` | `mhcflurry` |
| BigMHC immunogenicity score and rank | `BigMHC_IM {MT,WT} [Immunogenicity] {Score,Percentile}` | `bigmhc_im` |
| PRIME immunogenicity score and rank | `PRIME {MT,WT} Immunogenicity {Score,Percentile}` | `prime` |

For native non-affinity rows, a stated `score` is also the row's primary
`value`, matching Topiary's predictor output. A melted affinity algorithm that
states only a score keeps `value` and `affinity` null; Topiary does not borrow
the aggregate pVACseq median and attribute it to that algorithm. Aggregate TSV
presentation rows carry only a percentile, so their presentation `score` and
`value` are honestly null.

### Prediction-column vocabulary

The parser does not depend on one exact header spelling. It treats model
suffixes `EL` as presentation, `BA` / `Aff` / `Affinity` as affinity, and `IM`
as immunogenicity. Separators and capitalization are interchangeable, so
`NetMHCpanEL`, `NetMHCpan_EL`, and `netmhcpan-el` carry the same meaning.
`MT` and `WT` can appear on either side of `Score`, `Value`, or `Percentile`.

An explicit quantity in the metric wins over the model suffix. For example,
`MHCflurryEL Processing WT Percentile` is processing—not presentation—and is
stored in `wt_percentile_rank`. Affinity, processing, presentation, and
immunogenicity can all carry MT or WT percentile ranks. Known pVACtools and
mhctools model names may use a bare `MT Score`. For an unknown model, an
unqualified percentile inherits the kind only when that model has exactly one
explicit companion kind—for example, an `MT Percentile` beside an `MT IC50
Score` is an affinity rank. If there is no unique answer, the reader warns and
preserves the column under its original source name instead of guessing or
dropping it.

The same classifier is public as
`parse_prediction_metric(model_name, metric_name)` and is also the fallback
for new LENS predictor columns. This keeps the two readers from developing
different interpretations of the same header.

Calis immunogenicity rows are marked class I and allele-independent, following
mhctools' `AlleleFreePredictor` contract. The pVACtools report may repeat the
same Calis score beside several allele rows, but the allele is not an input to
that model.

### Derived columns (vaxrank-friendly)

Loader-derived columns aligned with `TopiaryPredictor` output so downstream code doesn't have to special-case the loader source:

| Column | Type | What it carries |
|--------|------|-----------------|
| `mhc_class` | `"I"` / `"II"` / `pd.NA` | Per-row class derived from the allele. Lets stacked multi-class results be filtered or split by class without re-parsing alleles. |
| `contains_mutant_residues` | `boolean` (nullable) | True iff the row's mutation position falls inside the candidate peptide. False for flanking-only peptides where pVACseq scored a 9-mer adjacent to the mutation but the mutation lies outside. |
| `mutation_start_in_peptide` / `mutation_end_in_peptide` | `Int64` | 0-based half-open mutation interval within the peptide. Derived from pVACseq's 1-based Pos (aggregated) or Mutation Position (all_epitopes). Single-residue semantics — multi-residue mutations collapse to a representative position. |
| `source` | `str` | Per-row provenance label, matching `read_tsv` convention so multi-file stacks stay distinguishable without rooting through `Metadata.sources`. |

### Annotation passthroughs

Expression and evidence are available under canonical cross-source names, so
DSL expressions such as `Column("rna_vaf") >= 0.1` work directly. These include
`gene_expression`, `rna_vaf`, `dna_vaf`, `rna_alt_expression`, and, where the
source supplies enough information, `n_rna_*` / `n_dna_*` counts with their
method and subject columns. all_epitopes also supplies the separately stated
`transcript_expression`; aggregated `RNA Expr` is gene-level in pVACseq's
schema and is not relabeled as transcript-level.

pVACseq's original values remain separately attributable under a `pvacseq_`
prefix: `pvacseq_tumor_dna_depth`, `pvacseq_tumor_dna_vaf`,
`pvacseq_tumor_rna_depth`, `pvacseq_tumor_rna_vaf`,
`pvacseq_normal_depth`, and `pvacseq_normal_vaf`. The aggregated flavor also
exposes `mane_select`, `canonical`, `transcript_support_level`, `aa_change`,
`pvacseq_tier`, and `pvacseq_evaluation`; all_epitopes additionally exposes
`hgvsc`, `hgvsp`, and `variant_type`.

## Loading + composing

### Single-class load

```python
from topiary import read_pvacseq, Affinity, apply_filter, apply_sort

r = read_pvacseq("HCC1395.MHC_I.all_epitopes.aggregated.tsv")
strong = apply_filter(r.df, Affinity.value <= 500)
ranked = apply_sort(strong, [Affinity.value])
print(ranked.head())
```

### MHC-I + MHC-II combined

`read_pvacseq()` doesn't expose a multi-file entry point — composition is just `topiary.stack_results`:

```python
from topiary import read_pvacseq, stack_results

combined = stack_results([
    read_pvacseq("HCC1395.MHC_I.all_epitopes.aggregated.tsv"),
    read_pvacseq("HCC1395.MHC_II.all_epitopes.aggregated.tsv"),
])
print(combined.df["mhc_class"].value_counts())
# I     317
# II    317
print(combined.df["source"].nunique())
# 2 (one per file)
```

The `source` and `mhc_class` columns keep the two halves distinguishable.

### Filter by class

The DSL has pre-built shortcuts (see [Ranking DSL: Categorical equality and membership](ranking.md#equality-membership-on-any-dtype)):

```python
from topiary import apply_filter, Affinity, Column, class_i, class_ii

# Class I only, strong binders
apply_filter(combined.df, class_i & (Affinity.value <= 500))

# Class I or II — drop unknown-class rows (none in pVACseq output today)
apply_filter(combined.df, class_i | class_ii)

# Class I + actually-mutant peptide (drops flanking-only candidates)
apply_filter(
    combined.df,
    class_i & Column("contains_mutant_residues").eq(True),
)
```

In the string DSL form (what a config file or CLI flag would carry):

```python
from topiary import parse

parse('affinity.value <= 500 & mhc_class == "I" & contains_mutant_residues == 1')
```

## Presentation scores are native rows

Presentation scores need no conversion after loading:

```python
from topiary import Presentation, apply_filter, read_pvacseq

r = read_pvacseq("HCC1395.MHC_I.all_epitopes.tsv")
presented = apply_filter(
    r.df,
    Presentation["mhcflurry"].rank <= 2,
)
```

Unqualified `Presentation.score` is ambiguous when several algorithms are
present. Qualify the method as above, or pass
`default_methods=resolve_default_methods(r.df)` to the ranking operation.

## Affinity algorithm scores: melt or pass through

The `all_epitopes` flavor also carries each binding algorithm's MT/WT IC50 + percentile side-by-side as columns:

```
NetMHCpan MT IC50 Score, NetMHCpan WT IC50 Score, NetMHCpan MT Percentile, NetMHCpan WT Percentile,
MHCflurry MT IC50 Score, MHCflurry WT IC50 Score, MHCflurry MT Percentile, MHCflurry WT Percentile, ...
```

`read_pvacseq()` snake-cases these into `pvacseq_<algo>_<field>_<mtwt>` annotation columns, reachable via `Column("...")`:

```python
from topiary import read_pvacseq, Column, apply_filter

r = read_pvacseq("HCC1395.MHC_I.all_epitopes.tsv")
# Strong on either NetMHCpan or MHCflurry
hits = apply_filter(
    r.df,
    (Column("pvacseq_netmhcpan_ic50_mt") <= 100)
    | (Column("pvacseq_mhcflurry_ic50_mt") <= 100),
)
```

Reachable but stringly-typed. If you want the DSL's `Affinity['mhcflurry'].value` selector to find pVACseq's per-algorithm scores natively, **melt** them into `prediction_method_name=<algo>` rows:

```python
from topiary import read_pvacseq, melt_pvacseq_algorithms, Affinity

r = read_pvacseq("HCC1395.MHC_I.all_epitopes.tsv")
m = melt_pvacseq_algorithms(r)
# Median rows are preserved; each (peptide, allele) now has N+1 rows
# (the Median + one per algorithm).

strong_in_mhcflurry = apply_filter(
    m.df, Affinity["mhcflurry"].value <= 100,
)
```

Melt extends `Metadata.extra["kind_support"]` to register each affinity
algorithm under the same MHC class as `"pvacseq"`. Existing processing,
presentation, and immunogenicity rows remain unchanged. On the aggregated
flavor (no per-algorithm binding columns to melt)
`melt_pvacseq_algorithms` is a no-op.

## WT peptide reconstruction

The aggregated TSV is the only flavor that doesn't ship `WT Epitope Seq` directly. For unambiguous missense rows (`AA Change` matches `^[A-Z]\d+[A-Z]$`), `read_pvacseq()` reconstructs the WT peptide from `Best Peptide` + `Pos` + `AA Change`:

| Best Peptide | Pos | AA Change | → wt_peptide |
|--------------|-----|-----------|-------------|
| `AERMGFTVV` | 8 | `E806V` | `AERMGFTEV` |

Rows where:

- The peptide doesn't carry the mutant residue at `Pos` (pVACseq's flanking-only peptides — mutation outside the window),
- `AA Change` is a frameshift / indel / multi-residue format (`FS342`, `EE764-765EK`, `SNNDRL233-238S`),

leave `wt_peptide` as NaN. On the HCC1395 sample (317 rows), 292 reconstruct cleanly. For full WT context on every row, use the unaggregated `all_epitopes.tsv` flavor, which ships `WT Epitope Seq` directly.

## Re-scoring with Topiary's own models

`read_pvacseq()` gives you peptides + alleles. To re-predict with Topiary's configured MHC stack:

```python
from topiary import read_pvacseq, TopiaryPredictor
from mhctools import NetMHCpan, MHCflurry

r = read_pvacseq("HCC1395.MHC_I.all_epitopes.aggregated.tsv")

alleles = sorted(r.df["allele"].dropna().unique())
peptides = {f"{v}_{p}": p for v, p in zip(r.df["variant"], r.df["peptide"])}

fresh = TopiaryPredictor(
    models=[NetMHCpan, MHCflurry],
    alleles=alleles,
).predict_from_named_peptides(peptides)
```

`fresh` is a long-form prediction DataFrame using Topiary's standard schema, identical in shape to `r.df`. Side-by-side comparison of pVACseq's scores against the fresh predictions is a plain pandas operation (`pd.concat` to stack, `pd.pivot_table` to align per `prediction_method_name`) — see the [Ranking DSL](ranking.md) guide for the DSL-side primitives.

## `derive_mhc_class` for non-pVACseq DataFrames

Fresh `TopiaryPredictor` output doesn't carry an `mhc_class` column — class lives in `TopiaryPredictor.kind_support` at the model level, not per row. To use the `class_i` / `class_ii` shortcuts (or any `Column("mhc_class")` expression) on such a frame, stamp the column first:

```python
from topiary import derive_mhc_class

df["mhc_class"] = derive_mhc_class(df["allele"])
```

`derive_mhc_class` maps each allele string to `"I"` (HLA-A/B/C), `"II"` (any HLA-D\* locus including heterodimers), or `pd.NA` (unrecognized).

## Metadata stamping

The result's `Metadata` records what the loader knows:

```python
r = read_pvacseq("HCC1395.MHC_I.all_epitopes.aggregated.tsv")

r.form                          # "long"
r.sources                       # ["pvacseq-aggregated:HCC1395.MHC_I.all_epitopes.aggregated.tsv"]
r.extra["pvacseq_format"]       # "aggregated"
r.extra["kind_support"]
# {"pvacseq": {
#     "pMHC_affinity": {"mhc_dependence": "single_allele", "mhc_class": "I"},
#     "pMHC_presentation": {"mhc_dependence": "single_allele", "mhc_class": "I"},
# }}
```

`kind_support` has the same shape as `TopiaryPredictor.kind_support`, so the result drops into call sites that expect that metadata:

```python
apply_filter(r.df, my_filter, kind_support=r.extra["kind_support"])
```

`pvacseq_format` is `"aggregated"` or `"all_epitopes"` (or a comma-joined string after melting / stacking).

## Caveats and known limitations

- **Sidecar `metrics.json` is not used.** pVACtools writes a sibling `.metrics.json` per output file with finer-grained data (per-algorithm scores on aggregated rows, the WT peptide sequence for non-missense, the underlying tool versions). `read_pvacseq()` works from the TSV alone — load the JSON yourself if you need that depth.
- **Flanking-only peptides survive in the candidate set.** pVACseq's `Best Peptide` can be a window that doesn't actually span the mutation (the mutation lies in the protein context but outside the predicted window). `contains_mutant_residues` flags these as `False`; filter with `Column("contains_mutant_residues").eq(True)` if your workflow rejects them.
- **Aggregate presentation has a rank but no score.** The aggregate TSV only
  supplies `Pres %ile MT/WT`; Topiary does not invent a presentation score.
  Use `Presentation.rank` for that flavor. The unaggregated file supplies the
  individual algorithms' scores as well.
- **No CLI flag.** The library API is the supported entry point; the existing `topiary` CLI is variant-pipeline-focused and doesn't surface `--pvacseq-input`. Pipelines that want CLI integration should wrap `read_pvacseq()` in their own script.

## See also

- [Ranking DSL](ranking.md) — filter and sort expressions, including `Column.eq` / `.isin` / `class_i` / `class_ii`
- [Cached Predictions](cached.md) — re-use external predictor output without re-running the predictor
- [API Reference](api.md) — full signatures
