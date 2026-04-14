# Cached predictions

Running an MHC predictor twice on the same peptide is wasted work.
`CachedPredictor` serves predictions from a pre-computed table — an
external predictor's output, a previous topiary run, or shards from
parallel jobs — and plugs into `TopiaryPredictor(models=…)` the same
way a live predictor does.

Use it when you want to:

- Iterate on filters / ranking / the DSL without re-running a slow predictor.
- Pin scores for reproducibility (papers, benchmarks).
- Predict per-allele / per-sample in parallel jobs, persist each shard, merge later.
- Ingest predictions from a tool topiary doesn't natively run.

## Basic usage

Load a cache, pass it as the model to `TopiaryPredictor`, go:

```python
from topiary import CachedPredictor, TopiaryPredictor

cache = CachedPredictor.from_topiary_output("run.parquet")
predictor = TopiaryPredictor(models=cache)
df = predictor.predict_from_variants(variants)
```

The cache answers `predict_proteins_dataframe` and
`predict_peptides_dataframe` calls from the table, so every
`predict_from_*` method on `TopiaryPredictor` works unchanged.

## Loaders

### From topiary's own prediction output

Round-trip a prior `TopiaryPredictor` run through Parquet (preferred)
or TSV:

```python
# Save
first_run_df.to_parquet("run.parquet", index=False)

# Reload on subsequent iterations
cache = CachedPredictor.from_topiary_output("run.parquet")
```

Schema: every column `_predict_raw*` produces is preserved. Topiary
overlay columns (`fragment_id`, `wt_peptide`, etc.) are kept in the
file but ignored on lookup — the cache is the predictor output, not
the full pipeline output.

### From mhcflurry output

```python
cache = CachedPredictor.from_mhcflurry("mhcflurry_predictions.csv")
```

The loader maps mhcflurry's `mhcflurry_affinity` /
`mhcflurry_affinity_percentile` / `mhcflurry_presentation_score`
columns onto topiary's canonical `affinity` / `percentile_rank` /
`score`. `predictor_version` is auto-composed from the installed
mhcflurry — see [mhcflurry version composition](#mhcflurry-version-composition)
below.

### Generic TSV / CSV with column mapping

For any tab- or comma-delimited file that doesn't match a format
topiary ships a dedicated loader for:

```python
cache = CachedPredictor.from_tsv(
    "third_party.tsv",
    columns={
        "peptide": "Peptide",
        "allele": "HLA",
        "affinity": "IC50_nM",
        "percentile_rank": "Rank%",
    },
    prediction_method_name="netchop",
    predictor_version="3.1",
)
```

`columns` maps canonical cache columns to the column names in your
file. `prediction_method_name` and `predictor_version` are required
when the file doesn't embed that identity.

Pass `sep=","` for CSV files.

### From an in-memory DataFrame

```python
cache = CachedPredictor.from_dataframe(
    df,
    prediction_method_name="custom",
    predictor_version="v7",
)
```

For programmatic construction — tests, scripts that built predictions
some other way, or intermediates in a larger pipeline.

## Version invariant

**A single `CachedPredictor` holds predictions from exactly one
`(prediction_method_name, predictor_version)` pair.** Scores from
different model versions aren't interchangeable — even percentile
ranks aren't directly comparable across versions. Mixing them would
produce an output that passes every downstream filter invisibly.

Enforcement:

- **On construction**: a DataFrame with multiple `(name, version)`
  pairs raises. `None` / `NaN` / empty-string values are also
  rejected — silent "I don't know" would mask the invariant.
- **On fallback attachment** (below): the fallback's `(name, version)`
  must equal the cache's, verified on the first fallback call.
- **On concat / `from_directory`** (sharding — upcoming): every shard
  must agree.

### Explicit opt-in equivalence

Sometimes two similar versions do produce identical predictions —
release candidate → final, or a timestamp-only model-data reflash.
Opt in explicitly:

```python
cache = CachedPredictor.from_mhcflurry(
    "path.csv",
    predictor_version="2.2.0rc2",
    also_accept_versions={"2.2.0rc1", "2.2.0"},
)
```

A fallback or shard passes if its version equals the cache's *or* is
in `also_accept_versions`. Names are still strict — mixing mhcflurry
and NetMHCpan is a real type mismatch, not a version wiggle.

## mhcflurry version composition

Unlike NetMHCpan (which bakes models into the binary), mhcflurry
fetches model weights separately via `mhcflurry-downloads fetch`.
Two systems on the same mhcflurry *package* version can produce
different predictions if they have different model bundles installed.
`predictor_version` must capture both.

`topiary.mhcflurry_composite_version()` introspects the installed
mhcflurry and returns a composite string like
`"2.2.1+release-2.2.0"`. `CachedPredictor.from_mhcflurry(path)`
calls it automatically when you omit `predictor_version` — **you
never have to enumerate model bundles manually**:

```python
# Auto-composed from the local install
cache = CachedPredictor.from_mhcflurry("predictions.csv")

# Explicit override if you want a custom label
cache = CachedPredictor.from_mhcflurry(
    "predictions.csv",
    predictor_version="my-project-20260414",
)
```

The helper raises with a clear message if mhcflurry isn't installed
or no model release is configured.

Other tools (NetMHCpan, NetMHCIIpan, etc.) don't need this — their
binary version is their full identity.

## Fallback: delegate misses to a live predictor

If you want a cache that falls back to a live predictor for peptides
not yet in the table:

```python
from mhctools import MHCflurry
live = MHCflurry(alleles=[...])

cache = CachedPredictor.from_mhcflurry(
    "partial.csv",
    fallback=live,
)

predictor = TopiaryPredictor(models=cache)
df = predictor.predict_from_variants(variants)   # misses delegate
```

Semantics:

- **Miss routed to `fallback`**, and the fallback's output is merged
  into the cache so subsequent queries for the same
  `(peptide, allele, peptide_length)` serve locally. Caching hits
  is always the right default — there's no separate flag.
- **Same version invariant**: the fallback's `(name, version)` must
  equal the cache's (or be in `also_accept_versions`), checked on the
  first fallback call.
- **Pure read-through mode** — empty cache, fallback-only — is
  supported: `CachedPredictor(fallback=live)`. The cache starts
  empty; identity is discovered from the fallback's first output.
- **No fallback** (default): misses raise `KeyError` with the missed
  peptides listed.

## Persisting a cache

```python
cache.save("cache.parquet")   # or .tsv / .tsv.gz / .csv
```

Writes the cache's internal table using the same schema the loaders
expect. Round-trips cleanly through `from_topiary_output`.

## When *not* to use

- You've never run the predictor on these peptides — just use the
  live predictor directly; there's nothing to cache yet.
- You want to mix predictions from multiple model versions in one
  pipeline. Don't. If you think you need this, revisit whether the
  versions are actually interchangeable; if they are, use
  `also_accept_versions`.
