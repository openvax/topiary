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

### From NetMHC-family stdout captures

The DTU NetMHC suite (NetMHCpan, NetMHC, NetMHCcons, NetMHCIIpan,
NetMHCstabpan) prints binding predictions to stdout when run from
the command line. If you've captured that output to a file, load
it directly — topiary reuses the parsers shipped by
[mhctools](https://github.com/openvax/mhctools) so every NetMHC
version they support is covered here too.

```python
cache = CachedPredictor.from_netmhcpan_stdout(
    "netmhcpan_run.out",
    mode="binding_affinity",     # or "elution_score" on 4+
)
cache = CachedPredictor.from_netmhc_stdout("netmhc_run.out", version="4")
cache = CachedPredictor.from_netmhcpan_cons_stdout("cons_run.out")
cache = CachedPredictor.from_netmhciipan_stdout("ii_run.out", version="4.3")
cache = CachedPredictor.from_netmhcstabpan_stdout("stab_run.out")
```

Each loader parses the version out of the stdout preamble (e.g.
`NetMHCpan version 4.1b`) and stamps it on `predictor_version`.
Pass `predictor_version="..."` if you want a different label, or
if your capture stripped the preamble.

The loaders parse the *stdout text* format. NetMHC's `-xlsfile`
tab-delimited output is a different format and isn't supported
today — open an issue if you need it.

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

### Sharding: merge multiple caches

Predict per-allele or per-sample in parallel, persist each shard
separately, then merge them:

```python
cache = CachedPredictor.concat([shard_a, shard_b, shard_c])

# Or: load every matching file from a directory
cache = CachedPredictor.from_directory(
    "caches/",
    pattern="*.parquet",
)
```

Every shard must share the same
`(prediction_method_name, predictor_version)` — the core invariant
applies across shards the same way it applies inside one.

**Overlap resolution** (`on_overlap=`):

- `"raise"` (default) — fail if any `(peptide, allele, peptide_length)`
  appears in more than one shard. A sample of conflicting keys is
  included in the error. Use this if shards should be disjoint.
- `"last"` — later shard in the input list wins. Useful when the
  sort order represents "newer overwrites older."
- `"first"` — earlier shard wins.
- `callable(row_a, row_b) -> row` — custom resolver. Called pairwise
  per duplicate group. Pattern for "keep stronger binder":

  ```python
  def keep_lower_affinity(a, b):
      return a if a["affinity"] <= b["affinity"] else b

  cache = CachedPredictor.concat(shards, on_overlap=keep_lower_affinity)
  ```

`from_directory` passes `on_overlap` through to `concat`; file order
is sorted lexicographically, so `shard_a.tsv` is always earlier than
`shard_b.tsv`.

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

## From the CLI

Every loader is exposed through `topiary`'s command-line interface via
`--mhc-cache-*` flags. The CLI runs the entire prediction pipeline
(variant effects, filtering, ranking, output formatting) against the
cached predictions — the live predictor is never invoked.

`--mhc-cache-format` is optional for NetMHC-family stdout captures,
mhcflurry CSVs, and topiary's own output — topiary sniffs the format
from file content. Only the generic `tsv` path needs an explicit
`--mhc-cache-format tsv` (generic tables don't carry identifying
signatures).

```bash
# NetMHCpan stdout capture — format sniffed from the preamble
topiary --peptide-csv peptides.csv \
    --mhc-cache-file netmhcpan_run.out \
    --output-csv results.csv

# mhcflurry CSV — format sniffed from column names; predictor_version
# auto-composed from the local install
topiary --peptide-csv peptides.csv \
    --mhc-cache-file mhcflurry_predictions.csv \
    --output-csv results.csv

# Topiary's own saved output (Parquet or TSV round-trip) — sniffed
topiary --peptide-csv peptides.csv \
    --mhc-cache-file prior_run.parquet \
    --output-csv results.csv

# Generic TSV with column mapping — format must be explicit
topiary --peptide-csv peptides.csv \
    --mhc-cache-file third_party.tsv \
    --mhc-cache-format tsv \
    --mhc-cache-predictor-name netchop \
    --mhc-cache-predictor-version 3.1 \
    --mhc-cache-tsv-column affinity=IC50_nM \
    --mhc-cache-tsv-column percentile_rank=Rank \
    --output-csv results.csv

# Sharded: merge every file in a directory
topiary --peptide-csv peptides.csv \
    --mhc-cache-directory ./caches \
    --mhc-cache-directory-pattern '*.parquet' \
    --output-csv results.csv
```

Full flag reference:

| Flag | Purpose |
|---|---|
| `--mhc-cache-file PATH` | Single cache file. Requires `--mhc-cache-format`. |
| `--mhc-cache-directory PATH` | Directory of shards; each file loaded via `from_topiary_output` and concatenated. Alternative to `--mhc-cache-file`. |
| `--mhc-cache-directory-pattern GLOB` | Pattern for `--mhc-cache-directory`. Default `*`. |
| `--mhc-cache-format FORMAT` | Optional — sniffed from file content when omitted (see above). One of `topiary_output`, `mhcflurry`, `tsv`, `netmhcpan`, `netmhc`, `netmhccons`, `netmhciipan`, `netmhcstabpan`. Only `tsv` strictly requires the explicit flag. |
| `--mhc-cache-predictor-name NAME` | Override `prediction_method_name` (required for `tsv` when not in the file). |
| `--mhc-cache-predictor-version V` | Override `predictor_version`. Auto-inferred for NetMHC stdout captures (parsed from preamble) and mhcflurry (composite from local install). |
| `--mhc-cache-tsv-column CANONICAL=FILE_COL` | Repeatable. Column-name mapping for `tsv` format. |
| `--mhc-cache-tsv-sep SEP` | Separator for `tsv`. Default tab. |
| `--mhc-cache-netmhcpan-mode MODE` | `binding_affinity` (default) or `elution_score` for NetMHCpan 4+. |
| `--mhc-cache-netmhc-version V` | `3`, `4`, or `4.1` for classic NetMHC output. Default `4`. |
| `--mhc-cache-netmhciipan-version V` | `legacy`, `4`, or `4.3` for NetMHCIIpan. Default `4.3`. |
| `--mhc-cache-netmhciipan-mode MODE` | `binding_affinity` or `elution_score` (default) for NetMHCIIpan 4+. |

`--mhc-predictor` and `--mhc-alleles` become optional when a cache is
in use — the cache supplies the predictions and its allele set.

## When *not* to use

- You've never run the predictor on these peptides — just use the
  live predictor directly; there's nothing to cache yet.
- You want to mix predictions from multiple model versions in one
  pipeline. Don't. If you think you need this, revisit whether the
  versions are actually interchangeable; if they are, use
  `also_accept_versions`.
