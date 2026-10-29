# Cached predictions

Running an MHC predictor twice on the same peptide is wasted work.
`CachedPredictor` serves predictions from a pre-computed table — an
external predictor's output, a previous topiary run, or shards from
parallel jobs — and plugs into `TopiaryPredictor(models=…)` the same
way a live predictor does.

Use it to iterate on filters, ranking and the DSL without re-running a slow
predictor; to pin scores for reproducibility in a paper or benchmark; to predict
per allele or per sample in parallel jobs, persist each shard and merge them
later; or to ingest predictions from a tool topiary does not run itself.

## Basic usage

Load a cache and pass it as the model:

```python
from topiary import CachedPredictor, TopiaryPredictor

cache = CachedPredictor.from_topiary_output("run.parquet")
predictor = TopiaryPredictor(models=cache)
df = predictor.predict_from_variants(variants)
```

The cache answers `predict_proteins_dataframe` and
`predict_peptides_dataframe` calls from the table, so every
`predict_from_*` method on `TopiaryPredictor` works unchanged.

For whole-peptide half-life tables, use `predict_from_named_peptides` to replay
the complete peptides originally scored. Known allele-independent kinds retain
`mhc_dependence="none"` without a live fallback; caching a measurement does not
make it allele-specific. A fallback's explicit kind metadata takes precedence.
Measurement contexts survive loading, queries and saving. Rows sharing a cache
key must agree on those semantics as well as their numerical measurements;
contradictory contexts are rejected. The cache does not yet index distinct
chemistry or assay conditions as separate queries: keep those experiments in
separate caches ([#288](https://github.com/openvax/topiary/issues/288)).

`save()` writes structured contexts using Topiary's typed CSV/TSV codec,
including metadata comments, or Parquet. Both delimited formats accept a `.gz`
suffix. Reload with `from_topiary_output`; the CLI recognizes these saved files
automatically. Generic pandas readers can skip metadata using `comment="#"`,
but use Topiary's readers when dictionaries and literal identifiers must retain
their types.

## Incomplete cache coverage

By default, a coverage gap raises `CachedPredictorCoverageError` and stops the
run. This includes missing peptides and incompatible flank, kind or genotype
contexts. `raise_on_error=False` only affects variant annotation errors.

To retain successful model/input pairs, explicitly supply a report handler:

```python
misses = []
predictor = TopiaryPredictor(models=cache, cache_miss_handler=misses.append)
df = predictor.predict_from_named_sequences(proteins)
# Persist misses alongside df. An empty list means no cache misses were skipped.
```

A failed sequence loses **all its rows for the affected model**, including any
covered windows. Other sequences and other models continue. Full sequences are
used when isolating misses, so flank-dependent predictions keep their original
context. Reports identify `source_sequence_name`, `model_key`, method/version,
`stage` (`protein`, `peptide`, `wildtype`, or `self_nearest`), error type and
message. WT or self-nearest failures leave their comparator scores missing;
they do not remove successful primary predictions. Successful rows remain
ordinary DataFrame rows, so retain the separate report even after filtering.
`PartialPredictionWarning` also signals incomplete results. Unrelated exceptions
and failures in the handler still raise.

The CLI uses a mandatory JSON sidecar for this opt-in mode:

```bash
topiary --fasta proteins.fasta --mhc-cache-file predictions.csv \
    --cache-miss-report missing-predictions.json --output-csv results.csv
```

The sidecar records `complete`, `prediction_rows` (after filters), and `failures`,
including when all inputs fail or all succeed. Exit status **3** means a partial
prediction run; **0** means no reported cache misses. A malformed command line
exits 2 with the usage text; any other fatal error, including a cache that
cannot answer without this option, exits 1 with one line. CSV stdout remains
parseable. The report is written before the
prediction output, and its path must differ from inputs and outputs.
`complete` concerns cache coverage only, not biological evidence, filtering,
or variant annotation skipped under `--skip-variant-errors`.

Lower-level callers can use the public `predict_with_cache_miss_report` policy.
See the [implementation and compatibility notes](cache-miss-reporting.md).

## Selecting lengths and retrieving different values

A cache replays stored measurements. Different peptide, allele, prediction-kind,
or stored context entries can have different scores. Selecting a different set
of entries does not recompute those scores. For example, a table containing
`SIINFEKL → 11`, `IINFEKLA → 22`, and `SIINFEKLA → 33` returns the first two
values when scanning 8-mers and the last value when scanning 9-mers. These
numbers are illustrative test data, not biological predictions.

```python
cache = CachedPredictor.from_topiary_output("predictions.csv")
print(cache.available_peptide_lengths)  # lengths in the table or its fallback
cache.default_peptide_lengths = [9]
predictor = TopiaryPredictor(models=cache)
df = predictor.predict_from_named_sequences({"protein": "SIINFEKLA"})
cache.default_peptide_lengths = None   # restore all available lengths
```

Length selection limits the protein windows generated before cache lookup.
An unavailable requested length raises an error; available lengths do not
guarantee that every peptide/allele/context is covered. An individual miss
still raises a coverage error or uses the configured fallback.

The selection leaves the cache table intact, including measurements at other
lengths. Explicit peptide inputs (`--peptide-csv`, `--peptide-fasta`, or
`predict_from_named_peptides`) query the supplied peptides as-is. Saving and
reloading a cache preserves its measurements; scan-length selection is a
per-run setting and must be applied again after loading.

From the CLI:

```bash
topiary --fasta proteins.fasta --mhc-cache-file predictions.csv \
    --mhc-peptide-lengths 9 --mhc-alleles 'HLA-A*02:01' \
    --subset-output-columns peptide allele kind value score percentile_rank \
    --output-csv -
```

Omitting the length flag uses all available lengths. The legacy
`--mhc-epitope-lengths` flag also works; `--mhc-peptide-lengths` takes precedence
if both are supplied. Requested length coverage is checked after selecting
the requested alleles.

To inspect other metrics, select columns such as `value`, `score`, and
`percentile_rank` when present; `kind` identifies which prediction each row
describes. Filters and ranking select or reorder results from the stored
measurements. To obtain new predictions for an existing entry, run the intended
live predictor and save a new table, then load that table as the cache.
`--mhc-cache-predictor-version` is a provenance label, not a request to run a
different model. CLI cache flags and `--mhc-predictor` are mutually exclusive.

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

Live MHCflurry output now records its package and official model release when
the weights are loaded (mhctools 3.44.26+, required by Topiary). That identity
survives replay without consulting the MHCflurry installation on the machine
reading the cache. Custom or injected weights need an explicit
`predictor_version` when constructing the MHCflurry wrapper; they are not labeled
as the default release.

For older output missing provenance, supply the identity of the **original
run**, not whichever model is installed now:

```bash
topiary --peptide-csv peptides.csv --mhc-cache-file old-run.csv \
    --mhc-cache-format topiary_output \
    --mhc-cache-predictor-version '2.2.1+release-2.2.0' \
    --output-csv replay.csv
```

The version above is an example: use it only if it describes the generating
run. `--mhc-cache-predictor-name` likewise fills missing method names. These
arguments fill absent columns and unstated cells; conflicting recorded values
raise an error, never silently relabel the predictions. The library loaders
accept the same `prediction_method_name` and `predictor_version` arguments,
including `from_directory`. Numeric-looking versions stay text: `2.10` is not
changed into `2.1`.

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
cache = CachedPredictor.from_netmhcpan_stdout("netmhcpan_run.out")
cache = CachedPredictor.from_netmhc_stdout("netmhc_run.out", version="4")
cache = CachedPredictor.from_netmhcpan_cons_stdout("cons_run.out")
cache = CachedPredictor.from_netmhciipan_stdout("ii_run.out", version="4.3")
cache = CachedPredictor.from_netmhcstabpan_stdout("stab_run.out")
```

Each loader parses the version out of the stdout preamble (e.g.
`NetMHCpan version 4.1b`) and stamps it on `predictor_version`.
Pass `predictor_version="..."` if you want a different label, or
if your capture stripped the preamble.

NetMHCpan run with `-BA` (which produces
both eluted-ligand and binding-affinity scores per peptide) is
loaded as **both** kinds in the cache — `pMHC_affinity` and
`pMHC_presentation` rows for each `(peptide, allele)`. The new
6-tuple index key distinguishes them, and downstream topiary DSL
scopes (`Affinity.*` vs `Presentation.*`) read the right rows.

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
below. Automatic composition is appropriate only when that installation and
bundle generated the file; otherwise pass its original `predictor_version`.

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
when the file doesn't embed that identity. They also fill unstated cells and
reject conflicts with recorded values.

Every row must carry `kind`, such as `pMHC_affinity`, either under that name
or mapped from another column (`columns={"kind": "Assay"}`). The loader does
not guess the kind from a score column or predictor label.

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
Malformed shards report their filename and the required `topiary_output`
format. Missing or unreadable files report the underlying file-access error.

`on_overlap=` decides what happens when shards are not disjoint:

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

A single `CachedPredictor` holds predictions from exactly one
`(prediction_method_name, predictor_version)` pair. Scores from
different model versions aren't interchangeable — even percentile
ranks aren't directly comparable across versions. Mixing them would
produce an output that passes every downstream filter invisibly.

This is enforced in three places. On construction, a DataFrame with several
`(name, version)` pairs raises, and so do `None`, `NaN` and empty-string
values — a silent "I don't know" would mask the invariant. On fallback
attachment (below), the fallback's pair must equal the cache's, verified on
every fallback batch. On `concat` and `from_directory`, every shard must
agree.

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
calls it automatically when you omit `predictor_version`, so model bundles
never have to be enumerated by hand:

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

## Exact peptide occurrences and context

`predict_peptide_occurrences` uses the cache's
`predict_contextual_peptides_dataframe` method to select exactly the requested
peptide, flanks, kinds and MHC scope:

```python
from topiary import CachedPredictor, predict_peptide_occurrences

cache = CachedPredictor.from_topiary_output("predictions.tsv", fallback=live)
cache.alleles = ["HLA-A*02:01", "HLA-B*07:02"]
predictions = predict_peptide_occurrences([
    dict(prediction_id="first", peptide="SIINFEKL",
         source_sequence="AASIINFEKLKK", peptide_offset=2),
    dict(prediction_id="second", peptide="SIINFEKL",
         source_sequence="CCSIINFEKLKK", peptide_offset=2),
], cache)
cache.save("filled.parquet")
```

Only uncovered exact contexts are sent to the fallback. It must declare
`kind_support`; returned peptides, kinds, alleles and context are validated
through the same occurrence workflow used for fresh prediction. Its model
identity is checked before inserting each batch. Existing measurements are
retained. Valid earlier batches remain cached if a later batch fails.

Every declared kind must cover each occurrence, including every selected
allele for per-allele kinds. Haplotype queries match the entire genotype,
independently of the reported best presenter. Assign `cache.alleles` to select
one genotype; its default is the union of stored and fallback alleles. A
haplotype fallback must be configured for that same genotype. Primary and
explicit wild-type comparator occurrences use their own flanks.

Missing request flanks can be derived from that request's nonempty
`source_sequence` and exact `peptide_offset`. This affects the new cache query,
not stored prediction metadata: historical flank values and
`prediction_flanks_supplied` are never reconstructed from a source annotation.

Two strings specify known inference inputs; `""` means an explicitly empty input.
Missing legacy context cannot satisfy a known-context request. A blank legacy
flank cannot establish a known terminus without `prediction_flanks_supplied=True`,
even when the other flank is populated. Peptide-only
queries require `use_flanks=False` (or omission of both flank sequences in the
direct contextual method). They accept explicitly peptide-only measurements
and legacy rows with absent context. Source annotations on a measurement
marked `prediction_flanks_supplied=False` do not become inference context.
The public `prediction_flanks_match` function exposes this matching rule.

The legacy `predict_peptides_dataframe` / `predict_dataframe` entry points
retain their flat lookup behavior, returning all stored contexts. Protein-scan
fallback remains separate, as described below.

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

A miss is routed to `fallback`, and the fallback's output is merged into the
cache, so later queries for the same `(peptide, allele, peptide_length)` are
served locally. There is no flag for this: caching a hit is always right.

The version invariant covers the fallback too — its
`(prediction_method_name, predictor_version)` must equal the cache's, or be in
`also_accept_versions`, and this is checked on every fallback batch.

An empty cache with only a fallback, `CachedPredictor(fallback=live)`, is a
supported read-through mode; the identity is discovered from the fallback's
first output. With no fallback, the default, a miss raises `KeyError` listing
the missed peptides.

Protein scans resolve missing occurrence contexts through the fallback's
`predict_proteins_dataframe` when it is available. Topiary sends each affected
original protein once, keeping its name, full sequence and coordinates. It does
not first run a peptide-only prediction. The cache's selected window lengths
apply, without modifying the fallback's lengths or alleles. Assigning
`cache.alleles` selects one patient genotype from a reusable cache for protein
scans as well as exact-context queries; `cache.alleles = None` restores the
union of stored and fallback scopes.

The scanner must declare `kind_support()` and its configured alleles. Every
requested window in the affected proteins must have all its declared kinds and
MHC scopes. Topiary validates source coordinates, peptide lengths, bounded
flanks, genotype, model/version and coverage before inserting anything. Only
missing compatible measurements are inserted; existing cache hits and stored
provenance stay intact. A failed scanner response leaves the cache, including
an empty cache's identity, unchanged. A different haplotype presenter can match
the same genotype; a different genotype cannot fill its gap.

Repeated queries and saved/reloaded caches reuse the new inference contexts
offline. With no explicit selection, protein scans retain every stored MHC
scope. With a selected genotype, per-allele rows outside it and haplotype rows
for another genotype remain stored but do not participate in lookup or coverage.
A cache-only scan may retain sparse kinds/genotypes, but a mismatched stored
context cannot disappear behind another kind's hit. A fallback with only a
peptide API can fill ordinary peptide misses; it cannot repair stored inference
flanks that disagree with the requested occurrence.

`prediction_flanks_supplied=False` makes flanks source annotations, so scan
output rebinds those annotations to the queried occurrence. With `True`, the
flanks are inference inputs. `prediction_n_flank_limit` and
`prediction_c_flank_limit` record the scanner's configured maximum windows.
Topiary obtains them from mhctools' public `protein_flank_lengths()` method;
MHCflurry resolves its limits from the loaded processing model. A short flank
can reflect a protein boundary, so its observed length never establishes the
window limit. Zero means that side supplies no residues, even inside a protein.

For example, a five-residue N window may supply only `MA` for a peptide near
the start of a protein. Extending that protein to `WWMA...` requires a fresh
prediction. A complete five-residue window still matches further inside a
protein when those five neighbours are unchanged. These limits describe
protein scanning; exact peptide queries continue to compare explicit input
strings without clipping them.

Both limits are nullable integers and participate in cache identity and
CSV/TSV/Parquet persistence. Wide output stores these declarations and the
flank-supplied flag under each model/kind prefix (for example,
`mhcflurry_affinity_n_flank_limit` and `mhcflurry_affinity_flanks_supplied`).
A corresponding `*_present` column records which measurements actually existed,
including rows with missing scores. Reading wide output restores each model's
own declarations; legacy flat metadata columns remain readable. Conflicting
input policies for one model/kind slot require long form or separate exports.
Combined-source observations retain their source/flank identity while these
model-specific declarations accompany their measurements.
`PROTEIN_SCAN_METADATA_COLUMNS` lists the three declaration fields. Legacy files retain unknown limits. A populated
legacy flank must match all available context on that side; a cache cannot
safely use its length to clip a longer protein. Such a query needs a compatible
fallback, a regenerated cache, or limits supplied from the original scanner's
documented configuration. Empty explicit inputs require the corresponding
boundary unless the limit is zero. Missing legacy annotations remain unknown.
The public `prediction_matches_protein_context` predicate exposes this rule;
`prediction_flank_limits` validates and decodes the nullable limits;
`annotate_protein_prediction_context` attaches a producer's declared windows
to fresh scanner output without changing the input frame.
`validate_protein_predictions` validates a configured scanner's complete
occurrence response for other consumers.

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

NetMHCpan stdout capture (format sniffed from preamble):

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-file netmhcpan_run.out \
    --output-csv results.csv
```

mhcflurry CSV (format sniffed from column names; `predictor_version`
auto-composed from the local install):

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-file mhcflurry_predictions.csv \
    --output-csv results.csv
```

Topiary's own saved output (Parquet or TSV round-trip — sniffed):

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-file prior_run.parquet \
    --output-csv results.csv
```

Generic TSV with column mapping (`--mhc-cache-format tsv` is required
here since generic tables can't be auto-detected). This example requires an
`Assay` column containing the kind of each measurement, e.g. `pMHC_affinity`:

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-file third_party.tsv \
    --mhc-cache-format tsv \
    --mhc-cache-predictor-name my-affinity-model \
    --mhc-cache-predictor-version 3.1 \
    --mhc-cache-tsv-column affinity=IC50_nM \
    --mhc-cache-tsv-column percentile_rank=Rank \
    --mhc-cache-tsv-column kind=Assay \
    --output-csv results.csv
```

Sharded — merge every matching file in a directory:

```bash
topiary --peptide-csv peptides.csv \
    --mhc-cache-directory ./caches \
    --mhc-cache-directory-pattern '*.parquet' \
    --output-csv results.csv
```

Full flag reference:

| Flag | Purpose |
|---|---|
| `--mhc-cache-file PATH` | Single cache file. Format is auto-detected; `--mhc-cache-format` only needed for generic TSVs. |
| `--mhc-cache-directory PATH` | Directory of shards; each file loaded via `from_topiary_output` and concatenated. Alternative to `--mhc-cache-file`. |
| `--mhc-cache-directory-pattern GLOB` | Pattern for `--mhc-cache-directory`. Default `*`. |
| `--mhc-cache-format FORMAT` | Optional — sniffed from file content when omitted (see above). One of `topiary_output`, `mhcflurry`, `tsv`, `netmhcpan`, `netmhc`, `netmhccons`, `netmhciipan`, `netmhcstabpan`. Only `tsv` strictly requires the explicit flag. |
| `--mhc-cache-predictor-name NAME` | Fill missing `prediction_method_name` in generic TSV, topiary output or directory shards; recorded values must agree. Other formats already identify their predictor. |
| `--mhc-cache-predictor-version V` | Supply version provenance. For generic TSV, topiary output and shards, fills missing values and rejects conflicts. NetMHC stdout defaults to its preamble; raw MHCflurry defaults to the configured official local bundle. |
| `--mhc-cache-tsv-column CANONICAL=FILE_COL` | Repeatable. Column-name mapping for `tsv` format. |
| `--mhc-cache-tsv-sep SEP` | Separator for `tsv`. Default tab. |
| `--mhc-cache-netmhc-version V` | `3`, `4`, or `4.1` for classic NetMHC output. Default `4`. |
| `--mhc-cache-netmhciipan-version V` | `legacy`, `4`, or `4.3` for NetMHCIIpan. Default `4.3`. |
| `--mhc-cache-netmhciipan-mode MODE` | `binding_affinity` or `elution_score` (default) for NetMHCIIpan 4+. |

`--mhc-predictor` is mutually exclusive with cache flags. `--mhc-alleles`
is optional: omit it to use the cache's allele set, or supply a covered subset.

## When not to use a cache

- You've never run the predictor on these peptides — just use the
  live predictor directly; there's nothing to cache yet.
- You want to mix predictions from multiple model versions in one
  pipeline. Don't. If you think you need this, revisit whether the
  versions are actually interchangeable; if they are, use
  `also_accept_versions`.
