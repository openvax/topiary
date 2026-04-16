# Changelog

## 5.7.0

**CachedPredictor — CLI, multi-kind, flanks, NetMHC fixtures (#136).**

**CLI support for cached predictions:**

- New `--mhc-cache-file` / `--mhc-cache-directory` CLI arguments let
  users run topiary entirely from pre-computed prediction files without
  invoking a live MHC predictor.
- `--mhc-cache-format` is optional — topiary sniffs the format from
  file content (NetMHC-family preamble lines, mhcflurry column names,
  topiary-output schema, Parquet magic bytes).  Only the generic `tsv`
  format requires an explicit flag.
- `--mhc-predictor` and `--mhc-alleles` become optional when a cache
  supplies predictions.

**Multi-kind cache (closes #137):**

- Cache index expanded from `(peptide, allele, peptide_length)` to a
  6-tuple `(peptide, allele, peptide_length, kind, n_flank, c_flank)`.
  A single cache holds every kind a predictor emits — mhcflurry's
  class1_presentation pipeline (affinity + presentation + processing),
  NetMHCpan `-BA` (affinity + presentation), etc.  No more silent
  data loss from single-kind heuristics.
- `from_mhcflurry` explodes wide-format CSVs into one row per
  `(peptide, allele, kind)`, preserving `n_flank` / `c_flank` /
  `source_sequence_name` / `peptide_offset` / `sample_name` per kind.
- `from_netmhcpan_stdout` switches to `parse_netmhcpan_to_preds`
  (mhctools' multi-kind API), returning all kinds instead of collapsing
  by mode.  Dropped `mode=` kwarg (no longer meaningful).
- Generic TSV loader (`from_tsv`) now requires a `kind` column per row.
  Multi-kind TSVs work natively — add a kind column and list one value
  per row.

**Multi-allele NetMHC parsing fixed:**

- `parse_netmhcpan_to_preds` handles multi-allele stdout correctly
  (per-allele header lines that crashed the old
  `parse_netmhcpan_stdout` are no longer an issue).  Multi-allele
  fixtures promoted from xfail to happy-path tests.

**Flank sensitivity in cache key:**

- `n_flank` / `c_flank` are now part of the composite key.  mhcflurry's
  processing and presentation predictions depend on flanking residues;
  the same peptide at different protein positions can produce different
  scores.  Absent flanks normalize to empty string `""` (no None/NaN
  handling quirks).

**Real NetMHC-family fixtures:**

- `tests/data/netmhc_fixtures/` — captured from netmhc-bundle binaries
  for peptide SLLQHLIGL at HLA-A*02:01 / A*24:02 / B*07:02.
  NetMHCpan 4.0 + 4.1, NetMHC 4.0, NetMHCstabpan, single-allele +
  multi-allele variants.  6 real-fixture tests pin actual numeric
  predictions through the loaders.

**README reorder:**

- "MHC prediction models" moved near the top (after "Predicting MHC
  binding"); "Cached predictions" moved to the end.

**Tests:**

- 1141 tests pass (up from 1111 in v5.6.0).  9 new CLI integration
  tests, 6 real-fixture tests, 3 promoted multi-allele happy-path
  tests, multi-kind + multi-flank regression tests.

## 5.6.0

**Closes #128 — `CachedPredictor` reaches feature-complete.**

**New loaders for the DTU NetMHC suite (#132):**

- `CachedPredictor.from_netmhcpan_stdout(path, mode=…)` — auto-detects
  NetMHCpan 2.8 / 3 / 4 / 4.1. `mode` selects `"binding_affinity"` or
  `"elution_score"` for 4+.
- `CachedPredictor.from_netmhc_stdout(path, version=…)` — classic
  NetMHC 3 / 4 / 4.1.
- `CachedPredictor.from_netmhcpan_cons_stdout(path)` — NetMHCcons.
- `CachedPredictor.from_netmhciipan_stdout(path, version=…)` —
  NetMHCIIpan legacy / 4 / 4.3.
- `CachedPredictor.from_netmhcstabpan_stdout(path)` — NetMHCstabpan
  pMHC-stability predictor.

Each loader wraps an existing `mhctools.parsing.*_stdout` function
(zero new parsing code) and parses the tool version out of the
stdout preamble onto `predictor_version`. Parses stdout text, not
the `-xlsfile` tab-delimited variant — flagged in `docs/cached.md`.

**Sharding — `concat` + `from_directory`:**

- `CachedPredictor.concat([caches], on_overlap=…)` — merge several
  caches into one. All shards must share `(name, version)` per the
  core invariant.
- `CachedPredictor.from_directory(path, pattern="*", on_overlap=…)` —
  glob a directory and concat every matching file through
  `from_topiary_output`.
- Overlap resolution policies (`on_overlap`): `"raise"` (default — fail
  if any `(peptide, allele, peptide_length)` appears in more than one
  shard), `"last"` (later shard wins), `"first"` (earlier wins), or a
  user-supplied `callable(row_a, row_b) -> row` resolver.

**Polish from vaxrank-consumer review on #130 (#131):**

- `_fallback_resolve` filters fallback output to keys not already in
  the index before merging, so a partial-allele cache (peptide P
  present for allele A, missing for B) doesn't see its `(P, A)` row
  silently overwritten by the fallback's all-alleles response.
- Class docstring now flags silent peptide-length lock-in and
  non-thread-safety.
- `save()` raises on an empty never-queried cache with no identity,
  so users don't write schema-only files that can't be round-tripped.

**Tests:**

- 59 tests in `tests/test_cached_predictor.py` (up from 41): 6 NetMHC
  loader tests, 12 sharding tests. Full suite 1111 passed (up from
  1093).

## 5.5.0

**New feature — `CachedPredictor`:**

- Pluggable prediction source (part 1 of #128) that loads MHC binding
  predictions from a pre-computed table and plugs into
  `TopiaryPredictor(models=…)` alongside live mhctools predictors.
  Use cases: reproducibility, iterating on filters/ranking without
  rerunning the predictor, per-allele / per-sample parallel
  predictions, ingesting output from tools topiary doesn't natively
  run.
- Loaders shipped: `CachedPredictor.from_dataframe`,
  `from_topiary_output` (Parquet / TSV), `from_tsv` (generic with
  column mapping), `from_mhcflurry` (maps `mhcflurry_*` columns onto
  canonical names).
- NetMHCpan / NetMHC / NetMHCstabpan / NetMHCIIpan / NetMHCcons
  loaders are queued for a follow-up PR.
- Sharding (`concat` / `from_directory`) is queued for a separate
  follow-up.

**Version invariant:**

- A single `CachedPredictor` holds exactly one
  `(prediction_method_name, predictor_version)` pair; `None` / `NaN`
  / empty-string values are rejected at construction. Mixing versions
  would produce outputs that pass downstream filters invisibly, so
  the invariant is enforced everywhere (load, fallback attach, concat).
- Explicit opt-in equivalence: pass `also_accept_versions={"…", …}`
  when two labels really are interchangeable (rc → final, timestamp-
  only model-data reflashes).

**mhcflurry-specific version composition:**

- New `topiary.mhcflurry_composite_version()` helper discovers the
  locally-installed mhcflurry package version plus its active model
  release and returns a composite string like `"2.2.1+release-2.2.0"`.
  `CachedPredictor.from_mhcflurry(path)` uses it automatically when
  no explicit `predictor_version` is passed — users never enumerate
  model bundles manually.

**Fallback mode:**

- Pass `fallback=<live_predictor>` to delegate cache misses; results
  are merged back into the cache so subsequent queries serve locally.
  No separate flag — caching fallback hits is always right for the
  batch-prediction workload.
- Pure read-through: `CachedPredictor(fallback=p)` with no df starts
  empty; identity is discovered from the fallback's first output.

**Documentation:**

- New `docs/cached.md` covering the full surface.
- `CachedPredictor` section added to `docs/api.md`.
- Subsection in `docs/quickstart.md`.
- README has a top-level "Cached predictions" section.
- Feature list in `docs/index.md` updated.

**Tests:**

- 38 new tests in `tests/test_cached_predictor.py` (up from 0),
  covering construction, version invariant (mixed rows, null rejection,
  name/version round-trip as string), predict_peptides +
  predict_proteins sliding-window, fallback hit + miss + version
  mismatch + empty-cache identity discovery, `also_accept_versions`,
  all four loaders, `mhcflurry_composite_version` via stubbed
  mhcflurry module (no tensorflow/libomp collisions), and
  integration with `TopiaryPredictor(models=cache)`.
- Full suite: 1090 passed (up from 1052), 3 skipped.

**Related upstream issue:**

- Filed `openvax/mhctools#193` — `predict_peptides_dataframe` misses
  `predictor_version` / `kind` / `value` columns returned by
  `predict_proteins_dataframe`. `CachedPredictor` currently backfills
  the gap internally; can simplify once the mhctools asymmetry is
  resolved.

## 5.4.0

**Breaking rename (no back-compat alias):**

- `AntigenFragment` → `ProteinFragment`. Describes what the object is
  (a slice of some protein — natural, chimeric, foreign, or designed)
  rather than what it's used for. Matches Isovar's convention.
- `topiary/antigen.py` → `topiary/protein_fragment.py`;
  `topiary/io_antigen.py` → `topiary/io_protein_fragment.py`;
  `docs/antigens.md` → `docs/fragments.md`.
- `TopiaryPredictor.predict_from_antigens(fragments)` →
  `predict_from_fragments(fragments)`.
- `read_antigens` / `write_antigens` / `iter_antigens` →
  `read_fragments` / `write_fragments` / `iter_fragments`.

**Downstream migration checklist:**

- `from topiary import AntigenFragment` → `from topiary import ProteinFragment`.
- `from topiary.antigen import …` / `from topiary.io_antigen import …` →
  `from topiary.protein_fragment import …` /
  `from topiary.io_protein_fragment import …`.
- `predictor.predict_from_antigens(fragments)` →
  `predictor.predict_from_fragments(fragments)`.
- `topiary.read_antigens(path)` / `write_antigens(fragments, path)` /
  `iter_antigens(path)` → `read_fragments` / `write_fragments` /
  `iter_fragments`.
- TSV files written by 5.2.x `write_antigens` remain readable by
  5.4.0 `read_fragments`: the new `transcript_name` column is
  optional and defaults to `None` when missing.  TSVs written by
  5.4.0 are **not** readable by ≤5.2.x (the old reader rejects
  unknown columns).
- Unaffected surface: `TopiaryPredictor`, `EvalContext`, `apply_filter`,
  `predict_from_variants` / `predict_from_mutation_effects` / the
  legacy column contract.

**Refactor (predict_from_variants now builds on ProteinFragment):**

- `predict_from_mutation_effects` builds a list of `ProteinFragment`s
  from varcode effects (via the new `_fragment_from_effect` adapter)
  and delegates to a shared `_build_fragment_rows` step — one prediction
  pipeline instead of two. The ~60-line row-by-row metadata loop is gone.
- New fragment-derived columns (`fragment_id`, `source_type`,
  `overlaps_target`, `wt_peptide` / `wt_peptide_length`) now flow
  through the variant path alongside the legacy columns.
- Legacy column contract preserved: absolute `peptide_offset`,
  `mutation_start_in_peptide` / `mutation_end_in_peptide`,
  `transcript_name`, `contains_mutant_residues`, `only_novel_epitopes`,
  and legacy `gene_expression_dict` / `transcript_expression_dict`
  plumbing all behave identically to 5.2.0.
- `source_type` classification aligned with `docs/fragments.md`
  vocabulary: `PrematureStop` → `variant:stop_gain`, multi-residue
  `Substitution` → `variant:indel`, unlisted effect classes fall back
  to `variant:<classname_lowered>`.
- Filter / sort now run after `peptide_offset` rebasing on the variant
  path, so filter expressions referencing `peptide_offset` see absolute
  protein coordinates (matches 5.1.x behavior).

**New field:**

- `ProteinFragment.transcript_name` — human-readable transcript label
  alongside `transcript_id`. Threaded through `from_dict`, `from_variant`,
  `from_junction`, and the TSV IO schema.

**Internal:**

- New `TopiaryPredictor._build_fragment_rows(fragments)` — fragment
  scanning + metadata overlay without filter / sort.  Public entry
  points layer filter / sort / `only_novel_epitopes` on top.
  Underscore-prefixed annotation keys are reserved for internal
  plumbing and never surface as DataFrame columns.
- 18 new regression tests covering legacy column contract, expression-
  dict plumbing, and the effect→fragment source_type classifier —
  including a parametrized grid pinning every entry of the documented
  `source_type` vocabulary (`variant:snv`, `variant:indel`,
  `variant:frameshift`, `variant:stop_gain`, `variant:stop_loss`,
  `variant:start_loss`, `variant:exon_loss`, `variant:alternate_start`,
  plus the `variant:<classname_lowered>` fallback).
- `tests/test_frameshift_fragments.py` — new regression suite (75
  cases) pinning `_fragment_from_effect` behavior on varcode
  `FrameShift` / `FrameShiftTruncation` effects: target_intervals
  span the full downstream novel tail, per-peptide `overlaps_target`
  agrees with ground truth across peptide lengths 8–11,
  `inframe=True`/`False` produce identical intervals for frameshift
  shapes, and `only_novel_epitopes=True` preserves every downstream
  9-mer.

## 5.2.0

**New features (core abstraction for antigens from any origin):**

- `AntigenFragment` — a universal record for a protein/peptide sequence
  with source-type, target-region, and comparator metadata. Carries
  variants, structural variants, ERVs, CTAs, viral proteins, allergens,
  autoantigens, and synthetic constructs through one pipeline. Free-form
  `source_type` tag (recommended vocabulary documented, not enforced);
  `target_intervals: list[tuple[int, int]]` for disjoint regions
  (breakpoints of tandem duplications, non-self regions of ERVs, etc.);
  `reference_sequence` + `germline_sequence` with germline-precedence
  `effective_baseline`. Equality/hash keyed on `fragment_id` (stable
  human-readable prefix + SHA-1 hash). Convenience constructors
  `from_variant`, `from_junction`. Stdlib-only serialization:
  `to_dict` / `from_dict` / `to_json` / `from_json`.
- `topiary.read_antigens(path)` / `write_antigens(fragments, path)` /
  `iter_antigens(path)` — TSV IO with JSON-serialized list/dict columns.
- `TopiaryPredictor.predict_from_antigens(fragments)` — new entry point
  that scans each fragment's sequence, propagates every fragment field
  (including arbitrary annotations) onto prediction rows, threads
  `fragment_id` through for downstream grouping (vaxrank vaccine-window
  selection), and emits an `overlaps_target` column computed from each
  peptide's position vs. the fragment's target intervals. Backwards-compat
  `contains_mutant_residues` alias for `source_type` prefixed with
  `variant`. `wt_peptide` derived by slicing `effective_baseline`;
  model-side WT predictions deferred to a follow-up PR.
- `self_nearest` — reserved DSL scope for cross-reactivity filtering
  ("closest peptide in essential healthy tissues"). Topiary does not
  compute these columns — producers populate via BLAST / edit distance
  against a healthy-tissue proteome with their own "self" definition.
  The scope reads `self_nearest_*` columns when present, returns NaN
  otherwise. See `docs/antigens.md` for the reserved column namespace.
- `fragment_id` is now preferred over `variant` as the group key in the
  DSL's group-by logic (falls back to `variant`, then
  `source_sequence_name`).

**Internal:**

- New module `topiary/antigen.py` (dataclass + helpers) and
  `topiary/io_antigen.py` (TSV IO).
- 63 new tests covering identity, serialization, geometry,
  `predict_from_antigens` propagation, `self_nearest` scope reads.

## 5.1.0

**New features:**

- `topiary.read_lens(path)` — load LENS (Landscape of Effective
  Neoantigens Software) reports into Topiary's wide-form schema.
  Handles the three observed schema variants (v1.4, v1.5.1, v1.9-dev)
  with column-based version detection. Binding columns are remapped to
  `{model}_{kind}_{field}`; per-model versions populate
  `Metadata.models`. LENS-specific columns (`erv_*`, `priority_score_*`,
  `b2m_*`, `hla_allele_*`, etc.) pass through as annotations and remain
  accessible via `Column("…")` in the DSL. See
  [#110](https://github.com/openvax/topiary/issues/110). Known losses:
  `peptide_offset` set to 0 (LENS doesn't record it);
  `contains_mutant_residues` / `mutation_start_in_peptide` left NaN
  (LENS's `mut_aa_pos` semantics are ambiguous); `n_flank` / `c_flank`
  derived from `pep_context` only for SNV / SPLICE / FUSION.
- `DSLNode.logistic_normalized(midpoint, width)` — logistic sigmoid
  rescaled to reach 1 as `x → -∞`, so the output is a proper
  `[0, 1]` score.  `.logistic(...)` is unchanged.
  ([#116](https://github.com/openvax/topiary/issues/116))
- Allele normalization uses `mhcgnomes` unconditionally (Class I,
  Class II, mouse all supported).

## 5.0.1

Polish pass on the v5.0.0 DSL refactor — no user-visible behavior
changes, just internal cleanup.

- `DSLNode.child_nodes()` — new abstract method on every node type.
  Generic tree walkers (column validation, future AST rewriters) no
  longer need a per-node `isinstance` ladder.  `_collect_column_names`
  now uses it.
- The scoped-field filter guard moves from four per-operator overrides
  on `Field` (`__le__` / `__ge__` / `__lt__` / `__gt__`) into a single
  check in `Comparison.__init__`.  Same error, less surface area.
- `apply_filter` now reindexes the evaluated Series to
  `ctx.group_index` before masking, so an index mismatch surfaces as
  NaN → False rather than as misaligned row selection.

## 5.0.0

**Breaking changes (DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- Filter leaves (`EpitopeFilter`, `ColumnFilter`, `ExprFilter`) and the
  composite (`RankingStrategy`, `SortSpec`) are removed. Every DSL
  expression is now a single `DSLNode` tree whose `.eval(ctx)` returns
  a `pandas.Series` indexed by peptide-allele group tuples.
- `Affinity <= 500` (and friends) now returns a `Comparison` node;
  `A | B` / `A & B` returns a `BoolOp`. Both classes inherit the full
  arithmetic operator set, so boolean-as-number composition
  (`(Affinity <= 500) * Affinity.score`) is allowed.
- `apply_ranking_strategy` is split into `apply_filter(df, node)` and
  `apply_sort(df, sort_nodes, sort_direction="auto")`.
- `parse_ranking`, `parse_filter`, `parse_expr` are collapsed into a
  single `parse()` that returns a `DSLNode`. The parser uses standard
  precedence for `&` / `|` (`&` binds tighter); mixed-operator strings
  are now accepted.
- `TopiaryPredictor` kwargs: `ranking_strategy`, `ranking`, `filter`,
  `rank_by`, `ic50_cutoff`, and `percentile_cutoff` are removed. Use
  `filter_by=` (a `DSLNode` or string) and `sort_by=` (a `DSLNode` or
  list).  The `TopiaryPredictor.ranking_strategy` property is replaced
  by the separate `.filter_by` / `.sort_by` attributes.
- `Field` gains an optional `version` parameter;
  `Affinity["netmhcpan", "4.1b"]` filters on both
  `prediction_method_name` and `predictor_version`.
- Ambiguity semantics tightened — unqualified `Affinity.value` on a
  DataFrame that contains multiple `prediction_method_name` values
  raises `ValueError` pointing at `Affinity["modelname"]`.
  Previously the old filter silently passed if *any* row satisfied the
  threshold.
- `apply_filter` now errors when the evaluated Series contains values
  outside `{True, False, 0, 1, 0.0, 1.0, NaN}`, pointing the user at
  `<=` / `>=`. NaN still maps to `False`.

**New:**

- `EvalContext`, `DSLNode`, `Const`, `Column`, `Field`, `BinOp`,
  `UnaryOp`, `NormExpr`, `SurvivalExpr`, `LogisticExpr`, `ClipExpr`,
  `AggExpr`, `Comparison`, `BoolOp` exported from `topiary.ranking`.
- `apply_filter`, `apply_sort`, `parse` exported as the top-level DSL
  entry points.
- Every `DSLNode` has a `to_expr_string()` that round-trips through
  `parse()`.

## 4.12.0

**Breaking changes:**

- `topiary.read_tsv` and `topiary.read_csv` now return a `TopiaryResult`
  instead of an `(DataFrame, Metadata)` tuple. Callers using tuple
  unpacking must migrate: `df, meta = read_tsv(path)` →
  `result = read_tsv(path); df, meta = result.df, result.metadata`.

**New features:**

- `TopiaryResult` class bundling a predictions DataFrame with provenance
  (model versions, source files, form, filter/sort history).  Delegates
  common DataFrame operations (`len`, `iter`, `columns`, `shape`, `head`,
  `iterrows`, etc.) so most existing DataFrame-style code continues to
  work.  Provides `to_wide()`, `to_long()`, `to_tsv()`, `to_csv()`,
  `filter_by()`, `sort_by()`.
- `topiary.concat([r1, r2, ...])` merges `TopiaryResult`s, unioning
  models (warns on version conflicts), concatenating sources, and
  preserving filter/sort history only if all inputs agree.
- `read_tsv` / `read_csv` accept a `tag=` kwarg to label the source of
  the loaded rows; defaults to the filename.  Auto-populates a `source`
  column on the DataFrame.
- `Metadata` gains a `sources: list[str]` field; the comment block
  supports multiple `#source=...` lines.

**Deprecations (removed in 5.0 alongside the DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- `EpitopeFilter`, `ColumnFilter`, `ExprFilter`, `RankingStrategy`
  replaced by a unified `Comparison` / `BoolOp` DSL tree. See the 5.0.0
  entry above for migration details.

## 4.9.0

- Require `mhctools>=3.7.0`.
- Rename CLI sorting flags to `--sort-by` and `--sort-direction`.
- Add Python API `sort_by=` and `.sort_by(...)`, while keeping `rank_by` as a compatibility alias.
- Treat comma-separated `--sort-by` keys as lexicographic tie breakers, with fallthrough on missing values.
- Document upstream `mhctools 3.7.0+` support for multi-predictor CLI invocations, the simplified `Kind` API, and the updated NetChop/Pepsickle behavior.
