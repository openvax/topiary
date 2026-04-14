# Changelog

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
