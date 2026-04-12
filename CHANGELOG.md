# Changelog

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

**Deprecations (to be removed in 5.0 alongside the DSL refactor,
[#111](https://github.com/openvax/topiary/issues/111)):**

- `EpitopeFilter`, `ColumnFilter`, `ExprFilter`, `RankingStrategy`
  will be replaced with a unified `Comparison` / `BoolOp` DSL tree.
  Their `to_expr_string()` and `to_ast_string()` methods, added in
  this release, are a stopgap for round-tripping filter metadata.

## 4.9.0

- Require `mhctools>=3.7.0`.
- Rename CLI sorting flags to `--sort-by` and `--sort-direction`.
- Add Python API `sort_by=` and `.sort_by(...)`, while keeping `rank_by` as a compatibility alias.
- Treat comma-separated `--sort-by` keys as lexicographic tie breakers, with fallthrough on missing values.
- Document upstream `mhctools 3.7.0+` support for multi-predictor CLI invocations, the simplified `Kind` API, and the updated NetChop/Pepsickle behavior.
