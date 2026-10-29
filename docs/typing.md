# Public API type checks

Topiary checks its table and cache boundary incrementally. Install the
development tools with `pip install -r requirements-dev.txt`, then run
`./lint.sh`. The supported checker is mypy 2.4.0 with pandas-stubs
3.0.5.260914, pinned together for reproducible pandas signatures.

The current checked scope is:

- `TopiaryResult` tables, metadata, long/wide views and result operations;
- native TSV/CSV readers and writers, including nullable `Metadata` fields;
- `CachedPredictor` loading, querying and saving;
- Isovar hypothesis and patient-background export readers.

`tests/typing/public_tables.py` checks a consumer that loads a table, changes
views, queries a cache and saves the result. It also includes invalid calls
whose expected errors must remain errors. Function bodies in the declared
modules are checked, including functions without complete annotations.
Imported source modules supply their available type information but are not
yet part of this diagnostic scope (`follow_imports = "silent"`).

Prediction-table cells, custom metadata and dynamically supplied predictor
plugins still use `Any` where the input schema is open. Pandas-compatible
indexing can return a Series or another TopiaryResult; use `.df` when the
consumer needs pandas' full indexing API. Three specific MHCTools/MHCgnomes
imports have local `import-untyped` allowances because those upstream
packages do not publish type information. There is no global missing-import
ignore and no blanket module error suppression.

This is a development gate, not a package-wide typing promise. Topiary does
not yet publish `py.typed`. Expand the named checked scope and qualify
installed-package consumers before adding that marker. Runtime validation
and error messages remain necessary for notebook and CLI inputs.
