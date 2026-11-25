# Static Typing & `py.typed`

Topiary ships a PEP 561 `topiary/py.typed` marker and runs package-wide static
type checking as part of `./lint.sh` and CI.

## Supported checker and stubs

Install the development tools with `pip install -r requirements-dev.txt`, then
run `./lint.sh`. The supported checker baseline is mypy 2.4.0 with
pandas-stubs 3.0.5.260914 (`mypy>=2.4,<3` and `pandas-stubs>=3,<4` in
`requirements-dev.txt`), pinned together for reproducible pandas 3 signatures.

## Checked scope

`[tool.mypy]` in `pyproject.toml` checks the entire `topiary` package and
`tests/typing` (`files = ["topiary", "tests/typing"]`) with:

- `python_version = "3.11"` (matching `requires-python`);
- `check_untyped_defs = true` (function bodies across all package modules are
  checked even where annotations are partial);
- `no_implicit_optional = true`;
- `warn_unused_ignores = true` and `warn_redundant_casts = true`.

There is no global `ignore_missing_imports`, no `follow_imports = "silent"`,
and no per-module error suppression. Only the four upstream dependencies that
do not yet publish type information or a `py.typed` marker (`gtfparse`,
`mhcgnomes`, `mhctools`, and `osteosarc`) are listed in the narrow
`[[tool.mypy.overrides]]` entry in `pyproject.toml`.

## Consumer verification and packaging

- `tests/typing/public_tables.py` checks table, metadata, view, cache, and
  Isovar hypothesis workflows, including negative cases whose expected errors
  must stay errors.
- `tests/typing/public_api.py` checks expanded public APIs including
  `ProteinFragment` and fragment I/O, LENS loaders, read/DNA/RNA evidence
  helpers, allele class derivation, and cleavage tables, along with negative
  boundary checks.
- `tests/check_distributions.py` verifies that `topiary/py.typed` is included
  in both the wheel and source distribution, and the `installed-wheel` CI job
  verifies `py.typed` in the installed package.

## Relationship to runtime validation

Prediction-table cells, custom metadata, and dynamically supplied predictor
plugins still use `Any` where the input schema is open. Pandas-compatible
indexing on `TopiaryResult` can return a `Series` or another `TopiaryResult`;
use `.df` when a consumer needs pandas' full indexing API.

Static typing complements, and does not replace, runtime input validation and
clear error messages at public, notebook, and CLI boundaries.
