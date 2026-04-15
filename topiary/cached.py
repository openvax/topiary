"""CachedPredictor — serve MHC binding predictions from a pre-computed table.

Plugs into :class:`topiary.TopiaryPredictor` alongside live mhctools
predictors.  Supports three producer paths:

1. External predictor output files (mhcflurry CSV, NetMHC-family
   stdout captures, generic TSV/CSV with column mapping).
2. Topiary's own saved predictions (``predict_*`` output round-tripped
   through Parquet / TSV).
3. Caller-supplied DataFrames (programmatic / in-memory construction).

All three load into the same internal index keyed by
``(peptide, allele, peptide_length)``.

Core invariant
--------------
A single :class:`CachedPredictor` holds predictions from exactly one
``(prediction_method_name, predictor_version)`` pair — never mixes
versions.  Enforced at load and on fallback-predictor attachment.

Fallback semantics
------------------
- ``fallback=None`` (default): a miss raises ``KeyError``.
- ``fallback=<predictor>``: misses delegate to the fallback, and the
  result is merged back into the cache so subsequent queries for the
  same ``(peptide, allele, peptide_length)`` are served locally.
  The fallback's ``(prediction_method_name, predictor_version)`` must
  match the cache's — verified lazily on the first fallback call.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Union

import pandas as pd


# Columns a cache row must carry so the core invariant and lookup work.
_REQUIRED_COLUMNS = (
    "peptide", "allele", "peptide_length", "kind",
    "prediction_method_name", "predictor_version",
)

# All columns the cache preserves on load and round-trip.
_CACHE_COLUMNS = (
    "peptide", "allele", "peptide_length", "kind",
    "score", "affinity", "percentile_rank", "value",
    "prediction_method_name", "predictor_version",
    # Provenance / context — carried through when the source file has
    # them.  n_flank / c_flank matter for mhcflurry's processing
    # prediction (flanking residues influence the score);
    # source_sequence_name + peptide_offset identify which protein +
    # position a peptide came from; sample_name discriminates multi-
    # sample inputs.  All optional; absent → None.
    "source_sequence_name", "peptide_offset",
    "n_flank", "c_flank", "sample_name",
)

# Composite key for the cache index.
#
# - (peptide, allele, peptide_length): basic identity.
# - kind: distinguishes affinity / presentation / processing / stability
#   — needed because mhcflurry's class1_presentation pipeline and
#   NetMHCpan's -BA flag emit multiple kinds per (peptide, allele).
# - n_flank, c_flank: some predictors (mhcflurry's antigen_processing,
#   mhcflurry's pMHC_presentation) take flanking residues as input, so
#   the same peptide at two different protein contexts produces
#   different scores.  Absent flanks (None) coexist cleanly with
#   populated flanks — predictors that don't use flanks just produce
#   a single (None, None) entry per (peptide, allele, kind).
_KEY_COLS = (
    "peptide", "allele", "peptide_length", "kind",
    "n_flank", "c_flank",
)


class CachedPredictor:
    """Predictor that answers MHC binding queries from a pre-computed table.

    Implements enough of the mhctools predictor protocol
    (``predict_peptides_dataframe``, ``predict_proteins_dataframe``,
    ``alleles``, ``default_peptide_lengths``) to drop into
    :class:`topiary.TopiaryPredictor` as a model.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized rows (one per ``(peptide, allele, peptide_length)``)
        with at least the columns in ``_REQUIRED_COLUMNS``.
    fallback : mhctools predictor or CachedPredictor, optional
        If set, cache misses are delegated here and the results are
        merged back into the cache.  Its
        ``(prediction_method_name, predictor_version)`` must match the
        cache's.

    Notes
    -----
    **Peptide-length coverage.**  ``default_peptide_lengths`` is derived
    from the lengths actually present in the cache (union with the
    fallback's lengths when one is set).  A cache loaded from a file
    that only contains 9-mers will silently scan only 9-mer windows in
    ``predict_proteins_dataframe``.  If your source table was intended
    to cover multiple lengths but doesn't, you won't notice it here —
    check ``cache.default_peptide_lengths`` after loading.

    **Thread safety.**  :meth:`predict_peptides_dataframe` and
    :meth:`predict_proteins_dataframe` mutate internal state (the
    ``(peptide, allele, peptide_length)`` index and the backing
    DataFrame) when a fallback is configured.  The class is **not
    thread-safe**; wrap calls in a lock if the cache is shared across
    worker threads.
    """

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        fallback=None,
        *,
        also_accept_versions: Optional[Iterable[str]] = None,
    ):
        self.fallback = fallback
        self.also_accept_versions = (
            frozenset(also_accept_versions) if also_accept_versions else frozenset()
        )
        self._fallback_verified = False

        # Empty-cache + fallback: (name, version) discovered lazily on
        # the first fallback call.
        if df is None or len(df) == 0:
            if fallback is None:
                raise ValueError(
                    "CachedPredictor: pass either `df` (pre-loaded rows) "
                    "or `fallback` (a live predictor).  An empty cache "
                    "with no fallback has no way to answer queries."
                )
            self._df = pd.DataFrame(columns=list(_CACHE_COLUMNS))
            self.prediction_method_name = None
            self.predictor_version = None
            self._index = {}
            return

        self._df = self._normalize(df)
        self.prediction_method_name, self.predictor_version = \
            self._unique_version_pair(self._df)
        self._index = self._build_index(self._df)

    # --- normalization + invariants ---------------------------------

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        missing = set(_REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"CachedPredictor rows missing required columns: "
                f"{sorted(missing)}.  Provide them in the DataFrame or "
                f"pass predictor_name / predictor_version to the loader."
            )
        # Reject null / empty identity columns before coercing to str
        # ("".astype(str) → "", "None" etc. pass a naive check).
        for col in ("prediction_method_name", "predictor_version"):
            na_mask = df[col].isna() | (
                df[col].astype(str).str.strip().isin(["", "None", "nan", "NaN"])
            )
            if na_mask.any():
                raise ValueError(
                    f"CachedPredictor: column {col!r} must be a non-empty "
                    f"string on every row (got {int(na_mask.sum())} "
                    f"null/empty value(s)).  Silent None/NaN would mask "
                    f"the version invariant — supply a value via the "
                    f"loader's predictor_name / predictor_version args."
                )
        # Reject null / empty kind the same way — multi-kind cache
        # keys on (peptide, allele, length, kind); NaN/missing kind
        # would silently collide across kinds.
        na_mask = df["kind"].isna() | (
            df["kind"].astype(str).str.strip().isin(["", "None", "nan", "NaN"])
        )
        if na_mask.any():
            raise ValueError(
                f"CachedPredictor: column 'kind' must be a non-empty "
                f"string on every row (got {int(na_mask.sum())} "
                f"null/empty value(s)).  The cache keys on "
                f"(peptide, allele, peptide_length, kind); missing "
                f"kind would collide across kinds."
            )
        keep = [c for c in _CACHE_COLUMNS if c in df.columns]
        out = df[keep].copy()
        # n_flank / c_flank are part of the composite key — backfill
        # with None when the source file doesn't provide them so every
        # row has a well-defined key shape.
        for flank_col in ("n_flank", "c_flank"):
            if flank_col not in out.columns:
                out[flank_col] = None
        out["peptide"] = out["peptide"].astype(str)
        out["allele"] = out["allele"].astype(str)
        out["peptide_length"] = out["peptide_length"].astype(int)
        out["kind"] = out["kind"].astype(str)
        # Version strings may look numeric ("1.0") and get coerced by
        # pandas on TSV/CSV reload — force to str so the invariant
        # compares the same shape on both sides of a round-trip.
        out["prediction_method_name"] = out["prediction_method_name"].astype(str)
        out["predictor_version"] = out["predictor_version"].astype(str)
        return out

    @staticmethod
    def _unique_version_pair(df: pd.DataFrame) -> tuple:
        pairs = df[["prediction_method_name", "predictor_version"]].drop_duplicates()
        if len(pairs) == 0:
            raise ValueError(
                "CachedPredictor: empty DataFrame has no "
                "(prediction_method_name, predictor_version) pair."
            )
        if len(pairs) > 1:
            pair_list = ", ".join(
                f"({r.prediction_method_name!r}, {r.predictor_version!r})"
                for r in pairs.itertuples(index=False)
            )
            raise ValueError(
                f"CachedPredictor rows span multiple "
                f"(prediction_method_name, predictor_version) pairs: "
                f"{pair_list}.  A single cache must hold predictions from "
                f"exactly one model version."
            )
        row = pairs.iloc[0]
        return row["prediction_method_name"], row["predictor_version"]

    @staticmethod
    def _build_index(df: pd.DataFrame) -> dict:
        """Build ``(peptide, allele, peptide_length, kind, n_flank,
        c_flank) → row_dict`` index.  The full key lets a single cache
        hold multiple kinds per (peptide, allele) — and, within a kind
        that depends on flanking context (mhcflurry processing and
        presentation), distinguish the same peptide at different
        source-protein positions."""
        return {
            (
                str(r["peptide"]),
                str(r["allele"]),
                int(r["peptide_length"]),
                str(r["kind"]),
                _flank_key(r.get("n_flank")),
                _flank_key(r.get("c_flank")),
            ): r.to_dict()
            for _, r in df.iterrows()
        }

    # --- mhctools protocol ------------------------------------------

    @property
    def alleles(self):
        a = set(self._df["allele"].unique().tolist())
        if self.fallback is not None:
            a.update(getattr(self.fallback, "alleles", []))
        return sorted(a)

    @property
    def default_peptide_lengths(self):
        lengths = set(int(x) for x in self._df["peptide_length"].unique().tolist())
        if self.fallback is not None:
            lengths.update(getattr(self.fallback, "default_peptide_lengths", []))
        return sorted(lengths)

    def _cache_alleles(self):
        return sorted(set(self._df["allele"].unique().tolist()))

    def predict_peptides_dataframe(
        self, peptides: Iterable[str],
    ) -> pd.DataFrame:
        """Return one row per ``(peptide, allele, kind)`` — every kind
        the cache carries for a given ``(peptide, allele)`` produces
        its own row.  mhcflurry's class1_presentation pipeline stores
        three kinds per key (affinity + presentation + processing);
        NetMHCpan ``-BA`` stores two (affinity + elution score).  This
        matches the shape ``mhctools.predict_proteins_dataframe`` emits.

        Misses are resolved through ``self.fallback`` (which merges
        into the cache) if set; else raise ``KeyError``.
        """
        peptides = [str(p) for p in peptides]
        query_alleles = self._cache_alleles()
        if self.fallback is not None:
            query_alleles = sorted(
                set(query_alleles)
                | set(getattr(self.fallback, "alleles", []))
            )

        # The cache's kind set — typically one or two entries per
        # predictor.  Multi-kind predictors produce the full set.
        cache_kinds = self._cache_kinds()

        # Identify peptides with at least one missing (peptide, allele)
        # across any known kind.  Resolving at peptide granularity keeps
        # the fallback call simple (it returns all kinds for each peptide
        # anyway) and matches the prior behavior's semantics.
        missed_peptides = set()
        for peptide in peptides:
            length = len(peptide)
            for allele in query_alleles:
                # At least one cached row must exist at (pep, allele,
                # length) across any kind / flank combo.
                if not self._lookup_by_prefix(peptide, allele, length):
                    missed_peptides.add(peptide)
                    break

        # Resolve misses through fallback (populates self._index in place).
        if missed_peptides:
            self._fallback_resolve(sorted(missed_peptides))

        # Assemble output — every cache row matching (peptide, allele)
        # across every kind + flank combo.  Single-kind / single-flank
        # caches return one row per (peptide, allele); multi-kind
        # (mhcflurry -BA, class1_presentation) or multi-flank (same
        # peptide across protein contexts) return multiple rows.
        rows = []
        for peptide in peptides:
            length = len(peptide)
            for allele in query_alleles:
                rows.extend(self._lookup_by_prefix(peptide, allele, length))

        if not rows:
            return pd.DataFrame(columns=list(_CACHE_COLUMNS))
        return pd.DataFrame(rows).reindex(columns=list(_CACHE_COLUMNS))

    def _lookup_by_prefix(self, peptide, allele, length):
        """Return every cached row matching ``(peptide, allele,
        peptide_length)`` — all kinds and flank contexts."""
        prefix = (str(peptide), str(allele), int(length))
        return [
            row for key, row in self._index.items()
            if key[:3] == prefix
        ]

    def _cache_kinds(self):
        """Distinct kinds present in the cache."""
        if len(self._df) == 0:
            return []
        return sorted(set(self._df["kind"].unique().tolist()))

    # mhctools compat: some code paths probe for ``predict_dataframe``.
    predict_dataframe = predict_peptides_dataframe

    def predict_proteins_dataframe(
        self, name_to_sequence: Mapping[str, str],
    ) -> pd.DataFrame:
        """Sliding-window lookup — generate peptides from each input
        sequence at every ``default_peptide_lengths``, then resolve
        through :meth:`predict_peptides_dataframe`.
        """
        unique_peptides = set()
        per_peptide_positions: dict[str, list[tuple[str, int, int]]] = {}
        for name, seq in name_to_sequence.items():
            for length in self.default_peptide_lengths:
                if length > len(seq):
                    continue
                for offset in range(len(seq) - length + 1):
                    peptide = seq[offset:offset + length]
                    unique_peptides.add(peptide)
                    per_peptide_positions.setdefault(peptide, []).append(
                        (name, offset, length)
                    )

        if not unique_peptides:
            return pd.DataFrame(
                columns=list(_CACHE_COLUMNS) + [
                    "source_sequence_name", "offset",
                ]
            )
        df = self.predict_peptides_dataframe(sorted(unique_peptides))

        # Expand: one row per (peptide, allele) × (name, offset) matching length.
        expanded = []
        for _, row in df.iterrows():
            positions = per_peptide_positions.get(row["peptide"], [])
            for (name, offset, length) in positions:
                if int(row["peptide_length"]) != length:
                    continue
                r = row.to_dict()
                r["source_sequence_name"] = name
                r["offset"] = offset
                expanded.append(r)

        if not expanded:
            return pd.DataFrame(
                columns=list(_CACHE_COLUMNS) + [
                    "source_sequence_name", "offset",
                ]
            )
        return pd.DataFrame(expanded)

    # --- fallback resolution ----------------------------------------

    def _fallback_resolve(self, peptides) -> None:
        """Run the fallback on ``peptides``, verify the version invariant,
        and merge results into ``self._df`` and ``self._index``.

        No return value — the caller re-reads from ``self._index``.
        """
        if self.fallback is None:
            missed_preview = peptides[:5]
            extra = "" if len(peptides) <= 5 else \
                f" (and {len(peptides) - 5} more)"
            raise KeyError(
                f"CachedPredictor: {len(peptides)} peptide(s) missed and "
                f"no fallback set.  Missed peptides: {missed_preview}{extra}."
            )

        if hasattr(self.fallback, "predict_peptides_dataframe"):
            fb_df = self.fallback.predict_peptides_dataframe(peptides)
        elif hasattr(self.fallback, "predict_dataframe"):
            fb_df = self.fallback.predict_dataframe(peptides)
        else:
            raise TypeError(
                f"fallback does not implement predict_peptides_dataframe "
                f"or predict_dataframe: {type(self.fallback).__name__}"
            )

        fb_df = self._conform_predictor_output(fb_df)
        self._verify_fallback_version(fb_df)
        new_rows = self._normalize(fb_df)

        # Filter to only keys not already in the cache.  The fallback
        # typically predicts across all its alleles for each missed
        # peptide; without this guard, a partial-allele cache (peptide
        # P present for allele A, missing for allele B) would see its
        # (P, A) row silently overwritten by the fallback's output
        # when (P, B) triggers a fallback call.  Preserve user intent.
        def _row_key(row):
            return (
                str(row["peptide"]),
                str(row["allele"]),
                int(row["peptide_length"]),
                str(row["kind"]),
                _flank_key(row.get("n_flank")),
                _flank_key(row.get("c_flank")),
            )

        novel_mask = new_rows.apply(
            lambda r: _row_key(r) not in self._index, axis=1,
        )
        novel_rows = new_rows[novel_mask]
        for _, r in novel_rows.iterrows():
            self._index[_row_key(r)] = r.to_dict()

        if len(novel_rows) == 0:
            return
        # Empty-cache mode: self._df is an empty all-NA frame from
        # __init__; concatenating through it trips a pandas FutureWarning.
        # Assign directly instead when there's nothing to preserve.
        if len(self._df) == 0:
            self._df = novel_rows.copy()
        else:
            self._df = pd.concat(
                [self._df, novel_rows], ignore_index=True,
            )

    @staticmethod
    def _conform_predictor_output(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce an mhctools predictor output frame into the cache
        column conventions: ``peptide_length`` not ``length``,
        ``prediction_method_name`` not ``predictor_name``, and
        ``predictor_version`` populated (NaN ok — still satisfies the
        required-column check)."""
        df = df.copy()
        if "peptide_length" not in df.columns and "length" in df.columns:
            df = df.rename(columns={"length": "peptide_length"})
        if "peptide_length" not in df.columns and "peptide" in df.columns:
            df["peptide_length"] = df["peptide"].str.len()
        if ("prediction_method_name" not in df.columns
                and "predictor_name" in df.columns):
            df = df.rename(columns={"predictor_name": "prediction_method_name"})
        if "predictor_version" not in df.columns:
            df["predictor_version"] = None
        return df

    def _verify_fallback_version(self, fb_df: pd.DataFrame):
        if self._fallback_verified:
            return
        fb_pair = self._unique_version_pair(fb_df)

        # Empty-cache mode: adopt the fallback's identity on first call.
        if self.prediction_method_name is None:
            self.prediction_method_name, self.predictor_version = fb_pair
            self._fallback_verified = True
            return

        cache_pair = (self.prediction_method_name, self.predictor_version)
        # Names must match exactly — different predictors are different.
        if fb_pair[0] != cache_pair[0]:
            raise ValueError(
                f"CachedPredictor fallback predictor_name mismatch: "
                f"cache={cache_pair[0]!r}, fallback={fb_pair[0]!r}."
            )
        # Versions match if equal OR in the caller's equivalence set.
        if (fb_pair[1] != cache_pair[1]
                and fb_pair[1] not in self.also_accept_versions):
            raise ValueError(
                f"CachedPredictor fallback version mismatch: cache="
                f"({cache_pair[0]!r}, {cache_pair[1]!r}), fallback="
                f"({fb_pair[0]!r}, {fb_pair[1]!r}).  Mixing versions is "
                f"not allowed by default — pass "
                f"also_accept_versions={{{fb_pair[1]!r}}} at cache "
                f"construction to opt in to treating these as "
                f"interchangeable."
            )
        self._fallback_verified = True

    # --- persistence ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the cache as a DataFrame (cache schema)."""
        return self._df.copy()

    def save(self, path) -> None:
        """Write the cache as Parquet (``.parquet``/``.pq``) or TSV
        (``.tsv``, optionally ``.tsv.gz``).

        Raises ``ValueError`` if the cache has no identity yet — an
        empty cache built with only a fallback hasn't learned its
        ``(prediction_method_name, predictor_version)`` until the
        first query runs.  Saving in that state would produce a
        schema-only file that can't be round-tripped through
        :meth:`from_topiary_output`.
        """
        if self.prediction_method_name is None or len(self._df) == 0:
            raise ValueError(
                "CachedPredictor.save(): no rows to persist.  An empty "
                "cache built with `fallback=` learns its "
                "(prediction_method_name, predictor_version) identity "
                "from the fallback's output on the first query.  Run "
                "at least one query that populates the cache, then save."
            )
        path_str = str(path)
        if path_str.endswith((".parquet", ".pq")):
            self._df.to_parquet(path_str, index=False)
        elif path_str.endswith((".tsv", ".tsv.gz")):
            self._df.to_csv(path_str, sep="\t", index=False)
        else:
            self._df.to_csv(path_str, index=False)

    # --- loaders ----------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        prediction_method_name: Optional[str] = None,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Construct from an in-memory DataFrame.

        ``prediction_method_name`` / ``predictor_version`` backfill
        columns when the DataFrame doesn't already carry them — one of
        the two sources (column or argument) must populate each of the
        required columns.
        """
        df = df.copy()
        if ("prediction_method_name" not in df.columns
                and prediction_method_name is not None):
            df["prediction_method_name"] = prediction_method_name
        if ("predictor_version" not in df.columns
                and predictor_version is not None):
            df["predictor_version"] = predictor_version
        if "peptide_length" not in df.columns and "length" in df.columns:
            df = df.rename(columns={"length": "peptide_length"})
        if "peptide_length" not in df.columns and "peptide" in df.columns:
            df["peptide_length"] = df["peptide"].str.len()
        return cls(
            df, fallback=fallback, also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_topiary_output(
        cls, path, *, fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a DataFrame previously written by topiary's prediction
        output (Parquet or TSV/CSV).  The expected columns match
        topiary's ``_predict_raw*`` schema; extraneous columns are
        dropped."""
        path_str = str(path)
        if path_str.endswith((".parquet", ".pq")):
            df = pd.read_parquet(path_str)
        elif path_str.endswith((".tsv", ".tsv.gz")):
            df = pd.read_csv(path_str, sep="\t")
        else:
            df = pd.read_csv(path_str)
        return cls.from_dataframe(
            df, fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_tsv(
        cls,
        path,
        *,
        columns: Optional[Mapping[str, str]] = None,
        sep: str = "\t",
        prediction_method_name: Optional[str] = None,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Generic tab- or comma-delimited loader for third-party
        prediction output.

        ``columns`` maps canonical cache column names to the column
        names actually present in the file, e.g.
        ``{"affinity": "ic50", "percentile_rank": "rank", "kind": "score_kind"}``.
        Pass ``sep=","`` for CSV files.

        The file **must** have a ``kind`` column per row (either
        natively or via the ``columns=`` mapping) — one of
        ``"pMHC_affinity"``, ``"pMHC_presentation"``,
        ``"pMHC_stability"``, ``"antigen_processing"``.  The DSL's
        ``Affinity.*`` / ``Presentation.*`` / ``Stability.*`` scopes
        dispatch on it.  A single-kind table is easy: add a ``kind``
        column with the same value on every row.  Multi-kind tables
        (e.g. affinity + presentation in the same file) work out of
        the box.
        """
        df = pd.read_csv(path, sep=sep)
        if columns:
            df = df.rename(columns={v: k for k, v in columns.items()})
        return cls.from_dataframe(
            df,
            prediction_method_name=prediction_method_name,
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_mhcflurry(
        cls,
        path,
        *,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load mhcflurry-predict CSV output.

        Maps mhcflurry's ``mhcflurry_affinity`` /
        ``mhcflurry_affinity_percentile`` /
        ``mhcflurry_presentation_score`` columns onto the cache's
        canonical ``affinity`` / ``percentile_rank`` / ``score``.

        **mhcflurry-specific version note.** Unlike NetMHCpan (which
        bakes models into the binary), mhcflurry fetches model weights
        separately via ``mhcflurry-downloads fetch``.  Two systems with
        the same package version can produce different predictions if
        they have different model bundles installed.  The cache's
        ``predictor_version`` must capture both.

        When ``predictor_version`` is omitted, the loader auto-composes
        it via :func:`mhcflurry_composite_version` — e.g.
        ``"2.2.1+release-2.2.0"``.  Users never have to enumerate the
        installed model bundle manually.  Pass an explicit string only
        when you need a custom label.
        """
        df = pd.read_csv(path)
        # mhcflurry's wide-format output carries up to three kinds of
        # prediction per (peptide, allele) row: binding affinity,
        # presentation, antigen processing.  Explode into one row per
        # (peptide, allele, kind), skipping kinds whose columns aren't
        # populated.  The cache keys on the 4-tuple so all kinds
        # coexist cleanly.
        df = _explode_mhcflurry_kinds(df)
        if predictor_version is None:
            predictor_version = mhcflurry_composite_version()
        return cls.from_dataframe(
            df,
            prediction_method_name="mhcflurry",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    # --- NetMHC-family loaders ---
    # Thin adapters over mhctools.parsing.*_stdout functions.  They
    # read an NetMHC-tool stdout capture, run the mhctools parser,
    # convert the resulting list[BindingPrediction] into a DataFrame
    # shaped for the cache, and hand off to from_dataframe.

    @classmethod
    def from_netmhcpan_stdout(
        cls,
        path,
        *,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a NetMHCpan stdout capture (2.8 / 3 / 4 / 4.1).

        Uses ``mhctools.parsing.parse_netmhcpan_to_preds`` which auto-
        detects the version and returns **every kind** present in the
        output — a ``-BA`` run carries both binding-affinity and
        elution-score rows per (peptide, allele), and both land in the
        cache as separate ``pMHC_affinity`` + ``pMHC_presentation``
        rows.

        When ``predictor_version`` is omitted, the version string is
        parsed from the stdout preamble (e.g. ``"4.1b"``).
        """
        from mhctools.parsing import parse_netmhcpan_to_preds
        text = _read_text(path)
        if predictor_version is None:
            predictor_version = _version_from_header(text) or "unknown"
        preds = parse_netmhcpan_to_preds(
            text,
            predictor_name="netmhcpan",
            predictor_version=predictor_version,
        )
        return cls.from_dataframe(
            _predictions_to_dataframe(preds),
            prediction_method_name="netmhcpan",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_netmhc_stdout(
        cls,
        path,
        *,
        version: str = "4",
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a classic NetMHC stdout capture (3 / 4 / 4.1).

        ``version`` selects the parser: ``"3"``, ``"4"``, or ``"4.1"``.
        """
        from mhctools import parsing as _p
        parsers = {
            "3": _p.parse_netmhc3_stdout,
            "4": _p.parse_netmhc4_stdout,
            "4.1": _p.parse_netmhc41_stdout,
        }
        if version not in parsers:
            raise ValueError(
                f"NetMHC version {version!r} not supported — choose "
                f"from {sorted(parsers)}."
            )
        text = _read_text(path)
        preds = parsers[version](text)
        if predictor_version is None:
            predictor_version = _version_from_header(text) or version
        return cls.from_dataframe(
            _bindings_to_dataframe(preds, kind="pMHC_affinity"),
            prediction_method_name="netmhc",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_netmhcpan_cons_stdout(
        cls,
        path,
        *,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a NetMHCcons stdout capture (consensus of multiple
        methods)."""
        from mhctools.parsing import parse_netmhccons_stdout
        text = _read_text(path)
        preds = parse_netmhccons_stdout(text)
        if predictor_version is None:
            predictor_version = _version_from_header(text) or "unknown"
        return cls.from_dataframe(
            _bindings_to_dataframe(preds, kind="pMHC_affinity"),
            prediction_method_name="netmhccons",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_netmhciipan_stdout(
        cls,
        path,
        *,
        version: str = "4.3",
        mode: str = "elution_score",
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a NetMHCIIpan stdout capture (class II).

        ``version`` selects the parser: ``"legacy"``, ``"4"``, or
        ``"4.3"`` (default — latest).  ``mode`` picks between
        ``"elution_score"`` (default, NetMHCIIpan 4+) and
        ``"binding_affinity"``.
        """
        from mhctools import parsing as _p
        parsers = {
            "legacy": _p.parse_netmhciipan_stdout,
            "4": _p.parse_netmhciipan4_stdout,
            "4.3": _p.parse_netmhciipan43_stdout,
        }
        if version not in parsers:
            raise ValueError(
                f"NetMHCIIpan version {version!r} not supported — "
                f"choose from {sorted(parsers)}."
            )
        text = _read_text(path)
        parser = parsers[version]
        # The legacy parser doesn't take mode; guard the call.
        if version == "legacy":
            preds = parser(text)
        else:
            preds = parser(text, mode=mode)
        if predictor_version is None:
            predictor_version = _version_from_header(text) or version
        kind = (
            "pMHC_presentation" if mode == "elution_score"
            else "pMHC_affinity"
        )
        return cls.from_dataframe(
            _bindings_to_dataframe(preds, kind=kind),
            prediction_method_name="netmhciipan",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    # --- sharding ---

    @classmethod
    def concat(
        cls,
        caches: List["CachedPredictor"],
        *,
        on_overlap: Union[str, Callable] = "raise",
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Merge several CachedPredictors into one.

        All shards must share the same
        ``(prediction_method_name, predictor_version)`` — the core
        invariant applies across shards the same way it applies
        inside one.  Use ``also_accept_versions`` on the returned
        cache to widen what its fallback attachment will accept.

        Parameters
        ----------
        caches : list of CachedPredictor
            Shards to merge.  Must be non-empty.
        on_overlap : ``"raise"`` | ``"last"`` | ``"first"`` | callable
            Policy when two shards have the same
            ``(peptide, allele, peptide_length)`` key:

            - ``"raise"`` (default): refuse to merge, showing a
              sample of the conflicting keys.
            - ``"last"``: the later shard in ``caches`` wins.
            - ``"first"``: the earlier shard wins.
            - callable ``(row_a_dict, row_b_dict) -> row_dict``:
              user-supplied resolver, called pairwise per duplicate
              group.
        fallback : predictor, optional
            Attached to the merged cache for miss-resolution.
        """
        if not caches:
            raise ValueError("CachedPredictor.concat: no caches given.")

        # Mixed (name, version) across shards fails the core invariant
        # when we hand the combined df to the constructor below.
        combined = pd.concat(
            [c._df for c in caches], ignore_index=True,
        )
        key_cols = list(_KEY_COLS)

        dup_mask = combined.duplicated(subset=key_cols, keep=False)
        if dup_mask.any():
            combined = cls._resolve_overlaps(
                combined, key_cols, dup_mask, on_overlap,
            )

        return cls(
            combined,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @staticmethod
    def _resolve_overlaps(df, key_cols, dup_mask, on_overlap):
        if on_overlap == "raise":
            dupes = df[dup_mask][key_cols].drop_duplicates()
            sample = dupes.head(5).to_dict("records")
            extra = "" if len(dupes) <= 5 else \
                f" (and {len(dupes) - 5} more)"
            raise ValueError(
                f"CachedPredictor.concat: {len(dupes)} overlapping "
                f"(peptide, allele, peptide_length, kind) key(s) across "
                f"shards.  Sample: {sample}{extra}.  Pass "
                f"on_overlap='last' / 'first' / callable to resolve."
            )
        if on_overlap == "last":
            return df.drop_duplicates(subset=key_cols, keep="last")
        if on_overlap == "first":
            return df.drop_duplicates(subset=key_cols, keep="first")
        if callable(on_overlap):
            singletons = df[~dup_mask]
            resolved = []
            # dropna=False: keep groups where a key column is None/NaN
            # (flank columns are None for predictors that don't supply
            # them; default groupby would drop those groups entirely).
            for _, group in df[dup_mask].groupby(
                key_cols, sort=False, dropna=False,
            ):
                rows = [r.to_dict() for _, r in group.iterrows()]
                merged = rows[0]
                for nxt in rows[1:]:
                    merged = on_overlap(merged, nxt)
                resolved.append(merged)
            return pd.concat(
                [singletons, pd.DataFrame(resolved)], ignore_index=True,
            )
        raise ValueError(
            f"CachedPredictor.concat: on_overlap must be 'raise', "
            f"'last', 'first', or a callable; got {on_overlap!r}."
        )

    @classmethod
    def from_directory(
        cls,
        path,
        *,
        pattern: str = "*",
        on_overlap: Union[str, Callable] = "raise",
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load every matching cache file in a directory and concat.

        ``pattern`` is a glob passed to :meth:`pathlib.Path.glob`.
        Files are loaded via :meth:`from_topiary_output` (any
        extension it handles — Parquet, TSV, TSV.gz, CSV — works).
        All files must share ``(name, version)`` per the core
        invariant.  ``on_overlap`` follows :meth:`concat` semantics.
        """
        directory = Path(path)
        if not directory.is_dir():
            raise ValueError(
                f"CachedPredictor.from_directory: {path!r} is not a "
                f"directory."
            )
        files = sorted(
            f for f in directory.glob(pattern) if f.is_file()
        )
        if not files:
            raise ValueError(
                f"CachedPredictor.from_directory: no files matching "
                f"{pattern!r} in {path!r}."
            )
        shards = [cls.from_topiary_output(f) for f in files]
        return cls.concat(
            shards,
            on_overlap=on_overlap,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )

    @classmethod
    def from_netmhcstabpan_stdout(
        cls,
        path,
        *,
        predictor_version: Optional[str] = None,
        fallback=None,
        also_accept_versions: Optional[Iterable[str]] = None,
    ) -> "CachedPredictor":
        """Load a NetMHCstabpan stdout capture (pMHC stability)."""
        from mhctools.parsing import parse_netmhcstabpan
        text = _read_text(path)
        preds = parse_netmhcstabpan(text)
        if predictor_version is None:
            predictor_version = _version_from_header(text) or "unknown"
        return cls.from_dataframe(
            _bindings_to_dataframe(preds, kind="pMHC_stability"),
            prediction_method_name="netmhcstabpan",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )


# ---------------------------------------------------------------------------
# Module-level helpers used by the NetMHC loaders
# ---------------------------------------------------------------------------


def _read_text(path) -> str:
    with open(path, "r") as f:
        return f.read()


def _flank_key(value):
    """Normalize a flank column value into something hashable + stable
    across NaN / None / empty-string representations.

    Returns ``None`` for any missing / NaN / empty value, else the
    uppercased string.  Ensures a row whose ``n_flank`` is NaN and a
    row whose ``n_flank`` is ``None`` hash to the same key (they
    represent the same absence-of-flank)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s.upper()


def _bindings_to_dataframe(preds, *, kind: str) -> pd.DataFrame:
    """Convert an mhctools ``list[BindingPrediction]`` into a DataFrame
    shaped for :class:`CachedPredictor`.  Uses ``length`` (the
    mhctools attribute name) → ``peptide_length``.

    ``kind`` is required and stamped on every row — DSL expressions
    like ``Affinity.value <= 500`` filter on the ``kind`` column, and
    mhctools' ``BindingPrediction`` objects don't carry ``kind`` as
    an attribute.  The caller knows the right value based on the tool.
    """
    rows = [
        {
            "peptide": p.peptide,
            "allele": p.allele,
            "peptide_length": p.length,
            "kind": kind,
            "score": p.score,
            "affinity": p.affinity,
            "percentile_rank": p.percentile_rank,
            "value": getattr(p, "value", None),
            "source_sequence_name": getattr(p, "source_sequence_name", None),
            "peptide_offset": getattr(p, "offset", None),
        }
        for p in preds
    ]
    return pd.DataFrame(rows)


def _predictions_to_dataframe(preds) -> pd.DataFrame:
    """Convert an mhctools ``list[Prediction]`` (multi-kind) into a
    DataFrame shaped for :class:`CachedPredictor`.

    Unlike ``BindingPrediction``, ``Prediction`` exposes ``.kind``
    directly, so callers don't supply it — a NetMHCpan ``-BA`` run
    naturally produces affinity + presentation rows interleaved.
    ``n_flank`` / ``c_flank`` are carried through when present.
    """
    rows = [
        {
            "peptide": p.peptide,
            "allele": p.allele,
            "peptide_length": len(p.peptide),
            "kind": str(p.kind),
            "score": p.score,
            "affinity": getattr(p, "value", None) if p.kind == "pMHC_affinity" else None,
            "percentile_rank": p.percentile_rank,
            "value": p.value,
            "source_sequence_name": getattr(p, "source_sequence_name", None),
            "peptide_offset": getattr(p, "offset", None),
            "n_flank": getattr(p, "n_flank", None),
            "c_flank": getattr(p, "c_flank", None),
        }
        for p in preds
    ]
    return pd.DataFrame(rows)


# NetMHC-family tools embed their version in the stdout preamble,
# e.g. "NetMHCpan version 4.1b" or "# NetMHCstabpan version 1.0".
# Capture the token after "version" (ignoring surrounding whitespace
# and case) — this is the stamp we put in predictor_version.
_NETMHC_VERSION_RE = re.compile(
    r"(?:NetMHCpan|NetMHCIIpan|NetMHCcons|NetMHCstabpan|NetMHC)\s+version\s+(\S+)",
    re.IGNORECASE,
)


def _version_from_header(text: str) -> Optional[str]:
    """Parse a NetMHC-family version string (e.g. ``"4.1b"``) from
    the stdout preamble.  Returns ``None`` if no version line found.

    Searches the first 10kB of the text; NetMHC tools put the version
    line within the first ~100 lines but the argument-dump preamble
    before it can stretch to several KB when verbose flags are set."""
    m = _NETMHC_VERSION_RE.search(text[:10000])
    return m.group(1) if m else None


def _explode_mhcflurry_kinds(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide-format mhcflurry output into one row per
    ``(peptide, allele, kind)``.

    mhcflurry's class1_presentation pipeline emits up to three kinds
    in a single CSV row:

    - ``pMHC_affinity``: ``mhcflurry_affinity`` (nM) +
      ``mhcflurry_affinity_percentile``
    - ``pMHC_presentation``: ``mhcflurry_presentation_score`` +
      ``mhcflurry_presentation_percentile``
    - ``antigen_processing``: ``mhcflurry_processing_score``

    Rows whose columns for a given kind are all NaN are skipped — so
    an affinity-only CSV (no presentation columns) produces only the
    ``pMHC_affinity`` rows, and vice versa.
    """
    # (kind, affinity_col, rank_col, score_col) — None = N/A for that kind.
    kind_specs = [
        ("pMHC_affinity",
         "mhcflurry_affinity",
         "mhcflurry_affinity_percentile",
         None),
        ("pMHC_presentation",
         None,
         "mhcflurry_presentation_percentile",
         "mhcflurry_presentation_score"),
        ("antigen_processing",
         None,
         None,
         "mhcflurry_processing_score"),
    ]
    # Provenance / context columns that round-trip with every kind.
    # mhcflurry's predict output emits n_flank / c_flank when the
    # processing model is active; source_sequence_name + offset when
    # called via predict-scan on full proteins; sample_name for
    # multi-sample CSVs.  'offset' is renamed to 'peptide_offset' to
    # match topiary's canonical column naming.
    base_col_aliases = {
        "peptide": "peptide",
        "allele": "allele",
        "peptide_length": "peptide_length",
        "source_sequence_name": "source_sequence_name",
        "offset": "peptide_offset",
        "peptide_offset": "peptide_offset",
        "n_flank": "n_flank",
        "c_flank": "c_flank",
        "sample_name": "sample_name",
    }

    rows = []
    for _, r in df.iterrows():
        base = {
            target: r[source]
            for source, target in base_col_aliases.items()
            if source in df.columns
        }
        for kind, aff_col, rank_col, score_col in kind_specs:
            cols = [c for c in (aff_col, rank_col, score_col)
                    if c and c in df.columns]
            if not cols or all(pd.isna(r[c]) for c in cols):
                continue
            row = dict(base)
            row["kind"] = kind
            row["affinity"] = (
                r[aff_col] if aff_col and aff_col in df.columns else None
            )
            row["percentile_rank"] = (
                r[rank_col] if rank_col and rank_col in df.columns else None
            )
            row["score"] = (
                r[score_col]
                if score_col and score_col in df.columns else None
            )
            # 'value' is the kind's primary metric: nM for affinity,
            # score for presentation/processing.
            row["value"] = (
                row["affinity"] if kind == "pMHC_affinity"
                else row["score"]
            )
            rows.append(row)

    if not rows:
        return df.head(0)
    return pd.DataFrame(rows)


def mhcflurry_composite_version() -> str:
    """Compose mhcflurry's package version + active model-release
    identifier into a single string for :attr:`CachedPredictor.predictor_version`.

    Returns a string like ``"2.2.1+release-2.2.0"`` — the Python
    package version joined to the mhcflurry model-data release
    currently installed via ``mhcflurry-downloads fetch``.  Two
    systems whose :func:`mhcflurry_composite_version` outputs match
    should produce interchangeable mhcflurry predictions.

    The helper introspects the locally-installed mhcflurry; the user
    never has to enumerate model bundles manually.

    Raises
    ------
    RuntimeError
        If mhcflurry isn't installed, or no model release is configured
        (run ``mhcflurry-downloads fetch`` first).
    """
    try:
        import mhcflurry
        import mhcflurry.downloads
    except ImportError as e:
        raise RuntimeError(
            "mhcflurry is not installed — cannot derive a composite "
            "version.  Install mhcflurry or pass predictor_version "
            "explicitly."
        ) from e
    pkg = getattr(mhcflurry, "__version__", None)
    if not pkg:
        raise RuntimeError(
            "mhcflurry is installed but exposes no __version__; cannot "
            "derive a composite version — pass predictor_version "
            "explicitly."
        )
    try:
        release = mhcflurry.downloads.get_current_release()
    except Exception as e:
        raise RuntimeError(
            f"Could not read mhcflurry's current model release: {e!r}.  "
            f"Pass predictor_version explicitly."
        ) from e
    if not release:
        raise RuntimeError(
            "mhcflurry has no active model release.  Run "
            "`mhcflurry-downloads fetch` or pass predictor_version "
            "explicitly."
        )
    return f"{pkg}+release-{release}"
