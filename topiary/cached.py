"""CachedPredictor — serve MHC binding predictions from a pre-computed table.

Plugs into :class:`topiary.TopiaryPredictor` alongside live mhctools
predictors.  Supports three producer paths:

1. External predictor output files (mhcflurry CSV today; NetMHCpan
   `.xls` and others via follow-up loaders).
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

from typing import Iterable, Mapping, Optional

import pandas as pd


# Columns a cache row must carry so the core invariant and lookup work.
_REQUIRED_COLUMNS = (
    "peptide", "allele", "peptide_length",
    "prediction_method_name", "predictor_version",
)

# All columns the cache preserves on load and round-trip.
_CACHE_COLUMNS = (
    "peptide", "allele", "peptide_length",
    "score", "affinity", "percentile_rank", "value", "kind",
    "prediction_method_name", "predictor_version",
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
        keep = [c for c in _CACHE_COLUMNS if c in df.columns]
        out = df[keep].copy()
        out["peptide"] = out["peptide"].astype(str)
        out["allele"] = out["allele"].astype(str)
        out["peptide_length"] = out["peptide_length"].astype(int)
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
        return {
            (str(r["peptide"]), str(r["allele"]), int(r["peptide_length"])):
                r.to_dict()
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
        """Return one row per ``(peptide, allele)`` for each input peptide
        crossed with every allele in the cache (or in the fallback, if set).

        Misses are resolved through ``self.fallback`` (which merges into
        the cache) if set; else raise ``KeyError``.
        """
        peptides = [str(p) for p in peptides]
        query_alleles = self._cache_alleles()
        if self.fallback is not None:
            query_alleles = sorted(
                set(query_alleles)
                | set(getattr(self.fallback, "alleles", []))
            )

        # Identify peptides with at least one missing (peptide, allele).
        missed_peptides = set()
        for peptide in peptides:
            length = len(peptide)
            for allele in query_alleles:
                if (peptide, allele, length) not in self._index:
                    missed_peptides.add(peptide)
                    break

        # Resolve misses through fallback (populates self._index in place).
        if missed_peptides:
            self._fallback_resolve(sorted(missed_peptides))

        # Assemble output from the (updated) cache.
        rows = []
        for peptide in peptides:
            length = len(peptide)
            for allele in query_alleles:
                row = self._index.get((peptide, allele, length))
                if row is not None:
                    rows.append(row)

        if not rows:
            return pd.DataFrame(columns=list(_CACHE_COLUMNS))
        return pd.DataFrame(rows).reindex(columns=list(_CACHE_COLUMNS))

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
        def _is_new(row):
            key = (
                str(row["peptide"]),
                str(row["allele"]),
                int(row["peptide_length"]),
            )
            return key not in self._index

        novel_mask = new_rows.apply(_is_new, axis=1)
        novel_rows = new_rows[novel_mask]
        for _, r in novel_rows.iterrows():
            key = (
                str(r["peptide"]), str(r["allele"]), int(r["peptide_length"]),
            )
            self._index[key] = r.to_dict()

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
        ``{"affinity": "ic50", "percentile_rank": "rank"}``.
        Pass ``sep=","`` for CSV files.
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
        col_map = {}
        if "mhcflurry_affinity" in df.columns:
            col_map["mhcflurry_affinity"] = "affinity"
        if "mhcflurry_affinity_percentile" in df.columns:
            col_map["mhcflurry_affinity_percentile"] = "percentile_rank"
        if "mhcflurry_presentation_score" in df.columns:
            col_map["mhcflurry_presentation_score"] = "score"
        df = df.rename(columns=col_map)
        if predictor_version is None:
            predictor_version = mhcflurry_composite_version()
        return cls.from_dataframe(
            df,
            prediction_method_name="mhcflurry",
            predictor_version=predictor_version,
            fallback=fallback,
            also_accept_versions=also_accept_versions,
        )


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
