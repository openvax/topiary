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
    """

    def __init__(self, df: pd.DataFrame, fallback=None):
        self._df = self._normalize(df)
        self.prediction_method_name, self.predictor_version = \
            self._unique_version_pair(self._df)
        self._index = self._build_index(self._df)
        self.fallback = fallback
        self._fallback_verified = False

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
        crossed with every allele in the cache.

        Misses go to ``self.fallback`` if set, else raise ``KeyError``.
        """
        peptides = [str(p) for p in peptides]
        cache_alleles = self._cache_alleles()
        rows = []
        misses = []
        for peptide in peptides:
            length = len(peptide)
            for allele in cache_alleles:
                row = self._index.get((peptide, allele, length))
                if row is None:
                    misses.append(peptide)
                    break  # defer: fallback call covers all alleles at once
                rows.append(row)

        if misses:
            missed_unique = sorted(set(misses))
            fb_rows = self._fallback_resolve(missed_unique)
            rows.extend(fb_rows)

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

    def _fallback_resolve(self, peptides):
        """Run the fallback on ``peptides`` (missed peptide strings),
        verify the version invariant, merge the results into the cache,
        and return the new row dicts for the requested peptides."""
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
        for _, r in new_rows.iterrows():
            key = (
                str(r["peptide"]), str(r["allele"]), int(r["peptide_length"]),
            )
            self._index[key] = r.to_dict()
        self._df = pd.concat([self._df, new_rows], ignore_index=True) \
            .drop_duplicates(
                subset=["peptide", "allele", "peptide_length"], keep="last",
            )
        return [r.to_dict() for _, r in new_rows.iterrows()]

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
        cache_pair = (self.prediction_method_name, self.predictor_version)
        if fb_pair != cache_pair:
            raise ValueError(
                f"CachedPredictor fallback version mismatch: cache="
                f"({cache_pair[0]!r}, {cache_pair[1]!r}), "
                f"fallback=({fb_pair[0]!r}, {fb_pair[1]!r}).  Mixing model "
                f"versions is not allowed — use a fallback whose "
                f"(prediction_method_name, predictor_version) matches the "
                f"cache."
            )
        self._fallback_verified = True

    # --- persistence ------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the cache as a DataFrame (cache schema)."""
        return self._df.copy()

    def save(self, path) -> None:
        """Write the cache as Parquet (``.parquet``/``.pq``) or TSV
        (``.tsv``, optionally ``.tsv.gz``)."""
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
        return cls(df, fallback=fallback)

    @classmethod
    def from_topiary_output(
        cls, path, *, fallback=None,
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
        return cls.from_dataframe(df, fallback=fallback)

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
        )

    @classmethod
    def from_mhcflurry(
        cls,
        path,
        *,
        predictor_version: str,
        fallback=None,
    ) -> "CachedPredictor":
        """Load mhcflurry-predict CSV output.  Maps mhcflurry's
        ``mhcflurry_affinity`` / ``mhcflurry_affinity_percentile`` /
        ``mhcflurry_presentation_score`` columns onto the cache's
        canonical ``affinity`` / ``percentile_rank`` / ``score``.

        ``predictor_version`` is required — mhcflurry's CSV doesn't
        embed it and the version invariant must be satisfied."""
        df = pd.read_csv(path)
        col_map = {}
        if "mhcflurry_affinity" in df.columns:
            col_map["mhcflurry_affinity"] = "affinity"
        if "mhcflurry_affinity_percentile" in df.columns:
            col_map["mhcflurry_affinity_percentile"] = "percentile_rank"
        if "mhcflurry_presentation_score" in df.columns:
            col_map["mhcflurry_presentation_score"] = "score"
        df = df.rename(columns=col_map)
        return cls.from_dataframe(
            df,
            prediction_method_name="mhcflurry",
            predictor_version=predictor_version,
            fallback=fallback,
        )
