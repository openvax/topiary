"""HistoricalPredictor — serve predictions from sparse historical tables.

Preserves input-table predictions as an alternative prediction source for
Vaxrank and Topiary workflows (#368).

Unlike CachedPredictor:
1. Allows unknown / missing predictor versions (pd.NA, None, "") without
   inferring them from installed packages or relabeling aggregates.
2. Allows multiple prediction methods (e.g. NetMHCpan and MHCflurry) in the
   same source.
3. Supports sparse peptide-allele coverage without requiring or fabricating
   a full peptide x allele Cartesian cross-product.
4. Preserves all measurement values, kinds, flanks, genotype scopes and
   comparators (wt_*).
5. Detects conflicting predictions for the same logical query and raises
   deterministically without silently averaging or arbitrarily choosing.
6. Restricts live-model fallback so historical rows with unknown versions
   never masquerade as hits for a current live model.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import mhcgnomes

from .cached import (
    CachedPredictorCoverageError,
    PREDICTION_VALUE_COLUMNS,
    PROTEIN_SCAN_COLUMNS,
    _cache_comparison_frame,
    _dependence_tag,
    _flank_flag,
    prediction_flanks_match,
)
from .predictor import _backfill_value_from_score
from .ranking import (
    KIND_MHC_DEPENDENCE,
    format_allele_set,
    is_stated,
    mhc_dependence,
    split_allele_set,
    stated_values,
)
from .result import TopiaryResult


HISTORICAL_KEY_COLUMNS = (
    "peptide",
    "allele",
    "peptide_length",
    "kind",
    "prediction_method_name",
    "predictor_version",
    "n_flank",
    "c_flank",
    "allele_set",
    "prediction_flanks_supplied",
    "prediction_mhc_dependence",
)


def conflicting_historical_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Rows sharing a historical prediction key but disagreeing about values.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical prediction rows.

    Returns
    -------
    pandas.DataFrame
        The conflicting rows, empty when there are none.
    """
    if df.empty:
        return df

    key = [c for c in HISTORICAL_KEY_COLUMNS if c in df.columns]
    value = [c for c in PREDICTION_VALUE_COLUMNS if c in df.columns]
    if not key or not value:
        return df.iloc[:0]

    conflict_indices = set()
    comparison = _cache_comparison_frame(df)
    for _, group in comparison.groupby(key, dropna=False, sort=False):
        if len(group) <= 1:
            continue
        has_conflict = False
        for col in value:
            if col in group.columns:
                s = group[col].dropna()
                if len(s) > 1:
                    if pd.api.types.is_numeric_dtype(s):
                        min_v = float(s.min())
                        max_v = float(s.max())
                        if (max_v - min_v) > 1e-9 * (abs(min_v) + 1.0):
                            has_conflict = True
                            break
                    elif s.nunique() > 1:
                        has_conflict = True
                        break
        if has_conflict:
            conflict_indices.update(group.index)

    if not conflict_indices:
        return df.iloc[:0]
    return df.loc[sorted(conflict_indices)]


def _historical_conflict_message(where: str, conflicts: pd.DataFrame) -> str:
    """Informative, deterministic message for conflicting predictions."""
    key = [c for c in HISTORICAL_KEY_COLUMNS if c in conflicts.columns]
    value = [c for c in PREDICTION_VALUE_COLUMNS if c in conflicts.columns]
    groups = _cache_comparison_frame(conflicts).groupby(key, dropna=False, sort=False)
    disagreed = sorted({
        column
        for _, group in groups
        for column in value
        if group[column].dropna().nunique() > 1
    })
    sample = conflicts[key].drop_duplicates().head(3).to_dict("records")
    return (
        f"{where}: {groups.ngroups} prediction key(s) with more than one "
        f"answer. Disagreed on: {disagreed}. Sample key(s): {sample}. "
        f"A historical prediction key with conflicting values has no defined result -- "
        f"drop or reconcile the losing rows before loading."
    )


def _coalesce_historical_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce duplicate historical rows sharing the same key, keeping non-null values."""
    if df.empty:
        return df
    key = [c for c in HISTORICAL_KEY_COLUMNS if c in df.columns]
    val = [c for c in PREDICTION_VALUE_COLUMNS if c in df.columns]
    if not key:
        return df.drop_duplicates()
    if val:
        non_null_count = df[val].notna().sum(axis=1)
        sorted_df = df.assign(_non_null_count=non_null_count).sort_values(
            by="_non_null_count", ascending=False, kind="stable"
        )
        return sorted_df.drop_duplicates(subset=key, keep="first").drop(columns=["_non_null_count"])
    return df.drop_duplicates(subset=key, keep="first")


class HistoricalPredictor:
    """Predictor that answers queries from pre-existing historical tables.

    Implements the mhctools predictor protocol
    (``predict_peptides_dataframe``, ``predict_proteins_dataframe``,
     ``predict_contextual_peptides_dataframe``, ``predict_dataframe``,
     ``alleles``, ``default_peptide_lengths``, ``kind_support``,
     ``supported_kinds``) so it can drop into :class:`topiary.TopiaryPredictor`
    as a model, or be passed to :func:`topiary.rescore_candidates` and
    :func:`topiary.predict_peptide_occurrences`.

    Parameters
    ----------
    df : pandas.DataFrame, TopiaryResult, str, or Path, optional
        Historical prediction rows. Can be loaded from pVACseq, LENS,
        or generic Topiary output tables.
    fallback : mhctools predictor or CachedPredictor, optional
        If set, queries missing from the historical table can be computed by
        the fallback predictor if compatible. Historical rows keep their true
        historical provenance and never masquerade as fallback results.
    alleles : iterable of str, optional
        Restricts queries to a specific subset of alleles. By default (None),
        all reported alleles present in the table are available.
    default_peptide_lengths : iterable of int, optional
        Selected protein-scan window lengths. By default uses all available
        lengths found in the table.
    missing : {'raise', 'ignore', 'omit'}
        Policy when an exact queried sequence has no coverage in the table.
        Default 'raise' raises :class:`CachedPredictorCoverageError`.
        'ignore' or 'omit' omits missing predictions.
    prediction_method_name : str, optional
        Default prediction method name to assign if the input table omits it.
    predictor_version : str, optional
        Default predictor version to assign if explicitly supplied. If None,
        unknown versions in the table remain unknown.
    """

    def __init__(
        self,
        df: Optional[Union[pd.DataFrame, TopiaryResult, str, Path]] = None,
        *,
        fallback=None,
        alleles: Optional[Iterable[str]] = None,
        default_peptide_lengths: Optional[Iterable[int]] = None,
        missing: str = "raise",
        prediction_method_name: Optional[str] = None,
        predictor_version: Optional[str] = None,
    ):
        if missing not in ("raise", "ignore", "omit"):
            raise ValueError(f"Unknown missing policy {missing!r}; expected 'raise', 'ignore', or 'omit'")
        self.missing = missing
        self.fallback = fallback
        self._default_method_name = prediction_method_name
        self._default_predictor_version = predictor_version
        self._requested_alleles = None
        self._requested_peptide_lengths = None

        if df is None:
            raw_df = pd.DataFrame(columns=list(HISTORICAL_KEY_COLUMNS))
        elif isinstance(df, TopiaryResult):
            raw_df = df.to_long().df if hasattr(df, "to_long") else df.df
        elif isinstance(df, (str, Path)):
            raw_df = self._load_file(df)
        elif isinstance(df, pd.DataFrame):
            raw_df = df
        else:
            raise TypeError(f"df must be a DataFrame, TopiaryResult, or file path, got {type(df).__name__}")

        if raw_df.empty and fallback is None and df is not None:
            # Empty table without fallback
            self._df = pd.DataFrame(columns=list(HISTORICAL_KEY_COLUMNS))
        elif raw_df.empty and fallback is not None:
            self._df = pd.DataFrame(columns=list(HISTORICAL_KEY_COLUMNS))
        else:
            self._df = self._normalize(raw_df)
            conflicts = conflicting_historical_predictions(self._df)
            if not conflicts.empty:
                raise ValueError(_historical_conflict_message("HistoricalPredictor", conflicts))
            self._df = _coalesce_historical_rows(self._df).reset_index(drop=True)

        if alleles is not None:
            self.alleles = alleles
        if default_peptide_lengths is not None:
            self.default_peptide_lengths = default_peptide_lengths

    @classmethod
    def _load_file(cls, path: Union[str, Path]) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix in (".tsv", ".txt") or path.name.endswith(".tsv.gz"):
            return pd.read_csv(path, sep="\t", comment="#")
        elif suffix == ".csv" or path.name.endswith(".csv.gz"):
            return pd.read_csv(path, sep=",", comment="#")
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(path)
        else:
            return pd.read_csv(path, sep=None, engine="python", comment="#")

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate required fields and preserve original provenance without inventing data."""
        out = df.copy()

        if "peptide" not in out.columns:
            raise ValueError("HistoricalPredictor rows missing required column: 'peptide'")
        na_peptide = ~stated_values(out["peptide"])
        if na_peptide.any():
            raise ValueError(
                f"HistoricalPredictor: column 'peptide' must be a non-empty string on every row "
                f"(got {int(na_peptide.sum())} null/empty value(s))."
            )
        out["peptide"] = out["peptide"].astype(str)

        if "kind" not in out.columns:
            raise ValueError("HistoricalPredictor rows missing required column: 'kind'")
        na_kind = ~stated_values(out["kind"])
        if na_kind.any():
            raise ValueError(
                f"HistoricalPredictor: column 'kind' must be a non-empty string on every row "
                f"(got {int(na_kind.sum())} null/empty value(s))."
            )
        out["kind"] = out["kind"].astype(str)

        if "allele" not in out.columns:
            out["allele"] = ""
        else:
            # Canonicalize stated alleles through mhcgnomes where valid
            out["allele"] = out["allele"].map(
                lambda v: mhcgnomes.parse(str(v)).to_string() if is_stated(v) else ""
            )

        if "peptide_length" not in out.columns:
            out["peptide_length"] = out["peptide"].str.len().astype(int)
        else:
            out["peptide_length"] = pd.to_numeric(out["peptide_length"], errors="coerce").fillna(
                out["peptide"].str.len()
            ).astype(int)

        if "prediction_method_name" not in out.columns:
            if self._default_method_name is not None:
                out["prediction_method_name"] = str(self._default_method_name)
            else:
                out["prediction_method_name"] = "historical"
        else:
            if self._default_method_name is not None:
                out["prediction_method_name"] = out["prediction_method_name"].fillna(self._default_method_name)
            out["prediction_method_name"] = out["prediction_method_name"].map(
                lambda v: str(v) if is_stated(v) else "historical"
            )

        if "predictor_version" not in out.columns:
            if self._default_predictor_version is not None:
                out["predictor_version"] = str(self._default_predictor_version)
            else:
                out["predictor_version"] = pd.NA
        else:
            if self._default_predictor_version is not None:
                out["predictor_version"] = out["predictor_version"].fillna(self._default_predictor_version)
            out["predictor_version"] = out["predictor_version"].map(
                lambda v: str(v) if is_stated(v) else pd.NA
            )

        for col in ("score", "affinity", "percentile_rank", "value"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

        _backfill_value_from_score(out)

        for col in ("wt_score", "wt_affinity", "wt_percentile_rank", "wt_value"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

        if "n_flank" in out.columns:
            out["n_flank"] = out["n_flank"].map(lambda v: str(v) if is_stated(v) else None)
        if "c_flank" in out.columns:
            out["c_flank"] = out["c_flank"].map(lambda v: str(v) if is_stated(v) else None)

        if "prediction_flanks_supplied" in out.columns:
            out["prediction_flanks_supplied"] = out["prediction_flanks_supplied"].map(_flank_flag)

        if "allele_set" in out.columns:
            out["allele_set"] = out["allele_set"].map(
                lambda v: format_allele_set(split_allele_set(v)) if is_stated(v) else ""
            )

        if "prediction_mhc_dependence" in out.columns:
            out["prediction_mhc_dependence"] = out["prediction_mhc_dependence"].map(_dependence_tag)

        return out

    # --- Properties -------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """The underlying historical DataFrame."""
        return self._df.copy()

    @property
    def alleles(self) -> list[str]:
        """Selected or available alleles in the historical source."""
        if self._requested_alleles is not None:
            return list(self._requested_alleles)
        column = self._df["allele"]
        a = set(column[stated_values(column)].unique().tolist())
        if "allele_set" in self._df:
            for genotype in self._df["allele_set"]:
                if is_stated(genotype):
                    a.update(split_allele_set(genotype))
        if self.fallback is not None:
            a.update(getattr(self.fallback, "alleles", []))
        return sorted(a)

    @alleles.setter
    def alleles(self, values):
        if values is None:
            self._requested_alleles = None
            return
        if isinstance(values, str):
            raise ValueError("alleles must be a sequence of strings, not a single string")
        parsed = tuple(sorted({mhcgnomes.parse(str(v)).to_string() for v in values if is_stated(v)}))
        if not parsed:
            raise ValueError("alleles sequence cannot be empty; pass None to restore defaults")
        self._requested_alleles = parsed

    @property
    def available_peptide_lengths(self) -> list[int]:
        """Lengths present in the historical rows or fallback."""
        lengths = set(int(x) for x in self._df["peptide_length"].unique().tolist() if pd.notna(x))
        if self.fallback is not None:
            lengths.update(getattr(self.fallback, "default_peptide_lengths", []))
        return sorted(lengths)

    @property
    def default_peptide_lengths(self) -> list[int]:
        """Selected lengths for protein scanning."""
        if self._requested_peptide_lengths is None:
            return self.available_peptide_lengths
        return list(self._requested_peptide_lengths)

    @default_peptide_lengths.setter
    def default_peptide_lengths(self, lengths):
        if lengths is None:
            self._requested_peptide_lengths = None
            return
        requested = list(lengths)
        if not requested or any(
            isinstance(l, bool) or not isinstance(l, Integral) or l <= 0 for l in requested
        ):
            raise ValueError("default_peptide_lengths must be a non-empty sequence of positive integers")
        requested = sorted(set(int(l) for l in requested))
        available = self.available_peptide_lengths
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Requested peptide lengths {missing} are not covered by the historical table ({available})")
        self._requested_peptide_lengths = tuple(requested)

    @property
    def prediction_method_names(self) -> list[str]:
        """Unique prediction method names present in the table."""
        return sorted(set(self._df["prediction_method_name"].dropna().unique().tolist()))

    @property
    def predictor_name(self) -> str:
        """Single method name or comma-separated summary."""
        names = self.prediction_method_names
        if len(names) == 1:
            return names[0]
        elif len(names) > 1:
            return f"historical({','.join(names)})"
        return "historical"

    @property
    def prediction_method_name(self) -> Optional[str]:
        names = self.prediction_method_names
        return names[0] if len(names) == 1 else None

    @property
    def predictor_version(self) -> Optional[str]:
        """Predictor version if unique and stated; None otherwise."""
        versions = self._df["predictor_version"].dropna().unique()
        if len(versions) == 1 and is_stated(versions[0]):
            return str(versions[0])
        return None

    @property
    def predictor_versions(self) -> list[str]:
        """Unique stated predictor versions in the table."""
        return sorted({str(v) for v in self._df["predictor_version"].dropna().unique() if is_stated(v)})

    def kind_support(self) -> dict[str, dict]:
        """MHC context for kinds present in the historical source."""
        support = {}
        fallback_support = {}
        if self.fallback is not None and hasattr(self.fallback, "kind_support"):
            fallback_support = {
                str(getattr(kind, "value", kind)): spec
                for kind, spec in self.fallback.kind_support().items()
            }

        cached_kinds = sorted(self._df["kind"].dropna().unique().tolist())
        for kind in sorted(set(cached_kinds) | set(fallback_support)):
            rows = self._df.loc[self._df["kind"] == kind]
            if "prediction_mhc_dependence" in rows and rows["prediction_mhc_dependence"].notna().any():
                dependence = mhc_dependence(
                    kind,
                    rows=rows.rename(columns={"prediction_mhc_dependence": "source_prediction_mhc_dependence"}),
                )
            elif kind in fallback_support:
                dependence = fallback_support[kind].get("mhc_dependence", "single_allele")
            elif KIND_MHC_DEPENDENCE.get(kind) == "none":
                dependence = "none"
            elif "allele_set" in rows and stated_values(rows["allele_set"]).any():
                dependence = "haplotype"
            else:
                dependence = "single_allele"

            if dependence == "none":
                mhc_class = "none"
            elif kind in fallback_support:
                mhc_class = fallback_support[kind].get("mhc_class", "I")
            elif not rows.empty and rows["allele"].dropna().any():
                from .io_pvacseq import _summarize_mhc_class
                mhc_class = _summarize_mhc_class(rows["allele"])
            else:
                mhc_class = "I"

            support[kind] = {
                "mhc_dependence": dependence,
                "mhc_class": mhc_class,
            }
        return support

    @property
    def supported_kinds(self) -> tuple[str, ...]:
        return tuple(self.kind_support())

    # --- Query Methods ----------------------------------------------

    def predict_peptides_dataframe(
        self, peptides: Iterable[str],
    ) -> pd.DataFrame:
        """Query peptides from the historical table without forcing Cartesian product.

        Parameters
        ----------
        peptides : iterable of str
            Peptides to query.

        Returns
        -------
        pandas.DataFrame
            One row per reported (peptide, allele, kind, model) match.
            Sparse coverage is respected: unmeasured combinations are neither
            demanded nor fabricated.
        """
        peptides = [str(p) for p in peptides]
        if not peptides:
            return pd.DataFrame(columns=list(self._df.columns))

        matched = self._df.loc[self._df["peptide"].isin(peptides)].copy()

        if self._requested_alleles is not None:
            matched = matched.loc[matched["allele"].isin(self._requested_alleles)]

        if self._requested_peptide_lengths is not None:
            matched = matched.loc[matched["peptide_length"].isin(self._requested_peptide_lengths)]

        found_peptides = set(matched["peptide"])
        missing_peptides = [p for p in peptides if p not in found_peptides]

        if missing_peptides and self.fallback is not None:
            fallback_res = self._fallback_predict_peptides(missing_peptides)
            if not fallback_res.empty:
                matched = pd.concat([matched, fallback_res], ignore_index=True)
                found_peptides = set(matched["peptide"])
                missing_peptides = [p for p in peptides if p not in found_peptides]

        if missing_peptides and self.missing == "raise":
            raise CachedPredictorCoverageError(
                f"HistoricalPredictor: no coverage for peptide(s) {missing_peptides[:5]!r}"
            )

        return matched.reset_index(drop=True)

    def predict_contextual_peptides_dataframe(
        self, peptides: Iterable[str], *, n_flanks=None, c_flanks=None,
    ) -> pd.DataFrame:
        """Query exact peptides with contextual flanks from the historical table.

        Parameters
        ----------
        peptides : iterable of str
            Exact sequences to query.
        n_flanks, c_flanks : sequence of str, optional
            Parallel flanking contexts.

        Returns
        -------
        pandas.DataFrame
            Matching rows, preserving explicit context and reported scores.
        """
        peptides = list(peptides)
        if not peptides:
            return pd.DataFrame(columns=list(self._df.columns))

        if (n_flanks is None) != (c_flanks is None):
            raise ValueError("Supply both n_flanks and c_flanks or neither")

        supplied = n_flanks is not None
        if supplied:
            n_flanks = list(n_flanks)
            c_flanks = list(c_flanks)
            if len(n_flanks) != len(peptides) or len(c_flanks) != len(peptides):
                raise ValueError("Flanks must align positionally with peptides")
        else:
            n_flanks = c_flanks = [None] * len(peptides)

        output_rows = []
        misses = []

        for p, nf, cf in zip(peptides, n_flanks, c_flanks):
            candidates = self._df.loc[self._df["peptide"] == p]
            if self._requested_alleles is not None:
                candidates = candidates.loc[candidates["allele"].isin(self._requested_alleles)]

            if candidates.empty:
                misses.append((p, nf, cf))
                continue

            matching = [
                row for row in candidates.to_dict("records")
                if prediction_flanks_match(row, n_flank=nf, c_flank=cf)
            ]
            if not matching:
                misses.append((p, nf, cf))
                continue

            for m in matching:
                row_copy = dict(m)
                row_copy["n_flank"] = nf
                row_copy["c_flank"] = cf
                output_rows.append(row_copy)

        if misses and self.fallback is not None:
            fallback_rows = self._fallback_predict_contextual(misses, supplied=supplied)
            output_rows.extend(fallback_rows)
            # Recheck remaining misses
            covered_misses = {(r["peptide"], r.get("n_flank"), r.get("c_flank")) for r in fallback_rows}
            misses = [m for m in misses if m not in covered_misses]

        if misses and self.missing == "raise":
            raise CachedPredictorCoverageError(
                f"HistoricalPredictor: no coverage for contextual query {misses[:5]!r}"
            )

        if not output_rows:
            return pd.DataFrame(columns=list(self._df.columns))
        return pd.DataFrame(output_rows).reset_index(drop=True)

    def predict_dataframe(self, peptides: Iterable[str], **kwargs) -> pd.DataFrame:
        """Generic prediction entry point conforming to mhctools protocol."""
        if "n_flanks" in kwargs or "c_flanks" in kwargs:
            return self.predict_contextual_peptides_dataframe(
                peptides,
                n_flanks=kwargs.get("n_flanks"),
                c_flanks=kwargs.get("c_flanks"),
            )
        return self.predict_peptides_dataframe(peptides)

    def predict_proteins_dataframe(self, proteins: Mapping[str, str]) -> pd.DataFrame:
        """Scan sliding windows across proteins and look up in historical rows."""
        lengths = self.default_peptide_lengths
        if not lengths:
            return pd.DataFrame(columns=list(PROTEIN_SCAN_COLUMNS))

        scanned = []
        for seq_name, seq in proteins.items():
            for length in lengths:
                if len(seq) < length:
                    continue
                for offset in range(len(seq) - length + 1):
                    pep = seq[offset: offset + length]
                    n_flank = seq[max(0, offset - 10): offset]
                    c_flank = seq[offset + length: offset + length + 10]
                    scanned.append({
                        "source_sequence_name": seq_name,
                        "peptide": pep,
                        "offset": offset,
                        "peptide_offset": offset,
                        "n_flank": n_flank,
                        "c_flank": c_flank,
                    })

        if not scanned:
            return pd.DataFrame(columns=list(PROTEIN_SCAN_COLUMNS))

        peptides = [s["peptide"] for s in scanned]
        n_flanks = [s["n_flank"] for s in scanned]
        c_flanks = [s["c_flank"] for s in scanned]

        pred_df = self.predict_contextual_peptides_dataframe(
            peptides, n_flanks=n_flanks, c_flanks=c_flanks,
        )
        if pred_df.empty:
            return pred_df

        # Merge with sequence coordinates
        scan_df = pd.DataFrame(scanned)
        merged = scan_df.merge(pred_df, on="peptide", how="inner", suffixes=("_scan", ""))
        return merged.reset_index(drop=True)

    # --- Fallback Handling ------------------------------------------

    def _fallback_predict_peptides(self, peptides: list[str]) -> pd.DataFrame:
        """Call fallback predictor, keeping fallback provenance distinct from historical rows."""
        if hasattr(self.fallback, "predict_dataframe"):
            res = self.fallback.predict_dataframe(peptides)
        elif hasattr(self.fallback, "predict_peptides_dataframe"):
            res = self.fallback.predict_peptides_dataframe(peptides)
        else:
            raise TypeError("Fallback must expose predict_dataframe or predict_peptides_dataframe")
        return self._normalize(res)

    def _fallback_predict_contextual(self, misses: list[tuple], supplied: bool) -> list[dict]:
        """Call contextual fallback predictor, keeping fallback provenance distinct."""
        peptides = [m[0] for m in misses]
        n_flanks = [m[1] for m in misses] if supplied else None
        c_flanks = [m[2] for m in misses] if supplied else None

        if hasattr(self.fallback, "predict_contextual_peptides_dataframe"):
            res = self.fallback.predict_contextual_peptides_dataframe(
                peptides, n_flanks=n_flanks, c_flanks=c_flanks,
            )
        elif hasattr(self.fallback, "predict_dataframe"):
            kwargs = {}
            if supplied:
                kwargs = dict(n_flanks=n_flanks, c_flanks=c_flanks)
            res = self.fallback.predict_dataframe(peptides, **kwargs)
        else:
            raise TypeError("Fallback must expose predict_contextual_peptides_dataframe or predict_dataframe")

        norm = self._normalize(res)
        return norm.to_dict("records")

    # --- Factory Classmethods ---------------------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **kwargs) -> HistoricalPredictor:
        return cls(df, **kwargs)

    @classmethod
    def from_table(cls, path_or_df: Union[str, Path, pd.DataFrame], **kwargs) -> HistoricalPredictor:
        return cls(path_or_df, **kwargs)

    @classmethod
    def from_topiary_output(cls, path_or_result: Union[str, Path, TopiaryResult], **kwargs) -> HistoricalPredictor:
        return cls(path_or_result, **kwargs)

    @classmethod
    def from_pvacseq(cls, path_or_result: Union[str, Path, TopiaryResult], **kwargs) -> HistoricalPredictor:
        """Construct HistoricalPredictor from a pVACseq report."""
        if isinstance(path_or_result, (str, Path)):
            from .io_pvacseq import read_pvacseq
            res = read_pvacseq(path_or_result)
            df = res.long_df if hasattr(res, "long_df") and not res.long_df.empty else res.df
        elif isinstance(path_or_result, TopiaryResult):
            df = path_or_result.long_df if hasattr(path_or_result, "long_df") and not path_or_result.long_df.empty else path_or_result.df
        else:
            df = path_or_result
        return cls(df, **kwargs)

    @classmethod
    def from_lens(cls, path_or_result: Union[str, Path, TopiaryResult], **kwargs) -> HistoricalPredictor:
        """Construct HistoricalPredictor from a LENS report."""
        if isinstance(path_or_result, (str, Path)):
            from .io_lens import read_lens
            res = read_lens(path_or_result)
            df = res.long_df if hasattr(res, "long_df") and not res.long_df.empty else res.df
        elif isinstance(path_or_result, TopiaryResult):
            df = path_or_result.long_df if hasattr(path_or_result, "long_df") and not path_or_result.long_df.empty else path_or_result.df
        else:
            df = path_or_result
        return cls(df, **kwargs)

    # --- Serialization ----------------------------------------------

    def to_result(self, **metadata_extra) -> TopiaryResult:
        """Wrap the historical table as a TopiaryResult."""
        meta_extra = dict(metadata_extra)
        meta_extra["historical_prediction_methods"] = self.prediction_method_names
        meta_extra["historical_predictor_versions"] = self.predictor_versions
        return TopiaryResult(self.df, form="long", extra=meta_extra)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the backing DataFrame."""
        return self.df

    def to_tsv(self, path: Union[str, Path], **kwargs):
        from .io import to_tsv
        to_tsv(self.df, path, **kwargs)

    def to_csv(self, path: Union[str, Path], **kwargs):
        from .io import to_csv
        to_csv(self.df, path, **kwargs)

    def to_parquet(self, path: Union[str, Path], **kwargs):
        self.df.to_parquet(path, **kwargs)
