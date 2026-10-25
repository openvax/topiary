"""Read-only prediction sources for sparse, provenance-bearing historical tables."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
from numbers import Integral, Real
import math
from pathlib import Path
from uuid import uuid4

import mhcgnomes
import pandas as pd

from .cached import CachedPredictorCoverageError, prediction_flanks_match, _flank_flag, _dependence_tag
from .io import Metadata, read_csv, read_tsv, _decode_json, _encode_json, _json_columns
from .ranking import format_allele_set, is_stated, mhc_dependence, split_allele_set
from .result import TopiaryResult
from .wide import _concat_frames, _group_token
from .wide import SOURCE_PREDICTION_COLUMNS


# Historical observations belong to their source and occurrence. Unlike a live
# cache, they are not interchangeable merely because their peptide is the same.
HISTORICAL_KEY_COLUMNS = (
    "historical_source_id", "source_label", "source_row", "source_observation_id",
    "sample_name", "candidate_sample", "prediction_id", "source_sequence_name",
    "peptide_offset", "gene_id", "transcript_id", "peptide", "allele",
    "peptide_length", "kind", "prediction_method_name", "predictor_version",
    "n_flank", "c_flank", "allele_set", "prediction_flanks_supplied",
    "prediction_mhc_dependence", "wt_peptide", "wt_n_flank", "wt_c_flank",
    "wt_peptide_offset",
)

_OCCURRENCE_COLUMNS = (
    "source_observation_id", "prediction_id", "source_sequence_name", "peptide_offset",
    "source_label", "source_row", "gene_id", "transcript_id",
)
_QUERY_INDEX = "historical_query_index"


def historical_prediction_key_columns(df):
    """Return the columns identifying reported historical observations.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical prediction rows; an empty schema is allowed.

    Returns
    -------
    list of str
        Standard identity axes followed by additional source annotations.
        Primary/comparator measurements, units and measurement context are
        values rather than identity. Different coding sequences or read
        evidence describe distinct observations and must remain separate.
    """
    values = {"value", "score", "affinity", "percentile_rank", "value_unit", "measurement_context"}
    values.update(f"{scope}_{field}" for scope in ("wt", "self", "self_nearest", "shuffled")
                  for field in ("value", "score", "affinity", "percentile_rank", "value_unit",
                                "measurement_context", "prediction_method_name", "predictor_version"))
    base = [column for column in HISTORICAL_KEY_COLUMNS if column in df]
    return base + [column for column in df if column not in base and column not in values]


def _copy_frame(frame):
    """Copy object cells too: pandas' deep copy does not copy nested containers."""
    copied = frame.copy(deep=True)
    for column in copied:
        if pd.api.types.is_object_dtype(copied[column]):
            copied[column] = copied[column].map(deepcopy)
    copied.attrs = deepcopy(frame.attrs)
    return copied


def _comparison(frame):
    return frame.map(_group_token)


def conflicting_historical_predictions(df):
    """Return distinct, contradictory rows for the same historical identity.

    Parameters
    ----------
    df : pandas.DataFrame
        Normalized historical rows. Source, occurrence, context, model and
        comparator identities follow ``historical_prediction_key_columns``.

    Returns
    -------
    pandas.DataFrame
        Rows whose keys agree but whose remaining cells differ. All stored
        measurements are compared exactly, including missing
        values and nested containers. No tolerance or non-null preference is
        applied. Empty input or a table without key columns returns no conflicts.
    """
    key = historical_prediction_key_columns(df)
    if df.empty or not key:
        return df.iloc[:0].copy()
    comparison = _comparison(df).reset_index(drop=True)
    distinct = ~comparison.duplicated()
    positions = comparison.loc[distinct].duplicated(subset=key, keep=False)
    return df.iloc[positions.index[positions]]


def _source_id(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _genotype(value):
    return format_allele_set(mhcgnomes.parse(name).to_string() for name in split_allele_set(value)) if is_stated(value) else ""


def _sample(row):
    return next((row[column] for column in ("sample_name", "candidate_sample") if is_stated(row.get(column))), None)


class HistoricalPredictor:
    """A read-only source of reported predictions, with explicit sparse coverage.

    Parameters
    ----------
    df : pandas.DataFrame, TopiaryResult, path-like, or None
        Long or wide historical rows. Paths name typed Topiary CSV/TSV
        (including gzip) or Parquet. Use ``from_lens`` / ``from_pvacseq`` for
        those report formats. Empty input is a valid empty source.
    source_id : str, optional
        Artifact identity for rows without ``historical_source_id``. File
        loaders use the file's SHA-256; in-memory input receives an opaque UUID
        unless supplied explicitly. Saved records retain their IDs on reload.
        Existing stated IDs cannot be relabelled with this argument.
    alleles : iterable of str, optional
        Query selection. Per-allele rows are restricted to this set; a genotype
        measurement requires an exact genotype match. Allele-free rows remain
        available. This never creates a peptide/allele cross-product.
    default_peptide_lengths : iterable of int, optional
        Query lengths, drawn from the stored lengths. No protein scanning runs.
    missing : {'raise', 'omit', 'ignore'}
        Raise on an entirely unreported query (the default), or omit it. Partial
        model/allele coverage remains sparse in either mode. ``ignore`` is an
        alias for ``omit``. Reapply query options when loading a saved source.

    Notes
    -----
    Unknown versions remain unknown, including when another row names a version.
    Exact duplicate records collapse; conflicting observations raise. Every
    other column and result metadata is preserved. No live fallback is run.
    Add fresh predictions explicitly with ``rescore_candidates`` or combine a
    separately computed source with ``combine_sources``.

    Use ``lookup_occurrences`` or ``rescore_candidates`` for scoped replay.
    This is not a complete-coverage live mhctools model: ``TopiaryPredictor`` and
    ``predict_peptide_occurrences`` accept live models, not this sparse source.
    """

    def __init__(self, df=None, *, source_id=None, alleles=None,
                 default_peptide_lengths=None, missing="raise"):
        if missing not in {"raise", "omit", "ignore"}:
            raise ValueError("missing must be 'raise' or 'omit' ('ignore' is an alias)")
        if source_id is not None and (not isinstance(source_id, str) or not is_stated(source_id)):
            raise ValueError("source_id must be a nonempty string")
        self.missing = "omit" if missing == "ignore" else missing
        if isinstance(df, (str, Path)):
            path = Path(df)
            result = self._read_table(path)
            # Persisted IDs name the original artifact, not this later encoding.
            if source_id is None and "historical_source_id" not in result.to_long().df:
                source_id = _source_id(path)
        elif isinstance(df, TopiaryResult):
            result = df
        elif df is None or isinstance(df, pd.DataFrame):
            result = TopiaryResult(df if df is not None else pd.DataFrame(), form=None if df is not None else "long")
        else:
            raise TypeError("df must be a DataFrame, TopiaryResult, or file path")
        self._metadata = deepcopy(result.metadata)
        raw = result.to_long().df
        self._df = self._normalize(raw, source_id)
        conflicts = conflicting_historical_predictions(self._df)
        if not conflicts.empty:
            raise ValueError("HistoricalPredictor: more than one answer for a historical key; "
                             "conflicting measurements or annotations must be reconciled")
        self._df = self._df.loc[~_comparison(self._df).duplicated()].reset_index(drop=True)
        self._annotation_keys = (set(historical_prediction_key_columns(self._df))
                                 - set(HISTORICAL_KEY_COLUMNS)
                                 - set(SOURCE_PREDICTION_COLUMNS.values()))
        self._requested_alleles = None
        self._requested_peptide_lengths = None
        self.alleles = alleles
        self.default_peptide_lengths = default_peptide_lengths

    @staticmethod
    def _read_table(path):
        if str(path).endswith((".parquet", ".pq")):
            frame = pd.read_parquet(path)
            envelope = frame.attrs.pop("topiary_historical", None)
            if envelope is None:
                return TopiaryResult(frame)
            if envelope.get("version") != 1:
                raise ValueError("Unsupported historical Parquet metadata version")
            for column in envelope["json_columns"]:
                frame[column] = pd.Series([_decode_json(value, allow_scalar=True)
                                           for value in frame[column]], dtype=object)
            return TopiaryResult(frame, metadata=Metadata(**envelope["metadata"]))
        if str(path).endswith((".tsv", ".tsv.gz", ".txt")):
            return read_tsv(path)
        if str(path).endswith((".csv", ".csv.gz")):
            return read_csv(path)
        raise ValueError("Historical tables must be CSV, TSV, or Parquet")

    def _normalize(self, frame, source_id):
        if not frame.columns.is_unique:
            raise ValueError("Historical columns must be unique")
        if _QUERY_INDEX in frame:
            raise ValueError(f"{_QUERY_INDEX} is reserved for lookup output")
        out = _copy_frame(frame).reset_index(drop=True)
        if out.empty:
            for column in HISTORICAL_KEY_COLUMNS:
                if column not in out:
                    out[column] = pd.Series(dtype=object)
            return out
        for column in ("peptide", "kind"):
            if column not in out or not out[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
                raise ValueError(f"Historical rows require nonempty string {column}")
        if "peptide_length" in out and not out.peptide_length.map(
                lambda value: not isinstance(value, bool) and isinstance(value, Integral)).all():
            raise ValueError("peptide_length must contain integers")
        if "peptide_length" in out and not out.peptide_length.eq(out.peptide.str.len()).all():
            raise ValueError("peptide_length disagrees with peptide")
        out["peptide_length"] = out.peptide.str.len()
        if "source_sequence_name" not in out:
            out["source_sequence_name"] = None
        if "peptide_offset" not in out:
            out["peptide_offset"] = pd.array([None] * len(out), dtype="Int64")
        for column in ("peptide_offset", "wt_peptide_offset"):
            if column in out and not out[column].dropna().map(
                    lambda value: not isinstance(value, bool) and isinstance(value, Real)
                    and math.isfinite(value) and value >= 0 and value == int(value)).all():
                raise ValueError(f"{column} must be a nonnegative integer or missing")
        for column in ("prediction_method_name", "predictor_version"):
            if column not in out:
                out[column] = None
            out[column] = out[column].map(lambda value: str(value) if is_stated(value) else None)
        if "allele" not in out:
            out["allele"] = ""
        out["allele"] = out.allele.map(lambda value: mhcgnomes.parse(str(value)).to_string() if is_stated(value) else "")
        for column in ("n_flank", "c_flank", "wt_n_flank", "wt_c_flank"):
            if column in out and not out[column].map(
                    lambda value: isinstance(value, str) or pd.api.types.is_scalar(value) and pd.isna(value)).all():
                raise ValueError(f"{column} must contain strings or missing values")
        if "allele_set" in out:
            out["allele_set"] = out.allele_set.map(_genotype)
        if "prediction_flanks_supplied" in out:
            out["prediction_flanks_supplied"] = out.prediction_flanks_supplied.map(_flank_flag)
        if "prediction_mhc_dependence" in out:
            out["prediction_mhc_dependence"] = out.prediction_mhc_dependence.map(_dependence_tag)
        modes = []
        for row in out.to_dict("records"):
            mode = row.get("prediction_mhc_dependence")
            source_mode = row.get("source_prediction_mhc_dependence")
            if mode in {"none", "single_allele", "haplotype"}:
                if source_mode in {"none", "single_allele", "haplotype"} and mode != source_mode:
                    raise ValueError("Stored prediction and source MHC dependence disagree")
                row["source_prediction_mhc_dependence"] = mode
            modes.append(mhc_dependence(row["kind"], rows=pd.DataFrame([row])))
        out["prediction_mhc_dependence"] = modes
        measurement_columns = [f"{scope}{field}" for scope in ("", "wt_", "self_", "self_nearest_", "shuffled_")
                               for field in ("value", "score", "affinity", "percentile_rank")]
        for column in measurement_columns:
            if column in out and not out[column].dropna().map(
                    lambda value: isinstance(value, Real) and not isinstance(value, bool)).all():
                raise ValueError(f"{column} must contain numeric measurements or missing values")
        if "historical_source_id" not in out:
            out["historical_source_id"] = source_id or "historical:" + str(uuid4())
        elif source_id is not None and not out.historical_source_id.eq(source_id).all():
            raise ValueError("source_id cannot relabel existing historical records")
        if not out.historical_source_id.map(lambda value: isinstance(value, str) and is_stated(value)).all():
            raise ValueError("Every historical row requires a source identity")
        return out

    @property
    def df(self):
        """A detached copy of all records, including nested annotation cells."""
        return _copy_frame(self._df)

    @property
    def alleles(self):
        """Selected alleles, or the reported union (never a coverage promise)."""
        if self._requested_alleles is not None:
            return list(self._requested_alleles)
        alleles = {value for value in self._df.allele if is_stated(value)}
        for value in self._df.get("allele_set", ()):
            alleles.update(split_allele_set(value))
        return sorted(alleles)

    @alleles.setter
    def alleles(self, values):
        if values is None:
            self._requested_alleles = None
        else:
            if isinstance(values, str):
                raise ValueError("alleles must be a sequence, not one string")
            values = tuple(sorted({mhcgnomes.parse(str(value)).to_string() for value in values}))
            if not values:
                raise ValueError("alleles must be nonempty; use None for all reported alleles")
            self._requested_alleles = values

    @property
    def available_peptide_lengths(self):
        """Lengths actually present in the source."""
        return sorted(set(self._df.peptide_length.dropna()))

    @property
    def default_peptide_lengths(self):
        """Selected lengths, or all reported lengths."""
        return self.available_peptide_lengths if self._requested_peptide_lengths is None else list(self._requested_peptide_lengths)

    @default_peptide_lengths.setter
    def default_peptide_lengths(self, values):
        if values is None:
            self._requested_peptide_lengths = None
            return
        values = list(values)
        if not values or any(isinstance(value, bool) or not isinstance(value, Integral) or value <= 0 for value in values):
            raise ValueError("Peptide lengths must be nonempty positive integers")
        if set(values) - set(self.available_peptide_lengths):
            raise ValueError("Requested lengths are not reported in the historical source")
        self._requested_peptide_lengths = tuple(sorted(set(values)))

    @property
    def prediction_method_names(self):
        """Stated method names, preserving aggregate labels."""
        return sorted(set(self._df.prediction_method_name.dropna()))

    @property
    def predictor_versions(self):
        """Stated versions; unknown rows remain present in the backing table."""
        return sorted(set(self._df.predictor_version.dropna()))

    @property
    def predictor_version(self):
        """The version only when every row states that same version."""
        return self.predictor_versions[0] if len(self.predictor_versions) == 1 and self._df.predictor_version.notna().all() else None

    @property
    def prediction_method_name(self):
        """The method only when every row states that same method."""
        return self.prediction_method_names[0] if len(self.prediction_method_names) == 1 and self._df.prediction_method_name.notna().all() else None

    def kind_support(self):
        """Describe homogeneous stored kinds; mixed scopes require row lookup.

        This reports source metadata, not complete live-model coverage.
        """
        support = {}
        for kind, rows in self._df.groupby("kind", sort=False):
            modes = rows.prediction_mhc_dependence.unique()
            if len(modes) != 1:
                raise ValueError("Historical kind has multiple MHC scopes; use lookup_occurrences")
            support[kind] = {"mhc_dependence": modes[0]}
        return support

    def _selected(self):
        rows = self._df
        if self._requested_peptide_lengths is not None:
            rows = rows.loc[rows.peptide_length.isin(self._requested_peptide_lengths)]
        if self._requested_alleles is not None:
            genotype = format_allele_set(self._requested_alleles)
            rows = rows.loc[[row["prediction_mhc_dependence"] == "none" or (
                _genotype(row.get("allele_set")) == genotype if row["prediction_mhc_dependence"] == "haplotype"
                else row["allele"] in self._requested_alleles) for row in rows.to_dict("records")]]
        return rows

    def _matches(self, row, query, *, use_flanks):
        if row["peptide"] != query["peptide"]:
            return False
        for column in _OCCURRENCE_COLUMNS:
            if is_stated(row.get(column)) and _group_token(row[column]) != _group_token(query.get(column)):
                return False
        if is_stated(_sample(row)) and _sample(row) != _sample(query):
            return False
        if is_stated(query.get("historical_source_id")) and query["historical_source_id"] != row["historical_source_id"]:
            return False
        if is_stated(query.get("allele")) and row["prediction_mhc_dependence"] != "none":
            if row["prediction_mhc_dependence"] == "single_allele" and row["allele"] != mhcgnomes.parse(str(query["allele"])).to_string():
                return False
        genotype = _genotype(row.get("allele_set"))
        if genotype and genotype != _genotype(query.get("allele_set")):
            return False
        if row["prediction_mhc_dependence"] == "haplotype" and not genotype:
            return False  # Unknown genotype cannot establish a matching scope.
        if use_flanks:
            # Replay a reported occurrence, without claiming that the original
            # model used its annotated flanks as inference inputs.
            if any(_group_token(row.get(column)) != _group_token(query.get(column))
                   for column in ("n_flank", "c_flank")):
                return False
        elif not prediction_flanks_match(row):
            return False
        if any(column in query and _group_token(row.get(column)) != _group_token(query[column])
               for column in self._annotation_keys):
            return False
        for column in ("wt_peptide", "wt_n_flank", "wt_c_flank", "wt_peptide_offset"):
            if column in query and _group_token(query[column]) != _group_token(row.get(column)):
                return False
        return True

    def lookup_occurrences(self, occurrences, *, use_flanks=True):
        """Return exact historical matches and their zero-based query positions.

        Parameters
        ----------
        occurrences : DataFrame or iterable of mappings
            Each query requires a nonempty peptide. Every known stored source
            name, prediction ID, offset, sample, source label/row, gene and
            transcript ID must be supplied and match.
            Optional allele/source-ID selections narrow matches. Genotypes must
            match exactly when recorded. Query measurements are ignored.
        use_flanks : bool
            Match the recorded occurrence's flank annotations exactly, without
            inferring whether the original model used them. False explicitly
            selects peptide-only predictions. Unknown context cannot satisfy
            known flanks; existing prediction_flanks_supplied flags are retained.

        Returns
        -------
        pandas.DataFrame
            Original records plus ``historical_query_index``. Duplicate queries
            retain separate positions; unreported combinations are not invented.
            Missing entire occurrences raise or are omitted according to
            ``missing``. Empty queries return an empty frame without inference.
        """
        if not isinstance(use_flanks, bool):
            raise TypeError("use_flanks must be a boolean")
        queries = pd.DataFrame(occurrences).reset_index(drop=True)
        if not queries.columns.is_unique:
            raise ValueError("Occurrence columns must be unique")
        if not queries.empty and ("peptide" not in queries or not queries.peptide.map(
                lambda value: isinstance(value, str) and bool(value.strip())).all()):
            raise ValueError("Occurrence queries require nonempty peptide strings")
        for column in ("n_flank", "c_flank"):
            if column in queries and not queries[column].map(
                    lambda value: isinstance(value, str) or pd.api.types.is_scalar(value) and pd.isna(value)).all():
                raise ValueError(f"{column} must contain strings or missing values")
        frames, missing = [], []
        selected = self._selected()
        for index, query in enumerate(queries.to_dict("records")):
            candidates = selected.loc[selected.peptide.eq(query["peptide"])]
            matches = candidates.loc[[self._matches(row, query, use_flanks=use_flanks)
                                      for row in candidates.to_dict("records")]].copy()
            if matches.empty:
                missing.append(index)
            matches[_QUERY_INDEX] = index
            frames.append(matches)
        if missing and self.missing == "raise":
            raise CachedPredictorCoverageError(f"Historical source has no matching reported occurrence at query positions {missing[:5]}")
        if not frames:
            return self.df.iloc[:0].assign(**{_QUERY_INDEX: pd.Series(dtype=int)})
        return _copy_frame(_concat_frames(frames))

    def predict_peptides_dataframe(self, peptides):
        """Retrieve all reported records for sequences, without reassigning context.

        Unlike occurrence lookup, this is an inventory query: all recorded
        proteins, samples and flanks are returned unchanged. Missing sequences
        raise or are omitted according to ``missing``; empty input is empty.
        """
        if isinstance(peptides, str):
            raise ValueError("peptides must be a sequence of strings")
        peptides = list(dict.fromkeys(peptides))
        if any(not isinstance(value, str) or not value.strip() for value in peptides):
            raise ValueError("peptides must contain nonempty strings")
        rows = self._selected()
        missing = set(peptides) - set(rows.peptide)
        if missing and self.missing == "raise":
            raise CachedPredictorCoverageError(f"Historical source has no reported peptide: {sorted(missing)[:5]}")
        return _copy_frame(_concat_frames([rows.loc[rows.peptide.eq(peptide)] for peptide in peptides])) if peptides else self.df.iloc[:0]

    def to_result(self):
        """Return all backing rows and original result metadata, detached from this source."""
        return TopiaryResult(self.df, metadata=deepcopy(self._metadata), form="long")

    def to_dataframe(self, *, selected=False):
        """Return detached records without changing their identities.

        Parameters
        ----------
        selected : bool
            False returns all backing rows. True applies configured allele
            and peptide-length selections, including exact genotype scope.

        Returns
        -------
        pandas.DataFrame
            A copy preserving original cells, possibly empty. Nested annotation
            containers are copied too, so edits cannot mutate this source.
        """
        return _copy_frame(self._selected()) if selected else self.df

    def to_tsv(self, path):
        """Save records and metadata through Topiary's typed TSV codec."""
        self.to_result().to_tsv(path)

    def to_csv(self, path):
        """Save records and metadata through Topiary's typed CSV codec."""
        self.to_result().to_csv(path)

    def to_parquet(self, path):
        """Save records and metadata, encoding heterogeneous cells losslessly."""
        frame = self.df
        columns = _json_columns(frame)
        for column in columns:
            frame[column] = [_encode_json(value, column=column, allow_scalar=True) for value in frame[column]]
        frame.attrs["topiary_historical"] = dict(version=1, metadata=asdict(self._metadata), json_columns=columns)
        frame.to_parquet(path, index=False)

    @classmethod
    def from_dataframe(cls, df, **kwargs):
        """Load normalized rows with the same validation as the constructor."""
        return cls(df, **kwargs)

    @classmethod
    def from_table(cls, path_or_df, **kwargs):
        """Load typed Topiary tables, preserving stored historical source IDs."""
        return cls(path_or_df, **kwargs)

    from_topiary_output = from_table

    @classmethod
    def from_pvacseq(cls, path_or_result, **kwargs):
        """Load pVACseq through its public reader, keeping aggregate identities."""
        if isinstance(path_or_result, (str, Path)):
            from .io_pvacseq import read_pvacseq
            kwargs.setdefault("source_id", _source_id(path_or_result))
            path_or_result = read_pvacseq(path_or_result)
        return cls(path_or_result, **kwargs)

    @classmethod
    def from_lens(cls, path_or_result, **kwargs):
        """Load LENS through its public reader, preserving model and source metadata."""
        if isinstance(path_or_result, (str, Path)):
            from .io_lens import read_lens
            kwargs.setdefault("source_id", _source_id(path_or_result))
            path_or_result = read_lens(path_or_result)
        return cls(path_or_result, **kwargs)
