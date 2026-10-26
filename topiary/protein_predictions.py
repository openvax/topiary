"""Validate protein-scan output before using its occurrence predictions."""

from numbers import Integral, Real
from itertools import repeat
import math
from collections.abc import Mapping

import mhcgnomes
import pandas as pd

from .predictor import from_predictions
from .ranking import format_allele_set, is_stated, prediction_mhc_scope


def annotate_protein_prediction_context(predictions, model):
    """Record the configured scanner window without inferring it from rows.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Fresh output from this model's protein scanner. Existing explicit
        inference flags are retained. The input frame is not modified.
    model : predictor
        A producer exposing ``protein_flank_lengths()`` and
        ``uses_flanking_sequences``. Legacy producers without that public
        declaration retain unknown limits; cached producers preserve their
        original row metadata instead of relabelling it.

    Returns
    -------
    pandas.DataFrame
        Rows with declared N/C window limits on flank-supplied predictions.
        A returned limit contradicting the configured producer raises
        ValueError. Explicit peptide-only rows carry no inference window.
        Empty frames retain the same declared metadata columns.
    """
    from .cached import _flank_flag, prediction_flank_limits

    frame = predictions.copy()
    resolver = getattr(model, "protein_flank_lengths", None)
    supplied = getattr(model, "uses_flanking_sequences", None)
    if not callable(resolver) or supplied is None:
        return frame
    declared = tuple(resolver())
    columns = ("prediction_n_flank_limit", "prediction_c_flank_limit")
    if len(declared) != 2:
        raise ValueError("protein_flank_lengths must return two nonnegative integers")
    limits = prediction_flank_limits(dict(zip(columns, declared)))
    if any(limit is None for limit in limits):
        raise ValueError("protein_flank_lengths must declare both window limits")
    flags = frame.get("prediction_flanks_supplied", pd.Series(None, index=frame.index, dtype=object)).map(_flank_flag)
    flags = flags.where(flags.notna(), bool(supplied))
    frame["prediction_flanks_supplied"] = pd.array(flags, dtype="boolean")
    values = []
    for flag, n_limit, c_limit in zip(
            frame.prediction_flanks_supplied,
            frame.get(columns[0], repeat(None, len(frame))),
            frame.get(columns[1], repeat(None, len(frame)))):
        expected = limits if flag else (None, None)
        recorded_limits = prediction_flank_limits(dict(zip(columns, (n_limit, c_limit))))
        for recorded, limit in zip(recorded_limits, expected):
            if recorded is not None and recorded != limit:
                raise ValueError("Returned protein flank limit differs from the configured scanner")
        values.append(expected)
    for index, column in enumerate(columns):
        frame[column] = pd.array([pair[index] for pair in values], dtype="Int64")
    return frame


def validate_protein_predictions(predictions, sequences, model, *, peptide_lengths):
    """Validate and normalize one configured model's protein-scan response.

    Parameters
    ----------
    predictions : pandas.DataFrame
        Scanner output with explicit source names and ``offset`` (or
        ``peptide_offset``), peptides, kinds, alleles and model provenance.
    sequences : mapping of str to str
        Original named proteins supplied to the scanner. Coordinates must
        identify the reported peptide exactly within one of these sequences.
    model : predictor
        Configured producer declaring ``kind_support()`` and ``alleles``.
        Missing inference flags and haplotype labels are filled from this
        explicit invocation, never from a different cached prediction.
    peptide_lengths : iterable of int
        Requested positive window lengths. Extra lengths emitted by a scanner
        are validated but excluded from the returned frame. Model configuration
        is not changed. Empty proteins or lengths with no windows require no rows.

    Returns
    -------
    pandas.DataFrame
        Topiary long-form rows for the requested lengths. Every requested
        occurrence must have each declared kind and configured MHC scope exactly
        once. Repeated peptide sequences keep their own coordinates. Malformed,
        out-of-scope, duplicate or incomplete responses raise ValueError.
        Numerical/model-version cache consistency is checked by CachedPredictor
        before insertion; this function validates the producer's geometry and
        declared inference scope.
    """
    from .cached import _flank_flag, _flank_key, prediction_matches_protein_context

    if not isinstance(sequences, Mapping) or any(
            not isinstance(name, str) or not name or not isinstance(sequence, str)
            for name, sequence in sequences.items()):
        raise ValueError("Protein inputs must map nonempty names to sequence strings")
    lengths = list(peptide_lengths)
    if any(isinstance(length, bool) or not isinstance(length, Integral) or length <= 0 for length in lengths):
        raise ValueError("Protein window lengths must be positive integers")
    if not callable(getattr(model, "kind_support", None)):
        raise TypeError("Protein fallback must declare kind_support")
    support = {str(getattr(kind, "value", kind)): spec for kind, spec in model.kind_support().items()}
    if not support or any(spec.get("mhc_dependence") not in {"none", "single_allele", "haplotype"}
                          for spec in support.values()):
        raise ValueError("Protein fallback must declare MHC dependence for every kind")
    alleles = sorted({mhcgnomes.parse(str(allele)).to_string() for allele in getattr(model, "alleles", ())})
    if any(spec["mhc_dependence"] != "none" for spec in support.values()) and not alleles:
        raise ValueError("Protein fallback requires configured alleles")
    expected = set()
    for name, sequence in sequences.items():
        for length in set(lengths):
            for offset in range(len(sequence) - length + 1):
                for kind, spec in support.items():
                    mode = spec["mhc_dependence"]
                    for allele in alleles if mode == "single_allele" else [None]:
                        expected.add((name, offset, length, kind,
                                      prediction_mhc_scope(allele, dependence=mode, allele_set=alleles)))
    raw = pd.DataFrame(predictions).copy()
    if not raw.columns.is_unique:
        raise ValueError("Protein fallback columns must be unique")
    if raw.empty:
        if expected:
            raise ValueError("Protein fallback returned no predictions for requested windows")
        return from_predictions(raw)
    required = {"peptide", "source_sequence_name", "kind", "allele"}
    offset_column = "offset" if "offset" in raw else "peptide_offset"
    if required - set(raw) or offset_column not in raw:
        raise ValueError("Protein fallback requires explicit source, coordinate, peptide, kind and allele")
    for row in raw.to_dict("records"):
        peptide, name, offset = row["peptide"], row["source_sequence_name"], row[offset_column]
        if not isinstance(peptide, str) or not peptide or not isinstance(name, str) or name not in sequences:
            raise ValueError("Protein fallback returned an unknown source or invalid peptide")
        if (isinstance(offset, bool) or not isinstance(offset, Real) or not math.isfinite(offset)
                or offset < 0 or offset != int(offset)):
            raise ValueError("Protein fallback offsets must be nonnegative integers")
        if sequences[name][int(offset):int(offset) + len(peptide)] != peptide:
            raise ValueError("Protein fallback peptide does not match its source coordinate")
        for column in ("peptide_length", "length"):
            if column in row and (isinstance(row[column], bool) or row[column] != len(peptide)):
                raise ValueError("Protein fallback peptide length disagrees with its sequence")
    frame = annotate_protein_prediction_context(from_predictions(raw), model)
    frame["allele"] = frame.allele.map(lambda value: mhcgnomes.parse(str(value)).to_string() if is_stated(value) else "")
    actual, records = set(), []
    for row in frame.to_dict("records"):
        kind = row["kind"]
        if kind not in support:
            raise ValueError("Protein fallback returned an undeclared prediction kind")
        mode = support[kind]["mhc_dependence"]
        reported = row.get("prediction_mhc_dependence")
        if reported is not None and not pd.isna(reported) and reported != mode:
            raise ValueError("Protein fallback returned a different MHC dependence")
        row["prediction_mhc_dependence"] = mode
        allowed_alleles = {""} if mode == "none" else set(alleles)
        if mode == "haplotype":
            allowed_alleles.add("")
        if row["allele"] not in allowed_alleles:
            raise ValueError("Protein fallback returned an allele outside its configured scope")
        if mode == "haplotype":
            genotype = prediction_mhc_scope(None, dependence=mode, allele_set=row.get("allele_set"))
            configured = prediction_mhc_scope(None, dependence=mode, allele_set=alleles)
            if genotype is not None and genotype != configured:
                raise ValueError("Protein fallback returned a different genotype")
            row["allele_set"] = format_allele_set(alleles)
        flag = _flank_flag(row.get("prediction_flanks_supplied"))
        if flag is None:
            # Nested caches may retain legacy unknown context. Absence of a
            # model declaration is not evidence that the inputs were unused.
            declared_flanks = getattr(model, "uses_flanking_sequences", None)
            flag = None if declared_flanks is None else bool(declared_flanks)
        row["prediction_flanks_supplied"] = flag
        sequence, offset = sequences[row["source_sequence_name"]], int(row["peptide_offset"])
        for column in ("n_flank", "c_flank"):
            value = row.get(column)
            if flag and not isinstance(value, str):
                raise ValueError("Protein fallback must report both supplied inference flanks")
            if is_stated(value) and not isinstance(value, str):
                raise ValueError("Protein fallback flanks must be strings")
            flank = _flank_key(value)
            observed = (sequence[max(0, offset - len(flank)):offset] if column == "n_flank"
                        else sequence[offset + len(row["peptide"]):offset + len(row["peptide"]) + len(flank)])
            if flank and observed.upper() != flank:
                raise ValueError("Protein fallback returned flank context from another occurrence")
        if not prediction_matches_protein_context(row, sequence, offset):
            raise ValueError("Protein fallback inference context does not match the occurrence")
        if len(row["peptide"]) not in lengths:
            continue
        scope = prediction_mhc_scope(row["allele"], dependence=mode, allele_set=row.get("allele_set"))
        key = (row["source_sequence_name"], offset, len(row["peptide"]), kind, scope)
        if key in actual:
            raise ValueError("Protein fallback returned duplicate occurrence predictions")
        actual.add(key)
        records.append(row)
    if actual != expected:
        raise ValueError(f"Protein fallback lacks complete kind/allele/occurrence coverage ({len(expected - actual)} missing)")
    return pd.DataFrame(records, columns=list(dict.fromkeys([*frame.columns, "prediction_mhc_dependence",
                                                          "prediction_flanks_supplied", "allele_set",
                                                          "prediction_n_flank_limit", "prediction_c_flank_limit"])))
