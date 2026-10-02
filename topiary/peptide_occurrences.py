"""Predict explicitly supplied peptide occurrences without scanning proteins."""

import numpy as np
import pandas as pd
import mhcgnomes

from .predictor import from_predictions
from .prediction_batch import predict_with_cache_miss_report
from .ranking import format_allele_set, split_allele_set, prediction_mhc_scope
from .wide import PREDICTION_COLUMNS, _concat_frames


def predict_peptide_occurrences(occurrences, model, *, use_flanks=True, predict_wt=False, on_miss=None):
    """Score exact peptide occurrences with one configured mhctools model.

    Parameters
    ----------
    occurrences : pandas.DataFrame or iterable of mappings
        Nonempty string ``prediction_id`` and ``peptide`` identify each row
        together with ``peptide_offset`` and any supplied ``sample_name`` /
        ``candidate_sample``. This compound identity must be unique; an ID may
        name several peptide windows in one source.
        Optional ``peptide_offset`` is a nonnegative source coordinate; absent
        coordinates remain unknown. ``n_flank`` and ``c_flank`` are strings or
        missing: an empty string states a known terminus. Other non-prediction
        columns (source IDs, genes, RNA evidence, etc.) survive unchanged.
        Prediction columns and allele-specific ``candidate_id`` /
        ``candidate_allele`` / ``candidate_mhc_class`` are refused: keep historical measurements in their
        original table and use :func:`rescore_candidates` for additive features.
        A supplied ``allele_set`` must match a haplotype model's configuration.
    model : mhctools predictor
        Configured model exposing ``predict_dataframe`` and ``kind_support``.
        Its alleles define the requested prediction scope. Every declared kind
        must cover every input, including every configured allele for per-allele
        kinds. Unsupported inputs, missing coverage and ambiguous outputs raise.
    use_flanks : bool
        Forward known flanks through mhctools' flank-aware API. Models declaring
        ``uses_flanking_sequences`` require both flanks unless False explicitly
        requests peptide-only inference. Original context is retained either way.
    predict_wt : bool
        Also score supplied ``wt_peptide`` comparators, using only explicitly
        supplied ``wt_n_flank`` / ``wt_c_flank`` context. Missing comparators keep
        missing scores. No baseline, coordinate correspondence or flank is
        inferred from the primary peptide.
    on_miss : callable, optional
        Explicitly report and skip cache coverage failures, using
        :func:`predict_with_cache_miss_report`. The report's source name is the
        occurrence ID, with peptide, offset and supplied sample keys identifying
        the window. Other errors still raise; partial results emit a warning.

    Returns
    -------
    pandas.DataFrame
        Topiary long-form predictions in occurrence order, with model/version,
        ``prediction_mhc_dependence``, ``prediction_flanks_supplied`` and genotype
        scope. Equal inputs within a sample/flank/genotype partition are inferred
        once and expanded back to all occurrences; biological support is never
        summed. Empty input calls no model and returns an empty prediction frame.
        No sliding windows, additional peptide sequences or source candidates
        are created. Model errors propagate; input tables are never mutated.
    """
    if not isinstance(use_flanks, bool) or not isinstance(predict_wt, bool):
        raise TypeError("use_flanks and predict_wt must be explicit booleans")
    if on_miss is not None and not callable(on_miss):
        raise TypeError("on_miss must be callable or None")
    frame = pd.DataFrame(occurrences).copy().reset_index(drop=True)
    if not frame.columns.is_unique:
        raise ValueError("Occurrence columns must be unique")
    if frame.empty:
        return from_predictions([], extra_columns={column: [] for column in frame})
    for column in ("prediction_id", "peptide"):
        if column not in frame or not frame[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Occurrences require nonempty string {column}")
    reserved = (PREDICTION_COLUMNS | {"allele", "value_unit", "measurement_context", "predictor_name",
                                    "offset", "prediction_mhc_dependence", "prediction_flanks_supplied",
                                    "candidate_id", "candidate_allele", "candidate_mhc_class"}) & set(frame)
    if reserved:
        raise ValueError(f"Occurrence input contains prediction columns: {sorted(reserved)}")
    if not callable(getattr(model, "predict_dataframe", None)) or not callable(getattr(model, "kind_support", None)):
        raise TypeError("model must expose predict_dataframe and kind_support")
    support = {str(getattr(kind, "value", kind)): spec for kind, spec in model.kind_support().items()}
    if not support or any(spec.get("mhc_dependence") not in {"none", "single_allele", "haplotype"}
                          for spec in support.values()):
        raise ValueError("Model must declare MHC dependence for every prediction kind")
    alleles = sorted({mhcgnomes.parse(str(allele)).to_string()
                      for allele in (getattr(model, "alleles", ()) or ())})
    if any(spec["mhc_dependence"] != "none" for spec in support.values()) and not alleles:
        raise ValueError("Allele-dependent models require configured alleles")
    for column in ("n_flank", "c_flank"):
        if column not in frame:
            frame[column] = None
        if not frame[column].map(lambda value: isinstance(value, str) or
                                 (pd.api.types.is_scalar(value) and pd.isna(value))).all():
            raise ValueError(f"{column} must contain strings or missing values")
    if "peptide_offset" not in frame:
        frame["peptide_offset"] = pd.array([None] * len(frame), dtype="Int64")
    offsets = frame.peptide_offset.dropna()
    if not offsets.map(lambda value: not isinstance(value, (bool, np.bool_))
                       and isinstance(value, (int, float, np.integer, np.floating))
                       and np.isfinite(value) and value >= 0 and value == int(value)).all():
        raise ValueError("peptide_offset must be a nonnegative integer or missing")
    identity_columns = ["prediction_id", "peptide", "peptide_offset"] + [
        column for column in ("sample_name", "candidate_sample") if column in frame]
    if frame.duplicated(identity_columns).any():
        raise ValueError("Occurrence identity must be unique by prediction_id, peptide, peptide_offset and sample")
    if "peptide_length" in frame and not frame.peptide_length.eq(frame.peptide.str.len()).all():
        raise ValueError("peptide_length disagrees with peptide")
    flank_model = bool(getattr(model, "uses_flanking_sequences", False))
    has_flanks = frame.n_flank.map(lambda value: isinstance(value, str)) & frame.c_flank.map(lambda value: isinstance(value, str))
    if use_flanks and flank_model and not has_flanks.all():
        raise ValueError("Flank-dependent prediction requires both flanks; set use_flanks=False explicitly")
    if predict_wt and "wt_peptide" in frame:
        comparators = frame.loc[frame.wt_peptide.notna()]
        if not comparators.wt_peptide.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError("wt_peptide must be a nonempty string or missing")
        if use_flanks and flank_model and not comparators.empty:
            if any(column not in comparators or not comparators[column].map(lambda value: isinstance(value, str)).all()
                   for column in ("wt_n_flank", "wt_c_flank")):
                raise ValueError("Flank-dependent comparator prediction requires both flanks for the comparator")
    genotypes = frame.get("allele_set", pd.Series(None, index=frame.index, dtype=object)).map(
        lambda value: "" if pd.isna(value) else format_allele_set(
            mhcgnomes.parse(name).to_string() for name in split_allele_set(value)))
    if any(spec["mhc_dependence"] == "haplotype" for spec in support.values()):
        if any(genotype and genotype != format_allele_set(alleles) for genotype in genotypes):
            raise ValueError("Haplotype occurrence allele_set must match the configured model")
    if on_miss is not None:
        def report_miss(report):
            row = frame.iloc[int(report["source_sequence_name"])]
            identity = {column: None if pd.isna(row[column]) else row[column] for column in identity_columns}
            on_miss(dict(report, **identity, source_sequence_name=row.prediction_id))

        reported = predict_with_cache_miss_report(
            lambda inputs: predict_peptide_occurrences(
                frame.iloc[[int(index) for index in inputs]], model,
                use_flanks=use_flanks, predict_wt=predict_wt),
            {str(index): peptide for index, peptide in frame.peptide.items()}, on_miss=report_miss,
            context=dict(stage="occurrence", prediction_method_name=str(
                getattr(model, "prediction_method_name", getattr(model, "predictor_name", type(model).__name__))),
                predictor_version=str(getattr(model, "predictor_version", "")),
                configured_alleles=alleles, use_flanks=use_flanks))
        return reported if not reported.empty else predict_peptide_occurrences(frame.iloc[:0], model)

    partitions = {}
    for index, row in frame.iterrows():
        flanks = (row.n_flank, row.c_flank) if use_flanks and has_flanks.iloc[index] else (None, None)
        samples = tuple(None if pd.isna(row.get(column)) else row[column]
                        for column in ("sample_name", "candidate_sample"))
        partitions.setdefault((*samples, *flanks, genotypes.iloc[index]), []).append(index)
    expanded = []
    for (*_, n_flank, c_flank, genotype), indices in partitions.items():
        inputs = frame.iloc[indices]
        peptides = inputs.peptide.drop_duplicates().tolist()
        kwargs = {} if n_flank is None else dict(n_flanks=[n_flank] * len(peptides), c_flanks=[c_flank] * len(peptides))
        predicted = from_predictions(model.predict_dataframe(peptides, **kwargs))
        if predicted.empty:
            raise ValueError(f"Model returned no prediction for {peptides}")
        if not predicted.peptide.isin(peptides).all():
            raise ValueError("Model returned a different peptide than the requested occurrences")
        if not predicted.kind.isin(support).all():
            raise ValueError("Model returned an undeclared prediction kind")
        predicted["allele"] = predicted.allele.map(
            lambda value: mhcgnomes.parse(str(value)).to_string() if isinstance(value, str) and value.strip() else "")
        for peptide in peptides:
            for kind, spec in support.items():
                rows = predicted[predicted.peptide.eq(peptide) & predicted.kind.eq(kind)]
                dependence = spec["mhc_dependence"]
                expected = set(alleles) if dependence == "single_allele" else None
                actual = set(rows.allele)
                if rows.empty or (expected is not None and expected != actual):
                    raise ValueError(f"Model returned no complete {kind} prediction for {peptide}/{','.join(alleles)}")
                if dependence == "none" and actual != {""}:
                    raise ValueError(f"Allele-free {kind} output must not name an allele")
                if dependence == "haplotype" and not actual <= set(alleles) | {""}:
                    raise ValueError("Haplotype output names an allele outside the configured genotype")
                if rows.duplicated(["allele"]).any() or (dependence != "single_allele" and len(rows) != 1):
                    raise ValueError(f"Conflicting or ambiguous {kind} predictions for {peptide}")
        for column, context in (("n_flank", n_flank), ("c_flank", c_flank)):
            if column in predicted:
                reported = predicted[column].dropna()
                if context is None:
                    matches = reported.eq("")
                elif flank_model:
                    matches = reported.map(lambda value: isinstance(value, str) and
                                           (context.endswith(value) if column == "n_flank" else context.startswith(value)))
                else:
                    matches = pd.Series(True, index=reported.index)
                if not matches.all():
                    raise ValueError("Returned flank context does not match the requested occurrence")
        predicted["prediction_mhc_dependence"] = predicted.kind.map(lambda kind: support[kind]["mhc_dependence"])
        predicted["prediction_flanks_supplied"] = bool(kwargs)
        if "allele_set" in predicted:
            reported = predicted.loc[predicted.prediction_mhc_dependence.eq("haplotype"), "allele_set"]
            if any(value and value != format_allele_set(alleles)
                   for value in reported.map(lambda value: "" if pd.isna(value) else format_allele_set(
                       mhcgnomes.parse(name).to_string() for name in split_allele_set(value)))):
                raise ValueError("Returned haplotype allele_set differs from the configured model")
        predicted["allele_set"] = np.where(predicted.prediction_mhc_dependence.eq("haplotype"),
                                           format_allele_set(alleles), genotype)
        # Input context is authoritative; prediction values remain model-owned.
        payload = predicted.drop(columns=[column for column in frame if column != "peptide"] +
                                 ["source_sequence_name", "peptide_offset", "sample_name"], errors="ignore")
        if "allele_set" in frame:
            payload["allele_set"] = predicted.allele_set
        context = inputs.drop(columns=["allele_set"], errors="ignore")
        expanded.append(context.merge(payload, on="peptide", how="left", validate="many_to_many"))
    output = _concat_frames(expanded)
    positions = pd.MultiIndex.from_frame(frame[identity_columns]).get_indexer(
        pd.MultiIndex.from_frame(output[identity_columns]))
    order = np.argsort(positions, kind="stable")
    output = output.iloc[order].reset_index(drop=True)
    if "source_sequence_name" not in output:
        output["source_sequence_name"] = output.prediction_id
    if "sample_name" not in output:
        output["sample_name"] = ""
    if predict_wt:
        output = _attach_comparators(output, frame, positions[order], model, use_flanks)
    return output


def _attach_comparators(output, occurrences, positions, model, use_flanks):
    """Plumbing for the explicit comparator pass of predict_peptide_occurrences."""
    present = occurrences.get("wt_peptide", pd.Series(None, index=occurrences.index, dtype=object)).map(
        lambda value: isinstance(value, str) and bool(value))
    fields = ("value", "score", "affinity", "percentile_rank", "prediction_method_name", "predictor_version")
    if not present.any():
        for field in fields:
            output["wt_" + field] = np.nan
        return output
    comparators = occurrences.loc[present].copy()
    # Comparator sequence/offset need not match the primary occurrence. Use
    # its original row position for this private pass, retaining public IDs.
    comparators["prediction_id"] = comparators.index.astype(str)
    comparators["peptide"] = comparators.wt_peptide
    comparators = comparators.drop(columns=["peptide_length"], errors="ignore")
    for column in ("n_flank", "c_flank", "peptide_offset"):
        comparators[column] = comparators.get("wt_" + column)
    comparators = comparators.drop(columns=[column for column in comparators if column.startswith("wt_")])
    scores = predict_peptide_occurrences(comparators, model, use_flanks=use_flanks)
    # A haplotype's deconvolved presenter may change for the comparator. Its
    # score still describes the same configured genotype, not that one allele.
    def key(identity, row):
        return (identity, row.kind, prediction_mhc_scope(
            row.allele, dependence=row.prediction_mhc_dependence, allele_set=row.allele_set))
    by_scope = {key(row.prediction_id, row): row for row in scores.itertuples(index=False)}
    matches = [by_scope.get(key(str(position), row))
               for position, row in zip(positions, output.itertuples(index=False))]
    for field in fields:
        output["wt_" + field] = [getattr(row, field, np.nan) for row in matches]
    return output
