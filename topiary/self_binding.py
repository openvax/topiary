"""Binding-conditioned summaries of bounded, all-origin self evidence."""

import math
from numbers import Real

import pandas as pd

from .ranking import is_stated
from .result import TopiaryResult
from .self_evidence import match_self_peptides, _digest, _evidence_records


def binding_aware_self_evidence(reference, peptides, *, alleles, predictions,
                                prediction_method_name, predictor_version,
                                kind="pMHC_affinity", binding_metric="percentile_rank",
                                lower_is_better=None, measurement_context=None,
                                value_unit=None, max_mismatches=0, max_edits=None,
                                excluded_gene_ids=(), observations=None):
    """Summarize nearby self predictions without inferring tissue or TCR effects.

    Parameters
    ----------
    reference, peptides, alleles, max_mismatches, max_edits, excluded_gene_ids, observations
        Arguments to :func:`match_self_peptides`. Every query occurrence and
        reference origin survives. Use an unfiltered reference and explicit
        exclusions when shared CTA/non-CTA origins must remain distinguishable.
    predictions : DataFrame or iterable of mappings
        Supplied predictions for both queries and nearby self peptides, using
        the schema of ``match_self_peptides``. No model is run. Conflicting or
        missing selected measurements remain unknown, never a negative result.
    prediction_method_name, predictor_version : str
        Required, stated, exact model identity. Different models and versions
        are never pooled, even for percentile ranks.
    kind : str
        ``pMHC_affinity`` (default) or ``pMHC_presentation``. Only measurements
        explicitly scoped to one allele are comparable; a haplotype's best
        presenter label does not establish that allele's binding measurement.
    binding_metric : str
        ``percentile_rank`` (default), ``value`` or ``score``. Rank is lower
        better and must be finite in [0, 100]. Other metrics need an explicit
        ``lower_is_better`` boolean; raw values additionally need ``value_unit``.
    lower_is_better : bool or None
        Direction for the selected metric; percentile rank defaults to True
        and cannot be inverted. This is a caller policy, not calibration.
    measurement_context : JSON-compatible object or None
        Exact selected measurement context. None selects only unstated
        contexts, not every context. Distinct flank contexts within the same
        selected measurement are ambiguous, even if their numeric values agree.
    value_unit : str or None
        Required exact unit selector for ``binding_metric='value'``.

    Returns
    -------
    TopiaryResult
        One row per query occurrence and allele, with the selected query
        measurement and structured ``self_nearest_candidates`` containing all
        matches, origins and supplied evidence. ``self_mimic_candidates`` holds
        every nearest measured in-scope match whose selected metric is at least as strong
        as the query. ``self_strongest_nearby_candidates`` holds every strongest
        in-scope match within the requested radius, independently of query score.
        Tied scalar fields are set only when all tied candidates agree.

        Each axis states ``matched``, ``no_match_within_scope`` or ``unassessed``
        and a separate completeness boolean. A known match can coexist with
        incomplete coverage; it is then nearest/strongest only among available
        selected measurements. Absence is bounded by this reference, search and
        supplied prediction coverage; it never means no biological risk.
        No position-specific TCR/anchor weights or recognition model are used.
        Empty input returns the same empty schema. Typed CSV/TSV preserves the
        structured candidates and the explicit comparison policy.
    """
    for label, value in (("prediction_method_name", prediction_method_name),
                         ("predictor_version", predictor_version)):
        if not isinstance(value, str) or not is_stated(value):
            raise ValueError(f"{label} must be a stated exact model identifier")
    if kind not in {"pMHC_affinity", "pMHC_presentation"}:
        raise ValueError("Binding evidence requires pMHC_affinity or pMHC_presentation")
    if binding_metric not in {"percentile_rank", "value", "score"}:
        raise ValueError("binding_metric must be percentile_rank, value or score")
    if binding_metric == "percentile_rank":
        if lower_is_better is not None and lower_is_better is not True:
            raise ValueError("Percentile rank is lower-is-better")
        lower_is_better = True
    elif type(lower_is_better) is not bool:
        raise ValueError("A value/score metric requires explicit lower_is_better")
    if binding_metric == "value" and (not isinstance(value_unit, str) or not is_stated(value_unit)):
        raise ValueError("A value metric requires an explicit value_unit")
    context_key = _digest(measurement_context)
    records = _evidence_records(predictions, observation=False)
    matches = match_self_peptides(
        reference, peptides, alleles=alleles, predictions=records,
        max_mismatches=max_mismatches, max_edits=max_edits,
        excluded_gene_ids=excluded_gene_ids, observations=observations)
    selected = {}
    for record in records:
        if (record["prediction_method_name"] == prediction_method_name
                and record["predictor_version"] == predictor_version
                and record["kind"] == kind
                and record["prediction_mhc_dependence"] == "single_allele"
                and _digest(record.get("measurement_context")) == context_key
                and (value_unit is None or record.get("value_unit") == value_unit)):
            selected.setdefault((record["peptide"], record["allele"]), []).append(record)

    rows = []
    for query_index, group in matches.df.groupby("query_index", sort=False):
        query = group.iloc[0]
        measurement, query_status = _measurement(
            selected.get((query.peptide, query.allele), []), binding_metric)
        candidates = []
        for row in group.to_dict("records"):
            if row["self_peptide"] is None:
                continue
            evidence, status = _measurement(
                selected.get((row["self_peptide"], query.allele), []), binding_metric)
            # Structured facts are copied, including every origin and observation.
            candidate = {key: value for key, value in row.items() if key.startswith("self_")}
            candidate.update(binding_status=status, binding_measurement=evidence)
            candidates.append(candidate)
        in_scope = [c for c in candidates if c["self_in_scope"] is True]
        measured = [c for c in in_scope if c["binding_status"] == "reported"]
        complete = (bool(query.self_search_complete) and query.allele is not None
                    and all(c["self_in_scope"] is not None for c in candidates)
                    and len(measured) == len(in_scope))
        mimic = []
        if measurement is not None:
            target = measurement[binding_metric]
            mimic = [c for c in measured if (
                c["binding_measurement"][binding_metric] <= target if lower_is_better else
                c["binding_measurement"][binding_metric] >= target)]
        if mimic:
            distance = min(c["self_edit_distance"] for c in mimic)
            mimic = [c for c in mimic if c["self_edit_distance"] == distance]
        strongest = []
        if measured:
            best = (min if lower_is_better else max)(c["binding_measurement"][binding_metric] for c in measured)
            strongest = [c for c in measured if c["binding_measurement"][binding_metric] == best]
        result = dict(query_index=int(query_index), peptide=query.peptide,
                      peptide_length=len(query.peptide), allele=query.allele, kind=kind,
                      prediction_method_name=prediction_method_name, predictor_version=predictor_version,
                      prediction_mhc_dependence="single_allele", measurement_context=measurement_context,
                      self_reference_version=query.self_reference_version,
                      self_search_complete=bool(query.self_search_complete),
                      self_search_reason=query.self_search_reason,
                      self_query_prediction_status=query_status,
                      self_prediction_complete=complete,
                      self_nearest_candidates=candidates)
        for field in ("value", "score", "percentile_rank", "value_unit"):
            result[field] = measurement.get(field) if measurement else None
        for prefix, choices, covered in (
                ("self_mimic", mimic, complete and measurement is not None),
                ("self_strongest_nearby", strongest, complete)):
            result[prefix + "_candidates"] = choices
            result[prefix + "_complete"] = covered
            result[prefix + "_status"] = ("matched" if choices else "no_match_within_scope"
                                            if covered else "unassessed")
            for field, key in (("peptide", "self_peptide"), ("edit_distance", "self_edit_distance")):
                result[prefix + "_" + field] = _agreement([c[key] for c in choices])
            for field in ("value", "score", "percentile_rank", "value_unit"):
                result[prefix + "_" + field] = _agreement([
                    c["binding_measurement"].get(field) for c in choices])
        rows.append(result)
    columns = ["query_index", "peptide", "peptide_length", "allele", "kind", "prediction_method_name",
               "predictor_version", "prediction_mhc_dependence", "measurement_context",
               "self_reference_version", "self_search_complete", "self_search_reason",
               "self_query_prediction_status", "self_prediction_complete", "self_nearest_candidates",
               "value", "score", "percentile_rank", "value_unit"]
    columns += [prefix + "_" + suffix for prefix in ("self_mimic", "self_strongest_nearby")
                for suffix in ("candidates", "complete", "status", "peptide", "edit_distance",
                               "value", "score", "percentile_rank", "value_unit")]
    return TopiaryResult(pd.DataFrame(rows, columns=columns), extra={
        **matches.extra, "self_binding": dict(
            prediction_method_name=prediction_method_name, predictor_version=predictor_version,
            kind=kind, binding_metric=binding_metric, lower_is_better=lower_is_better,
            measurement_context=measurement_context, value_unit=value_unit)})


def _measurement(records, metric):
    if not records:
        return None, "not_reported"
    # Repeated source observations may repeat one prediction. Distinct flank
    # contexts and distinct measurement values cannot be collapsed as coverage.
    fields = ("value", "score", "percentile_rank", "value_unit", "n_flank", "c_flank")
    identities = {_digest({field: record.get(field) for field in fields}) for record in records}
    if len(identities) != 1:
        return None, "ambiguous"
    record = records[0]
    value = record.get(metric)
    if (not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value)
            or (metric == "percentile_rank" and not 0 <= value <= 100)):
        return None, "invalid_metric"
    return dict(record), "reported"


def _agreement(values):
    return values[0] if values and all(value == values[0] for value in values) else None
