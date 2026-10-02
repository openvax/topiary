"""Coverage and comparisons over retained policy decisions, without inference."""

from copy import deepcopy

import numpy as np
import pandas as pd

from .policy_evaluation import PolicyEvaluation, _json_value, _key
from .ranking import is_stated, mhc_dependence, prediction_field_values, prediction_mhc_scope
from .result import TopiaryResult


_STATUSES = ("assessed", "missing", "failed", "not_applicable", "not_evaluated")
_MODEL_KEYS = ["kind", "prediction_method_name", "predictor_version", "field"]
_COLUMNS = ["occurrence_id", "level", "criterion", "role", *_MODEL_KEYS, "mhc_dependence", "requested_allele_set", "status", "reason",
            "native_status", "value", "detail", "raw_score", "score", "eligible",
            "filter_value", "filter_retained", "decision_reason", "evidence_rows", "observed_evidence_rows"]

_COMPARISON_FIELDS = ["present", "assessable", "score_status", "score_reason", "unknown_criteria", "criteria",
                      "occurrence_id", "raw_score", "score", "eligible", "reason", "filter_value",
                      "filter_retained", "evidence_rows", "supporting_rows"]
_COMPARISON_COLUMNS = [*[name + "_" + key for name in ("left", "right") for key in _COMPARISON_FIELDS],
                       "assessment_set", "raw_score_delta"]


def _metadata(evaluation):
    if not isinstance(evaluation, PolicyEvaluation):
        raise TypeError("evaluation must be a PolicyEvaluation")
    return evaluation.evidence.extra["policy_evaluation"]


def _report_context(evaluation):
    # Retain definitions/provenance once, without duplicating every occurrence
    # already represented in the report's records.
    context = {key: value for key, value in _metadata(evaluation).items() if key != "occurrences"}
    context["evidence_sources"] = evaluation.evidence.sources
    context["evidence_extra"] = {key: value for key, value in evaluation.evidence.extra.items()
                                 if key != "policy_evaluation"}
    return deepcopy(context)


def _identity_keys(evaluation, keys):
    metadata = _metadata(evaluation)
    if keys is None:
        groups = [part["context"]["group_keys"] for part in metadata["partitions"]]
        if any(group != groups[0] for group in groups[1:]):
            raise ValueError("Different source groupings require explicit coverage keys")
        keys = list(groups[0]) if groups else []
        if any(part["source_label"] is not None for part in metadata["partitions"]) and "source_label" not in keys:
            keys.insert(0, "source_label")
    keys = list(keys)
    if not keys or len(set(keys)) != len(keys) or set(keys) & set(_COLUMNS):
        raise ValueError("keys must be unique identity columns, distinct from report columns")
    _index(evaluation.occurrences, keys)
    return keys


def _index(frame, keys):
    if not frame.columns.is_unique:
        raise ValueError("Identity frames must have distinct column names")
    missing = set(keys) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing identity columns: {sorted(missing)}")
    rows = frame.to_dict("records")
    identities = [_key([row[key] for key in keys]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("keys must uniquely identify occurrences; include source and context identity")
    return dict(zip(identities, rows))


def _universe(evaluation, keys, universe):
    observed = _index(evaluation.occurrences, keys)
    expected = observed if universe is None else _index(universe, keys)
    if set(observed) - set(expected):
        raise ValueError("universe must include every evaluated occurrence")
    return expected, observed


def _score_status(row):
    if row is None:
        return "not_evaluated", "no_evaluation"
    if not row["filter_retained"]:
        return "not_evaluated", "filtered_before_scoring"
    if pd.isna(row["raw_score"]) or not np.isfinite(row["raw_score"]):
        return "missing", "missing_score"
    return "assessed", "observed_zero" if row["raw_score"] == 0 else "observed_value"


def _base(identity, row):
    return dict(identity, occurrence_id=None if row is None else row["occurrence_id"],
                raw_score=None if row is None else row["raw_score"],
                score=None if row is None else row["score"],
                eligible=False if row is None else row["eligible"],
                filter_value=None if row is None else row["filter_value"],
                filter_retained=None if row is None else row["filter_retained"],
                decision_reason="no_evaluation" if row is None else row["reason"],
                evidence_rows=[], observed_evidence_rows=[])


def _prediction_record(evaluation, row, request):
    # Model availability is independent of which DSL expression read the model.
    frame = evaluation.evidence.long_df
    positions = [] if row is None else row["supporting_rows"]
    selected = frame.iloc[positions]
    for key in _MODEL_KEYS[:-1]:
        wanted = _key(request[key])
        if key not in selected:
            selected = selected.iloc[:0]
        else:
            selected = selected.loc[selected[key].map(_key).eq(wanted)]
    support = {}
    for partition in _metadata(evaluation)["partitions"]:
        if partition["source_label"] is None or partition["source_label"] == (row or request).get("source_label"):
            support = partition["context"]["kind_support"]
            break
    # A request that returned no rows still names its model. Give the public
    # resolver that declaration so unrelated models cannot decide its scope.
    scope_rows = selected if len(selected) else pd.DataFrame([request])
    dependence = mhc_dependence(request["kind"], kind_support=support, rows=scope_rows)
    scope = prediction_mhc_scope(request.get("allele"), dependence=dependence,
                                 allele_set=request.get("allele_set"))
    if scope is None:
        selected = selected.iloc[:0]
    else:
        mask = []
        for item in selected.to_dict("records"):
            matches = prediction_mhc_scope(item.get("allele"), dependence=dependence,
                                           allele_set=item.get("allele_set")) == scope
            # Peptide-level rows explicitly credited to one allele retain that
            # restriction, matching peptide_view; allele-free rows can project.
            if dependence == "none" and is_stated(item.get("allele")):
                matches = matches and prediction_mhc_scope(item["allele"], dependence="single_allele") == (
                    prediction_mhc_scope(request.get("allele"), dependence="single_allele"))
            mask.append(matches)
        selected = selected.loc[np.asarray(mask, dtype=bool)]
    links = [int(position) for position in selected.index]
    status, reason, value, detail = "missing", "no_prediction_rows", None, None
    field = request["field"]
    observed_links = []
    if field in selected:
        numeric = pd.to_numeric(selected[field], errors="coerce")
        observed_links = [int(index) for index in selected.index[np.isfinite(numeric).fillna(False)]]
    if len(selected):
        reason = "missing_field" if field not in selected else "missing_evidence"
        if field in selected:
            try:
                values = prediction_field_values(selected.assign(_coverage_group=0), field,
                                                 group_keys=["_coverage_group"])
                value = values.iloc[0]
                if not pd.isna(value) and np.isfinite(value):
                    status, reason = "assessed", "observed_zero" if value == 0 else "observed_value"
            except ValueError as exc:
                reason, detail = "conflicting_evidence", str(exc)
    declared = request.get("status")
    if declared is not None and not pd.isna(declared):
        if declared not in _STATUSES[1:]:
            raise ValueError("Request diagnostics must declare a non-assessed coverage status")
        if status == "assessed":
            raise ValueError("Request diagnostic contradicts an observed prediction")
        if not isinstance(request.get("reason"), str) or not request["reason"]:
            raise ValueError("Request diagnostics require a nonempty reason")
        status, reason = declared, request["reason"]
        detail = request.get("detail", detail)
    return dict(level="prediction", **{key: request[key] for key in _MODEL_KEYS},
                status=status, native_status=status, reason=reason, detail=detail,
                mhc_dependence=dependence, requested_allele_set=(",".join(scope[1:]) if scope and dependence == "haplotype" else None),
                value=value, evidence_rows=links, observed_evidence_rows=observed_links)


def policy_coverage(evaluation, *, keys=None, universe=None, prediction_requests=None):
    """Retain coverage independently of policy acceptance and score filling.

    Parameters
    ----------
    evaluation : PolicyEvaluation
        Complete evaluation, including rejected occurrences and raw evidence.
    keys : sequence of str, optional
        Unique occurrence identity. Defaults to retained grouping, plus source
        label for partitioned evaluations. Incompatible groupings require keys.
    universe : pandas.DataFrame, optional
        Expected identities, including all evaluated occurrences. Absent input
        groups are explicitly not evaluated. None uses evaluated groups only;
        no Cartesian product of peptides, alleles or models is guessed.
    prediction_requests : pandas.DataFrame, optional
        Exact expected model assessments: identity keys, ``kind``,
        ``prediction_method_name``, ``predictor_version`` and numeric ``field``.
        Model names/kinds use retained canonical spellings; a null version means
        an unstated version, never a wildcard. Include ``allele_set`` for a
        haplotype request. Optional ``status``, ``reason`` and ``detail`` carry
        explicit missing/failed/not_applicable/not_evaluated diagnostics. A
        diagnostic contradicting a measured value is rejected. Requests report
        model availability, not criterion-to-model lineage. None omits this
        level: returned rows alone cannot establish a requested denominator.

    Returns
    -------
    TopiaryResult
        One score and each named criterion per expected occurrence, plus one
        record per explicit prediction request. ``status`` distinguishes
        assessed/missing/failed/not_applicable/not_evaluated; native decisions,
        reasons, raw and effective scores, and eligibility stay separate.
        Prediction links are positions in the retained evidence (shared across
        allele-free projections), not new observations. Empty input produces a
        typed table with no records. Metadata retains policy/context/provenance
        and the exact reporting inputs for CSV/TSV persistence and replay.
    """
    keys = _identity_keys(evaluation, keys)
    expected, observed = _universe(evaluation, keys, universe)
    records = []
    for identity, item in expected.items():
        row = observed.get(identity)
        base = _base({key: item[key] for key in keys}, row)
        status, reason = _score_status(row)
        records.append(dict(base, level="score", status=status, native_status=status,
                            reason=reason, value=None if row is None else row["raw_score"]))
        criteria = (row.get("criteria", []) if row is not None else
                    [dict(criterion=criterion.name, status="not_evaluated", reason="no_evaluation")
                     for criterion in evaluation.policy.criteria])
        for criterion in criteria:
            native = criterion["status"]
            status = "assessed" if native in {"pass", "fail", "value"} else "missing" if native == "unknown" else native
            records.append(dict(base, **dict(criterion, level="criterion", native_status=native, status=status)))
    requests = [] if prediction_requests is None else prediction_requests.to_dict("records")
    seen = set()
    for request in requests:
        missing = set([*keys, *_MODEL_KEYS]) - set(request)
        if missing:
            raise ValueError(f"Missing prediction request columns: {sorted(missing)}")
        for key in ("kind", "prediction_method_name", "field"):
            if not isinstance(request[key], str) or not request[key].strip():
                raise ValueError(f"Prediction request {key} must be a nonempty string")
        identity = _key([request[key] for key in keys])
        if identity not in expected:
            raise ValueError("Prediction requests must belong to the explicit occurrence universe")
        row = observed.get(identity)
        prediction = _prediction_record(evaluation, row, request)
        request_id = _key([identity, *[request[key] for key in _MODEL_KEYS],
                           prediction["mhc_dependence"], prediction["requested_allele_set"]])
        if request_id in seen:
            raise ValueError("Duplicate prediction request")
        seen.add(request_id)
        base = _base({key: request[key] for key in keys}, row)
        records.append(dict(base, **prediction))
    columns = list(dict.fromkeys([*keys, *_COLUMNS, *(key for row in records for key in row)]))
    frame = pd.DataFrame(records, columns=columns)
    return TopiaryResult(frame, form="long", extra={"policy_coverage": _json_value(dict(
        schema_version=1, keys=keys, evaluation=_report_context(evaluation),
        universe=None if universe is None else universe.to_dict("records"),
        prediction_requests=None if prediction_requests is None else requests,
    ))})


def summarize_policy_coverage(coverage, *, by=None):
    """Count assessment coverage with explicit, independent denominators.

    Parameters
    ----------
    coverage : TopiaryResult
        Output of :func:`policy_coverage`, including after typed CSV/TSV reload.
    by : sequence of str, optional
        Grouping columns. Defaults to level, criterion, kind, method, version
        and field; add ``reason`` or occurrence keys to inspect missingness.
        An empty list gives one overall group (across assessment types).

    Returns
    -------
    pandas.DataFrame
        ``n_assessments`` and counts for every status; ``assessed_fraction`` uses
        all requested assessments, including inapplicable/unevaluated ones.
        ``n_occurrences`` counts distinct report identities; ``n_candidates``
        counts distinct peptide/canonical-allele/genotype identities when available (or
        occurrence identities otherwise), across repeated source discoveries.
        ``n_prediction_rows`` counts the union of linked raw row positions,
        never their sum across projections; ``n_observed_prediction_rows`` counts
        the subset with finite numeric fields (even when rows conflict). These
        raw-row counts cannot include requests that returned no rows. Equal repeated raw rows remain raw
        rows, not independent biological evidence. Empty input has no groups.
    """
    metadata = coverage.extra["policy_coverage"]
    keys = metadata["keys"]
    frame = coverage.df
    by = ["level", "criterion", *_MODEL_KEYS] if by is None else list(by)
    if len(set(by)) != len(by) or set(by) - set(frame.columns):
        raise ValueError("by must name unique coverage columns")
    count_columns = ["n_assessments", "n_occurrences", "n_candidates", "n_prediction_rows", "n_observed_prediction_rows",
                     *["n_" + status for status in _STATUSES], "assessed_fraction"]
    records = []
    groups = frame.groupby(by, dropna=False, sort=False) if by else [((), frame)]
    for identity, group in groups:
        if group.empty:
            continue
        identity = identity if isinstance(identity, tuple) else (identity,)
        counts = group.status.value_counts()
        candidate_keys = [key for key in ("sample_name", "candidate_sample", "peptide", "allele", "allele_set") if key in keys]
        # Without peptide identity a subset of dimensions cannot define a candidate.
        if "peptide" not in candidate_keys:
            candidate_keys = keys
        candidates = group[candidate_keys].drop_duplicates().to_dict("records")
        for candidate in candidates:
            if "allele" in candidate:
                dependence = "haplotype" if is_stated(candidate.get("allele_set")) else "single_allele"
                candidate["allele"] = prediction_mhc_scope(candidate["allele"], dependence=dependence,
                                                            allele_set=candidate.pop("allele_set", None))
        records.append(dict(zip(by, identity), n_assessments=len(group),
                            n_occurrences=len(group[keys].drop_duplicates()),
                            n_candidates=len({_key(candidate) for candidate in candidates}),
                            n_prediction_rows=len({index for links in group.evidence_rows for index in links}),
                            n_observed_prediction_rows=len({index for links in group.observed_evidence_rows for index in links}),
                            **{"n_" + status: int(counts.get(status, 0)) for status in _STATUSES},
                            assessed_fraction=float(counts.get("assessed", 0) / len(group))))
    return pd.DataFrame(records, columns=[*by, *count_columns])


def compare_policy_evaluations(left, right, *, keys=None, universe=None):
    """Align full membership and compare raw scores only where both are assessable.

    Parameters
    ----------
    left, right : PolicyEvaluation
        Retained evaluations. They may have different policies and input rows.
    keys : sequence of str, optional
        Unique shared occurrence identity. Defaults to identical saved grouping;
        ambiguous or incompatible identities raise rather than joining loosely.
    universe : pandas.DataFrame, optional
        Expected identities including both evaluations. None uses their union.

    Returns
    -------
    TopiaryResult
        One record per identity, preserving both memberships, raw/effective
        scores, eligibility and native criterion decisions. ``assessment_set``
        is both/left_only/right_only/neither. Assessability requires an observed
        finite raw score and no unknown referenced criterion; not-applicable
        and unused criteria do not disqualify it. Prefiltered groups are not
        score-assessable. ``raw_score_delta`` (right minus left) exists only for
        both, with no normalization across policy scales. Metadata preserves
        both definitions, digests, runtime contexts and execution provenance.
        Empty inputs produce an empty table; absent groups remain explicit.
    """
    left_keys = _identity_keys(left, keys)
    right_keys = _identity_keys(right, keys)
    if left_keys != right_keys:
        raise ValueError("Comparison groupings differ; supply explicit keys")
    keys = left_keys
    if set(keys) & set(_COMPARISON_COLUMNS):
        raise ValueError("keys collide with reserved comparison columns")
    lhs, rhs = _index(left.occurrences, keys), _index(right.occurrences, keys)
    expected = {**lhs, **rhs} if universe is None else _index(universe, keys)
    if (set(lhs) | set(rhs)) - set(expected):
        raise ValueError("universe must include every evaluated occurrence")
    records = []
    for identity, item in expected.items():
        record = {key: item[key] for key in keys}
        for name, rows in (("left", lhs), ("right", rhs)):
            row = rows.get(identity)
            status, reason = _score_status(row)
            criteria = [] if row is None else row.get("criteria", [])
            unknown = [criterion["criterion"] for criterion in criteria if criterion["status"] == "unknown"]
            record.update({name + "_" + key: value for key, value in dict(
                present=row is not None, assessable=status == "assessed" and not unknown,
                score_status=status, score_reason=reason, unknown_criteria=unknown, criteria=_json_value(criteria),
                **{key: None if row is None else row[key] for key in
                   ("occurrence_id", "raw_score", "score", "eligible", "reason", "filter_value",
                    "filter_retained", "evidence_rows", "supporting_rows")},
            ).items()})
        a, b = record["left_assessable"], record["right_assessable"]
        record["assessment_set"] = "both" if a and b else "left_only" if a else "right_only" if b else "neither"
        record["raw_score_delta"] = record["right_raw_score"] - record["left_raw_score"] if a and b else None
        records.append(record)
    columns = [*keys, *_COMPARISON_COLUMNS]
    return TopiaryResult(pd.DataFrame(records, columns=columns), form="long", extra={
        "policy_comparison": _json_value(dict(schema_version=1, keys=keys,
            left=_report_context(left), right=_report_context(right),
            universe=None if universe is None else universe.to_dict("records"))),
    })
