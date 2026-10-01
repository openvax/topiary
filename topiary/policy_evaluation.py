"""Occurrence-level policy evaluation with retained evidence and replay context."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .ranking import EvalContext, as_dsl_node, is_stated
from .ranking.apply import evaluate_filter
from .ranking.nodes import _PeptideAlleleLookup, _peptide_keys
from .result import TopiaryResult
from .selection_policy import SelectionPolicy
from .serialization import normalize_python_types
from .criterion_evaluation import evaluate_selection_criteria


_DECISION_COLUMNS = ["occurrence_id", "evidence_rows", "supporting_rows", "filter_value",
                     "filter_retained", "raw_score", "score", "eligible", "reason"]


def _json_value(value):
    value = normalize_python_types(value)
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return None if value is None or pd.isna(value) else value


def _key(values):
    return json.dumps(_json_value(values), sort_keys=True, separators=(",", ":"), allow_nan=False)


def describe_evaluation_context(context):
    """Materialize a DSL runtime context for portable occurrence replay.

    Parameters
    ----------
    context : EvalContext
        Context on the complete source frame. Genotype callbacks are called
        once per peptide identity, including peptides later excluded.

    Returns
    -------
    dict
        JSON-compatible grouping, model defaults, kind support, and allele
        declarations. No Python callable is retained. Empty frames yield empty
        allele declarations. None and an explicitly empty declaration differ.
        Model defaults remain ambiguity-resolution choices, not strict model
        requirements; the retained evidence records the actual measurements.
    """
    if not isinstance(context, EvalContext):
        raise TypeError("context must be an EvalContext")
    alleles = context.alleles
    declaration = None
    if alleles is not None:
        keys = _peptide_keys(context.group_keys)
        if callable(alleles) or isinstance(alleles, Mapping):
            lookup = _PeptideAlleleLookup(alleles, keys)
            rows = context.key_frame[keys].drop_duplicates().to_dict("records")
            declaration = [dict(keys=row, alleles=list(lookup.for_peptide(row))) for row in rows]
            lookup.check_all_used()
        else:
            # A flat declaration is also valid for an allele-only grouping.
            declaration = list(alleles)
    return _json_value(dict(
        group_keys=list(context.group_keys), default_methods=dict(context.default_methods),
        default_versions=[dict(kind=k, method=m, version=v)
                          for (k, m), v in sorted(context.default_versions.items())],
        kind_support=deepcopy(context.kind_support), alleles=declaration,
    ))


def _context_options(record):
    alleles = record["alleles"]
    if alleles and isinstance(alleles[0], dict):
        lookup = {_key(item["keys"]): item["alleles"] for item in alleles}
        alleles = lambda keys: lookup.get(_key(keys), ())
    elif alleles == []:
        alleles = lambda keys: ()
    return dict(
        group_keys=record["group_keys"], alleles=alleles,
        default_methods=record["default_methods"], kind_support=record["kind_support"],
        default_versions={(item["kind"], item["method"]): item["version"]
                          for item in record["default_versions"]},
    )


@dataclass
class PolicyEvaluation:
    """Full source evidence and occurrence decisions from a selection policy.

    ``evidence`` is a TopiaryResult containing every original row, with policy,
    runtime context and decisions in ``extra['policy_evaluation']``. Save this
    object through Topiary's CSV/TSV writers to retain rejected alternatives.
    ``occurrences`` has one row per evaluation group, including projected allele
    groups with no direct measurement. Row links are positional in
    ``evidence.long_df``; supporting rows are references, never additional
    biological observations or additive read support.
    """

    evidence: TopiaryResult
    occurrences: pd.DataFrame

    @property
    def selected(self):
        """Eligible occurrence records, without collapsing candidates."""
        return self.occurrences.loc[self.occurrences.eligible].copy()

    @property
    def policy(self):
        """The immutable effective policy evaluated against this evidence."""
        return SelectionPolicy.from_dict(self.evidence.extra["policy_evaluation"]["definition"])

    @property
    def audit(self):
        """One criterion decision per occurrence, with all identity/row links."""
        records = []
        for row in self.occurrences.to_dict("records"):
            criteria = row.pop("criteria", [])
            records.extend(dict(row, **criterion) for criterion in criteria)
        return pd.DataFrame(records)


def _partition_decisions(frame, positions, context, policy, partition):
    options = _context_options(context)
    ctx = EvalContext(frame, **options)
    if set(ctx.group_keys) & set(_DECISION_COLUMNS):
        raise ValueError("group_keys collide with reserved policy decision columns")
    expanded = policy.expanded
    filter_criteria = None
    filter_node = policy.filter_by
    if policy.criteria:
        ctx = ctx.derive(preserve_unknown=True)
        filter_refs = [] if expanded["filter_by"] is None else expanded["filter_by"]["references"]
        filter_criteria = evaluate_selection_criteria(policy, ctx.derive(filter_context=True), references=filter_refs)
        filter_node = None if policy.filter_by is None else filter_criteria.expression(policy.filter_by)
    filtered = evaluate_filter(frame, filter_node, context=ctx, unknown=policy.unknown)
    retained = filtered.retained.to_numpy()
    kept = frame.iloc[np.flatnonzero(retained[ctx.row_group_codes()])]
    score_ctx = EvalContext(kept, **options, preserve_unknown=bool(policy.criteria))
    score_criteria = None
    score_node = as_dsl_node(expanded["score_by"]["expression"])
    if policy.criteria:
        score_refs = list(dict.fromkeys([
            *expanded["score_by"]["references"],
            *(name for term in expanded["ranking_by"] for name in term["references"]),
        ]))
        score_criteria = evaluate_selection_criteria(policy, score_ctx, references=score_refs)
        score_node = score_criteria.expression(policy.score_by)
    raw = score_node.eval(score_ctx).reindex(score_ctx.group_index)
    raw = pd.to_numeric(raw, errors="raise").reindex(ctx.group_index)
    effective = raw.fillna(policy.score_fill) if policy.score_fill is not None else raw.copy()
    # A removed occurrence must never be restored by score filling.
    effective = effective.where(filtered.retained)
    groups = ctx.group_index.to_frame(index=False)
    tie_scores = []
    for term, resolved in zip(policy.ranking_by, expanded["ranking_by"]):
        node = score_criteria.expression(term.expression) if score_criteria else as_dsl_node(resolved["expression"])
        tie_scores.append(node.eval(score_ctx).reindex(ctx.group_index))
    criterion_records = {}
    for measurements in (filter_criteria, score_criteria):
        if measurements is None:
            continue
        for record in measurements.decisions.to_dict("records"):
            identity = _key({key: record.pop(key) for key in ctx.group_keys})
            slot = criterion_records.setdefault(identity, {})
            if record["status"] != "not_evaluated" or record["criterion"] not in slot:
                slot[record["criterion"]] = record
    peptide_keys = _peptide_keys(ctx.group_keys)
    by_peptide = {}
    for row, position in zip(ctx.key_frame.to_dict("records"), positions):
        identity = _key([row[k] for k in peptide_keys])
        by_peptide.setdefault(identity, []).append(int(position))
    by_group = {}
    for code, position in zip(ctx.row_group_codes(), positions):
        by_group.setdefault(int(code), []).append(int(position))
    decisions = []
    for i, identity in enumerate(groups.to_dict("records")):
        rows = by_group.get(i, [])
        support = by_peptide.get(_key([identity[k] for k in peptide_keys]), [])
        score = effective.iloc[i]
        eligible = bool(retained[i] and (policy.min_score is None or (
            pd.notna(score) and score >= policy.min_score)))
        reason = ("filtered" if not retained[i] else
                  "below_min_score" if pd.notna(score) and not eligible else
                  "missing_score" if pd.isna(raw.iloc[i]) else "eligible")
        record = dict(identity)
        # Preserve source identity/annotations which are constant in a group.
        local = frame.loc[rows if rows else support]
        for column in ("candidate_id", "candidate_mhc_class", "candidate_sample", "source_observation_id", "source_label"):
            if not rows and column in {"candidate_id", "candidate_mhc_class"}:
                # Supporting rows may name several other alleles. The projected
                # candidate's identity is derived from its own allele below.
                continue
            if column in local and column not in record:
                values = local[column].drop_duplicates()
                if len(values) > 1:
                    raise ValueError(f"Occurrence group collapses distinct {column}; include it in group_keys")
                record[column] = values.iloc[0] if len(values) else None
        if not rows and is_stated(record.get("candidate_sample")):
            from .candidates import candidate_identifier
            from .io_pvacseq import derive_mhc_class
            record["candidate_id"] = candidate_identifier(record["candidate_sample"], identity.get("peptide"), identity.get("allele"))
            record["candidate_mhc_class"] = derive_mhc_class(pd.Series([identity.get("allele")])).iloc[0]
        record.update(
            occurrence_id=hashlib.sha256(_key([partition, identity]).encode()).hexdigest(),
            evidence_rows=rows, supporting_rows=support,
            filter_value=filtered.value.iloc[i], filter_retained=bool(retained[i]),
            raw_score=raw.iloc[i], score=score, eligible=eligible, reason=reason,
        )
        for j, values in enumerate(tie_scores):
            record[f"ranking_{j}"] = values.iloc[i]
        if policy.criteria:
            audit = criterion_records.get(_key(identity), {})
            record["criteria"] = []
            for criterion in policy.criteria:
                detail = dict(audit.get(criterion.name, dict(
                    criterion=criterion.name, role=criterion.role, value=None,
                    status="not_evaluated", reason="not_referenced", detail=None)))
                if detail["status"] == "not_evaluated" and not retained[i] and criterion.name in score_refs:
                    detail["reason"] = "filtered_before_scoring"
                record["criteria"].append(detail)
        decisions.append(record)
    return decisions


def evaluate_selection_policy(result, policy, *, group_keys=None, alleles=None,
                              kind_support=None, source_contexts=None, provenance=None):
    """Evaluate a saved policy before selecting candidate representatives.

    Parameters
    ----------
    result : TopiaryResult
        Complete long/wide evidence, including enriched source columns. A
        combined result is optional when explicit group_keys identify the input.
        Empty inputs with their identity schema return empty decisions.
    policy : SelectionPolicy
        Filter, scoring, fill and post-score minimum. No predictor is invoked.
    group_keys : sequence of str, optional
        Occurrence identity; for Vaxrank typically prediction_id, peptide,
        peptide_offset, allele. None uses EvalContext's inference.
    alleles : sequence, mapping or callable, optional
        Per-occurrence genotype, with exactly EvalContext's semantics. A
        callback is materialized before evaluation and is not saved as code.
    kind_support : mapping, optional
        Model MHC context. None uses the source result's recorded metadata.
    source_contexts : mapping, optional
        Source-label to context overrides (default_methods, default_versions,
        kind_support, alleles). If supplied, every input source must be named;
        each source is evaluated separately. Values replace global defaults,
        rather than implicitly merging model choices across sources.
    provenance : mapping, optional
        JSON-compatible derivation information, outside the policy digest.

    Returns
    -------
    PolicyEvaluation
        All evidence plus decisions including rejected, missing-score, zero
        and projected groups. Raw scores for pre-filtered groups are missing:
        they were not scored. Missing scores stay eligible when min_score is
        None, matching candidate ranking's explicit unranked state. Set a gate
        to exclude them; score_fill is applied before the gate. Raw evidence
        and the input object remain unchanged.
    """
    from . import __version__

    if not isinstance(result, TopiaryResult) or not isinstance(policy, SelectionPolicy):
        raise TypeError("Expected a TopiaryResult and SelectionPolicy")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping or None")
    derivation = json.loads(json.dumps(provenance, allow_nan=False))
    frame = result.long_df.copy().reset_index(drop=True)
    if not frame.columns.is_unique:
        raise ValueError("Evidence must have distinct column names")
    options = dict(group_keys=group_keys, alleles=alleles,
                   default_methods=policy.default_methods, default_versions=policy.default_versions,
                   kind_support=result._kind_support() if kind_support is None else kind_support)
    if source_contexts is None:
        partitions = [(None, np.arange(len(frame)), {})]
    else:
        if not isinstance(source_contexts, Mapping) or "source_label" not in frame:
            raise ValueError("source_contexts requires a mapping and a source_label column")
        if not frame.source_label.map(is_stated).all() or set(source_contexts) != set(frame.source_label):
            raise ValueError("source_contexts must name every source_label exactly once")
        partitions = [(label, np.flatnonzero(frame.source_label.eq(label)), settings)
                      for label, settings in source_contexts.items()]
    executions, decisions = [], []
    for label, positions, overrides in partitions:
        if not isinstance(overrides, Mapping) or set(overrides) - {
                "default_methods", "default_versions", "kind_support", "alleles"}:
            raise ValueError("Unsupported source context settings")
        part = frame.iloc[positions]
        context = describe_evaluation_context(EvalContext(part, **dict(options, **overrides)))
        executions.append(dict(source_label=label, context=context))
        decisions.extend(_partition_decisions(part, positions, context, policy, label))
    occurrences = pd.DataFrame(decisions)
    if not decisions:
        columns = list(dict.fromkeys([*(group_keys or EvalContext(frame).group_keys), *_DECISION_COLUMNS]))
        occurrences = pd.DataFrame(columns=columns).astype({"eligible": bool, "filter_retained": bool})
    extra = deepcopy(result.extra)
    extra["policy_evaluation"] = dict(
        schema_version=1, definition=policy.to_dict(), sha256=policy.sha256,
        provenance=derivation, topiary_version=__version__,
        input_topiary_version=result.topiary_version, partitions=executions,
        occurrences=_json_value(occurrences.to_dict("records")),
    )
    evidence = TopiaryResult(frame, metadata=result.metadata, form="long", extra=extra)
    return PolicyEvaluation(evidence, occurrences)


def replay_selection_policy(result):
    """Re-evaluate a saved occurrence policy using its retained evidence.

    Parameters
    ----------
    result : TopiaryResult
        Evidence exported from PolicyEvaluation.evidence and optionally read
        back through CSV/TSV. A representative-only export is insufficient.

    Returns
    -------
    PolicyEvaluation
        Fresh decisions using the saved definition and materialized contexts.
        Missing/unsupported metadata or a mismatched policy digest raises.
        Empty saved evaluations replay as empty. Predictors are never called.
    """
    record = result.extra.get("policy_evaluation", {})
    if record.get("schema_version") != 1:
        raise ValueError("Missing or unsupported policy_evaluation metadata")
    policy = SelectionPolicy.from_dict(record["definition"])
    if policy.sha256 != record["sha256"]:
        raise ValueError("Saved selection policy digest does not match its definition")
    partitions = record["partitions"]
    if len(partitions) == 1 and partitions[0]["source_label"] is None:
        options = _context_options(partitions[0]["context"])
        groups = options.pop("group_keys")
        # Source-local model choices are already part of this runtime context.
        from dataclasses import replace
        runtime_policy = replace(policy, default_methods=options.pop("default_methods"),
                                 default_versions=options.pop("default_versions"))
        replay = evaluate_selection_policy(result, runtime_policy, group_keys=groups,
                                          provenance=record["provenance"], **options)
        replay.evidence.extra["policy_evaluation"].update(definition=policy.to_dict(), sha256=policy.sha256)
        return replay
    groups = [item["context"]["group_keys"] for item in partitions]
    if groups and any(keys != groups[0] for keys in groups):
        raise ValueError("Saved source contexts must share group_keys")
    contexts = {}
    for item in partitions:
        context = _context_options(item["context"])
        context.pop("group_keys")
        contexts[item["source_label"]] = context
    return evaluate_selection_policy(result, policy, group_keys=groups[0] if groups else None,
                                     source_contexts=contexts, provenance=record["provenance"])


def select_policy_representatives(evaluation, *, candidate_keys=("candidate_id",), strata=None):
    """Select actual scored occurrences without adding repeated evidence.

    Parameters
    ----------
    evaluation : PolicyEvaluation
        Complete occurrence decisions. This object is not narrowed or changed.
    candidate_keys : sequence of str
        Columns identifying candidates in the occurrence table. For a direct
        consumer, pass its explicit candidate identity (e.g. peptide, allele).
        Rows with absent candidate identity are supporting evidence, not targets.
    strata : sequence of str, optional
        Independent partitions; None uses the saved policy's strata.

    Returns
    -------
    pandas.DataFrame
        One eligible occurrence per candidate and stratum, with alternative
        occurrence IDs and an explicit selection rationale. Best/worst/error
        duplicate handling and score direction come from the saved policy.
        Missing scores sort last. Empty input returns an empty selection.
        Evidence links identify real input rows; counts are never summed.
    """
    policy = evaluation.policy
    frame = evaluation.selected
    strata = policy.strata if strata is None else strata
    if isinstance(candidate_keys, str) or isinstance(strata, str) or not candidate_keys:
        raise ValueError("candidate_keys and strata must be column-name sequences")
    keys = list(dict.fromkeys([*strata, *candidate_keys]))
    if set(keys) - set(frame):
        raise ValueError(f"Unknown representative keys: {sorted(set(keys) - set(frame))}")
    frame = frame.loc[frame[list(candidate_keys)].map(is_stated).all(axis=1)]
    all_alternatives = {
        _key(list(identity) if isinstance(identity, tuple) else [identity]): group.occurrence_id.tolist()
        for identity, group in evaluation.occurrences.groupby(keys, sort=False, dropna=False)
    }
    rows = []
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        if policy.duplicates == "error" and group.score.nunique(dropna=False) > 1:
            raise ValueError("Candidate occurrences have conflicting scores; choose duplicates='best' or 'worst'")
        ascending = policy.ascending if policy.duplicates != "worst" else not policy.ascending
        columns = ["score", *(f"ranking_{i}" for i in range(len(policy.ranking_by)))]
        directions = [ascending, *(term.ascending for term in policy.ranking_by)]
        row = group.sort_values(columns, ascending=directions, na_position="last", kind="stable").iloc[0].copy()
        row["alternative_occurrences"] = all_alternatives[_key([row[key] for key in keys])]
        row["representative_reason"] = (f"{policy.duplicates}; ordered keys {columns}; "
                                         "missing last; stable input order breaks ties")
        rows.append(row)
    return pd.DataFrame(rows, columns=[*frame.columns, "alternative_occurrences", "representative_reason"]).sort_values(
        "score", ascending=policy.ascending, na_position="last", kind="stable").reset_index(drop=True)
