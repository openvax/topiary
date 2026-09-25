"""Combine source observations and add explicitly requested prediction features."""

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import re

import numpy as np
import pandas as pd
import mhcgnomes
from mhctools.pred import value_unit

from .io_pvacseq import derive_mhc_class
from .predictor import from_predictions
from .ranking import (
    DSLNode, apply_filter, evaluate_scores, format_allele_set, is_stated,
    mhc_dependence, parse, split_allele_set,
)
from .result import TopiaryResult, stack_results
from .serialization import normalize_python_types
from .wide import PREDICTION_COLUMNS, SOURCE_PREDICTION_COLUMNS


# Comparator scores vary by model/kind/allele just like the primary scores.
# Their peptides, genes and context remain observation identity; only the
# explicitly named prediction fields belong to the measurement axes.
_PREDICTION_MEASUREMENTS = PREDICTION_COLUMNS | {
    f"{scope}_{field}"
    for scope in ("wt", "self", "self_nearest", "shuffled")
    for field in ("value", "score", "percentile_rank", "affinity",
                  "prediction_method_name", "predictor_version", "value_unit", "measurement_context")
}
_MEASUREMENT_FIELDS = (_PREDICTION_MEASUREMENTS - {"prediction_run_name"}) | {
    "allele", "allele_set", "mhc_class", "value_unit", "measurement_context",
    "peptide_input", "cache_key",
}
_ADDED_COLUMNS = (
    "source_label", "source_row", "source_observation_id", "candidate_id",
    "candidate_sample", "candidate_allele", "candidate_mhc_class",
    "protein_sequence_id", "source_prediction_mhc_dependence", *SOURCE_PREDICTION_COLUMNS.values(),
)


def _identity(values):
    def encode(value):
        value = normalize_python_types(value, dataclasses_as_dict=True)
        if isinstance(value, Mapping):
            return [[str(type(k)), str(k), encode(v)] for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))]
        if isinstance(value, np.ndarray):
            return encode(value.tolist())
        if isinstance(value, (set, frozenset)):
            return sorted([encode(v) for v in value], key=str)
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        return str(value) if is_stated(value) else None

    payload = json.dumps(encode(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _allele(value):
    return mhcgnomes.parse(str(value)).to_string() if is_stated(value) else ""


def combine_sources(sources, *, sample_name=None):
    """Combine prediction tables without running models or merging evidence.

    Parameters
    ----------
    sources : mapping of str to TopiaryResult or pandas.DataFrame
        Unique caller-chosen source labels and normalized prediction tables.
        LENS, pVACseq and Topiary results may mix long and wide forms. An
        ORF/RNA-only table supplies ``protein_sequence`` and needs no peptide,
        allele or prediction columns. Every original column is retained.
        ``protein_sequence`` means a full translated ORF/protein; contextual
        fragments in ``sequence``/``pep_context`` are not promoted to ORFs.
        ``protein_hypothesis_sequence`` also admits comparison-only translated
        windows. These have no candidate ID, and partial windows do not receive
        a full-protein identity or enter ``protein_evidence_view``.
    sample_name : str, optional
        Sample identity for rows whose ``sample_name`` is unstated. Existing
        sample names take precedence. Unlabelled nonempty input requires this
        argument: absent identity must not merge different patients silently.

    Returns
    -------
    TopiaryResult
        Long-form observations, with ``source_label`` and zero-based
        ``source_row`` identifying each normalized input row. ``candidate_id``
        links the same sample/peptide/canonical allele across sources;
        allele-free evidence has no candidate ID. ``source_observation_id``
        preserves each source/sequence/transcript/context for DSL evaluation.
        Counts and conflicting predictions are never summed or overwritten.
        ``protein_sequence_id`` links exact full protein sequences. A shared
        caller-normalized ``event_id`` may link the same variant/event across
        tools; native variant strings are not assumed equivalent.
        Original metadata is retained in ``extra['combined_sources']``.
        An empty mapping returns an empty result with the combined schema.

    Notes
    -----
    This combines the candidates actually reported, including missing scores.
    It neither establishes tumor specificity nor recovers candidates omitted
    by an aggregated report. Use :func:`rank_candidates` for an explicit
    choice among multiple observations of a candidate.
    """
    if not isinstance(sources, Mapping):
        raise TypeError("sources must map unique source labels to prediction tables")
    if sample_name is not None and (not isinstance(sample_name, str) or not is_stated(sample_name)):
        raise ValueError("sample_name must be a nonempty sample label")
    results, provenance = [], {}
    for label, source in sources.items():
        if not isinstance(label, str) or not is_stated(label):
            raise ValueError("source labels must be nonempty strings")
        if isinstance(source, pd.DataFrame):
            source = TopiaryResult(source)
        if not isinstance(source, TopiaryResult):
            raise TypeError(f"Source {label!r} must be a TopiaryResult or DataFrame")
        native = source.df if source.form == "unknown" else source.long_df
        frame = native.copy().reset_index(drop=True)
        if not frame.columns.is_unique:
            raise ValueError(f"Source {label!r} has duplicate columns")
        collisions = set(_ADDED_COLUMNS) & set(frame)
        if collisions:
            raise ValueError(f"Source {label!r} already has combined columns: {sorted(collisions)}")
        for column in ("peptide", "allele", "kind", "prediction_method_name", "predictor_version"):
            if column not in frame:
                frame[column] = None
        proteins = frame.get("protein_sequence", pd.Series(None, index=frame.index, dtype=object))
        hypotheses = frame.get("protein_hypothesis_sequence", pd.Series(None, index=frame.index, dtype=object))
        if not (frame.peptide.map(is_stated) | proteins.map(is_stated) | hypotheses.map(is_stated)).all():
            raise ValueError(f"Source {label!r} needs a peptide or explicit protein_sequence "
                             "or protein_hypothesis_sequence on every row")
        samples = frame.get("sample_name", pd.Series(None, index=frame.index, dtype=object))
        samples = samples.where(samples.map(is_stated), sample_name)
        if not samples.map(is_stated).all():
            raise ValueError(f"Source {label!r} needs sample_name for unlabelled rows")
        frame["source_label"] = label
        frame["source_row"] = np.arange(len(frame))
        frame["candidate_sample"] = samples
        frame["candidate_allele"] = frame.allele.map(_allele)
        frame["candidate_mhc_class"] = derive_mhc_class(frame.candidate_allele)
        frame["protein_sequence_id"] = proteins.map(lambda s: _identity([s]) if is_stated(s) else None)
        for column, source_column in SOURCE_PREDICTION_COLUMNS.items():
            frame[source_column] = frame[column] if column in frame else None
        frame["candidate_id"] = [
            _identity([sample, peptide, allele]) if allele and is_stated(peptide) else None
            for sample, peptide, allele in zip(samples, frame.peptide, frame.candidate_allele)
        ]
        # Differing annotations/abundance are separate source observations,
        # even when sequence identity agrees. Prediction kinds and alleles
        # remain axes within an observation, so peptide_view still composes.
        fields = sorted(set(native.columns) - _MEASUREMENT_FIELDS)
        frame["source_observation_id"] = [
            _identity([label, sample, fields, *values])
            for sample, values in zip(samples, frame[fields].itertuples(index=False, name=None))
        ]
        # The ordinary DSL reads one value per observation/model. Refuse
        # contradictory rows in that slot instead of letting its first-row
        # reduction silently choose one. Distinct source labels or run names
        # retain independent measurements as separate observations.
        axes = ["source_observation_id", "candidate_allele", "kind",
                "prediction_method_name", "predictor_version"]
        if "allele_set" in frame:
            axes.append("allele_set")
        values = sorted((_PREDICTION_MEASUREMENTS | {"value_unit", "measurement_context"})
                        & set(frame) - set(axes))
        slots = frame[axes].apply(lambda row: _identity(row.tolist()), axis=1)
        measurements = frame[values].apply(lambda row: _identity(row.tolist()), axis=1)
        if measurements.groupby(slots).nunique().gt(1).any():
            raise ValueError(f"Source {label!r} has conflicting predictions for one observation; "
                             "supply distinct source labels or prediction_run_name values")
        frame["source_prediction_mhc_dependence"] = None
        for (kind, _), rows in frame.groupby(["kind", "prediction_method_name"], dropna=False):
            if is_stated(kind):
                frame.loc[rows.index, "source_prediction_mhc_dependence"] = mhc_dependence(
                    kind, kind_support=source._kind_support(), rows=rows)
        metadata = deepcopy(source.metadata)
        provenance[label] = dict(
            sources=list(source.sources), models=dict(source.models),
            extra=deepcopy(source.extra), filter_by=source.filter_by_str,
            sort_by=source.sort_by_str, columns=list(native.columns),
        )
        results.append(TopiaryResult(frame, metadata=metadata, form="long"))
    if not results:
        return TopiaryResult(pd.DataFrame(columns=[
            "peptide", "allele", "kind", "prediction_method_name", *_ADDED_COLUMNS,
        ]), form="long", extra={"combined_sources": {}})
    combined = stack_results(results)
    combined.extra["combined_sources"] = provenance
    # Source-specific context declarations may differ for the same model.
    # Retain them above and on each row; a last-source global override would
    # silently reinterpret another source's predictions.
    combined.extra.pop("kind_support", None)
    return combined


def protein_evidence_view(result):
    """Link identical full protein products while retaining ORF observations.

    Parameters
    ----------
    result : TopiaryResult
        Combined tables. Only explicit ``protein_sequence`` observations
        participate; peptide-only and local-context records remain available
        in the original result without being misrepresented as full ORFs.

    Returns
    -------
    pandas.DataFrame
        One row per sample, event and exact protein sequence. Equal sequences
        share ``protein_sequence_id`` across sources; different sequences
        under the same explicit ``event_id`` remain alternative rows.
        Without event_id, observations remain separate (the exact sequence ID
        still shows sequence agreement). JSON ``source_labels``,
        ``source_observations`` and ``candidate_ids`` link back to the original
        abundance measurements and predictions; no abundance is averaged,
        summed or promoted from one biological context to another.
        ``event_key`` records the explicit event or the observation fallback.
        Empty/protein-free input returns an empty frame with this schema.

    Notes
    -----
    A caller-normalized event ID must include reference and variant identity
    as appropriate. This function cannot infer that unrelated source-specific
    identifiers denote the same genomic event.
    Equal translated sequences do not imply identical nucleotide ORFs:
    differing ORF/transcript/coding-sequence annotations remain independent
    source observations linked from the protein view.
    """
    frame = _combined_frame(result)
    columns = ["candidate_sample", "event_key", "protein_sequence_id", "protein_sequence",
               "source_labels", "source_observations", "candidate_ids"]
    if "protein_sequence" not in frame:
        return pd.DataFrame(columns=columns)
    frame = frame[frame.protein_sequence_id.notna()].copy()
    events = frame.get("event_id", pd.Series(None, index=frame.index, dtype=object))
    frame["event_key"] = [
        "event:" + str(event) if is_stated(event) else "observation:" + observation
        for event, observation in zip(events, frame.source_observation_id)
    ]
    rows = []
    for (sample, event, sequence), group in frame.groupby(
            ["candidate_sample", "event_key", "protein_sequence_id"], sort=False):
        rows.append(dict(candidate_sample=sample, event_key=event,
                         protein_sequence_id=sequence, protein_sequence=group.protein_sequence.iloc[0],
                         source_labels=json.dumps(sorted(set(group.source_label))),
                         source_observations=json.dumps(sorted(set(group.source_observation_id))),
                         candidate_ids=json.dumps(sorted(set(group.candidate_id.dropna())))))
    return pd.DataFrame(rows, columns=columns)


def _combined_frame(result):
    if not isinstance(result, TopiaryResult):
        raise TypeError("result must be a TopiaryResult from combine_sources")
    frame = result.long_df
    missing = set(_ADDED_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"Use combine_sources first; missing {sorted(missing)}")
    return frame


def _node(expression):
    if isinstance(expression, str):
        return parse(expression)
    if isinstance(expression, DSLNode):
        return expression
    raise TypeError("Use a DSL expression string or DSLNode")


def rank_candidates(result, score_by, *, filter_by=None, ascending=False,
                    duplicates="error", strata=("candidate_mhc_class",),
                    default_methods=None, default_versions=None):
    """Rank table candidates using existing DSL features and predictions.

    Parameters
    ----------
    result : TopiaryResult
        Output of :func:`combine_sources`, optionally with added features.
    score_by : str or DSLNode
        Explicit scoring expression. Original predictions remain the default
        namespace even after re-scoring. Missing scores remain missing.
    filter_by : str or DSLNode, optional
        Existing Topiary filter semantics, applied per source observation.
        The original result is not modified or narrowed.
    ascending : bool
        Whether smaller scores rank first (for example, IC50). Default False.
    duplicates : {'error', 'best', 'worst'}
        Explicit policy for multiple observations of one candidate. Default
        rejects conflicting scores, including scored vs unscored evidence.
        Best/worst selects an actual observation; it never adds evidence.
    strata : sequence of str
        Independent ranking partitions. Defaults to MHC class; include
        ``source_label`` or an antigen-category column for stratified lists.
        Pass an empty sequence only to request a cross-class ranking.
    default_methods, default_versions : dict, optional
        Passed unchanged to the DSL's filter and score evaluators.

    Returns
    -------
    pandas.DataFrame
        One representative row per candidate per stratum, with
        ``candidate_score``, nullable ``candidate_rank``, ``ranking_status``
        and JSON ``candidate_observations`` listing all contributing source
        observations in the selected view. Missing scores have no rank and
        sort last. The full, unchanged evidence remains in ``result``.
        Allele-free rows support DSL evaluation but are not candidate pMHCs.
    """
    frame = _combined_frame(result)
    # ORF/RNA observations have no peptide to score. They must not enter
    # normalization/aggregation denominators in a candidate ranking.
    frame = frame[frame.peptide.map(is_stated).astype(bool)]
    if duplicates not in {"error", "best", "worst"}:
        raise ValueError("duplicates must be 'error', 'best', or 'worst'")
    if isinstance(strata, str):
        raise TypeError("strata must be a sequence of column names")
    strata = list(strata)
    if len(set(strata)) != len(strata) or set(strata) - set(frame):
        raise ValueError("strata must name distinct existing columns")
    context = dict(default_methods=default_methods, default_versions=default_versions,
                   kind_support=result._kind_support())
    if filter_by is not None:
        frame = apply_filter(frame, _node(filter_by), **context)
    frame = frame.copy()
    frame["candidate_score"] = evaluate_scores(frame, _node(score_by), **context)
    frame = frame[frame.candidate_id.notna()]
    observation_keys = list(dict.fromkeys(["source_observation_id", "candidate_id", "allele",
                                           *( ["allele_set"] if "allele_set" in frame else []), *strata]))
    frame = frame.drop_duplicates(observation_keys)
    keys = list(dict.fromkeys([*strata, "candidate_id"]))
    rows = []
    for _, group in frame.groupby(keys, dropna=False, sort=False):
        if duplicates == "error" and group.candidate_score.nunique(dropna=False) > 1:
            raise ValueError("Candidate observations have conflicting scores; choose duplicates='best' or 'worst'")
        order = ascending if duplicates != "worst" else not ascending
        row = group.sort_values("candidate_score", ascending=order, na_position="last", kind="stable").iloc[0].copy()
        row["candidate_observations"] = json.dumps(sorted(set(group.source_observation_id)))
        rows.append(row)
    ranked = pd.DataFrame(rows, columns=[*frame.columns, "candidate_observations"])
    ranked = ranked.sort_values("candidate_score", ascending=ascending, na_position="last", kind="stable")
    scores = ranked.groupby(strata, dropna=False)["candidate_score"] if strata else ranked.candidate_score
    ranked["candidate_rank"] = scores.rank(method="min", ascending=ascending).astype("Int64")
    ranked["ranking_status"] = np.where(ranked.candidate_score.notna(), "ranked", "missing_score")
    return ranked.reset_index(drop=True)


def rescore_candidates(result, models, *, prefix, select=None, use_flanks=True):
    """Append prediction features for selected exact peptides on demand.

    Parameters
    ----------
    result : TopiaryResult
        Combined source observations. Original values, rows and metadata
        remain intact; this operation does not change the ranking policy.
    models : predictor instance or sequence of instances
        Configured mhctools models exposing ``predict_dataframe`` and
        ``kind_support()``. Their configured alleles must cover the selected
        candidates for every declared allele-dependent kind. Allele-free
        processing cannot satisfy missing affinity/presentation coverage;
        processing-only models need no allele coverage. Models are called
        only here, never during combination.
    prefix : str
        Unique run identifier used in added feature names, for example
        ``fresh__mhcflurry__pMHC_affinity__value``. Double separators keep
        features distinct from native prediction columns during wide/long IO.
    select : str or DSLNode, optional
        Filter selecting source observations to re-score. Other rows receive
        missing features. No new peptide windows or HLA candidates are added.
    use_flanks : bool
        Forward supplied flank sequences. A flank-dependent model requires
        both flanks to be stated (empty strings mean known termini). Set
        False explicitly to request predictions without flanking context.

    Returns
    -------
    TopiaryResult
        Long-form copy with DSL-addressable numeric feature columns. Model,
        version, kind, field, units and invocation settings are recorded in
        ``extra['candidate_rescoring']``. Same-peptide calls are shared only
        within matching sample, flank and genotype context. Empty selections
        call no predictor. Existing prefixes or conflicting outputs raise.
    """
    frame = _combined_frame(result)
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("prefix must be an identifier starting with a letter")
    metadata = deepcopy(result.metadata)
    runs = metadata.extra.setdefault("candidate_rescoring", {})
    if not isinstance(runs, dict):
        raise ValueError("candidate_rescoring metadata must be a mapping of run identifiers")
    if prefix in runs or any(c.startswith(prefix + "__") for c in frame):
        raise ValueError(f"Re-scoring prefix {prefix!r} already exists")
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    if not models or any(not callable(getattr(m, method, None))
                         for m in models for method in ("predict_dataframe", "kind_support")):
        raise TypeError("models must expose predict_dataframe and kind_support; pass configured mhctools instances")
    selected = apply_filter(frame, _node(select)) if select is not None else frame
    selected = selected[selected.candidate_id.notna()]
    selected = selected.drop_duplicates(["source_observation_id", "candidate_id", "allele_set"]
                                        if "allele_set" in selected else ["source_observation_id", "candidate_id"])
    features, descriptors, cache = {}, {}, {}
    for _, row in selected.iterrows():
        n_flank, c_flank = row.get("n_flank"), row.get("c_flank")
        flanks_known = isinstance(n_flank, str) and isinstance(c_flank, str)
        genotype = format_allele_set(split_allele_set(row.get("allele_set")))
        for model_index, model in enumerate(models):
            if use_flanks and getattr(model, "uses_flanking_sequences", False) and not flanks_known:
                raise ValueError("Flank-dependent re-scoring requires both flanks; set use_flanks=False explicitly")
            kwargs = dict(n_flanks=[n_flank], c_flanks=[c_flank]) if use_flanks and flanks_known else {}
            cache_key = (model_index, row.candidate_sample, row.peptide,
                         n_flank if kwargs else None, c_flank if kwargs else None, genotype)
            if cache_key not in cache:
                raw = model.predict_dataframe([row.peptide], **kwargs)
                support = model.kind_support()
                predicted = from_predictions(raw)
                if not predicted.empty and not predicted.peptide.eq(row.peptide).all():
                    raise ValueError("Re-scoring returned a different peptide")
                cache[cache_key] = (predicted, support)
            predicted, support = cache[cache_key]
            matched_kinds = set()
            for _, prediction in predicted.iterrows():
                kind = str(prediction.kind)
                spec = next((v for k, v in support.items() if str(getattr(k, "value", k)) == kind), {})
                dependence = spec.get("mhc_dependence")
                if dependence not in {"none", "single_allele", "haplotype"}:
                    raise ValueError(f"Model must declare MHC dependence for {kind!r}")
                if dependence == "haplotype":
                    actual = format_allele_set(getattr(model, "alleles", []))
                    if not genotype or genotype != actual:
                        raise ValueError("Haplotype re-scoring requires allele_set matching the configured model")
                elif dependence != "none" and _allele(prediction.get("allele")) != row.candidate_allele:
                    continue
                matched_kinds.add(kind)
                method = str(prediction.prediction_method_name)
                if not re.fullmatch(r"[A-Za-z0-9_]+", method):
                    raise ValueError(f"Predictor name {method!r} cannot form a DSL feature identifier")
                for field in ("value", "score", "percentile_rank"):
                    if field not in prediction or not is_stated(prediction[field]):
                        continue
                    column = f"{prefix}__{method}__{kind}__{field}"
                    measurement = normalize_python_types(prediction.get("measurement_context"),
                                                         dataclasses_as_dict=True)
                    unit = prediction.get("value_unit")
                    if not is_stated(unit):
                        unit = measurement.get("unit") if isinstance(measurement, Mapping) else value_unit(kind)
                    descriptor = dict(model=method, version=prediction.get("predictor_version"),
                                      kind=kind, field=field, mhc_dependence=dependence,
                                      unit=unit if field == "value" else ("percent" if field == "percentile_rank" else None))
                    descriptor = {k: str(v) if is_stated(v) else None for k, v in descriptor.items()}
                    if dependence != "none":
                        descriptor["configured_alleles"] = split_allele_set(
                            format_allele_set(getattr(model, "alleles", ())))
                    if isinstance(measurement, Mapping):
                        descriptor["measurement_context"] = dict(measurement)
                    if column in descriptors and descriptors[column] != descriptor:
                        raise ValueError("Multiple versions/units need separate re-scoring prefixes")
                    descriptors[column] = descriptor
                    key = (row.source_observation_id, row.candidate_id, genotype)
                    values = features.setdefault(column, {})
                    value = prediction[field]
                    if key in values and values[key] != value:
                        raise ValueError("Conflicting re-scoring values for the same observation")
                    values[key] = value
            # Processing output is useful on its own, but cannot stand in for
            # a missing affinity/presentation result for the selected allele.
            required_kinds = {
                str(getattr(kind, "value", kind)) for kind, spec in support.items()
                if spec.get("mhc_dependence") in {"single_allele", "haplotype"}
            }
            missing_kinds = required_kinds - matched_kinds
            if missing_kinds:
                raise ValueError(f"Model returned no {', '.join(sorted(missing_kinds))} prediction "
                                 f"for {row.peptide}/{row.candidate_allele}")
            if not matched_kinds:
                raise ValueError(f"Model returned no prediction for {row.peptide}/{row.candidate_allele}")
    output = frame.copy()
    genotypes = frame.get("allele_set", pd.Series(None, index=frame.index, dtype=object)).map(
        lambda value: format_allele_set(split_allele_set(value)))
    keys = list(zip(frame.source_observation_id, frame.candidate_id, genotypes))
    for column, values in features.items():
        output[column] = [values.get(key, np.nan) for key in keys]
    runs[prefix] = dict(producer="topiary", created_at=datetime.now(timezone.utc).isoformat(),
                        features=descriptors, use_flanks=use_flanks,
                        input_sources=sorted(set(selected.source_label)),
                        select=_node(select).to_expr_string() if select is not None else None)
    return TopiaryResult(output, metadata=metadata, form="long",
                         filter_by_ast=result.filter_by_ast, sort_by_ast=result.sort_by_ast)
