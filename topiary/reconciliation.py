"""Conservative identity and relational views for combined biological evidence."""

from collections.abc import Mapping
from copy import deepcopy
import math

import pandas as pd

from .candidates import _combined_frame, _identity
from .ranking import is_stated
from .result import TopiaryResult
from .serialization import normalize_python_types


_ORF_FIELDS = (
    "coding_sequence", "nucleotide_sequence", "transcript_path", "transcript_id",
    "orf_start", "orf_end", "reading_frame", "orf_completeness",
    "starts_at_start_codon", "ends_with_stop_codon", "linked_variants",
    "protein_sequence", "protein_hypothesis_sequence",
)
_ID_COLUMNS = ("biological_event_ids", "orf_hypothesis_id", "peptide_occurrence_id", "rna_observation_ids")


def _value(value):
    if isinstance(value, (Mapping, list, tuple)):
        return normalize_python_types(value)
    return normalize_python_types(value) if is_stated(value) else None


def _names(value, field):
    if not isinstance(value, (list, tuple)) and not is_stated(value):
        return []
    if not isinstance(value, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return sorted(set(value))


def normalize_rna_observation(observation, *, sample_name=None):
    """Validate an RNA measurement without changing its biological subject.

    Parameters
    ----------
    observation : mapping
        Required fields are ``entity_type`` (gene, transcript, ORF or variant),
        ``entity_id``, ``quantity``, ``unit`` and ``value`` (nonnegative or null).
        ``sample_name`` is required here or as an argument. Optional library,
        read-set, method, version and other provenance fields are preserved.
        ``evidence_unit_ids`` supplies exact members of a count, if known;
        an empty list means measured zero, whereas absent/null means unknown.
    sample_name : str, optional
        Fill an absent sample; a different stated sample raises.

    Returns
    -------
    dict
        Independent copy, with sorted unique evidence members. No expression
        is promoted from gene/transcript to ORF, and no counts are inferred.
        A missing observation is invalid: omit it from the observation list.
    """
    if not isinstance(observation, Mapping):
        raise ValueError("RNA observation must be a mapping")
    record = deepcopy(normalize_python_types(dict(observation)))
    sample = record.get("sample_name") or sample_name
    if sample_name is not None and sample != sample_name:
        raise ValueError("RNA observation sample_name disagrees with its source row")
    record["sample_name"] = sample
    for key in ("sample_name", "entity_type", "entity_id", "quantity", "unit"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"RNA observation requires {key}")
    if record["entity_type"] not in {"gene", "transcript", "orf", "variant"}:
        raise ValueError("RNA entity_type must be gene, transcript, orf or variant")
    value = record.setdefault("value", None)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("RNA value must be a finite nonnegative number or null")
        if record["quantity"] == "count":
            if int(value) != value:
                raise ValueError("RNA count must be an integer")
            record["value"] = int(value)
    members = record.get("evidence_unit_ids")
    if members is not None:
        members = _names(members, "evidence_unit_ids")
        if record["quantity"] != "count" or value != len(members):
            raise ValueError("RNA evidence membership must agree with a stated count")
        for field in ("library_id", "read_set_id"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Identified RNA evidence requires {field}")
        record["evidence_unit_ids"] = members
    return record


def union_rna_observations(observations):
    """Count the union of identified RNA evidence in one measurement scope.

    Parameters
    ----------
    observations : iterable of mapping
        Records accepted by :func:`normalize_rna_observation`. Every record
        must measure the same sample, library, read-set namespace, entity,
        count quantity and unit, with explicit evidence-unit membership.

    Returns
    -------
    dict
        Union count and sorted member IDs, plus original ``observations``.
        Shared reads count once even if several callers report them. Empty
        input raises; unknown memberships, differing entities/namespaces and
        expression quantities such as TPM raise rather than implying disjoint
        evidence. Different read-set IDs do not prove independence.
    """
    records = [normalize_rna_observation(r) for r in observations]
    if not records:
        raise ValueError("At least one RNA observation is required")
    fields = ("sample_name", "library_id", "read_set_id", "entity_type", "entity_id", "quantity", "unit")
    first = records[0]
    if any(r["quantity"] != "count" or r["unit"] not in {"reads", "fragments", "umis", "cells"}
           or r.get("evidence_unit_ids") is None for r in records):
        raise ValueError("RNA union requires counts with known evidence-unit membership")
    if any(any(r.get(k) != first.get(k) for k in fields) for r in records):
        raise ValueError("RNA union requires the same sample/library/read-set/entity/quantity/unit scope")
    members = sorted({member for r in records for member in r["evidence_unit_ids"]})
    return dict({k: first[k] for k in fields}, value=len(members), evidence_unit_ids=members,
                method="evidence_union", observations=records)


def reconcile_evidence(result):
    """Link events, ORFs and occurrences without selecting biological evidence.

    Parameters
    ----------
    result : TopiaryResult
        Output of ``combine_sources``, optionally rescored. Each row may name
        one ``orf_id`` and several ``event_ids`` (or one ``event_id``). Equal
        event names link across callers only with the same explicit
        ``reference_name`` and sample; missing references remain source-local.
        ORF descriptors use coding/nucleotide sequence, transcript path or
        transcript ID, zero-based half-open ``orf_start``/``orf_end``, reading
        frame, completeness, start/stop flags and linked variants. A local ORF
        ID may recur across peptide rows: supplied descriptors must agree.
        Cross-source ORF agreement requires a reference, coding sequence and
        explicit path (or transcript ID plus ORF bounds); every supplied ORF
        descriptor must match. Equal proteins alone never establish ORF identity.
        ``peptide_start``/``peptide_end`` are zero-based half-open coordinates in
        the supplied protein/hypothesis sequence. Absent coordinates retain a
        source-local occurrence rather than guessing from a shared short peptide.
        Optional ``rna_observations`` is a list of explicit measurements accepted
        by ``normalize_rna_observation``. Other original annotations survive.

    Returns
    -------
    TopiaryResult
        Copy with ``biological_event_ids``, ``orf_hypothesis_id``,
        ``peptide_occurrence_id``, normalized ``rna_observations`` and
        ``rna_observation_ids``. ``orf_descriptor`` records reconciled assertions
        without overwriting the caller's cells. No measurements are combined or
        selected and no peptides, alleles or predictions are generated. Partial
        windows retain their own ORFs and never acquire full-protein identity.
        Empty input yields the same enriched schema. Reapplying is idempotent.
        Use ``evidence_views`` for relational tables and ``rank_candidates`` to
        select an actual supporting observation with an explicit score policy.

    Notes
    -----
    This reconciles explicit, normalized identities, not genome assemblies or
    coordinate conventions. Native event names must first be normalized by the
    caller; there is no implicit liftover, contig aliasing or variant alignment.
    Unknown specificity and conflicting abundance remain unchanged.
    """
    frame = _combined_frame(result).copy()
    records = frame.to_dict("records")
    descriptors, local_keys, events_by_row = {}, [], []
    for row in records:
        sample, source = row["candidate_sample"], row["source_label"]
        reference = _value(row.get("reference_name"))
        scope = [sample, reference] if reference is not None else [sample, None, source]
        names = _names(row.get("event_ids"), "event_ids")
        event = _value(row.get("event_id"))
        if event is not None:
            if names and str(event) not in names:
                raise ValueError("event_id disagrees with event_ids")
            names = sorted(set([*names, str(event)]))
        events = [_identity(["event", scope, name]) for name in names]
        events_by_row.append(events)
        descriptor = {key: _value(row.get(key)) for key in _ORF_FIELDS}
        for key in ("orf_start", "orf_end", "reading_frame"):
            value = descriptor[key]
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value or value < 0:
                    raise ValueError(f"{key} must be a nonnegative integer")
                descriptor[key] = int(value)
        if descriptor["reading_frame"] not in (None, 0, 1, 2):
            raise ValueError("reading_frame must be 0, 1 or 2")
        if descriptor["orf_start"] is not None and descriptor["orf_end"] is not None:
            if descriptor["orf_start"] >= descriptor["orf_end"]:
                raise ValueError("orf_start must be smaller than orf_end")
        for key in ("starts_at_start_codon", "ends_with_stop_codon"):
            if descriptor[key] is not None and type(descriptor[key]) is not bool:
                raise ValueError(f"{key} must be boolean or null")
        local_id = _value(row.get("orf_id"))
        has_orf = local_id is not None or any(descriptor[k] is not None for k in
                    ("coding_sequence", "nucleotide_sequence", "protein_sequence", "protein_hypothesis_sequence"))
        local = _identity(["local_orf", scope, source, local_id or row["source_observation_id"]]) if has_orf else None
        local_keys.append(local)
        if local is not None:
            previous = descriptors.setdefault(local, dict(descriptor, sample_name=sample, reference_name=reference))
            for key, value in descriptor.items():
                if value is not None and previous[key] is not None and _identity([value]) != _identity([previous[key]]):
                    raise ValueError(f"Conflicting {key} for source ORF {local_id!r}")
                if value is not None:
                    previous[key] = value
    enriched = []
    for row, local, events in zip(records, local_keys, events_by_row):
        descriptor = deepcopy(descriptors.get(local))
        orf = local
        if descriptor is not None:
            located = descriptor["transcript_path"] is not None or (
                descriptor["transcript_id"] is not None and descriptor["orf_start"] is not None
                and descriptor["orf_end"] is not None)
            if descriptor["reference_name"] is not None and descriptor["coding_sequence"] is not None and located:
                orf = _identity(["orf", descriptor])
        peptide, occurrence = _value(row.get("peptide")), None
        if peptide is not None:
            start, end = _value(row.get("peptide_start")), _value(row.get("peptide_end"))
            if (start is None) != (end is None):
                raise ValueError("peptide_start and peptide_end must be supplied together")
            if start is not None:
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v for v in (start, end)):
                    raise ValueError("Peptide coordinates must be integers")
                start, end = int(start), int(end)
                if start < 0 or end - start != len(peptide):
                    raise ValueError("Peptide coordinates disagree with peptide length")
                sequence = (descriptor or {}).get("protein_sequence") or (descriptor or {}).get("protein_hypothesis_sequence")
                if sequence is not None and sequence[start:end] != peptide:
                    raise ValueError("Peptide coordinates disagree with the supporting protein")
            occurrence = _identity([
                                    "occurrence", row["candidate_sample"], orf, peptide, start, end,
                                    _value(row.get("gene_id")), _value(row.get("pep_context")),
                                    [isinstance(row.get("n_flank"), str), _value(row.get("n_flank"))],
                                    [isinstance(row.get("c_flank"), str), _value(row.get("c_flank"))],
                                    None if orf is not None and start is not None else row["source_observation_id"]])
        rna = row.get("rna_observations")
        if not isinstance(rna, list) and not is_stated(rna):
            rna = []
        if not isinstance(rna, list):
            raise ValueError("rna_observations must be a list of measurement mappings")
        rna = [normalize_rna_observation(r, sample_name=row["candidate_sample"]) for r in rna]
        enriched.append(dict(biological_event_ids=events, orf_hypothesis_id=orf,
                             orf_descriptor=descriptor, peptide_occurrence_id=occurrence,
                             rna_observations=rna,
                             rna_observation_ids=[_identity(["rna", row["source_label"], r]) for r in rna]))
    for column in (*_ID_COLUMNS, "orf_descriptor", "rna_observations"):
        frame[column] = pd.Series([r[column] for r in enriched], index=frame.index, dtype=object)
    metadata = deepcopy(result.metadata)
    metadata.extra["evidence_reconciliation"] = {"schema": "topiary.evidence.v1", "interval_convention": "zero_based_half_open"}
    return TopiaryResult(frame, metadata=metadata, form="long")


def evidence_views(result, *, source_labels=None):
    """Return relational event/ORF/protein/occurrence/RNA and observation tables.

    Parameters
    ----------
    result : TopiaryResult
        Combined or reconciled evidence. Uses ``reconcile_evidence`` for the
        same identity rules before materializing any view.
    source_labels : iterable of str, optional
        Explicit source-stratified view; None includes all sources. Unknown
        labels raise. No caller, ORF or RNA observation is preferred implicitly.

    Returns
    -------
    dict of pandas.DataFrame
        ``events``, ``orfs``, ``proteins``, ``occurrences``, ``candidates``,
        ``rna_observations`` and ``links``. Node tables have ``id`` and
        ``source_observations``; links retain one distinct combination of source
        observation, event, ORF, protein, occurrence, candidate and RNA IDs.
        A link's list-valued event/RNA columns express many-to-many support.
        Empty views retain their columns. These views never aggregate abundance;
        traverse links to retain the supporting hypothesis of a selected window.
    """
    frame = reconcile_evidence(result).df
    if source_labels is not None:
        if isinstance(source_labels, str):
            raise TypeError("source_labels must be an iterable of labels")
        labels = set(source_labels)
        if labels - set(frame.source_label):
            raise ValueError("Unknown source_labels")
        frame = frame[frame.source_label.isin(labels)]
    tables = {name: {} for name in ("events", "orfs", "proteins", "occurrences", "candidates", "rna_observations")}
    links = {}

    def add(table, identity, payload, observation):
        if identity is not None:
            node = tables[table].setdefault(identity, dict(id=identity, **payload, source_observations=[]))
            node["source_observations"] = sorted(set([*node["source_observations"], observation]))

    for row in frame.to_dict("records"):
        observation = row["source_observation_id"]
        sample = row["candidate_sample"]
        names = _names(row.get("event_ids"), "event_ids")
        if is_stated(row.get("event_id")):
            names = sorted(set([*names, str(row["event_id"])]))
        for identity, name in zip(row["biological_event_ids"], names):
            add("events", identity, dict(sample_name=sample, reference_name=_value(row.get("reference_name")),
                                        event_id=name), observation)
        descriptor = row["orf_descriptor"]
        sequence = (descriptor or {}).get("protein_sequence")
        protein_id = _identity([sequence]) if sequence is not None else None
        add("orfs", row["orf_hypothesis_id"], dict(descriptor=descriptor, protein_sequence_id=protein_id), observation)
        add("proteins", protein_id, dict(sequence=sequence), observation)
        add("proteins", _value(row["protein_sequence_id"]), dict(sequence=_value(row.get("protein_sequence"))), observation)
        add("occurrences", row["peptide_occurrence_id"], dict(sample_name=sample, peptide=_value(row.get("peptide")),
            orf_hypothesis_id=row["orf_hypothesis_id"], peptide_start=_value(row.get("peptide_start")),
            peptide_end=_value(row.get("peptide_end"))), observation)
        add("candidates", _value(row["candidate_id"]), dict(sample_name=sample, peptide=_value(row.get("peptide")),
            allele=row["candidate_allele"]), observation)
        for identity, rna in zip(row["rna_observation_ids"], row["rna_observations"]):
            add("rna_observations", identity, dict(measurement=rna), observation)
        link = {key: _value(row.get(key)) for key in
                ("source_label", "source_observation_id", "candidate_id", "protein_sequence_id", *_ID_COLUMNS)}
        links[_identity([link])] = link
    columns = {
        "events": ["sample_name", "reference_name", "event_id"],
        "orfs": ["descriptor", "protein_sequence_id"], "proteins": ["sequence"],
        "occurrences": ["sample_name", "peptide", "orf_hypothesis_id", "peptide_start", "peptide_end"],
        "candidates": ["sample_name", "peptide", "allele"], "rna_observations": ["measurement"],
    }
    views = {name: pd.DataFrame(list(nodes.values()), columns=["id", *columns[name], "source_observations"])
             for name, nodes in tables.items()}
    views["links"] = pd.DataFrame(list(links.values()), columns=[
        "source_label", "source_observation_id", "candidate_id", "protein_sequence_id", *_ID_COLUMNS])
    return views
