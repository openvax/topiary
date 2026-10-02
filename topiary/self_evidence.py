"""Bounded self-sequence searches and supplied presentation evidence."""

import hashlib
import json
from collections import defaultdict

import mhcgnomes
import numpy as np
import pandas as pd

from .amino_acids import AMINO_ACIDS, encode_amino_acids
from .ranking import mhc_dependence
from .result import TopiaryResult
from .serialization import normalize_python_types


def match_self_peptides(reference, peptides, *, max_mismatches=0, alleles=None,
                        excluded_gene_ids=(), observations=None, predictions=None):
    """Return every bounded self match, its origins and supplied evidence.

    Parameters
    ----------
    reference : SelfProteome
        The explicit corpus searched. Use an unfiltered corpus plus
        ``excluded_gene_ids`` to retain both CTA and non-CTA origins of shared
        peptides. A filtered corpus cannot recover omitted origins.
    peptides : iterable of str
        Query sequences, retaining order and duplicates via ``query_index``.
    max_mismatches : int
        Nonnegative same-length Hamming radius; zero means exact matching.
        Indels and TCR recognition are not evaluated. Reference comparisons
        are chunked, without a query-by-whole-proteome distance matrix.
    alleles : iterable of str or None, optional
        One allele per query, parsed with mhcgnomes. Missing alleles remain
        unknown; no genotype is borrowed from predictions or other queries.
    excluded_gene_ids : iterable of str
        Caller-resolved reference exclusions, for example from oncoref. These
        mark origins out of scope but never discard them. Topiary owns no CTA
        membership or tissue-reference catalog.
    observations : DataFrame or iterable of mappings, optional
        Supplied records with unique ``evidence_id``, ``peptide``, ``source``,
        ``evidence_kind`` (observed/predicted), and ``allele_assignment``
        (confirmed/predicted/unknown). Confirmed or predicted assignments need
        an ``allele``; an unresolved ``allele_set`` is not confirmed restriction.
        Optional gene_id restricts attribution to that origin. Tissue, assay,
        source version and other JSON-compatible facts survive unchanged.
    predictions : DataFrame or iterable of mappings, optional
        Long-form records with peptide, allele, kind, value,
        prediction_method_name and predictor_version. Unknown provenance stays
        unknown. All same-peptide records survive, including conflicts; only
        finite, per-allele measurements at the query allele count toward
        same-allele coverage. No prediction or model selection is run here.

    Returns
    -------
    TopiaryResult
        One row per query/matched-sequence/reference occurrence. No-hit and
        unassessed queries retain a row with a missing self peptide. Search
        status, reason and completeness are separate from observation and
        prediction coverage. ``self_in_scope`` marks non-excluded origins.
        Structured observation/prediction cells and complete search/evidence
        identity round-trip through ordinary Topiary CSV/TSV.

        Observation flags describe the supplied records, not biological
        absence. Unreported observations remain unknown. Peptide observations
        without gene attribution do not establish which matching gene produced
        them. Presentation, similarity and binding do not establish TCR
        recognition. ``no_match_in_scope`` is bounded negative search evidence,
        never a statement of no risk. Empty input returns an empty schema.
    """
    if isinstance(peptides, str):
        raise TypeError("peptides must be an iterable of sequences, not one string")
    peptides = list(peptides)
    if any(not isinstance(peptide, str) or not peptide for peptide in peptides):
        raise ValueError("Query peptides must be nonempty strings")
    if type(max_mismatches) is not int or max_mismatches < 0:
        raise ValueError("max_mismatches must be a nonnegative integer")
    if isinstance(alleles, str):
        raise TypeError("alleles must supply one allele per query")
    alleles = [None] * len(peptides) if alleles is None else list(alleles)
    if len(alleles) != len(peptides):
        raise ValueError("alleles must supply one allele per query")
    alleles = [_allele(value) for value in alleles]
    if isinstance(excluded_gene_ids, str):
        raise TypeError("excluded_gene_ids must be an iterable of gene IDs")
    excluded = sorted(set(excluded_gene_ids))
    if any(not isinstance(gene, str) or not gene for gene in excluded):
        raise ValueError("Excluded gene IDs must be nonempty strings")
    observation_records = _evidence_records(observations, observation=True)
    prediction_records = _evidence_records(predictions, observation=False)
    by_observation, by_prediction = defaultdict(list), defaultdict(list)
    for row in observation_records:
        by_observation[row["peptide"]].append(row)
    for row in prediction_records:
        by_prediction[row["peptide"]].append(row)
    supported = set(AMINO_ACIDS)
    excluded = set(excluded)
    reference_coverage = {}
    for length in sorted({len(peptide) for peptide in peptides}):
        array = reference._reference_arrays.get(length, [])
        unsupported = sum(int((array[start:start + 65536] >= len(AMINO_ACIDS)).any(axis=1).sum())
                          for start in range(0, len(array), 65536))
        reference_coverage[str(length)] = dict(n_peptides=len(array), n_unsupported=unsupported)
    searches, rows = {}, []
    for query_index, (peptide, allele) in enumerate(zip(peptides, alleles)):
        if peptide not in searches:
            searches[peptide] = _search(reference, peptide, max_mismatches, supported,
                                        reference_coverage[str(len(peptide))]["n_unsupported"])
        matches, complete, reason = searches[peptide]
        status = "matched" if matches else "no_match_in_scope" if complete else "unassessed"
        origins = [(match, distance, origin) for match, distance in matches
                   for origin in reference._provenance.get(match, [(None, None, None)])]
        for match, distance, (gene, transcript, offset) in origins or [(None, None, (None, None, None))]:
            observed = [record for record in by_observation[match]
                        if record.get("gene_id") is None or record["gene_id"] == gene]
            predicted = by_prediction[match]
            same_allele = [record for record in predicted if allele is not None
                           and record["allele"] == allele and record["prediction_mhc_dependence"] == "single_allele"
                           and isinstance(record["value"], (int, float)) and not isinstance(record["value"], bool)]
            measured = [record for record in observed if record["evidence_kind"] == "observed"]
            identity = None if match is None else _digest([reference.reference_version, match, gene, transcript, offset])
            rows.append(dict(
                query_index=query_index, peptide=peptide, peptide_length=len(peptide), allele=allele,
                self_match_id=identity, self_peptide=match, self_mismatches=distance,
                self_gene_id=gene, self_transcript_id=transcript, self_reference_offset=offset,
                self_in_scope=None if gene is None else gene not in excluded,
                self_reference_version=reference.reference_version,
                self_search_status=status, self_search_complete=complete, self_search_reason=reason,
                self_observation_status="not_supplied" if observations is None else "reported" if observed else "not_reported",
                self_observed=bool(measured) if observed else None,
                self_observed_confirmed_same_allele=(any(record["allele_assignment"] == "confirmed"
                    and record.get("allele") == allele for record in measured) if observed and allele else None),
                self_observed_predicted_same_allele=(any(record["allele_assignment"] == "predicted"
                    and record.get("allele") == allele for record in measured) if observed and allele else None),
                self_prediction_status=("not_supplied" if predictions is None else "query_allele_unknown"
                                        if allele is None else "reported_for_allele" if same_allele else "not_reported_for_allele"),
                self_observations=observed, self_predictions=predicted, self_same_allele_predictions=same_allele,
            ))
    columns = ["query_index", "peptide", "peptide_length", "allele", "self_match_id", "self_peptide", "self_mismatches",
               "self_gene_id", "self_transcript_id", "self_reference_offset", "self_in_scope",
               "self_reference_version", "self_search_status", "self_search_complete", "self_search_reason",
               "self_observation_status", "self_observed", "self_observed_confirmed_same_allele",
               "self_observed_predicted_same_allele", "self_prediction_status", "self_observations",
               "self_predictions", "self_same_allele_predictions"]
    scope = dict(reference_version=reference.reference_version, search="same_length_hamming",
                 reference_coverage=reference_coverage,
                 max_mismatches=max_mismatches, include_indels=False, excluded_gene_ids=sorted(excluded),
                 observation_input="not_supplied" if observations is None else "supplied",
                 observation_sha256=_digest(observation_records), prediction_input="not_supplied"
                 if predictions is None else "supplied", prediction_sha256=_digest(prediction_records))
    return TopiaryResult(pd.DataFrame(rows, columns=columns), extra={"self_search": scope})


def self_matches_in_windows(windows, reference, *, peptide_lengths, alleles=None,
                            excluded_gene_ids=(), observations=None, predictions=None):
    """Find exact reference peptides at every requested position in windows.

    Parameters
    ----------
    windows : mapping of str to str
        Window ID to complete proposed sequence. IDs and sequences must be
        nonempty strings. Repeated peptides and every source offset survive.
    reference : SelfProteome
        Corpus passed to :func:`match_self_peptides` without further filtering.
    peptide_lengths : iterable of int
        Explicit positive lengths to enumerate. Missing reference lengths
        remain unassessed; they are never silently removed from the request.
    alleles : mapping of str to iterable of str, optional
        Explicit alleles for each window. An absent/empty entry produces
        unknown allele context; another window's genotype is never borrowed.
    excluded_gene_ids, observations, predictions
        Passed to :func:`match_self_peptides` with its evidence semantics.

    Returns
    -------
    TopiaryResult
        Exact-match evidence with window_id, window_sequence and zero-based
        peptide_offset. The peptide columns denote window occurrences, not
        administered molecules for a cleavage model. Metadata also records
        requested window/length combinations too short to contain a peptide.
        Empty windows input returns an empty schema. No predictor runs and no
        window is chosen, trimmed or declared safe by this function.
    """
    windows = dict(windows)
    lengths = list(peptide_lengths)
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("peptide_lengths must contain positive integers")
    lengths = sorted(set(lengths))
    alleles = {} if alleles is None else dict(alleles)
    if set(alleles) - set(windows):
        raise ValueError("Alleles name an unknown window")
    queries, coverage = [], []
    for name, sequence in windows.items():
        if not isinstance(name, str) or not name or not isinstance(sequence, str) or not sequence:
            raise ValueError("Window IDs and sequences must be nonempty strings")
        named = alleles.get(name, ())
        if isinstance(named, str):
            raise TypeError("Window alleles must be an iterable, not one string")
        genotype = list(dict.fromkeys(_allele(allele) for allele in named)) or [None]
        for length in lengths:
            count = max(0, len(sequence) - length + 1)
            coverage.append(dict(window_id=name, peptide_length=length, n_occurrences=count,
                                 reason=None if count else "window_shorter_than_peptide_length"))
            for offset in range(count):
                for allele in genotype:
                    queries.append(dict(window_id=name, window_sequence=sequence, peptide_offset=offset,
                                        peptide=sequence[offset:offset + length], allele=allele))
    result = match_self_peptides(reference, [row["peptide"] for row in queries],
                                 alleles=[row["allele"] for row in queries], excluded_gene_ids=excluded_gene_ids,
                                 observations=observations, predictions=predictions)
    frame = result.df.copy()
    for column in ("window_id", "window_sequence", "peptide_offset"):
        frame[column] = [queries[index][column] for index in frame.query_index]
    return TopiaryResult(frame, extra={**result.extra, "self_window_coverage": coverage})


def _allele(value):
    return None if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)) or value == "" else mhcgnomes.parse(str(value)).to_string()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _evidence_records(data, *, observation):
    if data is None:
        return []
    frame = pd.DataFrame(data).copy()
    if not frame.columns.is_unique:
        raise ValueError("Evidence columns must be unique")
    if frame.empty:
        return []
    required = {"peptide", "evidence_id", "source", "evidence_kind", "allele_assignment"} if observation else {
        "peptide", "allele", "kind", "value", "prediction_method_name", "predictor_version"}
    if required - set(frame):
        raise ValueError(f"Missing self evidence columns: {sorted(required - set(frame))}")
    records = []
    for raw in frame.to_dict("records"):
        row = {key: None if pd.api.types.is_scalar(value) and pd.isna(value) else normalize_python_types(value)
               for key, value in raw.items()}
        if not isinstance(row["peptide"], str) or not row["peptide"]:
            raise ValueError("Evidence peptide must be a nonempty string")
        row["allele"] = _allele(row.get("allele"))
        if row.get("allele_set"):
            from .ranking import format_allele_set, split_allele_set
            row["allele_set"] = format_allele_set(_allele(name) for name in split_allele_set(row["allele_set"]))
        if observation:
            if any(not isinstance(row[key], str) or not row[key] for key in ("evidence_id", "source")):
                raise ValueError("Observation evidence_id and source must be nonempty strings")
            if row["evidence_kind"] not in {"observed", "predicted"} or row["allele_assignment"] not in {"confirmed", "predicted", "unknown"}:
                raise ValueError("Unknown observation evidence kind or allele assignment")
            if row["allele_assignment"] != "unknown" and not row["allele"]:
                raise ValueError("An assigned observation allele must be supplied explicitly")
        else:
            dependence = row.get("prediction_mhc_dependence") or mhc_dependence(row["kind"], rows=pd.DataFrame([row]))
            if dependence not in {"single_allele", "haplotype", "none"}:
                raise ValueError("Unknown prediction MHC dependence")
            row["prediction_mhc_dependence"] = dependence
        _digest(row)  # Reject non-JSON facts rather than silently stringifying them.
        records.append(row)
    if observation and len({row["evidence_id"] for row in records}) != len(records):
        raise ValueError("Observation evidence_id must be unique")
    return records


def _search(reference, peptide, radius, supported, unsupported):
    if not set(peptide) <= supported:
        return [], False, "query_sequence_unsupported"
    length = len(peptide)
    candidates = reference._reference_peptides.get(length, [])
    if not candidates:
        return [], False, "reference_length_unavailable"
    complete, reason = not unsupported, "unsupported_reference_sequences" if unsupported else None
    if radius == 0:
        # Exact matching requires no sequence-model assumption or distance matrix.
        return ([(peptide, 0)] if peptide in reference._reference_set(length) else []), complete, reason
    query = encode_amino_acids([peptide], length)[0]
    array = reference._reference_arrays[length]
    matches = []
    for start in range(0, len(array), 65536):
        distances = (array[start:start + 65536] != query).sum(axis=1)
        for index in np.flatnonzero(distances <= radius):
            candidate = candidates[start + int(index)]
            if not set(candidate) <= supported:
                continue
            matches.append((candidate, int(distances[index])))
    return matches, complete, reason
