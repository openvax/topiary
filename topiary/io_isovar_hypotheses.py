"""Read Isovar's comparison export without selecting prediction candidates."""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from .result import TopiaryResult


_SCHEMA = "isovar.protein_hypotheses.v1"
_COLUMNS = (
    "sample_name", "event_id", "hypothesis_id", "translation_id",
    "isovar_protein_sequence_id", "nucleotide_sequence_id", "nucleotide_sequence",
    "protein_hypothesis_sequence", "protein_sequence", "isovar_rank", "representative",
    "n_terminus", "c_terminus", "mutation_start", "mutation_end",
    "protein_hypotheses_complete", "protein_sequence_limit", "passes_all_filters",
    "isovar_source", "protein_segments", "protein_fragments", "protein_evidence_set_id",
    "translation_segments", "translation_fragments", "translation_evidence_set_id",
)


def _named(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Isovar export requires a nonempty {field}")
    return value


def _support_columns(support, prefix, export):
    """Project one wire-format measurement, checking its evidence reference."""
    support = support or {}
    if not isinstance(support, dict):
        raise ValueError("Isovar RNA support must be a mapping")
    for unit in ("segments", "fragments"):
        count = support.get(unit)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"Isovar {unit} count must be a nonnegative integer or null")
    key = support.get("evidence_set_id")
    if key is not None:
        sets = export["evidence_sets"]
        if key not in sets:
            raise ValueError(f"Unknown Isovar evidence_set_id: {key!r}")
        evidence = sets[key]
        if evidence.get("evidence_set_id") != key or evidence.get("evidence_scope") != export["evidence_scope"]:
            raise ValueError(f"Inconsistent Isovar evidence identity or scope: {key!r}")
        for unit in ("segments", "fragments"):
            identities = evidence.get("segment_ids" if unit == "segments" else "fragment_ids")
            if (not isinstance(identities, list) or any(not isinstance(i, str) for i in identities)
                    or len(set(identities)) != evidence.get(unit) or support.get(unit) != evidence.get(unit)):
                raise ValueError(f"Inconsistent Isovar {unit} count for {key!r}")
    return {f"{prefix}_{field}": support.get(field) for field in ("segments", "fragments", "evidence_set_id")}


def read_isovar_hypotheses(data, *, tag=None):
    """Import all exported Isovar hypotheses as comparison observations.

    Parameters
    ----------
    data : str, pathlib.Path or mapping
        Path to an ``isovar.protein_hypotheses.v1`` JSON export, or the mapping
        returned by ``isovar.export_protein_hypotheses``. Use the JSON export:
        its companion TSV omits evidence sets and other required provenance.
        An export with no hypotheses returns an empty table with the same
        columns, retaining its events and their outcomes in metadata.
    tag : str, optional
        File/provenance label. Defaults to the filename, or the export's source
        for a mapping. It never changes sample identity or RNA evidence scope.

    Returns
    -------
    TopiaryResult
        One comparison row per translation (one protein-only row when no
        translations were exported). Every hypothesis, including lower-ranked
        and filtered hypotheses, remains visible. ``isovar_rank`` records the
        producer's order and ``passes_all_filters`` its event-level outcome;
        neither field admits a row to prediction. No peptide, allele, score or
        candidate is generated, and existing fragment/CLI defaults are unchanged.

        ``protein_hypothesis_sequence`` retains the translated sequence even
        for a partial window. ``protein_sequence`` is populated only when that
        translation explicitly starts at the annotated start codon and the
        protein ends at a stop codon. This permits full-protein comparisons
        without promoting partial windows to full ORFs. The producer's sequence
        ID is ``isovar_protein_sequence_id``; ``combine_sources`` reserves the
        unprefixed name for its full-protein identity.

        The complete, independently copied export is retained in
        ``extra['isovar_hypotheses']``, including reference contexts, edits,
        filters, cap/completeness settings, empty events and evidence sets.
        Counts stay separate by protein and translation. To combine support,
        pass the referenced evidence sets to ``isovar.union_rna_support``;
        adding counts would double-count shared reads. Missing evidence IDs
        stay missing. Read/fragment counts are not expression or molecule counts.

    Raises
    ------
    ValueError
        The schema, required identities, intervals or evidence references are
        invalid. An unsupported schema is never interpreted as version 1.

    Notes
    -----
    Reading requires neither Isovar nor a predictor. Import does not rerank,
    reconstruct, apply filters, infer tumor specificity, or expand candidates.
    Use ``combine_sources`` to obtain a long-form result for the ranking DSL;
    the reader's native comparison table has no prediction axes.
    A truncated upstream export remains truncated; consult its completeness
    fields. Selecting alternatives for prediction is a separate explicit step.
    """
    if isinstance(data, Mapping):
        export = deepcopy(dict(data))
        label = export.get("source")
    else:
        path = Path(data)
        with path.open(encoding="utf-8") as handle:
            export = json.load(handle)
        label = path.name
    if not isinstance(export, dict) or export.get("schema") != _SCHEMA:
        raise ValueError(f"Expected {_SCHEMA} JSON export")
    if export.get("interval_convention") != "zero_based_half_open":
        raise ValueError("Isovar export requires zero_based_half_open intervals")
    sample = _named(export.get("sample_id"), "sample_id")
    source = _named(export.get("source"), "source")
    if export.get("evidence_scope") != [sample, source]:
        raise ValueError("Isovar evidence_scope must match sample_id and source")
    if not isinstance(export.get("events"), list) or not isinstance(export.get("evidence_sets"), dict):
        raise ValueError("Isovar export requires events and evidence_sets")
    rows, event_ids = [], set()
    try:
        for event in export["events"]:
            event_id = _named(event["event_id"], "event_id")
            if event_id in event_ids:
                raise ValueError(f"Duplicate Isovar event_id: {event_id!r}")
            event_ids.add(event_id)
            hypothesis_ids = set()
            for protein in event["protein_hypotheses"]:
                hypothesis_id = _named(protein["hypothesis_id"], "hypothesis_id")
                if hypothesis_id in hypothesis_ids:
                    raise ValueError(f"Duplicate Isovar hypothesis_id: {hypothesis_id!r}")
                hypothesis_ids.add(hypothesis_id)
                sequence = _named(protein["amino_acids"], "amino_acids")
                rank = protein["isovar_rank"]
                if type(rank) is not int or rank < 1:
                    raise ValueError("Isovar rank must be a positive integer")
                interval = protein["mutation_interval"]
                if (not isinstance(interval, list) or len(interval) != 2
                        or any(type(i) is not int for i in interval) or not 0 <= interval[0] <= interval[1] <= len(sequence)):
                    raise ValueError("Invalid Isovar mutation_interval")
                common = dict(
                    sample_name=sample, event_id=event_id, hypothesis_id=hypothesis_id,
                    isovar_protein_sequence_id=_named(protein["protein_sequence_id"], "protein_sequence_id"),
                    protein_hypothesis_sequence=sequence, isovar_rank=rank,
                    representative=protein["representative"], n_terminus=protein["n_terminus"],
                    c_terminus=protein["c_terminus"], mutation_start=interval[0], mutation_end=interval[1],
                    protein_hypotheses_complete=event["protein_hypotheses_complete"],
                    protein_sequence_limit=event["protein_sequence_limit"],
                    passes_all_filters=event["filters"]["passes_all_filters"], isovar_source=source,
                    **_support_columns(protein["rna_support"], "protein", export),
                )
                translation_ids = set()
                if not isinstance(protein["translations"], list):
                    raise ValueError("Isovar translations must be a list")
                for translation in protein["translations"] or [None]:
                    translation = translation or {}
                    if not isinstance(translation, dict):
                        raise ValueError("Isovar translation must be a mapping")
                    identity = translation.get("translation_id")
                    if translation:
                        _named(identity, "translation_id")
                        if identity in translation_ids:
                            raise ValueError(f"Duplicate Isovar translation_id: {identity!r}")
                        translation_ids.add(identity)
                    complete = (translation.get("starts_at_annotated_start_codon") is True
                                and protein["c_terminus"] == "stop_codon"
                                and protein["ends_with_stop_codon"] is True)
                    rows.append(dict(
                        common, translation_id=identity,
                        nucleotide_sequence_id=translation.get("nucleotide_sequence_id"),
                        nucleotide_sequence=translation.get("nucleotide_sequence"),
                        protein_sequence=sequence if complete else None,
                        **_support_columns(translation.get("rna_support"), "translation", export),
                    ))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed Isovar hypothesis export: {error}") from error
    return TopiaryResult(pd.DataFrame(rows, columns=_COLUMNS), sources=[tag or label],
                         extra={"isovar_hypotheses": export})
