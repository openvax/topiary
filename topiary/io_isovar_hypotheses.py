"""Read Isovar's comparison export without selecting prediction candidates."""

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import pandas as pd

from .result import TopiaryResult
from .isovar_rna_support import normalize_isovar_rna_support, RNA_SUPPORT_COUNTS, RNA_SUPPORT_FLAGS


_SCHEMAS = ("isovar.protein_hypotheses.v1", "isovar.protein_hypotheses.v2")
_SUPPORT_FIELDS = (*RNA_SUPPORT_COUNTS, *RNA_SUPPORT_FLAGS, "evidence_set_id")
_COLUMNS = (
    "sample_name", "event_id", "hypothesis_id", "translation_id",
    "isovar_protein_sequence_id", "nucleotide_sequence_id", "nucleotide_sequence",
    "protein_hypothesis_sequence", "protein_sequence", "isovar_rank", "representative",
    "n_terminus", "c_terminus", "mutation_start", "mutation_end",
    "protein_hypotheses_complete", "protein_sequence_limit", "passes_all_filters",
    "isovar_source", "protein_segments", "protein_fragments", "protein_evidence_set_id",
    "translation_segments", "translation_fragments", "translation_evidence_set_id",
    *(f"{prefix}_{field}" for prefix in ("protein", "translation")
      for field in _SUPPORT_FIELDS if field not in ("fragments", "evidence_set_id")),
)


def _named(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Isovar export requires a nonempty {field}")
    return value


def _support_columns(support, prefix, export):
    """Project one wire-format measurement, checking its evidence reference."""
    support = normalize_isovar_rna_support(
        support, evidence_sets=export["evidence_sets"], evidence_scope=export["evidence_scope"])
    return {f"{prefix}_{field}": support.get(field) for field in _SUPPORT_FIELDS} | {
        f"{prefix}_segments": support["reads"]}  # Retain the v1 reader's column alias.


def _interval(value, field, length):
    """Validate a required half-open interval in the exchange format."""
    if (not isinstance(value, list) or len(value) != 2
            or any(type(i) is not int for i in value) or not 0 <= value[0] <= value[1] <= length):
        raise ValueError(f"Invalid Isovar {field}")
    return value


def _flag(value, field):
    """Preserve unknown wire-format flags without accepting truthy strings."""
    if value is not None and type(value) is not bool:
        raise ValueError(f"Isovar {field} must be boolean or null")
    return value


def read_isovar_hypotheses(data, *, tag=None):
    """Import all exported Isovar hypotheses as comparison observations.

    Parameters
    ----------
    data : str, pathlib.Path or mapping
        Path to an ``isovar.protein_hypotheses.v1`` or ``v2`` JSON export, or the mapping
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
        normalize legacy evidence sets with ``normalize_isovar_rna_support``
        before passing them to Isovar 1.37+'s ``union_rna_support``;
        adding counts would double-count shared reads. Missing evidence IDs
        stay missing. Read/fragment counts are not expression or molecule counts.
        Both versions populate ``protein_reads`` and ``translation_reads``;
        the older ``*_segments`` columns remain aliases. Optional ``*_umis``,
        ``*_cells`` and ``*_complete`` fields retain unknown versus measured
        values. Label-resolution details remain in the original export.

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
    if not isinstance(export, dict) or export.get("schema") not in _SCHEMAS:
        raise ValueError("Expected isovar.protein_hypotheses.v1 or v2 JSON export")
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
            for allele in ("ref", "alt", "other", "total"):
                normalize_isovar_rna_support(event.get("allele_support", {}).get(allele),
                                            evidence_sets=export["evidence_sets"],
                                            evidence_scope=export["evidence_scope"])
            complete = _flag(event["protein_hypotheses_complete"], "protein_hypotheses_complete")
            passing = _flag(event["filters"]["passes_all_filters"], "passes_all_filters")
            hypothesis_ids = set()
            if not isinstance(event["protein_hypotheses"], list):
                raise ValueError("Isovar protein_hypotheses must be a list")
            for protein in event["protein_hypotheses"]:
                hypothesis_id = _named(protein["hypothesis_id"], "hypothesis_id")
                if hypothesis_id in hypothesis_ids:
                    raise ValueError(f"Duplicate Isovar hypothesis_id: {hypothesis_id!r}")
                hypothesis_ids.add(hypothesis_id)
                sequence = _named(protein["amino_acids"], "amino_acids")
                rank = protein["isovar_rank"]
                if type(rank) is not int or rank < 1:
                    raise ValueError("Isovar rank must be a positive integer")
                interval = _interval(protein["mutation_interval"], "mutation_interval", len(sequence))
                common = dict(
                    sample_name=sample, event_id=event_id, hypothesis_id=hypothesis_id,
                    isovar_protein_sequence_id=_named(protein["protein_sequence_id"], "protein_sequence_id"),
                    protein_hypothesis_sequence=sequence, isovar_rank=rank,
                    representative=_flag(protein["representative"], "representative"), n_terminus=protein["n_terminus"],
                    c_terminus=protein["c_terminus"], mutation_start=interval[0], mutation_end=interval[1],
                    protein_hypotheses_complete=complete,
                    protein_sequence_limit=event["protein_sequence_limit"],
                    passes_all_filters=passing, isovar_source=source,
                    **_support_columns(protein["rna_support"], "protein", export),
                )
                translation_ids = set()
                stop = _flag(protein["ends_with_stop_codon"], "ends_with_stop_codon")
                if not isinstance(protein["translations"], list):
                    raise ValueError("Isovar translations must be a list")
                # Only an empty list represents a protein-only hypothesis.
                # A malformed element must never become an anonymous row.
                translations = protein["translations"]
                for translation in translations or [{}]:
                    if not isinstance(translation, dict) or (translations and not translation):
                        raise ValueError("Isovar translation must be a nonempty mapping")
                    identity = translation.get("translation_id")
                    if translations:
                        _named(identity, "translation_id")
                        if identity in translation_ids:
                            raise ValueError(f"Duplicate Isovar translation_id: {identity!r}")
                        translation_ids.add(identity)
                        _named(translation["nucleotide_sequence_id"], "nucleotide_sequence_id")
                        nucleotide = _named(translation["nucleotide_sequence"], "nucleotide_sequence")
                        for field in ("translated_interval", "variant_cdna_interval"):
                            _interval(translation[field], field, len(nucleotide))
                        _flag(translation["starts_at_annotated_start_codon"], "starts_at_annotated_start_codon")
                    full_protein = (translation.get("starts_at_annotated_start_codon") is True
                                    and protein["c_terminus"] == "stop_codon" and stop is True)
                    rows.append(dict(
                        common, translation_id=identity,
                        nucleotide_sequence_id=translation.get("nucleotide_sequence_id"),
                        nucleotide_sequence=translation.get("nucleotide_sequence"),
                        protein_sequence=sequence if full_protein else None,
                        **_support_columns(translation.get("rna_support"), "translation", export),
                    ))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed Isovar hypothesis export: {error}") from error
    return TopiaryResult(pd.DataFrame(rows, columns=_COLUMNS), sources=[tag or label],
                         extra={"isovar_hypotheses": export})
