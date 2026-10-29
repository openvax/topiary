"""Read Isovar's comparison export without selecting prediction candidates."""

from collections.abc import Mapping
from copy import deepcopy
import gzip
import json
from pathlib import Path
from os import PathLike
from typing import Any

import pandas as pd

from .result import TopiaryResult
from .isovar_rna_support import normalize_isovar_rna_support, RNA_SUPPORT_COUNTS, RNA_SUPPORT_FLAGS


_SCHEMAS = ("isovar.protein_hypotheses.v1", "isovar.protein_hypotheses.v2")
_PATIENT_BACKGROUND_SCHEMA = "isovar.patient_background_pairs.v1"
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
_PATIENT_BACKGROUND_COLUMNS = (
    "sample_name", "event_id", "pair_id", "hypothesis_id", "translation_id",
    "transcript_id", "baseline_policy", "pair_state", "pair_reason",
    "patient_background_protein_comparison", "strand",
    "translation_table", "protein_hypotheses_complete", "protein_sequence_limit",
    "phased_variants_in_supporting_reads", "isovar_variant", "isovar_source",
    "nucleotide_sequence_id", "nucleotide_sequence", "mutant_coding_interval",
    "protein_hypothesis_sequence", "mutant_ends_with_stop_codon", "mutant_phase",
    "mutant_phase_scope", "mutant_normal_tissue_observed",
    "mutant_translation_observed", "mutant_rna_support", "mutant_coding_support",
    "patient_background_nucleotide_sequence_id",
    "patient_background_nucleotide_sequence", "patient_background_coding_interval",
    "patient_background_available_coding_interval", "patient_background_sequence",
    "patient_background_ends_with_stop_codon",
    "patient_background_trailing_incomplete_codon_bases",
    "patient_background_construction", "patient_background_observation",
    "patient_background_reference_allele_phase",
    "patient_background_normal_tissue_observed",
    "patient_background_translation_observed",
    "patient_background_reference_allele_coding_support",
    "reference_nucleotide_sequence_id", "reference_nucleotide_sequence",
    "reference_cdna_interval", "removed_edits", "retained_edits",
    "mutant_segments", "mutant_fragments", "mutant_evidence_set_id",
    *(f"mutant_{field}" for field in _SUPPORT_FIELDS
      if field not in ("fragments", "evidence_set_id")),
)


def _load_export(data):
    """Independently load a mapping, JSON file or gzip-compressed JSON file."""
    if isinstance(data, Mapping):
        export = deepcopy(dict(data))
        return export, export.get("source")
    path = Path(data)
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    with handle:
        return json.load(handle), path.name


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


def _coding_support(value, field, sequence_length, export):
    """Validate one exact coding-window support record and its evidence links."""
    if not isinstance(value, Mapping):
        raise ValueError(f"Isovar {field} must be a mapping")
    result = deepcopy(dict(value))
    if result.get("scope") != "exact_observed_coding_interval":
        raise ValueError(f"Invalid Isovar {field} scope")
    interval = _interval(result.get("coding_interval"), f"{field} coding_interval",
                         sequence_length)
    depths = result.get("fragment_depth")
    if (not isinstance(depths, list) or len(depths) != interval[1] - interval[0]
            or any(type(depth) is not int or depth < 0 for depth in depths)):
        raise ValueError(f"Invalid Isovar {field} fragment_depth")
    minimum = result.get("minimum_fragment_depth")
    expected_minimum = min(depths) if depths else 0
    if type(minimum) is not int or minimum != expected_minimum:
        raise ValueError(f"Invalid Isovar {field} minimum_fragment_depth")
    tiled = result.get("fully_tiled")
    if type(tiled) is not bool or tiled != (bool(depths) and all(depths)):
        raise ValueError(f"Invalid Isovar {field} fully_tiled")
    for name in ("allele_observation_span_support", "single_read_span_support"):
        result[name] = normalize_isovar_rna_support(
            result.get(name), evidence_sets=export["evidence_sets"],
            evidence_scope=export["evidence_scope"])
    return result


def _coding_phase(support):
    """Return the producer's declared phase tier from validated support."""
    if support["single_read_span_support"]["reads"]:
        return "single_original_read_span"
    if support["allele_observation_span_support"]["reads"]:
        return "allele_observation_span"
    return "assembly_only" if support["fully_tiled"] else "incomplete_rna_coverage"


def _translation_window(nucleotide, interval, protein, stop, field, *, allow_empty=False):
    """Validate one bounded cDNA/protein translation without inferring sequence."""
    if not isinstance(nucleotide, str) or not nucleotide:
        raise ValueError(f"Isovar export requires a nonempty {field} nucleotide_sequence")
    interval = _interval(interval, f"{field} coding_interval", len(nucleotide))
    if not isinstance(protein, str) or (not protein and not allow_empty):
        raise ValueError(f"Invalid Isovar {field} protein_sequence")
    stop = _flag(stop, f"{field} ends_with_stop_codon")
    if stop is None or interval[1] - interval[0] != 3 * (len(protein) + int(stop)):
        raise ValueError(f"Inconsistent Isovar {field} translation length")
    return interval, stop


def _edit_records(value, field):
    """Copy positioned edit provenance while checking its coordinate shape."""
    if not isinstance(value, list):
        raise ValueError(f"Isovar {field} must be a list")
    result = []
    for edit in value:
        if not isinstance(edit, Mapping):
            raise ValueError(f"Isovar {field} entries must be mappings")
        copied = deepcopy(dict(edit))
        _named(copied.get("origin"), f"{field} origin")
        cdna_interval = _edit_interval(
            copied.get("cdna_interval"), f"{field} cdna_interval")
        observed_interval = _edit_interval(
            copied.get("observed_interval"), f"{field} observed_interval")
        for name in ("alt_bases", "reference_bases"):
            if not isinstance(copied.get(name), str):
                raise ValueError(f"Invalid Isovar {field} {name}")
        if (len(copied["reference_bases"]) != cdna_interval[1] - cdna_interval[0]
                or len(copied["alt_bases"]) != observed_interval[1] - observed_interval[0]):
            raise ValueError(f"Inconsistent Isovar {field} edit sequence lengths")
        result.append(copied)
    return result


def _edit_interval(value, field):
    """Validate a nonnegative half-open interval without a local length bound."""
    if (not isinstance(value, list) or len(value) != 2
            or any(type(i) is not int for i in value)
            or not 0 <= value[0] <= value[1]):
        raise ValueError(f"Invalid Isovar {field}")
    return value


def read_isovar_hypotheses(
    data: str | PathLike[str] | Mapping[str, Any], *, tag: str | None = None,
) -> TopiaryResult:
    """Import all exported Isovar hypotheses as comparison observations.

    Parameters
    ----------
    data : str, pathlib.Path or mapping
        Path to an ``isovar.protein_hypotheses.v1`` or ``v2`` JSON or JSON.gz
        export, or the mapping returned by ``isovar.export_protein_hypotheses``.
        Use the JSON export: its companion TSV omits evidence sets and other
        required provenance.
        An export with no hypotheses returns an empty table with the same
        columns, retaining its events and their outcomes in metadata.
    tag : str, optional
        File/provenance label. Defaults to the filename, or the export's source
        for a mapping. It never changes sample identity or RNA evidence scope.

    Returns
    -------
    TopiaryResult
        One comparison row per distinct translation identity within each
        hypothesis (one protein-only row when no translations were exported).
        Exact repeated records are coalesced; conflicting records sharing an
        identity are rejected. Every hypothesis, including lower-ranked
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
        filters, cap/completeness settings, empty events, evidence sets and
        original repeated translation records.
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
    export, label = _load_export(data)
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
                translation_records: dict[str, str] = {}
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
                        identity = _named(identity, "translation_id")
                        # Compare the complete wire record, including metadata
                        # not projected into columns. JSON keeps true distinct
                        # from 1 and makes mapping key order irrelevant.
                        record = json.dumps(translation, sort_keys=True, allow_nan=False)
                        if identity in translation_records:
                            if record != translation_records[identity]:
                                raise ValueError(f"Conflicting Isovar translation_id: {identity!r}")
                            continue
                        translation_records[identity] = record
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


def read_isovar_patient_background_pairs(
    data: str | PathLike[str] | Mapping[str, Any], *, tag: str | None = None,
) -> TopiaryResult:
    """Import Isovar's paired patient-background coding windows.

    Parameters
    ----------
    data : str, pathlib.Path or mapping
        Path to an ``isovar.patient_background_pairs.v1`` JSON or JSON.gz
        export, or the mapping returned by
        ``isovar.export_patient_background_pairs``. Empty exports return an
        empty table while retaining their events and evidence in metadata.
    tag : str, optional
        File/provenance label. Defaults to the filename, or the export's source
        for a mapping. It never changes sample identity or evidence scope.

    Returns
    -------
    TopiaryResult
        One comparison observation per exact Isovar pair identity. Mutant and
        patient-background cDNA/protein windows remain separately named.
        ``pair_state`` preserves paired, zero-length and unavailable outcomes;
        unavailable baselines stay null with ``pair_reason``.
        ``patient_background_protein_comparison`` classifies the exact bounded
        windows as ``identical``, ``amino_acid_divergent`` or ``unavailable``.
        Construction, observation and phase fields distinguish a computed
        counterfactual from separately spanning reference-allele RNA.

        These are bounded coding windows, not necessarily full proteins or
        aligned fixed-length peptides. The reader therefore does not populate
        ``protein_sequence``, construct fragments, map peptides or select a
        comparator policy. The complete independent export remains in
        ``extra['isovar_patient_background_pairs']``.

    Raises
    ------
    ValueError
        The schema, identities, states, intervals, translation lengths or RNA
        evidence references are invalid. Unsupported future schemas are never
        interpreted as version 1.

    Notes
    -----
    Reading requires neither Isovar nor a predictor. Counts in structured
    coding-support fields retain their evidence-set membership and must not be
    added when supports overlap. ``computed_only`` is not observed normal
    tissue, protein expression or immunogenicity.
    """
    export, label = _load_export(data)
    if (not isinstance(export, dict)
            or export.get("schema") != _PATIENT_BACKGROUND_SCHEMA):
        raise ValueError(f"Expected {_PATIENT_BACKGROUND_SCHEMA} JSON export")
    if export.get("interval_convention") != "zero_based_half_open":
        raise ValueError("Isovar export requires zero_based_half_open intervals")
    sample = _named(export.get("sample_id"), "sample_id")
    source = _named(export.get("source"), "source")
    if export.get("evidence_scope") != [sample, source]:
        raise ValueError("Isovar evidence_scope must match sample_id and source")
    policy = export.get("baseline_policy")
    if policy not in ("focal", "supplied_somatic_haplotype"):
        raise ValueError("Invalid Isovar patient-background baseline_policy")
    if (not isinstance(export.get("events"), list)
            or not isinstance(export.get("evidence_sets"), Mapping)
            or not isinstance(export.get("supplied_variants"), Mapping)):
        raise ValueError(
            "Isovar patient-background export requires events, evidence_sets "
            "and supplied_variants")

    rows, event_ids, pair_ids = [], set(), set()
    try:
        for event in export["events"]:
            if not isinstance(event, Mapping):
                raise ValueError("Isovar patient-background event must be a mapping")
            event_id = _named(event["event_id"], "event_id")
            if event_id in event_ids:
                raise ValueError(f"Duplicate Isovar event_id: {event_id!r}")
            event_ids.add(event_id)
            pairs = event["pairs"]
            if not isinstance(pairs, list):
                raise ValueError("Isovar patient-background pairs must be a list")
            expected_event_state = "retained" if pairs else "no_protein_hypotheses"
            if event["state"] != expected_event_state:
                raise ValueError("Inconsistent Isovar patient-background event state")
            complete = _flag(
                event["protein_hypotheses_complete"], "protein_hypotheses_complete")
            limit = event["protein_sequence_limit"]
            if limit is not None and (type(limit) is not int or limit < 1):
                raise ValueError("Isovar protein_sequence_limit must be positive or null")
            phased = event["phased_variants_in_supporting_reads"]
            if (not isinstance(phased, list)
                    or any(not isinstance(value, str) or not value for value in phased)
                    or len(phased) != len(set(phased))):
                raise ValueError("Invalid Isovar phased_variants_in_supporting_reads")
            variant = event["variant"]
            if not isinstance(variant, Mapping):
                raise ValueError("Isovar patient-background variant must be a mapping")

            for pair in pairs:
                if not isinstance(pair, Mapping):
                    raise ValueError("Isovar patient-background pair must be a mapping")
                pair_id = _named(pair["pair_id"], "pair_id")
                if pair_id in pair_ids:
                    raise ValueError(f"Duplicate Isovar pair_id: {pair_id!r}")
                pair_ids.add(pair_id)
                if pair["baseline_policy"] != policy:
                    raise ValueError("Inconsistent Isovar patient-background baseline_policy")
                state = pair["state"]
                if state not in ("paired", "zero_length_baseline", "unavailable"):
                    raise ValueError("Invalid Isovar patient-background pair state")
                reason, baseline = pair["reason"], pair["baseline"]
                if state == "unavailable":
                    _named(reason, "unavailable reason")
                    if baseline is not None:
                        raise ValueError("Unavailable Isovar pair cannot have a baseline")
                elif reason is not None or not isinstance(baseline, Mapping):
                    raise ValueError("Available Isovar pair requires a baseline and no reason")

                strand = pair["strand"]
                table = pair["translation_table"]
                if strand not in ("+", "-") or type(table) is not int or table < 1:
                    raise ValueError("Invalid Isovar strand or translation_table")
                mutant = pair["mutant"]
                if not isinstance(mutant, Mapping):
                    raise ValueError("Isovar patient-background mutant must be a mapping")
                mutant_nucleotide = _named(
                    mutant["nucleotide_sequence"], "mutant nucleotide_sequence")
                mutant_interval, mutant_stop = _translation_window(
                    mutant_nucleotide, mutant["coding_interval"],
                    mutant["protein_sequence"], mutant["ends_with_stop_codon"], "mutant")
                mutant_support = normalize_isovar_rna_support(
                    mutant["rna_support"], evidence_sets=export["evidence_sets"],
                    evidence_scope=export["evidence_scope"])
                mutant_coding = _coding_support(
                    mutant["coding_support"], "mutant coding_support",
                    len(mutant_nucleotide), export)
                if mutant_coding["coding_interval"] != mutant_interval:
                    raise ValueError("Inconsistent Isovar mutant coding intervals")
                phase = pair["phase"]
                if (not isinstance(phase, Mapping)
                        or phase.get("state") not in (
                            "single_original_read_span", "allele_observation_span",
                            "assembly_only", "incomplete_rna_coverage")
                        or phase.get("scope") != "the_mutant_coding_window"
                        or phase.get("normal_tissue_observed") is not False):
                    raise ValueError("Invalid Isovar patient-background mutant phase")
                if phase["state"] != _coding_phase(mutant_coding):
                    raise ValueError("Inconsistent Isovar patient-background mutant phase")
                if pair["translation_observed"] is not False:
                    raise ValueError("Isovar mutant translation_observed must be false")

                reference = pair["reference"]
                reference_values = dict(
                    reference_nucleotide_sequence_id=None,
                    reference_nucleotide_sequence=None,
                    reference_cdna_interval=None,
                )
                if reference is not None:
                    if not isinstance(reference, Mapping):
                        raise ValueError("Isovar patient-background reference must be a mapping")
                    if reference["transcript_id"] != pair["transcript_id"]:
                        raise ValueError("Inconsistent Isovar reference transcript_id")
                    reference_nucleotide = _named(
                        reference["nucleotide_sequence"], "reference nucleotide_sequence")
                    reference_interval = _edit_interval(
                        reference["cdna_interval"], "reference cdna_interval")
                    if reference_interval[1] - reference_interval[0] != len(reference_nucleotide):
                        raise ValueError("Inconsistent Isovar reference cDNA length")
                    reference_values.update(
                        reference_nucleotide_sequence_id=_named(
                            reference["sequence_id"], "reference sequence_id"),
                        reference_nucleotide_sequence=reference_nucleotide,
                        # Transcript coordinates are absolute and need not start at zero.
                        reference_cdna_interval=reference_interval,
                    )

                background_values: dict[str, Any] = dict(
                    patient_background_nucleotide_sequence_id=None,
                    patient_background_nucleotide_sequence=None,
                    patient_background_coding_interval=None,
                    patient_background_available_coding_interval=None,
                    patient_background_sequence=None,
                    patient_background_ends_with_stop_codon=None,
                    patient_background_trailing_incomplete_codon_bases=None,
                    patient_background_construction=None,
                    patient_background_observation=None,
                    patient_background_reference_allele_phase=None,
                    patient_background_normal_tissue_observed=None,
                    patient_background_translation_observed=None,
                    patient_background_reference_allele_coding_support=None,
                )
                if baseline is not None:
                    background_nucleotide = _named(
                        baseline["nucleotide_sequence"],
                        "patient-background nucleotide_sequence")
                    background_interval, background_stop = _translation_window(
                        background_nucleotide, baseline["coding_interval"],
                        baseline["protein_sequence"], baseline["ends_with_stop_codon"],
                        "patient-background", allow_empty=state == "zero_length_baseline")
                    if ((state == "zero_length_baseline")
                            != (baseline["protein_sequence"] == "")):
                        raise ValueError("Inconsistent Isovar zero-length baseline state")
                    available = _interval(
                        baseline["available_coding_interval"],
                        "patient-background available_coding_interval",
                        len(background_nucleotide))
                    if (available[0] != background_interval[0]
                            or background_interval[1] > available[1]):
                        raise ValueError(
                            "Inconsistent Isovar patient-background available interval")
                    trailing = baseline["trailing_incomplete_codon_bases"]
                    if type(trailing) is not int or trailing not in (0, 1, 2):
                        raise ValueError(
                            "Invalid Isovar patient-background incomplete codon count")
                    if background_stop and trailing:
                        raise ValueError(
                            "Stopped Isovar patient-background cannot have an incomplete codon")
                    observation = baseline["observation"]
                    background_phase = baseline["reference_allele_phase"]
                    observed_phases = (
                        "single_original_read_span", "allele_observation_span")
                    if (baseline["construction"] != "transcript_cdna_somatic_reversion"
                            or observation not in (
                                "computed_only", "reference_allele_rna_span")
                            or background_phase not in (
                                *observed_phases, "assembly_only",
                                "incomplete_rna_coverage")
                            or (observation == "reference_allele_rna_span")
                            != (background_phase in observed_phases)
                            or baseline["normal_tissue_observed"] is not False
                            or baseline["translation_observed"] is not False):
                        raise ValueError("Invalid Isovar patient-background evidence state")
                    background_coding = _coding_support(
                        baseline["reference_allele_coding_support"],
                        "patient-background reference-allele coding_support",
                        len(background_nucleotide), export)
                    if background_coding["coding_interval"] != background_interval:
                        raise ValueError(
                            "Inconsistent Isovar patient-background coding intervals")
                    if background_phase != _coding_phase(background_coding):
                        raise ValueError(
                            "Inconsistent Isovar patient-background reference-allele phase")
                    background_values.update(
                        patient_background_nucleotide_sequence_id=_named(
                            baseline["nucleotide_sequence_id"],
                            "patient-background nucleotide_sequence_id"),
                        patient_background_nucleotide_sequence=background_nucleotide,
                        patient_background_coding_interval=background_interval,
                        patient_background_available_coding_interval=available,
                        patient_background_sequence=baseline["protein_sequence"],
                        patient_background_ends_with_stop_codon=background_stop,
                        patient_background_trailing_incomplete_codon_bases=trailing,
                        patient_background_construction=baseline["construction"],
                        patient_background_observation=observation,
                        patient_background_reference_allele_phase=background_phase,
                        patient_background_normal_tissue_observed=False,
                        patient_background_translation_observed=False,
                        patient_background_reference_allele_coding_support=background_coding,
                    )

                removed = _edit_records(pair["removed_edits"], "removed_edits")
                retained = _edit_records(pair["retained_edits"], "retained_edits")
                allowed_origins = {
                    "nominated_variant", "co_somatic_variant",
                    "germline_variant", "unexplained",
                }
                if any(edit["origin"] not in allowed_origins
                       for edit in (*removed, *retained)):
                    raise ValueError("Invalid Isovar patient-background edit origin")
                for edit in (*removed, *retained):
                    if edit["observed_interval"][1] > len(mutant_nucleotide):
                        raise ValueError("Isovar edit lies outside mutant nucleotide sequence")
                    if (reference is not None
                            and not (reference_interval[0] <= edit["cdna_interval"][0]
                                     <= edit["cdna_interval"][1]
                                     <= reference_interval[1])):
                        raise ValueError("Isovar edit lies outside reference cDNA interval")
                if reference is None:
                    if baseline is not None or removed or retained:
                        raise ValueError(
                            "Isovar baseline and positioned edits require a reference window")
                else:
                    cursor = reference_interval[0]
                    replayed = ""
                    positioned = sorted((*removed, *retained),
                                        key=lambda edit: edit["cdna_interval"])
                    for edit in positioned:
                        begin, end = edit["cdna_interval"]
                        if begin < cursor:
                            raise ValueError("Overlapping Isovar transcript edits")
                        replayed += reference_nucleotide[
                            cursor - reference_interval[0]:begin - reference_interval[0]]
                        replayed += edit["alt_bases"]
                        cursor = end
                    replayed += reference_nucleotide[cursor - reference_interval[0]:]
                    if replayed != mutant_nucleotide:
                        raise ValueError(
                            "Isovar transcript edits do not reproduce mutant nucleotide sequence")
                if baseline is not None:
                    if sum(edit["origin"] == "nominated_variant" for edit in removed) != 1:
                        raise ValueError("Isovar baseline must remove one nominated variant")
                    removable = ({"nominated_variant"} if policy == "focal" else
                                 {"nominated_variant", "co_somatic_variant"})
                    if (any(edit["origin"] not in removable for edit in removed)
                            or any(edit["origin"] in removable for edit in retained)):
                        raise ValueError("Inconsistent Isovar removed and retained edits")
                    reverted = mutant_nucleotide
                    for edit in reversed(removed):
                        begin, end = edit["observed_interval"]
                        reverted = (reverted[:begin] + edit["reference_bases"]
                                    + reverted[end:])
                    if reverted != background_nucleotide:
                        raise ValueError(
                            "Isovar removed edits do not reproduce patient-background sequence")

                if baseline is None:
                    protein_comparison = "unavailable"
                elif mutant["protein_sequence"] == baseline["protein_sequence"]:
                    protein_comparison = "identical"
                else:
                    protein_comparison = "amino_acid_divergent"

                rows.append(dict(
                    sample_name=sample, event_id=event_id, pair_id=pair_id,
                    hypothesis_id=_named(pair["hypothesis_id"], "hypothesis_id"),
                    translation_id=_named(pair["translation_id"], "translation_id"),
                    transcript_id=_named(pair["transcript_id"], "transcript_id"),
                    baseline_policy=policy, pair_state=state, pair_reason=reason,
                    patient_background_protein_comparison=protein_comparison,
                    strand=strand, translation_table=table,
                    protein_hypotheses_complete=complete, protein_sequence_limit=limit,
                    phased_variants_in_supporting_reads=deepcopy(phased),
                    isovar_variant=deepcopy(dict(variant)), isovar_source=source,
                    nucleotide_sequence_id=_named(
                        mutant["nucleotide_sequence_id"], "mutant nucleotide_sequence_id"),
                    nucleotide_sequence=mutant_nucleotide,
                    mutant_coding_interval=mutant_interval,
                    protein_hypothesis_sequence=mutant["protein_sequence"],
                    mutant_ends_with_stop_codon=mutant_stop,
                    mutant_phase=phase["state"], mutant_phase_scope=phase["scope"],
                    mutant_normal_tissue_observed=False,
                    mutant_translation_observed=False,
                    mutant_rna_support=mutant_support,
                    mutant_coding_support=mutant_coding,
                    removed_edits=removed, retained_edits=retained,
                    **reference_values, **background_values,
                    **_support_columns(mutant_support, "mutant", export),
                ))
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed Isovar patient-background export: {error}") from error

    return TopiaryResult(
        pd.DataFrame(rows, columns=_PATIENT_BACKGROUND_COLUMNS),
        sources=[tag or label],
        extra={"isovar_patient_background_pairs": export},
    )
