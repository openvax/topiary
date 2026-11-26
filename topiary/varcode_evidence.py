"""Portable provenance from Varcode's native candidate wrappers."""

import hashlib
import json
from typing import Any

from serializable.helpers import to_json


def _compact_genomic_positions(positions):
    """Replace huge per-base genomic coordinate arrays with a deterministic digest."""
    payload = json.dumps(positions, separators=(",", ":")).encode("utf-8")
    return len(positions), hashlib.sha256(payload).hexdigest()


def _plain_record(value):
    """Keep type identity as data, without Serializable's reconstruction tags."""
    cls = value.pop("__class__", None)
    if cls is not None:
        if cls["__module__"] == "builtins" and cls["__name__"] in ("tuple", "set"):
            return value["__value__"]
        value["record_type"] = cls["__module__"] + "." + cls["__name__"]
    if isinstance(value.get("genomic_positions"), list):
        positions = value.pop("genomic_positions")
        count, sha256 = _compact_genomic_positions(positions)
        value["genomic_positions_count"] = count
        value["genomic_positions_sha256"] = sha256
    return value


def varcode_candidate_provenance(candidate):
    """Retain the native evidence for one candidate without copying its siblings.

    Parameters
    ----------
    candidate : varcode.EffectCandidate or varcode.RealizedEffectCandidate
        An original wrapper from ``varcode.iter_candidate_paths``.

    Returns
    -------
    dict
        JSON-compatible source and evidence. For ``varcode>=13.0.0``
        ``EffectCandidate`` wrappers from ``varcode.dna``, evidence includes
        ``mechanisms``, ``phase``, ``uncertainty``, baseline/mutant products
        (including unavailable sequences), and ``MutantTranscript`` junction
        or open-template metadata when present. ``RealizedEffectCandidate``
        wrappers from explicit ``annotator="transcript_model"`` runs remain
        supported. The effect itself is omitted because it can point back to
        all alternatives.

    Raises
    ------
    TypeError
        If the input is not a supported native candidate wrapper.
    """
    from varcode.effect_candidates import EffectCandidate
    from varcode.effect_hypotheses import RealizedEffectCandidate

    if isinstance(candidate, EffectCandidate):
        evidence = dict(candidate.evidence)
        mechanisms = getattr(candidate, "mechanisms", ()) or ()
        phase = getattr(candidate, "phase", ()) or ()
        uncertainty = getattr(candidate, "uncertainty", ()) or ()
        if mechanisms:
            evidence["mechanisms"] = list(mechanisms)
        if phase:
            evidence["phase"] = [[str(v), state] for v, state in phase]
        if uncertainty:
            evidence["uncertainty"] = list(uncertainty)
        if candidate.source == "varcode.dna":
            effect = candidate.effect
            model = getattr(effect, "mutant_transcript", None)
            paired = getattr(candidate, "_paired_outcomes", None)
            if paired:
                baseline_cdna = paired[0].baseline.cdna_sequence
                baseline_protein = paired[0].baseline.protein_sequence
            else:
                baseline_protein = getattr(effect, "original_protein_sequence", None)
                transcript = getattr(effect, "transcript", None)
                baseline_cdna = (
                    getattr(transcript, "sequence", None)
                    if transcript is not None and baseline_protein is not None
                    else None
                )
            outcome: dict[str, Any] = dict(
                baseline=dict(
                    cdna_sequence=baseline_cdna,
                    protein_sequence=baseline_protein,
                ),
                mutant=dict(
                    cdna_sequence=getattr(model, "cdna_sequence", None) if model is not None else None,
                    protein_sequence=getattr(effect, "mutant_protein_sequence", None),
                ),
            )
            if model is not None:
                is_complete = getattr(model, "is_complete", None)
                if is_complete is not None:
                    outcome["is_complete"] = is_complete
                novel_junctions = getattr(model, "novel_junctions", ()) or ()
                if novel_junctions:
                    outcome["novel_junctions"] = [list(j) for j in novel_junctions]
                back_junctions = getattr(model, "back_junctions", ()) or ()
                if back_junctions:
                    outcome["back_junctions"] = [list(j) for j in back_junctions]
                mate_region = getattr(model, "mate_region", None)
                if mate_region is not None:
                    outcome["mate_region"] = list(mate_region)
                exit_frame = getattr(model, "exit_frame", None)
                if exit_frame is not None:
                    outcome["exit_frame"] = exit_frame
            evidence["outcomes"] = [outcome]
        return json.loads(
            to_json(dict(source=candidate.source, evidence=evidence)),
            object_hook=_plain_record,
        )
    if not isinstance(candidate, RealizedEffectCandidate):
        raise TypeError(f"Unsupported Varcode candidate {type(candidate).__name__}")
    evidence = dict(
        ordinal_key=candidate.ordinal_key,
        probability=candidate.probability,
        phase_mass=candidate.phase_mass,
        first_enumeration_index=candidate.first_enumeration_index,
        outcomes=[dict(
            hypothesis=outcome.hypothesis,
            baseline=outcome.baseline,
            mutant=outcome.mutant,
            change_aa_offset=outcome.change_aa_offset,
            change_cdna_offset=outcome.change_cdna_offset,
        ) for outcome in candidate.outcomes],
    )
    return dict(source="varcode.transcript_model",
                evidence=json.loads(to_json(evidence), object_hook=_plain_record))
