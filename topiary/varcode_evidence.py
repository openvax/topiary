"""Portable provenance from Varcode's native candidate wrappers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from serializable.helpers import to_json

_MISSING = object()


def _compact_genomic_positions(positions: list[Any]) -> tuple[int, str]:
    """Replace huge per-base genomic coordinate arrays with a deterministic digest."""
    payload = json.dumps(positions, separators=(",", ":")).encode("utf-8")
    return len(positions), hashlib.sha256(payload).hexdigest()


def _plain_record(value: dict[str, Any]) -> Any:
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


def _safe_effect_attr(effect: Any, name: str, default: Any = None) -> Any:
    """Read an optional effect attribute without failing on transcriptless effects."""
    try:
        return getattr(effect, name)
    except (AttributeError, ValueError):
        return default


def _transcript_product_provenance(
    model: Any,
    fallback_protein: str | None = None,
) -> dict[str, Any]:
    """Extract JSON-ready sequences and MutantTranscript metadata without cycles."""
    cdna_sequence = getattr(model, "cdna_sequence", None) if model is not None else None
    protein_sequence = (
        getattr(model, "mutant_protein_sequence", None)
        if model is not None
        else None
    )
    if protein_sequence is None:
        protein_sequence = fallback_protein
    product: dict[str, Any] = {
        "cdna_sequence": cdna_sequence,
        "protein_sequence": protein_sequence,
    }
    if model is not None:
        product["is_complete"] = bool(
            getattr(model, "is_complete", cdna_sequence is not None)
        )
        is_structural = getattr(model, "is_structural", None)
        if is_structural is not None:
            product["is_structural"] = bool(is_structural)
        annotator_name = getattr(model, "annotator_name", None)
        if annotator_name is not None:
            product["annotator_name"] = annotator_name
        product["novel_junctions"] = getattr(model, "novel_junctions", ())
        product["back_junctions"] = getattr(model, "back_junctions", ())
        product["wildtype_junctions"] = getattr(model, "wildtype_junctions", ())
        product["mate_region"] = getattr(model, "mate_region", None)
        product["exit_frame"] = getattr(model, "exit_frame", None)
        model_evidence = getattr(model, "evidence", None)
        if model_evidence:
            product["evidence"] = dict(model_evidence)
    return product


def varcode_candidate_provenance(candidate: Any) -> dict[str, Any]:
    """Retain the native evidence for one candidate without copying its siblings.

    Parameters
    ----------
    candidate : varcode.EffectCandidate or varcode.RealizedEffectCandidate
        An original wrapper from ``varcode.iter_candidate_paths``.

    Returns
    -------
    dict
        JSON-compatible provenance for one candidate. For Varcode 13+
        ``EffectCandidate`` wrappers (default ``"dna"`` annotator and external
        candidates), this retains ``source``, ``mechanisms``, ``phase``,
        ``uncertainty``, optional ``ordinal_key`` / ``probability``,
        ``baseline`` and ``mutant`` sequences and ``MutantTranscript``
        junction/template metadata (``is_complete``, ``novel_junctions``,
        ``back_junctions``, ``wildtype_junctions``, ``mate_region``,
        ``exit_frame``, and compacted evidence), plus ``evidence``. For
        explicit ``annotator="transcript_model"`` ``RealizedEffectCandidate``
        wrappers, this retains every hypothesis, full baseline/mutant product,
        ordinal preference, phase mass, and probability. The ``effect`` itself
        is omitted because it can point back to all alternatives.

    Raises
    ------
    TypeError
        If the input is not a supported native candidate wrapper.
    """
    from varcode.effect_candidates import EffectCandidate
    from varcode.effect_hypotheses import RealizedEffectCandidate

    if isinstance(candidate, EffectCandidate):
        effect = candidate.effect
        baseline_model = _safe_effect_attr(effect, "baseline_transcript")
        mutant_model = _safe_effect_attr(effect, "mutant_transcript")
        baseline_protein = _safe_effect_attr(effect, "original_protein_sequence")
        mutant_protein = _safe_effect_attr(effect, "mutant_protein_sequence")
        record: dict[str, Any] = {
            "source": candidate.source,
            "mechanisms": candidate.mechanisms,
            "phase": candidate.phase,
            "uncertainty": candidate.uncertainty,
            "baseline": _transcript_product_provenance(
                baseline_model,
                fallback_protein=baseline_protein,
            ),
            "mutant": _transcript_product_provenance(
                mutant_model,
                fallback_protein=mutant_protein,
            ),
            "evidence": dict(candidate.evidence),
        }
        ordinal_key = getattr(candidate, "ordinal_key", _MISSING)
        if ordinal_key is _MISSING and effect is not None:
            ordinal_key = getattr(effect, "mechanism_rank", _MISSING)
        if ordinal_key is not _MISSING:
            record["ordinal_key"] = ordinal_key

        probability = getattr(candidate, "probability", _MISSING)
        if probability is _MISSING and effect is not None:
            probability = getattr(effect, "probability", _MISSING)
        if probability is not _MISSING:
            record["probability"] = probability

        result: dict[str, Any] = json.loads(
            to_json(record),
            object_hook=_plain_record,
        )
        return result

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
    return dict(
        source="varcode.transcript_model",
        evidence=json.loads(to_json(evidence), object_hook=_plain_record),
    )
