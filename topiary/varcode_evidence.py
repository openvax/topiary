"""Portable provenance from Varcode's native candidate wrappers."""

import hashlib
import json

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
    candidate : varcode.EffectCandidate, varcode.RealizedEffectCandidate, or varcode.DNAHypothesis
        An original wrapper from ``varcode.iter_candidate_paths``.

    Returns
    -------
    dict
        JSON-compatible source and evidence. Realized and DNA-hypothesis
        evidence includes baseline/mutant products (including unavailable
        sequences), ordinal preference, and probability. The effect is omitted
        because it can point back to all alternatives.

    Raises
    ------
    TypeError
        If the input is not a supported native candidate wrapper.
    """
    import varcode
    from varcode.effect_candidates import EffectCandidate
    from varcode.effect_hypotheses import RealizedEffectCandidate

    dna_hypothesis_cls = getattr(varcode, "DNAHypothesis", None)

    if isinstance(candidate, EffectCandidate):
        return json.loads(to_json(dict(source=candidate.source, evidence=dict(candidate.evidence))),
                          object_hook=_plain_record)
    if dna_hypothesis_cls is not None and isinstance(candidate, dna_hypothesis_cls):
        outcome = dict(
            baseline=dict(
                cdna_sequence=candidate.baseline_cdna,
                protein_sequence=candidate.baseline_protein,
            ),
            mutant=dict(
                cdna_sequence=candidate.mutant_cdna,
                protein_sequence=candidate.mutant_protein,
            ),
            uncertainty=candidate.uncertainty,
            splice_mechanisms=candidate.splice_mechanisms,
            phase=candidate.phase,
            novel_junctions=candidate.novel_junctions,
        )
        evidence = dict(
            ordinal_key=candidate.ordinal_rank,
            probability=candidate.probability,
            outcomes=[outcome],
            **dict(candidate.evidence),
        )
        return dict(
            source=candidate.source,
            evidence=json.loads(to_json(evidence), object_hook=_plain_record),
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
