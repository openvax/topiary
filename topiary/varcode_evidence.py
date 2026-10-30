"""Portable provenance from Varcode's native candidate wrappers."""

import json

from serializable.helpers import to_json


def _plain_record(value):
    """Keep type identity as data, without Serializable's reconstruction tags."""
    cls = value.pop("__class__", None)
    if cls is None:
        return value
    if cls["__module__"] == "builtins" and cls["__name__"] in ("tuple", "set"):
        return value["__value__"]
    value["record_type"] = cls["__module__"] + "." + cls["__name__"]
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
        JSON-compatible source and evidence. Realized evidence includes every
        hypothesis and full baseline/mutant product, including unavailable
        sequences. Ordinal preference, phase mass and probability remain separate.
        The effect is omitted because it can point back to all alternatives.

    Raises
    ------
    TypeError
        If the input is not a supported native candidate wrapper.
    """
    from varcode.effect_candidates import EffectCandidate
    from varcode.effect_hypotheses import RealizedEffectCandidate

    if isinstance(candidate, EffectCandidate):
        return json.loads(to_json(dict(source=candidate.source, evidence=dict(candidate.evidence))),
                          object_hook=_plain_record)
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
