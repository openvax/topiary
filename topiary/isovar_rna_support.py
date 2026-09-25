"""Validate Isovar RNA measurements without importing the optional producer."""

from collections.abc import Mapping
from copy import deepcopy


RNA_SUPPORT_COUNTS = ("reads", "fragments", "umis", "cells", "unlabeled_reads", "unknown_library_reads")
RNA_SUPPORT_FLAGS = ("umis_complete", "cells_complete")


def normalize_isovar_rna_support(support, *, evidence_sets=None, evidence_scope=None):
    """Return an independently copied Isovar support record with read fields.

    Parameters
    ----------
    support : mapping or None
        A protein, translation, allele or SV RNA support record. Legacy
        ``segments`` and ``segment_ids`` become ``reads`` and ``read_ids``.
        Other fields, including lineage and label-resolution details, survive.
        None or an empty mapping returns unknown measurements, never zeros.
    evidence_sets : mapping, optional
        A protein-hypothesis export's evidence-set table. A non-null reference
        must resolve here; its counts, scope and memberships must agree with
        the support. Referenced memberships are included in the returned copy.
    evidence_scope : list of str, optional
        Expected ``[sample_id, source]``. An inline or referenced evidence set
        must explicitly match this scope. Count-only records need no scope.

    Returns
    -------
    dict
        Canonical read/fragment counts, optional UMI/cell counts and boolean
        completeness flags, and the supplied provenance. Missing measurements
        remain None. Empty identified sets remain empty identified sets; absent
        identities are never fabricated. Label counts are not molecule counts.

    Raises
    ------
    ValueError
        Counts, flags, aliases, memberships or evidence references disagree.
        This validates supplied identities, not their cryptographic derivation.

    Notes
    -----
    This function does not select hypotheses or combine counts. UMI/cell counts
    cannot be added across overlapping supports without their label identities.
    """
    if support is None:
        support = {}
    if not isinstance(support, Mapping):
        raise ValueError("Isovar RNA support must be a mapping")
    result = deepcopy(dict(support))
    if "evidence_scope" in result:
        scope = result["evidence_scope"]
        if (not isinstance(scope, list) or len(scope) != 2
                or any(not isinstance(part, str) or not part.strip() for part in scope)):
            raise ValueError("Isovar evidence_scope requires [sample_id, source]")
    for old, new in (("segments", "reads"), ("segment_ids", "read_ids")):
        if old in result:
            if new in result and result[new] != result[old]:
                raise ValueError(f"Conflicting Isovar {old}/{new} aliases")
            result[new] = result.pop(old)
    for field in RNA_SUPPORT_COUNTS:
        count = result.setdefault(field, None)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError(f"Isovar {field} count must be a nonnegative integer or null")
    reads = result["reads"]
    for field in RNA_SUPPORT_COUNTS[1:]:
        if reads is not None and result[field] is not None and result[field] > reads:
            raise ValueError(f"Isovar {field} count exceeds reads")
    for field in RNA_SUPPORT_FLAGS:
        flag = result.setdefault(field, None)
        if flag is not None and type(flag) is not bool:
            raise ValueError(f"Isovar {field} must be boolean or null")
        if flag is True and (not reads or result[field.removesuffix("_complete")] is None
                             or result["unknown_library_reads"] not in (None, 0)):
            raise ValueError(f"Inconsistent Isovar {field}")
    if result["umis_complete"] is True and result["unlabeled_reads"] not in (None, 0):
        raise ValueError("Inconsistent Isovar umis_complete")
    statuses = result.setdefault("label_statuses", None)
    if statuses is not None:
        if (not isinstance(statuses, Mapping)
                or any(not isinstance(k, str) or not k or type(v) is not int or v < 0
                       for k, v in statuses.items())
                or (reads is not None and sum(statuses.values()) != reads)):
            raise ValueError("Inconsistent Isovar label_statuses counts")
    key = result.get("evidence_set_id")
    if key is not None and (not isinstance(key, str) or not key):
        raise ValueError("Isovar evidence_set_id must be a nonempty string or null")
    if key is not None and evidence_sets is not None:
        if not isinstance(evidence_sets, Mapping) or key not in evidence_sets:
            raise ValueError(f"Unknown Isovar evidence_set_id: {key!r}")
        evidence = normalize_isovar_rna_support(evidence_sets[key], evidence_scope=evidence_scope)
        if evidence.get("evidence_set_id") != key or "evidence_scope" not in evidence:
            raise ValueError(f"Inconsistent Isovar evidence identity or scope: {key!r}")
        for field, unit in (("read_ids", "reads"), ("fragment_ids", "fragments")):
            if evidence.get(field) is None:
                raise ValueError(f"Missing Isovar {unit} membership for {key!r}")
            if result[unit] != evidence[unit]:
                raise ValueError(f"Inconsistent Isovar {unit} count for {key!r}")
        for field in ("read_ids", "fragment_ids", "evidence_scope"):
            if field in result and result[field] != evidence.get(field):
                raise ValueError(f"Inconsistent Isovar {field} for {key!r}")
            result[field] = evidence[field]
    has_membership = any(field in result for field in ("read_ids", "fragment_ids"))
    if evidence_scope is not None and (has_membership or "evidence_scope" in result):
        if result.get("evidence_scope") != evidence_scope:
            raise ValueError("Inconsistent Isovar evidence identity or scope")
    for field, unit in (("read_ids", "reads"), ("fragment_ids", "fragments")):
        if field not in result:
            continue
        identities = result[field]
        if (not isinstance(identities, list) or any(not isinstance(i, str) or not i for i in identities)
                or len(set(identities)) != len(identities) or len(identities) != result[unit]):
            raise ValueError(f"Inconsistent Isovar {unit} count or membership")
    return result
