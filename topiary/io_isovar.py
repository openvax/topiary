"""isovar → :class:`ProteinFragment`.

isovar assembles a mutant protein sequence from RNA reads rather than
translating one from the reference, and counts the reads that support it.
That makes it the only source topiary reads that can populate the read
counts natively — everything else derives or approximates them.

**isovar is optional in the strong sense**: not imported at module scope,
not in ``requirements.txt``, and topiary's shape is identical whether or
not it is installed. Only this module's functions need it, and they say
so clearly when it is missing. A consumer that reads LENS reports should
not pay for a package it never calls.
"""

from __future__ import annotations

from typing import Optional

from .protein_fragment import ProteinFragment
from .rna_evidence import ISOVAR_ASSEMBLY, RNA_READS

_MIN_ISOVAR = (1, 7, 2)
_MIN_ISOVAR_TEXT = "1.7.2"


def _check_isovar():
    """Import isovar or explain how to get it, the way pirlygenes is."""
    try:
        import isovar
    except ImportError:
        raise ImportError(
            f"isovar is required to build fragments from RNA-assembled "
            f"protein sequences. Install with: "
            f"pip install 'isovar>={_MIN_ISOVAR_TEXT}'"
        ) from None
    version = getattr(isovar, "__version__", "0.0.0")
    parts = []
    for piece in str(version).split(".")[:3]:
        digits = "".join(c for c in piece if c.isdigit())
        parts.append(int(digits) if digits else 0)
    if tuple(parts + [0, 0, 0][: 3 - len(parts)]) < _MIN_ISOVAR:
        raise ImportError(
            f"isovar>={_MIN_ISOVAR_TEXT} required; found {version}. "
            f"Upgrade with: pip install -U 'isovar>={_MIN_ISOVAR_TEXT}'"
        )
    return isovar


def fragment_from_isovar_result(
    isovar_result,
    *,
    gene_expression=None,
    transcript_expression=None,
) -> Optional[ProteinFragment]:
    """Build a :class:`ProteinFragment` from one ``isovar.IsovarResult``.

    The RNA arm of the multi-source fragment story. The sequence is
    *assembled from reads* rather than translated from the reference, so
    it carries the patient's other variants and whatever phasing the
    reads support — which is what makes ``sequence_source`` worth
    recording alongside it.

    The read counts are **native**: isovar counted them, so they are
    marked ``measured`` rather than carrying a derivation. That is the
    distinction the whole evidence vocabulary exists for — every other
    source either estimates them or counts something adjacent.

    Parameters
    ----------
    isovar_result : isovar.IsovarResult
        A result with a ``top_protein_sequence``.
    gene_expression, transcript_expression : float, optional
        Abundance to carry onto the fragment. Left ``None`` when the
        caller has none, which is not the same as zero.

    Returns
    -------
    ProteinFragment or None
        ``None`` when isovar assembled no protein sequence for the
        variant — an absence, not an error: a variant with no RNA
        support is a normal result.

    Examples
    --------
    >>> fragments = [                          # doctest: +SKIP
    ...     fragment_from_isovar_result(r) for r in isovar_results
    ... ]
    >>> fragments = [f for f in fragments if f is not None]  # doctest: +SKIP
    """
    protein_sequence = getattr(isovar_result, "top_protein_sequence", None)
    if protein_sequence is None:
        return None

    amino_acids = getattr(protein_sequence, "amino_acids", "") or ""
    if not amino_acids:
        return None

    start = getattr(protein_sequence, "mutation_start_idx", None)
    end = getattr(protein_sequence, "mutation_end_idx", None)
    variant = getattr(isovar_result, "variant", None)

    transcript_ids = list(getattr(protein_sequence, "transcript_ids", ()) or ())
    transcript_names = list(
        getattr(protein_sequence, "transcript_names", ()) or ()
    )

    # isovar counts these; nothing is derived, so they are measured.
    counts = dict(
        n_overlapping_reads=_as_count(
            getattr(isovar_result, "num_total_fragments", None)
        ),
        n_alt_reads=_as_count(
            getattr(isovar_result, "num_alt_fragments", None)
        ),
        n_ref_reads=_as_count(
            getattr(isovar_result, "num_ref_fragments", None)
        ),
        n_alt_reads_supporting_protein_sequence=_as_count(
            getattr(protein_sequence, "num_supporting_fragments", None)
        ),
    )
    provenance = {
        name: "measured" for name, value in counts.items() if value is not None
    }

    target_intervals = None
    if start is not None and end is not None:
        lo = max(0, min(int(start), len(amino_acids)))
        hi = max(lo, min(int(end), len(amino_acids)))
        target_intervals = [(lo, hi)]

    return ProteinFragment(
        fragment_id=_fragment_id(variant, amino_acids),
        source_type="variant:rna_assembled",
        sequence=amino_acids,
        target_intervals=target_intervals,
        variant=str(variant) if variant is not None else None,
        gene=getattr(protein_sequence, "gene_name", None),
        transcript_id=transcript_ids[0] if transcript_ids else None,
        transcript_name=transcript_names[0] if transcript_names else None,
        gene_expression=gene_expression,
        transcript_expression=transcript_expression,
        field_provenance=provenance,
        annotations={
            "sequence_source": ISOVAR_ASSEMBLY,
            "read_count_method": RNA_READS,
            # Every transcript consistent with the assembled sequence,
            # not just the one named above. A release mismatch that
            # leaves these unresolvable downstream is visible rather
            # than an empty list.
            "supporting_reference_transcripts": transcript_ids,
        },
        **counts,
    )


def fragments_from_isovar_results(isovar_results):
    """Fragments for every result that assembled a protein sequence.

    Results with no RNA support are dropped rather than yielding
    ``None`` entries a caller has to filter — the absence is the answer,
    and a list comprehension over it should not need a guard.
    """
    fragments = []
    for result in isovar_results:
        fragment = fragment_from_isovar_result(result)
        if fragment is not None:
            fragments.append(fragment)
    return fragments


def _as_count(value):
    """A non-negative int, or None when the value was not stated."""
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def _fragment_id(variant, amino_acids):
    from .protein_fragment import make_fragment_id
    prefix = str(variant) if variant is not None else "isovar"
    return make_fragment_id(prefix=prefix, sequence=amino_acids)
