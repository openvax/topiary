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


#: Default assembled window around a mutation, in amino acids.
#:
#: A fragment is scanned by a sliding window later, so the assembled
#: sequence has to be long enough to contain every peptide that could
#: cover the mutation — the peptide length plus padding on both sides.
#: 9 + 2 * 8 is the usual class-I shape.
DEFAULT_PROTEIN_SEQUENCE_LENGTH = 25


def fragments_from_variants(
    variants,
    alignment_file=None,
    *,
    protein_sequence_length: int = DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    padding_around_mutation: Optional[int] = None,
    allow_reference_fallback: bool = False,
    transcript_id_whitelist=None,
    filter_thresholds=None,
    **isovar_kwargs,
):
    """Fragments for *variants*, assembled from RNA when RNA is available.

    The point of entry that makes the sources interchangeable: give it a
    ``alignment_file`` and the protein context is **assembled from
    reads**, carrying the patient's other variants and whatever phasing
    the reads support; leave it out and the same variants are
    **translated from the reference**. Either way the result is a list
    of :class:`~topiary.ProteinFragment` with the same core, so the rest
    of a pipeline does not change when the RNA does or does not exist.

    The assembled sequence is deliberately longer than one peptide.  A
    fragment is scanned by a sliding window downstream, so it has to
    contain every peptide that could cover the mutation — hence
    *protein_sequence_length*, not a peptide length.

    Parameters
    ----------
    variants : varcode.VariantCollection or iterable of varcode.Variant
        The variants to build fragments for.
    alignment_file : pysam.AlignmentFile, optional
        RNA alignment. When given, isovar assembles the protein sequence
        from the reads covering each variant and counts the reads
        supporting it. When ``None``, every fragment comes from
        reference translation and carries no read counts.
    protein_sequence_length : int
        Amino acids of assembled context around the mutation.
    padding_around_mutation : int, optional
        Residues kept either side of the mutation on the *reference*
        path. Defaults to half the remaining context, so both paths
        produce comparable windows.
    allow_reference_fallback : bool
        When true, a variant isovar could not support is translated from
        the reference instead of being dropped. The fragments say which
        they are via ``annotations["sequence_source"]``, so a consumer
        can tell an RNA-backed candidate from an inferred one — the
        distinction is the reason to record it rather than blend them.
    transcript_id_whitelist, filter_thresholds
        Passed to :func:`isovar.run_isovar`.
    **isovar_kwargs
        Also passed through — ``read_collector``,
        ``min_shared_fragments_for_phasing`` and friends.

    Returns
    -------
    list of ProteinFragment

    Notes
    -----
    Requires isovar only when *alignment_file* is given. The reference
    path has no such dependency, so a caller without RNA never imports
    it.
    """
    if padding_around_mutation is None:
        padding_around_mutation = max(1, (protein_sequence_length - 1) // 2)

    variants = list(variants)
    if alignment_file is None:
        return _reference_fragments(variants, padding_around_mutation)

    isovar = _check_isovar()
    from isovar.protein_sequence_creator import ProteinSequenceCreator

    creator = isovar_kwargs.pop("protein_sequence_creator", None)
    if creator is None:
        creator = ProteinSequenceCreator(
            protein_sequence_length=protein_sequence_length,
        )
    results = isovar.run_isovar(
        variants=variants,
        alignment_file=alignment_file,
        transcript_id_whitelist=transcript_id_whitelist,
        protein_sequence_creator=creator,
        filter_thresholds=filter_thresholds,
        **isovar_kwargs,
    )

    fragments = []
    unsupported = []
    for result in results:
        fragment = fragment_from_isovar_result(result)
        if fragment is not None:
            fragments.append(fragment)
        else:
            unsupported.append(getattr(result, "variant", None))

    if unsupported and allow_reference_fallback:
        fragments.extend(
            _reference_fragments(
                [v for v in unsupported if v is not None],
                padding_around_mutation,
            )
        )
    return fragments


def _reference_fragments(variants, padding_around_mutation):
    """Translate variants from the reference — the no-RNA path.

    Kept in this module so both arms of :func:`fragments_from_variants`
    live together, but it imports nothing from isovar: a caller with no
    alignment file never touches the optional dependency.
    """
    from .predictor import fragment_from_effect

    fragments = []
    for variant in variants:
        try:
            effects = variant.effects()
        except AttributeError:
            continue
        effect = effects.top_priority_effect()
        if effect is None:
            continue
        fragment = fragment_from_effect(
            effect, padding_around_mutation=padding_around_mutation,
        )
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
