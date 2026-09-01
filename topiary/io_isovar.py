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
from .rna_evidence import ISOVAR_ASSEMBLY, RNA_ALIGNMENT

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

    # isovar reports both units for every count, so carry both under
    # names that say which is which. Putting the fragment count in a
    # field named for reads was the same mistake as the CDS-overlap
    # column: a real count of one thing under a name for another.
    counts = dict(
        n_overlapping_reads=_as_count(
            getattr(isovar_result, "num_total_reads", None)
        ),
        n_alt_reads=_as_count(
            getattr(isovar_result, "num_alt_reads", None)
        ),
        n_ref_reads=_as_count(
            getattr(isovar_result, "num_ref_reads", None)
        ),
        n_alt_reads_supporting_protein_sequence=_as_count(
            getattr(protein_sequence, "num_supporting_reads", None)
        ),
        n_overlapping_fragments=_as_count(
            getattr(isovar_result, "num_total_fragments", None)
        ),
        n_alt_fragments=_as_count(
            getattr(isovar_result, "num_alt_fragments", None)
        ),
        n_ref_fragments=_as_count(
            getattr(isovar_result, "num_ref_fragments", None)
        ),
        n_alt_fragments_supporting_protein_sequence=_as_count(
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
            "read_count_method": RNA_ALIGNMENT,
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
#: This matches isovar's own default rather than raising it: with
#: assembly on, a longer window still has to be reachable from the
#: reads, and asking for more than the data supports silently returns
#: fewer variants rather than longer sequences.
DEFAULT_PROTEIN_SEQUENCE_LENGTH = 21


def fragments_from_effects(
    effects,
    padding_around_mutation: int,
    *,
    gene_expression=None,
    transcript_expression=None,
):
    """Fragments translated from reference, one per variant effect group.

    The no-RNA arm of :func:`fragments_from_variants`, public because a
    caller with variants and no alignment file wants exactly this and
    should not have to reach into a private helper for it.

    Silent, non-coding and untranslatable effects are filtered first —
    several varcode effect classes expose a ``mutant_protein_sequence``
    while leaving the amino-acid offsets ``None``, and
    :func:`~topiary.fragment_from_effect` raises on those by design.
    Without the filter one such effect anywhere in a batch discards
    every fragment already built.

    Parameters
    ----------
    effects : varcode.EffectCollection or iterable of effects
        Effects to translate. Grouped by variant; the top-priority
        effect of each group becomes a fragment, or the
        top-*expression* effect when *transcript_expression* is given.
    padding_around_mutation : int
        Residues kept either side of the mutated span.
    gene_expression, transcript_expression : dict, optional
        ``{gene_id: value}`` / ``{transcript_id: value}``. When
        transcript expression is present it also drives transcript
        selection, matching what
        :meth:`TopiaryPredictor.predict_from_mutation_effects` does —
        the same variants must not pick a different transcript
        depending on which entry point was used.

    Returns
    -------
    list of ProteinFragment
    """
    from varcode import EffectCollection

    from .filters import filter_silent_and_noncoding_effects
    from .predictor import fragment_from_effect

    if not isinstance(effects, EffectCollection):
        effects = EffectCollection(list(effects))
    effects = filter_silent_and_noncoding_effects(effects)
    if len(effects) == 0:
        return []

    groups = effects.groupby_variant()
    if transcript_expression:
        top_effects = [
            group.top_expression_effect(transcript_expression)
            for group in groups.values()
        ]
    else:
        top_effects = [group.top_priority_effect() for group in groups.values()]

    fragments = []
    for effect in top_effects:
        if effect is None:
            continue
        fragment = fragment_from_effect(
            effect,
            padding_around_mutation,
            gene_expression=(
                gene_expression.get(effect.gene_id)
                if gene_expression else None
            ),
            transcript_expression=(
                transcript_expression.get(effect.transcript_id)
                if transcript_expression else None
            ),
        )
        if fragment is not None:
            fragments.append(fragment)
    return fragments


def fragments_from_variants(
    variants,
    alignment_file=None,
    *,
    protein_sequence_length: int = DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    padding_around_mutation: Optional[int] = None,
    epitope_lengths=(8, 9, 10, 11),
    allow_reference_fallback: bool = False,
    require_passing_filters: bool = True,
    gene_expression=None,
    transcript_expression=None,
    transcript_id_whitelist=None,
    filter_thresholds=None,
    **isovar_kwargs,
):
    """Fragments for *variants*, assembled from RNA when RNA is available.

    The entry point that makes the sources interchangeable: give it an
    ``alignment_file`` and the protein context is **assembled from
    reads**, carrying the patient's other variants and whatever phasing
    the reads support; leave it out and the same variants are
    **translated from the reference**. Either way the result is a list
    of :class:`~topiary.ProteinFragment` with the same core, so the rest
    of a pipeline does not change when the RNA does or does not exist.

    The assembled sequence is deliberately longer than one peptide — a
    fragment is scanned by a sliding window downstream, so it has to
    contain every peptide that could cover the mutation. Hence
    *protein_sequence_length*, not a peptide length.

    Parameters
    ----------
    variants : varcode.VariantCollection, iterable of Variant, or str
        Variants to build fragments for. A path is passed to isovar,
        which loads it; on the reference arm it is loaded with varcode.
    alignment_file : pysam.AlignmentFile, optional
        RNA alignment. When given, isovar assembles the protein sequence
        from reads covering each variant and counts the reads supporting
        it. When ``None``, every fragment comes from reference
        translation and carries no read counts.
    protein_sequence_length : int
        Amino acids of assembled context around the mutation.
    padding_around_mutation : int, optional
        Residues kept either side of the mutation on the reference arm.
        Validated against *epitope_lengths* by
        :func:`~topiary.check_padding_around_mutation`, so a padding too
        small to contain any epitope is refused rather than producing
        fragments the sliding window cannot use.
    epitope_lengths : sequence of int
        Peptide lengths the fragments must be able to contain; only used
        to validate the padding.
    allow_reference_fallback : bool
        When true, a variant isovar could not support is translated from
        the reference instead of dropped. Fragments say which they are
        via ``annotations["sequence_source"]``, so an RNA-backed
        candidate and an inferred one never blend.
    require_passing_filters : bool
        Drop isovar results that fail their filters. **isovar records
        filter outcomes but never drops anything**, so without this a
        caller's ``filter_thresholds`` — and isovar's own defaults —
        annotate results that then flow on as RNA-backed evidence.
    gene_expression, transcript_expression : dict, optional
        Expression to attach, and on the reference arm to drive
        transcript selection.
    transcript_id_whitelist, filter_thresholds
        Passed to :func:`isovar.run_isovar`.
    **isovar_kwargs
        Also passed through — ``read_collector``,
        ``protein_sequence_creator``, and friends. Rejected when no
        *alignment_file* is given, rather than silently ignored.

    Returns
    -------
    list of ProteinFragment
        In input-variant order.

    Notes
    -----
    Requires isovar only when *alignment_file* is given.
    """
    if protein_sequence_length < 1:
        raise ValueError(
            f"protein_sequence_length is a count of amino acids and must "
            f"be positive; got {protein_sequence_length}."
        )
    if padding_around_mutation is None:
        padding_around_mutation = max(
            max(epitope_lengths) - 1, (protein_sequence_length - 1) // 2
        )
    else:
        from .sequence_helpers import check_padding_around_mutation
        padding_around_mutation = check_padding_around_mutation(
            padding_around_mutation, epitope_lengths
        )

    if alignment_file is None:
        rejected = sorted(
            set(isovar_kwargs)
            | {k for k, v in (
                ("transcript_id_whitelist", transcript_id_whitelist),
                ("filter_thresholds", filter_thresholds),
            ) if v is not None}
        )
        if rejected:
            raise TypeError(
                f"{rejected} only apply when an alignment_file is given; "
                f"without one there is no isovar run to configure. Drop "
                f"them, or pass the alignment file."
            )
        return fragments_from_effects(
            _effects_for(variants),
            padding_around_mutation,
            gene_expression=gene_expression,
            transcript_expression=transcript_expression,
        )

    isovar = _check_isovar()
    creator = isovar_kwargs.pop("protein_sequence_creator", None)
    if creator is None:
        from isovar.protein_sequence_creator import ProteinSequenceCreator
        creator = ProteinSequenceCreator(
            protein_sequence_length=protein_sequence_length,
            # Without assembly a single read or fragment must span the
            # whole window, so a longer context quietly yields fewer
            # variants rather than longer sequences — and "assembled
            # from reads, carrying the phasing the reads support" would
            # not be true of the result.
            variant_sequence_assembly=True,
        )
    elif protein_sequence_length != DEFAULT_PROTEIN_SEQUENCE_LENGTH:
        raise ValueError(
            "protein_sequence_length and protein_sequence_creator both "
            "set the assembled window; pass one. The creator's length "
            "would have won silently."
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
        if require_passing_filters and not getattr(
            result, "passes_all_filters", True
        ):
            unsupported.append(getattr(result, "variant", None))
            continue
        fragment = fragment_from_isovar_result(
            result,
            gene_expression=(
                gene_expression.get(getattr(result, "gene_id", None))
                if gene_expression else None
            ),
            transcript_expression=None,
        )
        if fragment is not None:
            fragments.append(fragment)
        else:
            unsupported.append(getattr(result, "variant", None))

    if unsupported and allow_reference_fallback:
        fragments.extend(fragments_from_effects(
            _effects_for([v for v in unsupported if v is not None]),
            padding_around_mutation,
            gene_expression=gene_expression,
            transcript_expression=transcript_expression,
        ))
    return fragments


def _effects_for(variants):
    """Variant effects, with the loading varcode already knows how to do."""
    from varcode import EffectCollection, load_vcf

    if isinstance(variants, str):
        variants = load_vcf(variants)
    collected = []
    for variant in variants:
        collected.extend(variant.effects())
    return EffectCollection(collected)


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
