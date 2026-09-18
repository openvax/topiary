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

from numbers import Integral
from typing import Optional

from .protein_fragment import ProteinFragment, fragments_for_sample
from .evidence import ISOVAR_ASSEMBLY, RNA_ALIGNMENT
from .optional_dependencies import require_optional_dependency
from .serialization import normalize_python_types


def _check_isovar():
    """Load the Isovar API used to assemble variants from RNA reads.

    Refuses an installed release older than the ``isovar`` extra's floor:
    older releases run, but can return wrong evidence (1.17.x miscounts
    reads at insertion boundaries), which is worse than failing.
    """
    return require_optional_dependency(
        "isovar",
        feature="assembling protein fragments from RNA alignments",
        required_callables=("run_isovar", "ProteinSequenceCreator"),
    )


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
    extracted = _extract_isovar_result(isovar_result)
    amino_acids = extracted["protein_sequence"]
    if amino_acids is None:
        return None

    # isovar reports both units for every count, so carry both under
    # names that say which is which. Putting the fragment count in a
    # field named for reads was the same mistake as the CDS-overlap
    # column: a real count of one thing under a name for another.
    counts = {
        field: extracted[key] for field, key in _FRAGMENT_COUNT_SOURCES.items()
    }
    provenance = {
        name: "measured" for name, value in counts.items() if value is not None
    }
    start, end = extracted["mutation_start"], extracted["mutation_end"]
    transcript_ids = extracted["transcript_ids"]
    transcript_names = extracted["transcript_names"]
    variant = getattr(isovar_result, "variant", None)

    return ProteinFragment(
        fragment_id=_fragment_id(variant, amino_acids),
        source_type="variant:rna_assembled",
        sequence=amino_acids,
        target_intervals=None if start is None else [(start, end)],
        variant=str(variant) if variant is not None else None,
        gene=extracted["gene"],
        # Isovar's supporting transcripts are an unordered set; the
        # first after sorting is a reproducible choice, not a ranking.
        transcript_id=transcript_ids[0] if transcript_ids else None,
        transcript_name=transcript_names[0] if transcript_names else None,
        gene_expression=gene_expression,
        transcript_expression=transcript_expression,
        field_provenance=provenance,
        annotations={
            "sequence_source": ISOVAR_ASSEMBLY,
            "rna_evidence_method": RNA_ALIGNMENT,
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


def describe_isovar_result(isovar_result):
    """Describe reconstruction and filter outcomes, including empty results.

    Parameters
    ----------
    isovar_result : isovar.IsovarResult
        A completed result from ``run_isovar``. Acquisition failures and
        exceptions are not results and must be reported separately by callers.

    Returns
    -------
    dict
        JSON-compatible variant identity, native RNA counts, protein sequence,
        mutation interval, transcript IDs, named failed filters and ``status``.
        Status distinguishes ``passing``, ``filtered``, ``no_usable_reads``,
        ``no_alt_reads``, ``no_predicted_coding_change`` and
        ``no_protein_sequence``; a protein with unknown filter disposition has
        ``filter_status_unavailable``. Counts unavailable on an object remain null,
        not zero. ``passing`` means only that the supplied result has a protein
        and passes its recorded filters; it is not a clinical or presentation
        judgment, and does not imply the caller used default filters.

    Notes
    -----
    Fragment-producing APIs intentionally omit empty results. Use this function
    on every upstream result to retain the reason a variant yielded no accepted
    fragment. A reconstructed but filtered sequence remains visible for audit;
    it must not silently become an accepted prediction input. Native read and
    fragment counts remain separate and are not independent-molecule counts.
    """
    result = isovar_result
    extracted = _extract_isovar_result(result)
    variant = getattr(result, "variant", None)
    counts = {key: extracted[key] for key in _ISOVAR_COUNT_KEYS}
    passing = extracted["passes_all_filters"]
    effect = getattr(result, "predicted_effect", None)
    if extracted["protein_sequence"]:
        status = ("filter_status_unavailable" if passing is None else
                  "passing" if passing else "filtered")
    elif counts["num_total_reads"] == 0:
        status = "no_usable_reads"
    elif counts["num_alt_reads"] == 0:
        status = "no_alt_reads"
    elif getattr(effect, "modifies_protein_sequence", None) is False:
        status = "no_predicted_coding_change"
    else:
        status = "no_protein_sequence"
    filters = extracted["filter_values"]
    return normalize_python_types(dict(
        status=status,
        variant=str(variant) if variant is not None else None,
        reference_name=getattr(variant, "reference_name", None),
        contig=getattr(variant, "contig", None),
        start=getattr(variant, "start", None),
        ref=getattr(variant, "ref", None),
        alt=getattr(variant, "alt", None),
        **counts,
        passes_all_filters=passing,
        filter_values=filters,
        failed_filters=sorted(name for name, passed in filters.items() if not passed),
        predicted_effect_class=type(effect).__name__ if effect is not None else None,
        predicted_effect=getattr(effect, "short_description", None),
        protein_sequence=extracted["protein_sequence"],
        mutation_start=extracted["mutation_start"],
        mutation_end=extracted["mutation_end"],
        transcript_ids=extracted["transcript_ids"],
        protein_supporting_reads=extracted["protein_supporting_reads"],
        protein_supporting_fragments=extracted["protein_supporting_fragments"],
    ))


#: Isovar's native count attributes, in the order outcomes report them.
_ISOVAR_COUNT_KEYS = tuple(
    f"num_{category}_{unit}"
    for unit in ("reads", "fragments")
    for category in ("total", "ref", "alt", "other")
)

#: Fragment count field → the extracted value that fills it.
_FRAGMENT_COUNT_SOURCES = {
    "n_rna_overlapping_reads": "num_total_reads",
    "n_rna_alt_reads": "num_alt_reads",
    "n_rna_ref_reads": "num_ref_reads",
    "n_rna_other_reads": "num_other_reads",
    "n_rna_alt_reads_supporting_protein_sequence": "protein_supporting_reads",
    "n_rna_overlapping_fragments": "num_total_fragments",
    "n_rna_alt_fragments": "num_alt_fragments",
    "n_rna_ref_fragments": "num_ref_fragments",
    "n_rna_other_fragments": "num_other_fragments",
    "n_rna_alt_fragments_supporting_protein_sequence": "protein_supporting_fragments",
}


def _extract_isovar_result(result):
    """Everything topiary reads from one result, read one way.

    The single reader behind :func:`describe_isovar_result` and
    :func:`fragment_from_isovar_result`, so the outcome report and the
    fragment cannot disagree about a count, interval or transcript.
    Attributes are read by name, so duck-typed results work; a missing
    attribute is ``None`` (unknown), never zero.

    The mutation interval is clamped to the sequence, and dropped when
    either end is missing. Supporting transcripts are sorted by ID
    (Isovar keeps them as a set, whose order varies between processes),
    with each name kept parallel to its ID (``None`` when unavailable). Filter disposition is Isovar's
    ``passes_all_filters`` when the result states it, otherwise Isovar's
    own rule over ``filter_values`` (every recorded filter passed), and
    ``None`` when neither is available.
    """
    protein = getattr(result, "top_protein_sequence", None)
    sequence = getattr(protein, "amino_acids", None) or None
    start = getattr(protein, "mutation_start_idx", None)
    end = getattr(protein, "mutation_end_idx", None)
    if sequence is None or start is None or end is None:
        start = end = None
    else:
        start = max(0, min(int(start), len(sequence)))
        end = max(start, min(int(end), len(sequence)))
    ids = list(getattr(protein, "transcript_ids", ()) or ())
    names = list(getattr(protein, "transcript_names", ()) or ())
    if len(names) != len(ids):
        names = [None] * len(ids)
    transcripts = sorted(zip(ids, names), key=lambda pair: str(pair[0]))
    filters = dict(getattr(result, "filter_values", None) or {})
    if hasattr(result, "passes_all_filters"):
        passing = normalize_python_types(result.passes_all_filters)
        passing = None if passing is None else bool(passing)
    elif hasattr(result, "filter_values"):
        passing = all(filters.values())
    else:
        passing = None
    return dict(
        {key: _as_count(getattr(result, key, None)) for key in _ISOVAR_COUNT_KEYS},
        protein_sequence=sequence,
        gene=getattr(protein, "gene_name", None),
        mutation_start=start,
        mutation_end=end,
        transcript_ids=[transcript_id for transcript_id, _ in transcripts],
        transcript_names=[name for _, name in transcripts],
        protein_supporting_reads=_as_count(getattr(protein, "num_supporting_reads", None)),
        protein_supporting_fragments=_as_count(getattr(protein, "num_supporting_fragments", None)),
        filter_values=filters,
        passes_all_filters=passing,
    )


#: Historical context length, retained for explicit callers and reference
#: padding. The RNA default is now derived by Isovar from the desired peptide
#: size: the default 11-aa ligand objective still requests 21 aa. Available
#: RNA, mutation position and stop codons can limit the returned context.
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
    protein_sequence_length: Optional[int] = None,
    protein_context_peptide_length: Optional[int] = None,
    protein_sequence_preference: Optional[str] = None,
    min_protein_sequence_support_fraction: Optional[float] = None,
    min_variant_sequence_coverage: Optional[int] = None,
    protein_sequence_creator=None,
    padding_around_mutation: Optional[int] = None,
    epitope_lengths=(8, 9, 10, 11),
    allow_reference_fallback: bool = False,
    require_passing_filters: bool = True,
    gene_expression=None,
    transcript_expression=None,
    transcript_id_whitelist=None,
    filter_thresholds=None,
    sample_name: Optional[str] = None,
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

    The desired peptide size and the RNA context length are distinct.
    By default Isovar targets ``2 * protein_context_peptide_length - 1``
    residues, enough for every overlapping peptide placement around a
    centered single-residue substitution. Wider edits and deletion junctions
    have different overlap geometry, and reads may support less context.
    No reference residues are appended to an RNA-supported fragment.

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
    protein_sequence_length : int, optional
        Explicit positive RNA context target, in amino acids. When omitted,
        Isovar derives the target from *protein_context_peptide_length*.
        A target is not a promise that enough RNA context exists.
    protein_context_peptide_length : int, optional
        Peptide size used to evaluate mutation-overlapping RNA context.
        Defaults to ``max(epitope_lengths)`` for ligand workflows (11 aa
        with the default lengths). Set this to the vaccine peptide size
        for vaccine workflows; it does not change MHC prediction lengths.
    protein_sequence_preference : str, optional
        Isovar's ``balanced`` default favors useful context within the
        relative support budget. ``support`` prioritizes support and
        ``context`` prioritizes context without that relative budget.
        The absolute per-base floor applies to every preference.
    min_protein_sequence_support_fraction : float, optional
        Balanced selection's minimum fraction of the best mutant candidate's
        compatible read-name support (Isovar default 0.85). Not per-base
        depth, total-alt VAF, a confidence probability, or a count of names
        spanning the entire peptide.
    min_variant_sequence_coverage : int, optional
        Independent absolute floor of RNA read objects covering each retained
        base (Isovar default 2). It is never relaxed to meet a length target.
    protein_sequence_creator : isovar.ProteinSequenceCreator, optional
        Use this configured creator unchanged. Cannot be combined with any
        explicit creator option above, even one equal to its default. Unknown
        settings on custom creators are left unknown in provenance.
    padding_around_mutation : int, optional
        Residues kept either side of the mutation on the reference arm.
        Validated against *epitope_lengths* by
        :func:`~topiary.check_padding_around_mutation`, so a padding too
        small to contain any epitope is refused rather than producing
        fragments the sliding window cannot use. The default retains the
        historical padding rule using an explicit *protein_sequence_length*
        or 21 aa; the new RNA peptide objective does not change this rule.
    epitope_lengths : sequence of int
        MHC ligand lengths. Used for reference-padding validation and, when
        no RNA peptide size or custom creator is given, the RNA objective.
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
    sample_name : str, optional
        Label every returned fragment as this sample's observation, through
        :func:`~topiary.fragments_for_sample`: the ID is namespaced and
        ``annotations["sample_name"]`` fills the prediction frame's
        ``sample_name`` column. Needed to combine fragments from more than
        one alignment (or one alignment under two settings) in a single
        prediction; unlabelled, the same variant would get the same ID
        with different evidence, which prediction refuses.
    **isovar_kwargs
        Passed to ``run_isovar``, for example ``read_collector`` and
        ``filter_flags``. Creator options above are not passed here. Rejected when no
        *alignment_file* is given, rather than silently ignored.

    Returns
    -------
    list of ProteinFragment
        In input-variant order.

    Notes
    -----
    Needs the ``isovar`` extra (``pip install 'topiary[isovar]'``) only
    when *alignment_file* is given; an installed Isovar older than the
    extra's floor is refused rather than trusted. Explicit
    RNA-only options are rejected without an alignment file. RNA fragments
    record the Isovar version, creator class and available creator settings
    as ``isovar_*`` annotations, preserved by fragment IO and prediction.
    """
    if protein_sequence_length is not None and (
        isinstance(protein_sequence_length, bool)
        or not isinstance(protein_sequence_length, Integral)
        or protein_sequence_length < 1
    ):
        raise ValueError(
            f"protein_sequence_length is a count of amino acids and must "
            f"be positive (an integer); got {protein_sequence_length}."
        )
    creator_options = {
        name: value for name, value in (
            ("protein_sequence_length", protein_sequence_length),
            ("protein_context_peptide_length", protein_context_peptide_length),
            ("protein_sequence_preference", protein_sequence_preference),
            ("min_protein_sequence_support_fraction", min_protein_sequence_support_fraction),
            ("min_variant_sequence_coverage", min_variant_sequence_coverage),
        ) if value is not None
    }
    if padding_around_mutation is None:
        padding_around_mutation = max(
            max(epitope_lengths) - 1,
            ((protein_sequence_length or DEFAULT_PROTEIN_SEQUENCE_LENGTH) - 1) // 2,
        )
    else:
        from .sequence_helpers import check_padding_around_mutation
        padding_around_mutation = check_padding_around_mutation(
            padding_around_mutation, epitope_lengths
        )

    if alignment_file is None:
        rejected = sorted(
            set(isovar_kwargs)
            | (set(creator_options) - {"protein_sequence_length"})
            | {k for k, v in (
                ("transcript_id_whitelist", transcript_id_whitelist),
                ("filter_thresholds", filter_thresholds),
                ("protein_sequence_creator", protein_sequence_creator),
            ) if v is not None}
        )
        if rejected:
            raise TypeError(
                f"{rejected} only apply when an alignment_file is given; "
                f"without one there is no isovar run to configure. Drop "
                f"them, or pass the alignment file."
            )
        return _labelled(fragments_from_effects(
            _effects_for(variants),
            padding_around_mutation,
            gene_expression=gene_expression,
            transcript_expression=transcript_expression,
        ), sample_name)

    isovar = _check_isovar()
    creator = protein_sequence_creator
    if creator is not None and creator_options:
        raise ValueError(
            f"protein_sequence_creator and {sorted(creator_options)} both "
            "configure RNA reconstruction; pass one configuration source."
        )
    if creator is None:
        creator_options.setdefault("protein_context_peptide_length", max(epitope_lengths))
        creator = isovar.ProteinSequenceCreator(
            # Preserve Topiary's overlap-assembly policy; Isovar still
            # enforces the support floor on the retained RNA context.
            variant_sequence_assembly=True,
            **creator_options,
        )

    reconstruction_annotations = {
        "isovar_version": getattr(isovar, "__version__", None),
        "isovar_creator": f"{type(creator).__module__}.{type(creator).__qualname__}",
    }
    for name in (
        "protein_sequence_length", "protein_context_peptide_length",
        "protein_sequence_preference", "min_protein_sequence_support_fraction",
        "min_variant_sequence_coverage", "variant_sequence_assembly",
        "min_transcript_prefix_length", "max_transcript_mismatches",
        "count_mismatches_after_variant", "min_assembly_overlap_size",
        "max_protein_sequences_per_variant",
    ):
        value = getattr(creator, name, None)
        if value is not None:
            reconstruction_annotations[f"isovar_{name}"] = normalize_python_types(value)

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
        # A result whose filter disposition is unknown is not shown to pass.
        if require_passing_filters and (
            _extract_isovar_result(result)["passes_all_filters"] is not True
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
            fragment.annotations.update(reconstruction_annotations)
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
    return _labelled(fragments, sample_name)


def _labelled(fragments, sample_name):
    """*fragments*, labelled with *sample_name* when one was given."""
    return fragments if sample_name is None else fragments_for_sample(fragments, sample_name)


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
