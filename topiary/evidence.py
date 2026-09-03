"""Read-level DNA and RNA evidence, and saying where each number came from.

Readers report RNA support differently, and the difference matters. isovar
counts reads supporting an assembled protein sequence. pVACseq reports a
depth and a variant allele fraction, from which the split is arithmetic.
LENS counts reads overlapping the peptide's CDS, which is a real count of
something adjacent to what was asked for. A consumer weighting a candidate
by depth of support needs to know which of those it has.

So every derived number carries the name of its derivation:

============================  ==================================================
``rna_alignment``             Counted directly from an RNA alignment, in
                              whichever unit the aligner reported.
``rna_depth_x_vaf``           depth x VAF, rounded. Not counted.
``cds_overlap_reads``         Counted, but of reads overlapping the peptide's
                              coding sequence rather than supporting the
                              variant allele.
``tpm_x_dna_vaf``             Transcript abundance x DNA variant allele
                              fraction. An expression proxy, not a read count.
============================  ==================================================

The last one carries a bias worth stating: it assumes both alleles are
transcribed equally, so a variant on a transcriptionally silenced allele
looks expressed. That error is exactly the phenomenon allele-specific
counting exists to detect, which is why the estimate is labelled rather
than silently mixed in with measured values.

DNA support is carried in the same shape under a ``dna_`` prefix
(:func:`attach_dna_evidence`), so a caller writing a depth threshold writes
it the same way for either assay and can always tell which one it got.

Three layers of column, and which to reach for:

1. **Canonical cross-source** — ``n_rna_alt``, ``rna_vaf``, ``n_dna_alt``,
   ``dna_vaf`` and friends. Same meaning from every reader. Write
   filters against these.
2. **Canonical unit-specific** — ``n_rna_alt_reads`` / ``n_rna_alt_fragments``.
   Same meaning everywhere, present only where a source reports both
   units and they genuinely differ.
3. **Source-prefixed originals** — ``lens_vaf``, ``pvacseq_tumor_dna_depth``.
   Exactly the number the tool printed, never reinterpreted. See
   :data:`SOURCE_PREFIXES`.

``None`` throughout means the source could not answer, which is not the
same as answering zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from types import MappingProxyType

from .ranking import EvalContext, is_stated, stated_values

#: Counted directly from an RNA alignment.
#:
#: Named for the *source* of the number, not its unit: an aligner counts
#: reads and fragments alike, and which one you got is
#: ``rna_evidence_subject``. The old name ``rna_reads`` implied a unit
#: it never fixed.
RNA_ALIGNMENT = "rna_alignment"

#: Deprecated alias for :data:`RNA_ALIGNMENT`.
RNA_READS = RNA_ALIGNMENT

#: depth x VAF, rounded — arithmetic, not counted.
RNA_DEPTH_X_VAF = "rna_depth_x_vaf"

#: Counted, but of reads overlapping the peptide's coding sequence rather
#: than supporting the variant allele.
CDS_OVERLAP_READS = "cds_overlap_reads"

#: Transcript abundance x DNA variant allele fraction — an expression
#: proxy, not a read count.
TPM_X_DNA_VAF = "tpm_x_dna_vaf"

#: RNA depth x a VAF whose assay the source did not state.
#:
#: LENS carries one unqualified ``vaf`` column while naming its read
#: columns ``rna_*`` explicitly, so topiary cannot tell whether the
#: fraction is from RNA or DNA. The count is still the best available
#: estimate, but calling it ``rna_depth_x_vaf`` would assert an assay
#: nobody stated.
RNA_DEPTH_X_SOURCE_VAF = "rna_depth_x_source_vaf"

#: DNA depth x DNA VAF, rounded — arithmetic, not counted.
#:
#: The DNA twin of :data:`RNA_DEPTH_X_VAF`. Kept as a separate term
#: rather than reusing the RNA one so that a row naming its derivation
#: also names its assay: a caller reading ``rna_depth_x_vaf`` in a
#: ``dna_evidence_method`` column would have no way to tell a mislabelled
#: frame from a correct one.
DNA_DEPTH_X_VAF = "dna_depth_x_vaf"

#: Counted directly from a DNA alignment, in whichever unit it reported.
DNA_ALIGNMENT = "dna_alignment"

#: Reported by the source, which did not say how it got there.
#:
#: pVACseq's aggregated report supplies its own ``Allele Expr``. Passing
#: that through as if topiary had derived it would claim a derivation
#: nobody can check, and dropping it in favour of our own estimate would
#: discard the number the source actually stands behind. Neither
#: ``measured`` nor any of the arithmetic terms is true of it.
SOURCE_REPORTED = "source_reported"

#: Every named derivation, RNA and DNA alike.
#:
#: One set rather than one per assay: the question a caller asks is
#: "is this a derivation topiary knows about", and a term is already
#: self-identifying as to assay.
READ_COUNT_METHODS = frozenset({
    RNA_ALIGNMENT, RNA_DEPTH_X_VAF, RNA_DEPTH_X_SOURCE_VAF,
    CDS_OVERLAP_READS, TPM_X_DNA_VAF, SOURCE_REPORTED,
    DNA_ALIGNMENT, DNA_DEPTH_X_VAF,
})

#: How each derivation maps onto :data:`~topiary.PROVENANCE_VALUES`.
#:
#: Two vocabularies answer different questions — this one says *how* a
#: number was obtained, ``field_provenance`` says *how real* it is — and
#: the mapping between them belongs in one place so a reader and a
#: fragment builder cannot disagree about whether depth x VAF counts as
#: measured. It does not: only a direct count does.
METHOD_PROVENANCE = MappingProxyType({
    RNA_ALIGNMENT: "measured",
    RNA_DEPTH_X_VAF: "approximated",
    RNA_DEPTH_X_SOURCE_VAF: "approximated",
    CDS_OVERLAP_READS: "approximated",
    TPM_X_DNA_VAF: "approximated",
    # The source stands behind it, but did not say how it got there, so
    # it cannot be called measured.
    SOURCE_REPORTED: "approximated",
    DNA_ALIGNMENT: "measured",
    DNA_DEPTH_X_VAF: "approximated",
})


def provenance_for_method(method):
    """How real a value obtained by *method* is, or ``None`` if unstated.

    Parameters
    ----------
    method : str or None
        One of :data:`READ_COUNT_METHODS`.

    Returns
    -------
    str or None
        A ``field_provenance`` value, or ``None`` when no method was
        recorded — which means the number was not derived here and
        carries no claim either way.
    """
    if not is_stated(method):
        return None
    resolved = METHOD_PROVENANCE.get(str(method).strip())
    if resolved is None:
        raise ValueError(
            f"{method!r} is not a known derivation; use one of "
            f"{sorted(READ_COUNT_METHODS)}."
        )
    return resolved


#: What a read count counts.
#:
#: A count needs a subject as well as a derivation. isovar counts
#: **fragments**; a depth x VAF estimate is inherently about **reads**,
#: because depth is a read depth. Five fragments and five reads are
#: different bars, and ``n_rna_alt`` — which carries whichever unit the
#: source reported — cannot say which was cleared without this.
#:
#: Within one run the unit is internally consistent, so a ranking does
#: not change. The harm is in things that travel: a documented
#: ``n_rna_alt > 5`` threshold, a config copied between projects, a
#: number in a paper.
#:
#: Perfect cross-path comparability is not available and is not the
#: goal — converting a read estimate to fragments needs library
#: information the source does not carry. The goal is that every path
#: names its subject, and a source that cannot supply the subject a
#: caller wants says so rather than substituting.
FRAGMENTS = "fragments"
READS = "reads"

READ_SUBJECTS = frozenset({FRAGMENTS, READS})

#: What each derivation counts, where the derivation determines it.
#:
#: A depth x VAF estimate is about reads whatever produced it, because
#: depth is a read depth. ``rna_reads`` is left out deliberately: it
#: says the count came from an alignment, not whether the aligner was
#: counting reads or fragments, so the producer has to say.
METHOD_SUBJECT = MappingProxyType({
    RNA_DEPTH_X_VAF: READS,
    RNA_DEPTH_X_SOURCE_VAF: READS,
    CDS_OVERLAP_READS: READS,
    DNA_DEPTH_X_VAF: READS,
})


#: ``n_rna_*`` column → (fragment column, read column) it prefers.
#:
#: A paired-end fragment is one molecule read twice, so it is one piece
#: of evidence and two reads. Where a source reports both, fragments are
#: the better count; where it reports only reads, reads are what there
#: is. ``n_rna_*`` takes the better one and
#: ``rna_evidence_subject`` says which it took, so a threshold is
#: written once and a number that travels can still name its unit.
RNA_EVIDENCE_PREFERENCE = MappingProxyType({
    "n_rna_alt": ("n_rna_alt_fragments", "n_rna_alt_reads"),
    "n_rna_ref": ("n_rna_ref_fragments", "n_rna_ref_reads"),
    "n_rna_other": ("n_rna_other_fragments", "n_rna_other_reads"),
    "n_rna_overlapping": ("n_rna_overlapping_fragments", "n_rna_overlapping_reads"),
    "n_rna_supporting_protein_sequence": (
        "n_rna_alt_fragments_supporting_protein_sequence",
        "n_rna_alt_reads_supporting_protein_sequence",
    ),
})


def attach_rna_evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ``n_rna_*`` columns and ``rna_evidence_subject`` to *df*.

    The columns a threshold should be written against. Each takes the
    fragment count where the frame has one and the read count
    otherwise, and ``rna_evidence_subject`` records which — per row,
    since a frame can mix sources.

    Leaves the underlying ``n_*_reads`` / ``n_*_fragments`` columns
    alone: they say exactly what they hold, and a caller who needs one
    unit specifically should name it. A row that would mix units across
    its canonical columns is rejected because one subject label could
    not describe it truthfully.
    """
    out = df.copy()
    subject = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    for target, (fragment_col, read_col) in RNA_EVIDENCE_PREFERENCE.items():
        fragments = (
            out[fragment_col] if fragment_col in out.columns else None
        )
        reads = out[read_col] if read_col in out.columns else None
        if fragments is None and reads is None:
            continue
        if fragments is None:
            values, from_fragments = reads, pd.Series(False, index=out.index)
        elif reads is None:
            values, from_fragments = fragments, fragments.notna()
        else:
            from_fragments = fragments.notna()
            values = fragments.where(from_fragments, reads)
        if not values.notna().any():
            continue
        candidate_subject = from_fragments.map({True: FRAGMENTS, False: READS})
        mixed = (
            values.notna()
            & subject.notna()
            & candidate_subject.notna()
            & subject.fillna("").ne(candidate_subject.fillna(""))
        )
        if mixed.any():
            rows = list(out.index[mixed][:5])
            raise ValueError(
                f"Rows {rows} mix RNA evidence units across canonical "
                f"n_rna_* columns. Use one unit per row or read the "
                f"unit-specific columns directly."
            )
        out[target] = values
        subject = subject.where(
            values.isna() | subject.notna(),
            candidate_subject,
        )
    if subject.notna().any():
        out["rna_evidence_subject"] = subject
    elif (
        "rna_evidence_subject" in out.columns
        and not out["rna_evidence_subject"].notna().any()
    ):
        out = out.drop(columns=["rna_evidence_subject"])
    return out


#: How the protein sequence a prediction was made on came to exist.
#:
#: ``source_type`` says what an antigen *is* (``"variant:snv"``), which
#: is biology and deliberately says nothing about method. This says how
#: the sequence was obtained, which is the question you ask when
#: auditing a ranking: an assembled sequence carries the patient's other
#: variants and any phasing the reads support; a translated one carries
#: the reference everywhere except the variant itself.
ISOVAR_ASSEMBLY = "isovar_assembly"
VARCODE_TRANSLATION = "varcode_translation"
LENS_PEP_CONTEXT = "lens_pep_context"
PVACSEQ_EPITOPE = "pvacseq_epitope"
CALLER_SUPPLIED = "caller_supplied"

SEQUENCE_SOURCES = frozenset({
    ISOVAR_ASSEMBLY, VARCODE_TRANSLATION, LENS_PEP_CONTEXT,
    PVACSEQ_EPITOPE, CALLER_SUPPLIED,
})


def attach_sequence_source(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Record how every row's protein sequence was obtained.

    Parameters
    ----------
    df : pandas.DataFrame
    source : str
        One of :data:`SEQUENCE_SOURCES`.

    Returns
    -------
    pandas.DataFrame
    """
    if source not in SEQUENCE_SOURCES:
        raise ValueError(
            f"sequence_source must be one of {sorted(SEQUENCE_SOURCES)}, "
            f"got {source!r}. An unnamed provenance is the same as none."
        )
    out = df.copy()
    out["sequence_source"] = source
    return out


#: Columns this module writes, in the order a reader should expect them.
#:
#: One name per quantity, plus one column saying what unit the counts
#: are in and one saying where they came from. The unit-specific
#: ``n_*_reads`` / ``n_*_fragments`` fields live on
#: :class:`~topiary.ProteinFragment`, where a caller who needs one unit
#: specifically can name it; on a frame they were exact duplicates of
#: ``n_rna_*`` for every source that reports a single unit, which is
#: every reader.
READ_EVIDENCE_COLUMNS = (
    "n_rna_alt",
    "n_rna_ref",
    "n_rna_overlapping",
    "rna_evidence_subject",
    "rna_evidence_method",
    "rna_alt_expression",
    "rna_alt_expression_method",
    "sequence_source",
)


def _counts(values) -> pd.Series:
    """Non-negative integer counts, with unstated entries left as NA."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(numeric >= 0)
    return numeric.round().astype("Int64")


def _fractions(values) -> pd.Series:
    """Fractions in [0, 1]; anything else is not a VAF and reads as absent."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where((numeric >= 0) & (numeric <= 1))


def _aligned(values, index, argument):
    """*values* as a Series on *index*, or a clear error saying why not.

    The two attach functions used to disagree here, silently and in
    opposite directions: a Series whose index did not match the frame
    was aligned by pandas on the RNA side (yielding an all-null column)
    and assigned positionally on the DNA side (yielding a misaligned
    one). Both lose data without saying so, and twins that answer the
    same question differently are worse than either answer.

    So: an already-aligned Series passes through, a bare sequence is
    positional because it has no index to honour, and a Series carrying
    a different index raises rather than guessing which of the two
    silent behaviours the caller wanted.
    """
    if values is None:
        return None
    if isinstance(values, pd.Series):
        if values.index.equals(index):
            return values
        raise ValueError(
            f"{argument}: index does not match the frame's "
            f"({len(values)} vs {len(index)} rows"
            + (", same length but different labels"
               if len(values) == len(index) else "")
            + "). Pass a Series sharing the frame's index, or "
            f"{argument}.to_numpy() to align by position."
        )
    values = list(values)
    if len(values) != len(index):
        raise ValueError(
            f"{argument}: got {len(values)} values for {len(index)} rows."
        )
    return pd.Series(values, index=index)


def _vaf_from(stated, alt, depth, index):
    """The canonical variant allele fraction for one assay.

    Prefers the fraction the source stated over one recomputed from the
    counts. They agree to rounding where topiary derived the counts *from*
    the fraction, and where it did not, the source's own number is the
    one it stands behind.

    Absent wherever neither a stated fraction nor a usable alt/depth pair
    exists, so "no variant reads" and "nobody measured" stay apart.
    """
    if stated is not None:
        fraction = _fractions(stated)
        fraction.index = index
        if fraction.notna().any():
            return fraction
    if alt is None or depth is None:
        return pd.Series([np.nan] * len(index), index=index, dtype="float64")
    counts, total = _counts(alt), _counts(depth)
    counts.index, total.index = index, index
    usable = counts.notna() & total.notna() & (total > 0)
    return (
        counts.astype("Float64") / total.astype("Float64")
    ).where(usable).astype("float64")


def other_allele_count(overlapping, alt, ref):
    """Reads at the locus supporting neither the reference nor the alt allele.

    A third allele, a sequencing error, or a nearby indel all land here.
    Worth seeing separately: a variant with 40 alt, 10 ref and 50 "other"
    reads is a messy locus, and a caller weighting by depth of support
    should be able to tell it from a clean 40/60.

    Returns ``NA`` unless all three inputs are stated **and** *ref* was
    counted independently. Where *ref* was derived as ``depth - alt`` it
    already absorbs the other alleles, so the difference is
    definitionally zero and reporting it would assert a clean locus
    nobody checked.

    Parameters
    ----------
    overlapping : pandas.Series
        Total reads covering the position.
    alt, ref : pandas.Series
        Independently counted alt and reference support.

    Returns
    -------
    pandas.Series
        Nullable integer, clipped at zero — counts that do not add up
        are a source's inconsistency, not a negative quantity.
    """
    total, a, r = _counts(overlapping), _counts(alt), _counts(ref)
    usable = total.notna() & a.notna() & r.notna()
    return (total - a - r).where(usable).clip(lower=0)


def split_reads_by_vaf(depth, vaf):
    """``(alt, ref)`` counts from a depth and a variant fraction.

    Assay-agnostic: both :func:`attach_rna_evidence` and
    :func:`attach_dna_evidence` split their depth with this.

    Both are ``NA`` wherever either input is absent — an estimate needs
    both halves, and inventing one of them is how a missing value becomes
    a number nobody measured.

    Parameters
    ----------
    depth : pandas.Series
        Reads covering the position.
    vaf : pandas.Series
        Variant allele fraction, 0..1.

    Returns
    -------
    (pandas.Series, pandas.Series)
        Alt and reference counts, nullable integer.
    """
    depth = _counts(depth)
    fraction = _fractions(vaf)
    usable = depth.notna() & fraction.notna()
    alt = (depth.astype("Float64") * fraction).round().astype("Int64")
    alt = alt.where(usable)
    ref = (depth - alt).where(usable)
    return alt, ref.clip(lower=0)


def attach_rna_evidence(
    df: pd.DataFrame,
    *,
    overlapping=None,
    vaf=None,
    supporting=None,
    supporting_method=None,
    expression=None,
    dna_vaf=None,
    reported_rna_alt_expression=None,
    vaf_method=RNA_DEPTH_X_VAF,
) -> pd.DataFrame:
    """Write the read-evidence columns onto *df*, naming each derivation.

    Every argument is a Series (or ``None`` when the source has no such
    column). Nothing is invented: a quantity no row can answer is omitted;
    within a partially populated column, unanswered rows remain ``NA``.
    Thus "no support" and "cannot answer" stay distinguishable.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to write onto. Not mutated; a copy is returned.
    overlapping : pandas.Series, optional
        Reads covering the variant position. Taken as a direct count.
    vaf : pandas.Series, optional
        RNA variant allele fraction, used with *overlapping* to split
        alt from reference.
    supporting : pandas.Series, optional
        Reads supporting the assembled protein sequence.
    supporting_method : str, optional
        Which derivation *supporting* is — :data:`RNA_READS` when it
        counts reads carrying the variant, :data:`CDS_OVERLAP_READS`
        when it counts reads overlapping the peptide's CDS instead.
    expression : pandas.Series, optional
        Transcript or gene abundance, for the expression proxy.
    dna_vaf : pandas.Series, optional
        DNA variant allele fraction, used with *expression*.

    Returns
    -------
    pandas.DataFrame
    """
    overlapping = _aligned(overlapping, df.index, "overlapping")
    vaf = _aligned(vaf, df.index, "vaf")
    expression = _aligned(expression, df.index, "expression")
    dna_vaf = _aligned(dna_vaf, df.index, "dna_vaf")
    reported_rna_alt_expression = _aligned(
        reported_rna_alt_expression, df.index, "reported_rna_alt_expression",
    )
    out = df.copy()
    n_rows = len(out)
    empty = pd.Series([pd.NA] * n_rows, index=out.index, dtype="Int64")
    has_count = pd.Series(False, index=out.index)

    # Emitted only where the source can answer, exactly as on the DNA
    # side. A source with no RNA at all gets no n_rna_* columns rather
    # than a set full of nulls claiming it looked and found nothing.
    overlapping_counts = (
        _counts(overlapping) if overlapping is not None else None
    )
    if overlapping_counts is not None and overlapping_counts.notna().any():
        out["n_rna_overlapping"] = overlapping_counts
        has_count |= overlapping_counts.notna()
    if overlapping is not None and vaf is not None:
        alt, ref = split_reads_by_vaf(overlapping, vaf)
        method = vaf_method
    else:
        alt, ref, method = empty, empty, None
    if method is not None and alt.notna().any():
        out["n_rna_alt"] = alt
        out["n_rna_ref"] = ref
        out["rna_evidence_method"] = pd.Series(
            [method if pd.notna(a) else pd.NA for a in alt],
            index=out.index, dtype="object",
        )
        has_count |= alt.notna() | ref.notna()
    # No n_rna_other by construction: ref came from depth - alt, which
    # already absorbs any third allele. A source that counts ref
    # independently gets a real one via other_allele_count.
    if vaf is not None or method is not None:
        canonical_vaf = _vaf_from(vaf, alt, overlapping, out.index)
        if canonical_vaf.notna().any():
            out["rna_vaf"] = canonical_vaf

    if supporting is not None:
        if supporting_method not in READ_COUNT_METHODS:
            raise ValueError(
                f"supporting_method must name a derivation from "
                f"{sorted(READ_COUNT_METHODS)}, got "
                f"{supporting_method!r}. A count whose derivation is "
                f"unnamed cannot be told from one that was measured."
            )
        # A count of reads overlapping the peptide's CDS is not a count
        # of reads supporting the assembled protein sequence, and only
        # an assembler reports the latter. Rather than emit a column
        # whose name overstates what a reader has, the source's own
        # column is left to pass through under its own name; the
        # assembled count lives on ProteinFragment where it is real.
        pass

    reported = None
    if reported_rna_alt_expression is not None:
        # The source supplied the number. Keep it and say where it came
        # from, rather than overwriting it with our own estimate or
        # passing it through unlabelled as if we had derived it.
        reported = pd.to_numeric(
            reported_rna_alt_expression, errors="coerce"
        )
        if reported.notna().any():
            out["rna_alt_expression"] = reported
            out["rna_alt_expression_method"] = pd.Series(
                [SOURCE_REPORTED if pd.notna(v) else pd.NA for v in reported],
                index=out.index, dtype="object",
            )
    if (
        (reported is None or not reported.notna().any())
        and expression is not None
        and dna_vaf is not None
    ):
        abundance = pd.to_numeric(expression, errors="coerce")
        fraction = _fractions(dna_vaf)
        estimate = (abundance * fraction).where(
            abundance.notna() & fraction.notna()
        )
        if estimate.notna().any():
            out["rna_alt_expression"] = estimate
            out["rna_alt_expression_method"] = pd.Series(
                [TPM_X_DNA_VAF if pd.notna(v) else pd.NA for v in estimate],
                index=out.index, dtype="object",
            )
    # No else: a source with no abundance figure gets no expression
    # columns rather than two full of nulls, matching every other
    # evidence column. A null column claims the source looked and found
    # nothing; an absent one says it cannot answer.
    if has_count.any():
        out["rna_evidence_subject"] = pd.Series(
            [READS if value else pd.NA for value in has_count],
            index=out.index, dtype="object",
        )
    return out


def attach_dna_evidence(
    df: pd.DataFrame,
    *,
    depth=None,
    vaf=None,
    alt=None,
    ref=None,
    method=None,
    subject=None,
) -> pd.DataFrame:
    """Write the DNA-evidence columns onto *df*, naming the derivation.

    The DNA twin of :func:`attach_rna_evidence`, deliberately the same
    shape: ``n_dna_alt`` / ``n_dna_ref`` / ``n_dna_overlapping`` /
    ``dna_vaf`` / ``dna_evidence_subject`` / ``dna_evidence_method``.
    A caller who has written a filter against RNA depth can write the
    DNA one by changing three letters.

    A source supplies either counts or a depth-and-fraction pair. Counts
    are taken as measured; a depth x VAF split is arithmetic and is
    labelled :data:`DNA_DEPTH_X_VAF` so it is never mistaken for one.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to write onto. Not mutated; a copy is returned.
    depth : pandas.Series, optional
        Reads covering the variant position in the DNA alignment.
    vaf : pandas.Series, optional
        DNA variant allele fraction, used with *depth* to split alt from
        reference when *alt* is not given directly.
    alt, ref : pandas.Series, optional
        Direct counts, when the source reports them rather than a
        fraction. Take precedence over the *depth* x *vaf* split.
    method : str, optional
        Overrides the derivation name. Defaults to
        :data:`DNA_ALIGNMENT` for direct counts and
        :data:`DNA_DEPTH_X_VAF` for a split.
    subject : {"reads", "fragments"}, optional
        What the counts count. Only a caller supplying direct counts
        knows this; a depth x VAF split infers :data:`READS` because
        depth is a read depth. Where neither applies the column is
        omitted rather than guessed — a wrong unit is how a documented
        threshold silently changes meaning between sources.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    Nothing is invented. A source that reports no DNA at all — isovar,
    and LENS, whose single ``vaf`` never names its assay — gets no DNA
    columns rather than a column of nulls, so
    :func:`available_evidence_columns` keeps meaning "what this source
    could answer".
    """
    depth = _aligned(depth, df.index, "depth")
    vaf = _aligned(vaf, df.index, "vaf")
    alt = _aligned(alt, df.index, "alt")
    ref = _aligned(ref, df.index, "ref")
    if subject is not None and subject not in (READS, FRAGMENTS):
        raise ValueError(
            f"attach_dna_evidence: subject must be {READS!r} or "
            f"{FRAGMENTS!r}, got {subject!r}."
        )
    given_subject = subject
    out = df.copy()
    empty = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    if alt is not None:
        counted_alt = _counts(alt)
        counted_alt.index = out.index
        if ref is not None:
            counted_ref = _counts(ref)
            counted_ref.index = out.index
        elif depth is not None:
            total = _counts(depth)
            total.index = out.index
            counted_ref = (total - counted_alt).clip(lower=0)
        else:
            counted_ref = empty
        derivation = method or DNA_ALIGNMENT
    elif depth is not None and vaf is not None:
        counted_alt, counted_ref = split_reads_by_vaf(depth, vaf)
        counted_alt.index, counted_ref.index = out.index, out.index
        derivation = method or DNA_DEPTH_X_VAF
    else:
        counted_alt, counted_ref, derivation = empty, empty, None

    if depth is not None:
        overlapping = _counts(depth)
        overlapping.index = out.index
    else:
        overlapping = empty

    # Emit a column only where the source could answer it. A frame that
    # carries `n_dna_alt` full of nulls says "measured nothing"; one that
    # omits the column says "cannot measure this", and
    # available_evidence_columns is only a useful signal if that
    # distinction is kept at the column level as well as the value level.
    if derivation is not None and counted_alt.notna().any():
        out["n_dna_alt"] = counted_alt
        out["n_dna_ref"] = counted_ref
        out["dna_evidence_method"] = pd.Series(
            [derivation if pd.notna(a) else pd.NA for a in counted_alt],
            index=out.index, dtype="object",
        )
        # Only where something actually determines it. A depth x VAF
        # split is about reads whatever produced it, because depth is a
        # read depth; a direct count's unit is known only to whoever
        # counted, so an unstated one is left absent rather than
        # asserted. Hardcoding READS here would have made the column a
        # literal wearing a data column's clothes.
    if overlapping.notna().any():
        out["n_dna_overlapping"] = overlapping
    if ref is not None:
        other = other_allele_count(
            overlapping, counted_alt, counted_ref,
        )
        if other.notna().any():
            out["n_dna_other"] = other
    if vaf is not None or derivation is not None:
        canonical_vaf = _vaf_from(vaf, counted_alt, overlapping, out.index)
        if canonical_vaf.notna().any():
            out["dna_vaf"] = canonical_vaf

    has_count = counted_alt.notna() | counted_ref.notna() | overlapping.notna()
    inferred_subject = (
        READS if overlapping.notna().any()
        else METHOD_SUBJECT.get(derivation)
    )
    if (
        has_count.any()
        and given_subject is not None
        and inferred_subject is not None
        and given_subject != inferred_subject
    ):
        raise ValueError(
            f"attach_dna_evidence: subject {given_subject!r} contradicts "
            f"the supplied read depth or {derivation!r}, which counts "
            f"{inferred_subject}."
        )
    resolved_subject = inferred_subject or given_subject
    if resolved_subject is not None and has_count.any():
        out["dna_evidence_subject"] = pd.Series(
            [resolved_subject if value else pd.NA for value in has_count],
            index=out.index, dtype="object",
        )
    return out


#: Columns renamed since 5.46.0, old name -> new name.
#:
#: Exported because a consumer that reads these names out of a frame --
#: vaxrank does, with ``row.get(...)`` and ordered fallback tuples --
#: cannot recover from a rename by pattern-matching. Two of these
#: renames are actively hostile to guessing: ``vaf`` looks like it
#: should become ``rna_vaf`` and does not (that is the canonical
#: cross-source fraction; ``lens_vaf`` is LENS's own, whose assay the
#: file never states), and a ``.get()`` that misses returns ``None``
#: rather than raising, so the failure is a silent zero.
#:
#: Reader frames emit no compatibility aliases: two output columns for
#: one quantity is the ambiguity the renames existed to remove. Serialized
#: and direct ``ProteinFragment`` input accepts its old field names for 5.x
#: compatibility, while always emitting the new names.
RENAMED_COLUMNS = MappingProxyType({
    # LENS: its own numbers now carry its name.
    "vaf": "lens_vaf",
    "rna_reads_covering_genomic_origin":
        "lens_rna_reads_covering_genomic_origin",
    "rna_reads_covering_genomic_origin_with_peptide_cds":
        "lens_rna_reads_covering_genomic_origin_with_peptide_cds",
    "proportion_rna_reads_covering_genomic_origin_with_peptide_cds":
        "lens_proportion_rna_reads_covering_genomic_origin_with_peptide_cds",
    # pVACseq: same.
    "tumor_dna_depth": "pvacseq_tumor_dna_depth",
    "tumor_dna_vaf": "pvacseq_tumor_dna_vaf",
    "tumor_rna_depth": "pvacseq_tumor_rna_depth",
    "tumor_rna_vaf": "pvacseq_tumor_rna_vaf",
    "normal_depth": "pvacseq_normal_depth",
    "normal_vaf": "pvacseq_normal_vaf",
    # Unit-specific counts, now scoped by assay.
    "n_alt_reads": "n_rna_alt_reads",
    "n_alt_fragments": "n_rna_alt_fragments",
    "n_ref_reads": "n_rna_ref_reads",
    "n_ref_fragments": "n_rna_ref_fragments",
    "n_other_reads": "n_rna_other_reads",
    "n_other_fragments": "n_rna_other_fragments",
    "n_overlapping_reads": "n_rna_overlapping_reads",
    "n_overlapping_fragments": "n_rna_overlapping_fragments",
    "n_alt_reads_supporting_protein_sequence":
        "n_rna_alt_reads_supporting_protein_sequence",
    "n_alt_fragments_supporting_protein_sequence":
        "n_rna_alt_fragments_supporting_protein_sequence",
})


def renamed_column(name: str):
    """What *name* was renamed to, or ``None`` if it was not renamed.

    Parameters
    ----------
    name : str
        A column name from topiary 5.46.0 or earlier.

    Returns
    -------
    str or None

    Notes
    -----
    Prefer this to a fuzzy match. ``vaf`` is the case that proves the
    point: the closest surviving name is ``rna_vaf``, and that is the
    wrong answer -- it is the canonical cross-source fraction, while
    ``vaf`` became ``lens_vaf``, LENS's own fraction of unstated assay.
    """
    return RENAMED_COLUMNS.get(name)


#: Tool name -> the prefix its own columns carry on a topiary frame.
#:
#: A canonical column such as ``rna_vaf`` means the same thing whichever
#: reader produced it, because topiary derived it under a stated method.
#: A tool's own columns do not have that guarantee: pVACseq's
#: ``Tumor RNA VAF`` and LENS's ``vaf`` are both "a variant allele
#: fraction" and are not the same quantity — LENS never says which assay
#: its fraction is from. Landing both as a bare ``vaf`` would let a
#: stacked frame silently answer with whichever tool happened to fill
#: the column first.
#:
#: So a tool's own numbers keep the tool's name. ``lens_vaf`` and
#: ``pvacseq_tumor_rna_vaf`` can coexist in one frame, disagree, and
#: still be attributable; the unprefixed canonical columns are the ones
#: to filter on.
SOURCE_PREFIXES = MappingProxyType({
    "isovar": "isovar_",
    "lens": "lens_",
    "pvacseq": "pvacseq_",
})


def source_column(source: str, name: str) -> str:
    """The prefixed column name *source* uses for its own field *name*.

    Parameters
    ----------
    source : str
        A key of :data:`SOURCE_PREFIXES`.
    name : str
        The tool's own column name, already normalized to snake_case.

    Returns
    -------
    str
        e.g. ``source_column("lens", "vaf") == "lens_vaf"``.

    Raises
    ------
    ValueError
        If *source* is not a known tool. Guessing a prefix would let a
        typo create a column nobody can find again.
    """
    if source not in SOURCE_PREFIXES:
        raise ValueError(
            f"source_column: unknown source {source!r}; expected one of "
            f"{sorted(SOURCE_PREFIXES)}. Add it to SOURCE_PREFIXES "
            f"rather than passing a prefix directly, so every reader "
            f"spells the same tool the same way."
        )
    return f"{SOURCE_PREFIXES[source]}{name}"


def source_columns(df: pd.DataFrame, source: str = None) -> tuple:
    """The tool-specific columns present in *df*, in column order.

    Parameters
    ----------
    df : pandas.DataFrame
    source : str, optional
        Restrict to one tool. ``None`` (default) returns the columns of
        every known tool, which is what you want to see which tools a
        stacked frame is carrying evidence from.

    Returns
    -------
    tuple of str

    Notes
    -----
    Answers the question "what did each tool actually say", which is the
    one you ask when two sources disagree on a canonical column.
    """
    if source is not None and source not in SOURCE_PREFIXES:
        raise ValueError(
            f"source_columns: unknown source {source!r}; expected one "
            f"of {sorted(SOURCE_PREFIXES)}."
        )
    prefixes = (
        (SOURCE_PREFIXES[source],) if source is not None
        else tuple(SOURCE_PREFIXES.values())
    )
    return tuple(c for c in df.columns if c.startswith(prefixes))


#: Canonical RNA columns, in the order a reader writes them.
RNA_EVIDENCE_COLUMNS = (
    "n_rna_alt",
    "n_rna_ref",
    "n_rna_overlapping",
    "n_rna_other",
    "rna_vaf",
    "rna_evidence_subject",
    "rna_evidence_method",
    "rna_alt_expression",
    "rna_alt_expression_method",
    "gene_expression",
    "transcript_expression",
)

#: Canonical DNA columns — the same shape as
#: :data:`RNA_EVIDENCE_COLUMNS`, minus expression and abundance, which have
#: no DNA meaning.
DNA_EVIDENCE_COLUMNS = (
    "n_dna_alt",
    "n_dna_ref",
    "n_dna_overlapping",
    "n_dna_other",
    "dna_vaf",
    "dna_evidence_subject",
    "dna_evidence_method",
)

#: The evidence columns a reader emits when its source can answer.
#:
#: The canonical layer only. A tool's own numbers are prefixed and found
#: with :func:`source_columns`; the unit-specific ``n_rna_alt_reads`` /
#: ``n_rna_alt_fragments`` pair appears only where a source reports both.
EVIDENCE_COLUMNS = (
    RNA_EVIDENCE_COLUMNS + DNA_EVIDENCE_COLUMNS + ("sequence_source",)
)


_COUNT_EVIDENCE_COLUMNS = (
    "n_rna_alt",
    "n_rna_ref",
    "n_rna_overlapping",
    "n_rna_other",
    "n_dna_alt",
    "n_dna_ref",
    "n_dna_overlapping",
    "n_dna_other",
)


def _groupby(df, columns):
    """Group by one or more columns without pandas' length-one warning."""
    grouper = columns[0] if len(columns) == 1 else columns
    return df.groupby(grouper, sort=False, dropna=False, observed=True)


def _identity(columns, key):
    """Return a readable identity mapping for one groupby key."""
    values = (key,) if len(columns) == 1 else key
    return dict(zip(columns, values))


def _single_sample_evidence_rows(df, group_keys, evidence_columns):
    """Collapse repeated prediction rows to one evidence row per sample."""
    identity_columns = [*group_keys, "sample_name"]
    rows = []
    for key, group in _groupby(df, identity_columns):
        identity = _identity(identity_columns, key)
        row = dict(identity)
        for column in evidence_columns:
            stated = group.loc[stated_values(group[column]), column]
            distinct = stated.drop_duplicates()
            if len(distinct) > 1:
                raise ValueError(
                    f"Contradictory {column!r} values within one sample for "
                    f"{identity}: {distinct.tolist()!r}."
                )
            row[column] = distinct.iloc[0] if len(distinct) else pd.NA
        rows.append(row)
    return pd.DataFrame(rows, columns=[*identity_columns, *evidence_columns])


def _sum_complete_counts(group, column, identity):
    """Sum a count only when every represented sample states it."""
    values = group[column]
    stated = stated_values(values)
    stated_counts = values[stated]
    if stated_counts.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).any():
        raise ValueError(
            f"Evidence count {column!r} must be a non-negative integer for "
            f"{identity}; boolean values are not counts."
        )
    try:
        numeric = pd.to_numeric(stated_counts, errors="raise").astype(float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Evidence count {column!r} must be numeric for {identity}: "
            f"{stated_counts.tolist()!r}."
        ) from None
    if (
        not np.isfinite(numeric).all()
        or (numeric < 0).any()
        or (numeric % 1 != 0).any()
    ):
        raise ValueError(
            f"Evidence count {column!r} must contain non-negative finite "
            f"integers for {identity}: {stated_counts.tolist()!r}."
        )
    if not stated.all():
        return pd.NA
    return int(numeric.sum())


def _common_assay_metadata(group, assay, identity, pooled_counts):
    """Return metadata shared by every sample contributing assay counts."""
    if not pooled_counts:
        return {}
    result = {}
    for suffix in ("subject", "method"):
        column = f"{assay}_evidence_{suffix}"
        if column not in group.columns or not stated_values(group[column]).all():
            raise ValueError(
                f"Cannot pool {assay.upper()} evidence for {identity}: every "
                f"represented sample must state {column!r}."
            )
        distinct = group[column].drop_duplicates()
        if len(distinct) != 1:
            raise ValueError(
                f"Cannot pool {assay.upper()} evidence for {identity}: samples "
                f"use different {column!r} values {distinct.tolist()!r}."
            )
        result[column] = distinct.iloc[0]
    return result


def aggregate_evidence_across_samples(
    df: pd.DataFrame,
    *,
    group_keys=None,
) -> pd.DataFrame:
    """Pool canonical read counts across samples without losing attribution.

    Repeated prediction rows for the same candidate and sample are collapsed
    before summing, so evidence copied onto several prediction kinds or methods
    is counted once. A canonical count is emitted only when every represented
    sample states it. RNA and DNA VAFs are recomputed as pooled alternate count
    divided by pooled overlapping count; per-sample VAFs are never averaged.

    Assay counts can be pooled only when every represented sample names the
    same evidence subject and derivation method. Mixing reads with fragments,
    or measured counts with arithmetic estimates, raises instead of flattening
    unlike quantities into one number. Expression is intentionally not pooled:
    unlike read counts, it has no generally valid cross-sample sum.

    The returned DataFrame is separate from *df*, whose per-sample values stay
    unchanged and available for sample-specific reporting and thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form prediction rows with a stated ``sample_name`` and at least
        one canonical RNA or DNA count column.
    group_keys : sequence of str, optional
        Candidate identity excluding ``sample_name``. By default Topiary's
        normal prediction identity is inferred, including ``allele_set`` when
        populated but deliberately excluding the sample dimension.

    Returns
    -------
    pandas.DataFrame
        One row per candidate identity. ``n_samples`` is the number of samples
        represented by rows for that identity. Complete canonical counts keep
        their original names, and computable VAFs are pooled from those counts.

    Raises
    ------
    ValueError
        If samples are unnamed, evidence conflicts within a sample, a count is
        invalid, or pooled values would mix evidence subjects or methods.

    Notes
    -----
    A candidate absent from a sample has no row to aggregate. ``n_samples``
    makes that visible; completeness is evaluated across the samples actually
    represented for each candidate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "aggregate_evidence_across_samples expects a pandas.DataFrame, "
            f"got {type(df).__name__}."
        )
    if df.empty:
        raise ValueError("Cannot aggregate evidence from an empty DataFrame.")
    if "sample_name" not in df.columns:
        raise ValueError(
            "Cross-sample aggregation requires a 'sample_name' column."
        )
    if not stated_values(df["sample_name"]).all():
        raise ValueError(
            "Cross-sample aggregation requires every row to state "
            "'sample_name'."
        )

    if group_keys is None:
        keys = [
            key for key in EvalContext(df).group_keys
            if key != "sample_name"
        ]
        context = EvalContext(df, group_keys=keys)
    else:
        context = EvalContext(df, group_keys=group_keys)
        keys = context.group_keys
    if "sample_name" in keys:
        raise ValueError(
            "group_keys must exclude 'sample_name'; this function pools "
            "across that dimension."
        )
    if "n_samples" in keys:
        raise ValueError("group_keys cannot contain the output column 'n_samples'.")

    count_columns = [
        column for column in _COUNT_EVIDENCE_COLUMNS if column in df.columns
    ]
    if not count_columns:
        raise ValueError(
            "Cross-sample aggregation requires at least one canonical count "
            f"column from {list(_COUNT_EVIDENCE_COLUMNS)!r}."
        )
    metadata_columns = [
        column
        for assay in ("rna", "dna")
        for column in (
            f"{assay}_evidence_subject",
            f"{assay}_evidence_method",
        )
        if column in df.columns
    ]
    evidence_columns = [*count_columns, *metadata_columns]
    normalized_df = df.assign(**{
        key: context.key_frame[key] for key in keys
    })
    per_sample = _single_sample_evidence_rows(
        normalized_df,
        keys,
        evidence_columns,
    )

    rows = []
    for key, group in _groupby(per_sample, keys):
        identity = _identity(keys, key)
        row = {**identity, "n_samples": len(group)}
        for column in count_columns:
            row[column] = _sum_complete_counts(group, column, identity)
        for assay in ("rna", "dna"):
            assay_counts = [
                column for column in count_columns
                if column.startswith(f"n_{assay}_") and is_stated(row[column])
            ]
            row.update(
                _common_assay_metadata(group, assay, identity, assay_counts)
            )
            alt = row.get(f"n_{assay}_alt", pd.NA)
            overlapping = row.get(f"n_{assay}_overlapping", pd.NA)
            if is_stated(alt) and is_stated(overlapping) and overlapping > 0:
                row[f"{assay}_vaf"] = alt / overlapping
        rows.append(row)

    result = pd.DataFrame(rows)
    for column in count_columns:
        if column in result and stated_values(result[column]).any():
            result[column] = result[column].astype("Int64")
        elif column in result:
            result = result.drop(columns=column)
    for column in ("rna_vaf", "dna_vaf"):
        if column in result and not stated_values(result[column]).any():
            result = result.drop(columns=column)
    ordered = [
        *keys,
        "n_samples",
        *(column for column in EVIDENCE_COLUMNS if column in result.columns),
    ]
    return result.loc[:, ordered]


def available_evidence_columns(df: pd.DataFrame) -> tuple:
    """Which of :data:`EVIDENCE_COLUMNS` *df* actually has.

    Readers emit an evidence column only where the source can answer, so
    the set is the same *vocabulary* across readers rather than the same
    *columns*: a pVACseq aggregated report has gene-level abundance but no
    separately stated transcript abundance, and a LENS file without a
    ``tpm`` column has neither.

    That matters because naming an absent column in an expression
    **raises** rather than evaluating to NaN — which is the right
    behaviour, and the reason to check before writing a config that has
    to run against more than one source.

    The alternative, emitting all-null columns everywhere, is worse: a
    column that is present and empty asserts the question was asked and
    answered as nothing. Absent beats substituted here as everywhere
    else.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    tuple of str
        In :data:`EVIDENCE_COLUMNS` order.

    Examples
    --------
    >>> missing = set(EVIDENCE_COLUMNS) - set(available_evidence_columns(df))
    ... # doctest: +SKIP
    """
    return tuple(c for c in EVIDENCE_COLUMNS if c in df.columns)


def describe_read_evidence(df: pd.DataFrame) -> dict:
    """``{column: method}`` for every evidence column *df* populates.

    The frame-level summary of what the per-row method columns say, for
    a caller reporting how a run's numbers were obtained without walking
    the rows.
    """
    described = {}
    for column, method_column in (
        ("n_rna_alt", "rna_evidence_method"),
        ("n_rna_ref", "rna_evidence_method"),
        ("n_rna_overlapping", "rna_evidence_method"),
        ("rna_alt_expression", "rna_alt_expression_method"),
    ):
        if column not in df.columns or method_column not in df.columns:
            continue
        if not df[column].notna().any():
            continue
        methods = df[method_column][stated_values(df[method_column])]
        for method in sorted(set(methods.tolist())):
            described.setdefault(column, method)
    return described
