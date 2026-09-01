"""RNA read-level evidence, and saying where each number came from.

Readers report RNA support differently, and the difference matters. isovar
counts reads supporting an assembled protein sequence. pVACseq reports a
depth and a variant allele fraction, from which the split is arithmetic.
LENS counts reads overlapping the peptide's CDS, which is a real count of
something adjacent to what was asked for. A consumer weighting a candidate
by depth of support needs to know which of those it has.

So every derived number carries the name of its derivation:

============================  ==================================================
``rna_reads``                 Counted directly from an RNA alignment.
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

``None`` throughout means the source could not answer, which is not the
same as answering zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from types import MappingProxyType

from .ranking import is_stated, stated_values

#: Counted directly from an RNA alignment.
RNA_READS = "rna_reads"

#: depth x VAF, rounded — arithmetic, not counted.
RNA_DEPTH_X_VAF = "rna_depth_x_vaf"

#: Counted, but of reads overlapping the peptide's coding sequence rather
#: than supporting the variant allele.
CDS_OVERLAP_READS = "cds_overlap_reads"

#: Transcript abundance x DNA variant allele fraction — an expression
#: proxy, not a read count.
TPM_X_DNA_VAF = "tpm_x_dna_vaf"

#: Reported by the source, which did not say how it got there.
#:
#: pVACseq's aggregated report supplies its own ``Allele Expr``. Passing
#: that through as if topiary had derived it would claim a derivation
#: nobody can check, and dropping it in favour of our own estimate would
#: discard the number the source actually stands behind. Neither
#: ``measured`` nor any of the arithmetic terms is true of it.
SOURCE_REPORTED = "source_reported"

READ_COUNT_METHODS = frozenset({
    RNA_READS, RNA_DEPTH_X_VAF, CDS_OVERLAP_READS, TPM_X_DNA_VAF,
    SOURCE_REPORTED,
})

#: How each derivation maps onto :data:`~topiary.PROVENANCE_VALUES`.
#:
#: Two vocabularies answer different questions — this one says *how* a
#: number was obtained, ``field_provenance`` says *how real* it is — and
#: the mapping between them belongs in one place so a reader and a
#: fragment builder cannot disagree about whether depth x VAF counts as
#: measured. It does not: only a direct count does.
METHOD_PROVENANCE = MappingProxyType({
    RNA_READS: "measured",
    RNA_DEPTH_X_VAF: "approximated",
    CDS_OVERLAP_READS: "approximated",
    TPM_X_DNA_VAF: "approximated",
    # The source stands behind it, but did not say how it got there, so
    # it cannot be called measured.
    SOURCE_REPORTED: "approximated",
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
READ_EVIDENCE_COLUMNS = (
    "n_overlapping_reads",
    "n_alt_reads",
    "n_ref_reads",
    "n_alt_reads_supporting_protein_sequence",
    "read_count_method",
    "supporting_read_count_method",
    "variant_allele_expression",
    "variant_allele_expression_method",
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


def split_reads_by_vaf(depth, vaf):
    """``(n_alt_reads, n_ref_reads)`` from a depth and a variant fraction.

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


def attach_read_evidence(
    df: pd.DataFrame,
    *,
    overlapping=None,
    vaf=None,
    supporting=None,
    supporting_method=None,
    expression=None,
    dna_vaf=None,
    reported_variant_allele_expression=None,
) -> pd.DataFrame:
    """Write the read-evidence columns onto *df*, naming each derivation.

    Every argument is a Series (or ``None`` when the source has no such
    column). Nothing is invented: a quantity whose inputs are absent is
    left ``NA``, and its method column is ``NA`` too, so "no support" and
    "cannot answer" stay distinguishable.

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
    out = df.copy()
    n_rows = len(out)
    empty = pd.Series([pd.NA] * n_rows, index=out.index, dtype="Int64")

    out["n_overlapping_reads"] = (
        _counts(overlapping) if overlapping is not None else empty
    )
    if overlapping is not None and vaf is not None:
        alt, ref = split_reads_by_vaf(overlapping, vaf)
        method = RNA_DEPTH_X_VAF
    else:
        alt, ref, method = empty, empty, None
    out["n_alt_reads"] = alt
    out["n_ref_reads"] = ref
    out["read_count_method"] = pd.Series(
        [method if method and pd.notna(a) else pd.NA for a in alt],
        index=out.index, dtype="object",
    )

    if supporting is not None:
        if supporting_method not in READ_COUNT_METHODS:
            raise ValueError(
                f"supporting_method must name a derivation from "
                f"{sorted(READ_COUNT_METHODS)}, got "
                f"{supporting_method!r}. A count whose derivation is "
                f"unnamed cannot be told from one that was measured."
            )
        supporting_counts = _counts(supporting)
        out["n_alt_reads_supporting_protein_sequence"] = supporting_counts
        # Record the derivation, not just accept it. Without this the
        # frame carries the count and loses how it was obtained, so a
        # fragment built from the frame cannot say whether 45 reads were
        # counted supporting the variant or counted overlapping a CDS.
        out["supporting_read_count_method"] = pd.Series(
            [supporting_method if pd.notna(v) else pd.NA
             for v in supporting_counts],
            index=out.index, dtype="object",
        )
    else:
        out["n_alt_reads_supporting_protein_sequence"] = empty
        out["supporting_read_count_method"] = pd.Series(
            [pd.NA] * n_rows, index=out.index, dtype="object"
        )

    if reported_variant_allele_expression is not None:
        # The source supplied the number. Keep it and say where it came
        # from, rather than overwriting it with our own estimate or
        # passing it through unlabelled as if we had derived it.
        reported = pd.to_numeric(
            reported_variant_allele_expression, errors="coerce"
        )
        out["variant_allele_expression"] = reported
        out["variant_allele_expression_method"] = pd.Series(
            [SOURCE_REPORTED if pd.notna(v) else pd.NA for v in reported],
            index=out.index, dtype="object",
        )
    elif expression is not None and dna_vaf is not None:
        abundance = pd.to_numeric(expression, errors="coerce")
        fraction = _fractions(dna_vaf)
        estimate = (abundance * fraction).where(
            abundance.notna() & fraction.notna()
        )
        out["variant_allele_expression"] = estimate
        out["variant_allele_expression_method"] = pd.Series(
            [TPM_X_DNA_VAF if pd.notna(v) else pd.NA for v in estimate],
            index=out.index, dtype="object",
        )
    else:
        out["variant_allele_expression"] = pd.Series(
            [np.nan] * n_rows, index=out.index, dtype="float64"
        )
        out["variant_allele_expression_method"] = pd.Series(
            [pd.NA] * n_rows, index=out.index, dtype="object"
        )
    return out


def describe_read_evidence(df: pd.DataFrame) -> dict:
    """``{column: method}`` for every read-evidence column *df* populates.

    The frame-level summary of what the per-row method columns say, for a
    caller reporting how a run's numbers were obtained without walking
    the rows.
    """
    described = {}
    for column, method_column in (
        ("n_alt_reads", "read_count_method"),
        ("n_ref_reads", "read_count_method"),
        ("n_alt_reads_supporting_protein_sequence",
         "supporting_read_count_method"),
        ("variant_allele_expression", "variant_allele_expression_method"),
    ):
        if column not in df.columns or method_column not in df.columns:
            continue
        methods = df[method_column][stated_values(df[method_column])]
        for method in sorted(set(methods.tolist())):
            described.setdefault(column, method)
    return described
