"""ProteinFragment — a universal record for a protein/peptide sequence
with source-type, target-region, and comparator metadata.

Designed to carry antigens from any origin (somatic variant, structural
variant, ERV, CTA, viral, allergen, autoantigen, synthetic) through a
single prediction pipeline and into downstream tools (vaxrank, etc.)
without losing provenance or comparator information.

This module defines only the data model + helpers.  IO, prediction,
and format-specific loaders live in sibling modules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


# =============================================================================
# ProteinFragment
# =============================================================================


#: How real a field's value is, for :attr:`ProteinFragment.field_provenance`.
#:
#: A field not named in the mapping is unqualified — it means what it
#: says. These qualify a field that *has* a value, which is the case a
#: bare ``None`` cannot express: "populated but estimated" and
#: "populated but invented" are neither absent nor trustworthy.
MEASURED = "measured"
APPROXIMATED = "approximated"
SYNTHESIZED = "synthesized"

PROVENANCE_VALUES = frozenset({MEASURED, APPROXIMATED, SYNTHESIZED})

#: Provenance values a consumer must not interpret as biology.
_NOT_BIOLOGY = frozenset({SYNTHESIZED})


@dataclass(frozen=True, eq=False)
class ProteinFragment:
    """A protein/peptide sequence with source-type, target-region, and
    comparator metadata.

    Parameters
    ----------
    fragment_id : str
        Canonical identity.  Convention is
        ``{readable_prefix}__{short_hash}`` — see :func:`make_fragment_id`.
        Two fragments with the same ``fragment_id`` are treated as the
        same fragment (equality and hash key on this field alone).
    source_type : str, optional
        Free-form biological category (e.g. ``"variant:snv"``,
        ``"sv:fusion"``, ``"erv"``, ``"viral:hpv16"``,
        ``"allergen:peanut"``, ``"cta"``, ``"autoantigen"``,
        ``"synthetic"``).  Used for filtering and display; never
        interpreted by Topiary.  See ``docs/fragments.md`` for the
        recommended (not enforced) vocabulary.
    sequence : str
        The antigen's protein / peptide sequence.  Sliding-window scans
        produced by the predictor run over this string.
    reference_sequence : str, optional
        A canonical reference sequence (Ensembl/RefSeq, reference strain,
        reference allergen isoform) to diff against.  ``None`` when no
        natural reference exists (ERV, CTA, pure self, synthetic).
    germline_sequence : str, optional
        A patient-specific (or strain-specific) baseline that may differ
        from ``reference_sequence`` due to polymorphism.  The DSL's
        ``wt.*`` scope reads germline if present, otherwise falls back
        to reference.  Typically populated only for somatic-variant and
        autoantigen workflows.
    target_intervals : list of (int, int), optional
        Half-open intervals within ``sequence`` considered targetable /
        distinguishing.  Meaning depends on ``source_type``:
        for variants the mutated residues; for fusions the junction; for
        splice the residues downstream of a novel junction; for ERVs and
        CTAs the non-self regions (where "self" is whatever the producer
        cares about).  ``None`` = unspecified (downstream can treat as
        "whole sequence").  Empty list = explicitly nothing.
    variant, effect, effect_type : str, optional
        Variant-level provenance when applicable.  ``variant`` is a
        free-form identifier (``chr:pos:ref>alt``, HGVS, strain name);
        ``effect`` is typically HGVS protein notation; ``effect_type``
        is a coarse label (``Substitution``, ``FrameShift``, etc.).
    gene, gene_id, transcript_id, transcript_name : str, optional
        Source gene / transcript identifiers.  ``transcript_name`` is
        the human-readable label (e.g. ``"BRAF-204"``) alongside the
        Ensembl id.
    gene_expression, transcript_expression : float, optional
        Expression evidence carried forward into prediction rows.
    n_overlapping_reads, n_alt_reads, n_ref_reads, \
n_alt_reads_supporting_protein_sequence : int, optional
        RNA read-level evidence.  Not derivable from the aggregate
        expression fields above, and separately useful: a consumer that
        weights a candidate by depth of support needs the counts, not a
        TPM.  ``n_alt_reads_supporting_protein_sequence`` is deliberately
        distinct from ``n_alt_reads`` — it counts reads supporting *this
        assembled protein sequence*, not merely the variant allele.

        ``None`` means **unknown**, and is not the same as ``0``.  A
        source with no read data leaves these ``None``; a source that
        looked and found no support sets ``0``.  Collapsing the two
        would let a consumer read "no RNA support" out of "this source
        cannot answer".
    field_provenance : dict, optional
        Per-field statement of how real a value is, mapping a field name
        to one of :data:`PROVENANCE_VALUES`:

        - ``"measured"`` — observed directly from data.
        - ``"approximated"`` — derived or estimated, e.g. read counts
          reconstructed as depth × VAF rather than counted.
        - ``"synthesized"`` — a placeholder the loader invented because
          the source did not supply one.  **Anything interpreting such a
          field as biology must refuse rather than compute.**

        A field absent from this mapping is unqualified: it means what
        it says.  This exists so a consumer can tell a populated field
        that is real from one that merely has a value — which a
        multi-source abstraction cannot express any other way, since
        every source populates a different subset and some of them
        estimate.  Use :meth:`provenance_of`, :meth:`is_known` and
        :meth:`is_usable_as_biology` rather than reading the dict.
    annotations : dict
        Tool-specific signals that don't fit the above fields.
        Serialized as JSON in TSV IO; carried through prediction as
        additional output columns.  Underscore-prefixed keys are
        reserved for internal plumbing and are **not** surfaced as
        columns — use them for short-lived bookkeeping that should
        not leak into user-facing output.
    """

    fragment_id: str

    source_type: Optional[str] = None

    sequence: str = ""
    reference_sequence: Optional[str] = None
    germline_sequence: Optional[str] = None

    target_intervals: Optional[list] = None  # list[tuple[int, int]] | None

    variant: Optional[str] = None
    effect: Optional[str] = None
    effect_type: Optional[str] = None
    gene: Optional[str] = None
    gene_id: Optional[str] = None
    transcript_id: Optional[str] = None
    transcript_name: Optional[str] = None

    gene_expression: Optional[float] = None
    transcript_expression: Optional[float] = None

    n_overlapping_reads: Optional[int] = None
    n_alt_reads: Optional[int] = None
    n_ref_reads: Optional[int] = None
    n_alt_reads_supporting_protein_sequence: Optional[int] = None

    field_provenance: dict = field(default_factory=dict)

    annotations: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Identity: fragment_id is the canonical key.  Using all-field eq
    # would trip over unhashable list/dict members; keying on
    # fragment_id also matches the intent that id is a stable
    # content-derived handle.
    # ------------------------------------------------------------------

    def __eq__(self, other):
        return (
            isinstance(other, ProteinFragment)
            and self.fragment_id == other.fragment_id
        )

    def __hash__(self):
        return hash(self.fragment_id)

    def __post_init__(self):
        """Reject a provenance mapping that cannot mean anything.

        A typo'd field name or an unknown label would sit inert and
        silently stop protecting the field it was written to protect,
        which is worse than not writing it.
        """
        if not self.field_provenance:
            return
        if not isinstance(self.field_provenance, dict):
            raise TypeError(
                f"field_provenance must be a dict of field name -> "
                f"provenance, got {type(self.field_provenance).__name__}"
            )
        known = {f.name for f in dataclasses.fields(self)}
        for name, value in self.field_provenance.items():
            if name not in known:
                raise ValueError(
                    f"field_provenance names {name!r}, which is not a "
                    f"ProteinFragment field. Use annotations for "
                    f"tool-specific signals."
                )
            if value not in PROVENANCE_VALUES:
                raise ValueError(
                    f"field_provenance[{name!r}] is {value!r}; use one of "
                    f"{sorted(PROVENANCE_VALUES)}."
                )

    # ------------------------------------------------------------------
    # Knownness
    # ------------------------------------------------------------------

    def provenance_of(self, name: str) -> Optional[str]:
        """How real *name*'s value is, or ``None`` if unqualified."""
        return self.field_provenance.get(name)

    def is_known(self, name: str) -> bool:
        """Whether *name* carries a value at all.

        The distinction this exists for: ``n_alt_reads == 0`` means the
        source looked and found no support; ``n_alt_reads is None``
        means the source cannot answer. Both are legitimate and they are
        not the same claim.
        """
        if name not in {f.name for f in dataclasses.fields(self)}:
            raise ValueError(
                f"{name!r} is not a ProteinFragment field."
            )
        return getattr(self, name) is not None

    def is_approximate(self, name: str) -> bool:
        """Whether *name*'s value was derived rather than observed."""
        return self.provenance_of(name) == APPROXIMATED

    def read_count_subject(self) -> Optional[str]:
        """What this fragment's read counts count, or ``None`` if unstated.

        ``"fragments"`` for an RNA-assembled fragment (isovar counts
        fragments), ``"reads"`` for a depth-based estimate.
        """
        return self.annotations.get("read_count_subject")

    def count_in(self, name: str, subject: str) -> Optional[int]:
        """*name*'s value if it counts *subject*, else ``None``.

        The point of the subject vocabulary. Five fragments and five
        reads are different bars, so a threshold has to know which it
        cleared — and a source that counted the other thing should
        **say so rather than substitute**, because converting between
        them needs library information no source carries.

        Within one run the unit is consistent and a ranking does not
        change either way. The harm is in what travels: a documented
        ``n_alt_reads > 5``, a config copied between projects, a number
        in a paper.

        Parameters
        ----------
        name : str
            A read-count field, e.g. ``"n_alt_reads"``.
        subject : str
            One of :data:`~topiary.READ_SUBJECTS`.

        Returns
        -------
        int or None
            ``None`` when the field is absent, or when this fragment's
            counts are of the other subject.

        Examples
        --------
        >>> fragment.count_in("n_alt_reads", "fragments")  # doctest: +SKIP
        30
        >>> fragment.count_in("n_alt_reads", "reads")      # doctest: +SKIP
        None
        """
        from .rna_evidence import READ_SUBJECTS

        if subject not in READ_SUBJECTS:
            raise ValueError(
                f"subject must be one of {sorted(READ_SUBJECTS)}, got "
                f"{subject!r}. A count without a stated subject cannot "
                f"be compared against a threshold."
            )
        if not self.is_known(name):
            return None
        if self.read_count_subject() != subject:
            return None
        return getattr(self, name)

    def is_usable_as_biology(self, name: str) -> bool:
        """Whether *name* may be interpreted as a fact about the sample.

        False for a field that is absent, and for one whose value the
        loader synthesized because the source supplied none — a
        placeholder ref/alt, say, which anything doing variant effect
        annotation must refuse rather than compute on. An approximated
        value is usable but should be understood as an estimate; ask
        :meth:`is_approximate` when that matters.
        """
        return (
            self.is_known(name)
            and self.provenance_of(name) not in _NOT_BIOLOGY
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def effective_baseline(self) -> Optional[str]:
        """Sequence the DSL's ``wt.*`` scope reads.

        Resolution order: ``germline_sequence`` if populated, else
        ``reference_sequence``, else ``None``.
        """
        if self.germline_sequence is not None:
            return self.germline_sequence
        return self.reference_sequence

    @property
    def has_target(self) -> bool:
        """True iff ``target_intervals`` names at least one interval."""
        return bool(self.target_intervals)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def peptide_overlaps_target(self, peptide_start: int, peptide_length: int) -> bool:
        """Whether the window ``[peptide_start, peptide_start+peptide_length)``
        overlaps any target interval.

        Returns ``False`` when ``target_intervals is None`` (unspecified —
        downstream code decides whether to treat as "whole sequence")
        or an empty list.
        """
        if not self.target_intervals:
            return False
        p_end = peptide_start + peptide_length
        for t_start, t_end in self.target_intervals:
            if peptide_start < t_end and t_start < p_end:
                return True
        return False

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-dict representation (tuples → lists, JSON-compatible)."""
        d = dataclasses.asdict(self)
        if d["target_intervals"] is not None:
            d["target_intervals"] = [list(p) for p in d["target_intervals"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProteinFragment":
        """Construct from a plain dict (e.g. parsed JSON or a row-dict).

        Missing optional fields fall back to ``None`` / empty
        annotations.  Unknown keys are rejected to catch typos — pass
        them through ``annotations`` instead.
        """
        # Derived, not restated: a hand-maintained copy of the field
        # list silently rejects every field added after it was written.
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d.keys()) - known
        if unknown:
            raise ValueError(
                f"Unknown ProteinFragment field(s): {sorted(unknown)}. "
                f"Move them to the annotations dict."
            )
        ti = d.get("target_intervals")
        if ti is not None:
            ti = [tuple(pair) for pair in ti]
        return cls(
            fragment_id=d["fragment_id"],
            source_type=d.get("source_type"),
            sequence=d.get("sequence", ""),
            reference_sequence=d.get("reference_sequence"),
            germline_sequence=d.get("germline_sequence"),
            target_intervals=ti,
            variant=d.get("variant"),
            effect=d.get("effect"),
            effect_type=d.get("effect_type"),
            gene=d.get("gene"),
            gene_id=d.get("gene_id"),
            transcript_id=d.get("transcript_id"),
            transcript_name=d.get("transcript_name"),
            gene_expression=d.get("gene_expression"),
            transcript_expression=d.get("transcript_expression"),
            n_overlapping_reads=d.get("n_overlapping_reads"),
            n_alt_reads=d.get("n_alt_reads"),
            n_ref_reads=d.get("n_ref_reads"),
            n_alt_reads_supporting_protein_sequence=d.get(
                "n_alt_reads_supporting_protein_sequence"
            ),
            field_provenance=dict(d.get("field_provenance") or {}),
            annotations=dict(d.get("annotations") or {}),
        )

    def to_json(self, **kwargs) -> str:
        """JSON string. Extra kwargs are forwarded to :func:`json.dumps`
        (e.g. ``indent=2`` for pretty-printing).
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> "ProteinFragment":
        return cls.from_dict(json.loads(s))

    # ------------------------------------------------------------------
    # Stringification
    # ------------------------------------------------------------------

    # __repr__ stays as dataclass-generated (verbose, unambiguous — the
    # right thing for debugging / pytest failure output).

    def __str__(self) -> str:
        """Short human-friendly summary for logs."""
        bits = [self.fragment_id, f"{len(self.sequence)} aa"]
        if self.source_type:
            bits.append(self.source_type)
        if self.target_intervals:
            n = len(self.target_intervals)
            bits.append(f"{n} target {'interval' if n == 1 else 'intervals'}")
        if self.gene:
            bits.append(f"gene={self.gene}")
        return f"ProteinFragment({', '.join(bits)})"

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_variant(
        cls,
        *,
        sequence: str,
        reference_sequence: Optional[str] = None,
        germline_sequence: Optional[str] = None,
        mutation_start: int,
        mutation_end: int,
        inframe: bool,
        variant: Optional[str] = None,
        effect: Optional[str] = None,
        gene: Optional[str] = None,
        gene_id: Optional[str] = None,
        transcript_id: Optional[str] = None,
        transcript_name: Optional[str] = None,
        **extra_kwargs,
    ) -> "ProteinFragment":
        """Build a fragment for a variant-derived antigen.

        In-frame mutations: ``target_intervals = [(mutation_start, mutation_end)]``.
        Frameshifts: ``target_intervals = [(mutation_start, len(sequence))]``
        — everything downstream is novel (caller is responsible for
        having truncated ``sequence`` at the new stop codon if desired).
        """
        if inframe:
            intervals = [(mutation_start, mutation_end)]
            source_type = extra_kwargs.pop("source_type", None) or (
                "variant:indel" if mutation_end - mutation_start != 1 else "variant:snv"
            )
        else:
            intervals = [(mutation_start, len(sequence))]
            source_type = extra_kwargs.pop("source_type", None) or "variant:frameshift"
        prefix = extra_kwargs.pop("fragment_prefix", None)
        if prefix is None:
            prefix = _default_prefix(gene, effect, variant)
        fragment_id = make_fragment_id(prefix, sequence, variant=variant)
        return cls(
            fragment_id=fragment_id,
            source_type=source_type,
            sequence=sequence,
            reference_sequence=reference_sequence,
            germline_sequence=germline_sequence,
            target_intervals=intervals,
            variant=variant,
            effect=effect,
            effect_type=extra_kwargs.pop("effect_type", None),
            gene=gene,
            gene_id=gene_id,
            transcript_id=transcript_id,
            transcript_name=transcript_name,
            gene_expression=extra_kwargs.pop("gene_expression", None),
            transcript_expression=extra_kwargs.pop("transcript_expression", None),
            annotations=extra_kwargs.pop("annotations", {}) or {},
        )

    @classmethod
    def from_junction(
        cls,
        *,
        sequence: str,
        junction_position: int,
        novel_downstream: bool,
        reference_sequence: Optional[str] = None,
        germline_sequence: Optional[str] = None,
        source_type: Optional[str] = None,
        variant: Optional[str] = None,
        effect: Optional[str] = None,
        gene: Optional[str] = None,
        gene_id: Optional[str] = None,
        transcript_id: Optional[str] = None,
        transcript_name: Optional[str] = None,
        **extra_kwargs,
    ) -> "ProteinFragment":
        """Build a fragment for a fusion / splice / cryptic-exon /
        readthrough case.

        ``novel_downstream=False`` (in-frame coding-coding fusion, splice
        junction of known exons): targets the junction residue pair only.

        ``novel_downstream=True`` (splice into novel exon, coding→noncoding
        readthrough, frameshift from junction): targets the junction
        through the end of ``sequence``.
        """
        if novel_downstream:
            intervals = [(junction_position, len(sequence))]
        else:
            lo = max(0, junction_position - 1)
            hi = min(len(sequence), junction_position + 1)
            intervals = [(lo, hi)]
        if source_type is None:
            source_type = "sv:fusion"
        prefix = extra_kwargs.pop("fragment_prefix", None)
        if prefix is None:
            prefix = _default_prefix(gene, effect, variant) or source_type
        fragment_id = make_fragment_id(prefix, sequence, variant=variant)
        return cls(
            fragment_id=fragment_id,
            source_type=source_type,
            sequence=sequence,
            reference_sequence=reference_sequence,
            germline_sequence=germline_sequence,
            target_intervals=intervals,
            variant=variant,
            effect=effect,
            effect_type=extra_kwargs.pop("effect_type", None),
            gene=gene,
            gene_id=gene_id,
            transcript_id=transcript_id,
            transcript_name=transcript_name,
            gene_expression=extra_kwargs.pop("gene_expression", None),
            transcript_expression=extra_kwargs.pop("transcript_expression", None),
            annotations=extra_kwargs.pop("annotations", {}) or {},
        )


# =============================================================================
# Helpers
# =============================================================================


_SAFE_PREFIX_RE = re.compile(r"[^A-Za-z0-9._:\-]+")


def _sanitize_prefix(s: str) -> str:
    """Collapse any run of non-safe characters in *s* to a single ``_``."""
    return _SAFE_PREFIX_RE.sub("_", s).strip("_")


def _default_prefix(gene, effect, variant) -> str:
    parts = []
    for p in (gene, effect, variant):
        if isinstance(p, str) and p:
            parts.append(p)
    return _sanitize_prefix("_".join(parts))


def make_fragment_id(
    prefix: str,
    sequence: str,
    *,
    variant: Optional[str] = None,
    hash_length: int = 8,
) -> str:
    """Build a stable, human-readable fragment id.

    Format: ``{prefix}__{short_hash}``.  Prefix is sanitized to
    ``[A-Za-z0-9._:-]``; runs of other characters collapse to ``_``.
    Empty prefix yields just ``__{short_hash}``.

    The hash portion is a SHA-1 prefix over ``sequence`` + ``variant``
    (when provided), making it deterministic for the same content.
    """
    prefix = _sanitize_prefix(prefix or "")
    hasher = hashlib.sha1()
    hasher.update(sequence.encode("utf-8"))
    if variant:
        hasher.update(b"\x00")
        hasher.update(variant.encode("utf-8"))
    short = hasher.hexdigest()[:hash_length]
    return f"{prefix}__{short}"


# =============================================================================
# Iteration helpers
# =============================================================================


def collect_annotations(fragments: Iterable[ProteinFragment]) -> set:
    """Return the union of annotation keys across *fragments*.  Useful
    for TSV writers deciding whether to expand known keys into columns."""
    keys = set()
    for f in fragments:
        keys.update(f.annotations.keys())
    return keys


# ---------------------------------------------------------------------------
# Every path to a fragment
# ---------------------------------------------------------------------------

#: The fields every source is expected to speak to, whether or not it can
#: populate them. A source that cannot answer leaves them ``None``, which
#: :meth:`ProteinFragment.is_known` distinguishes from zero — that is what
#: makes one consumer code path work across every source.
SEMANTIC_CORE = (
    "fragment_id", "source_type", "sequence", "target_intervals",
    "variant", "gene", "gene_id", "transcript_id",
    "gene_expression", "transcript_expression",
    "n_overlapping_reads", "n_alt_reads", "n_ref_reads",
    "n_alt_reads_supporting_protein_sequence",
)

_FRAGMENT_IDENTITY = ("source_sequence_name", "variant", "peptide")

_COUNT_FIELDS = (
    "n_overlapping_reads", "n_alt_reads", "n_ref_reads",
    "n_alt_reads_supporting_protein_sequence",
)


def fragments_from_dataframe(df, *, sequence_column=None):
    """Fragments from a reader's frame — the LENS / pVACseq path.

    :func:`~topiary.fragment_from_effect` covers varcode and
    :func:`~topiary.fragment_from_isovar_result` covers isovar; this
    covers the sources that arrive as a table. All three produce the
    same :data:`SEMANTIC_CORE`, differing only in which fields they can
    fill, so a consumer reads one shape rather than branching on where
    the data came from.

    Read counts carry the provenance their derivation implies —
    ``rna_reads`` is ``measured``, ``rna_depth_x_vaf`` and
    ``cds_overlap_reads`` are ``approximated`` — via one mapping in
    :mod:`topiary.rna_evidence`, so a frame and a fragment cannot
    disagree about whether a number was counted.

    Parameters
    ----------
    df : pandas.DataFrame
        A frame from :func:`~topiary.read_lens` or
        :func:`~topiary.read_pvacseq`, or anything with the same
        columns.
    sequence_column : str, optional
        Which column holds the fragment's sequence. Defaults to the
        first of ``sequence`` / ``pep_context`` / ``peptide`` present —
        so a reader that carries surrounding context uses it, and one
        that carries only the peptide produces the degenerate fragment
        whose sequence *is* the peptide.

    Returns
    -------
    list of ProteinFragment
        One per distinct (source, variant, sequence). Rows with no
        sequence are skipped: a fragment with nothing to present is not
        a fragment.
    """
    import pandas as pd

    from .rna_evidence import provenance_for_method
    from .ranking import is_stated

    if df is None or len(df) == 0:
        return []

    if sequence_column is None:
        for candidate in ("sequence", "pep_context", "peptide"):
            if candidate in df.columns:
                sequence_column = candidate
                break
    if sequence_column is None or sequence_column not in df.columns:
        raise ValueError(
            f"No sequence column found. Looked for 'sequence', "
            f"'pep_context', 'peptide'; frame has "
            f"{sorted(df.columns)[:8]}... Pass sequence_column= to say "
            f"which column holds the fragment's sequence."
        )

    identity = [c for c in _FRAGMENT_IDENTITY if c in df.columns]
    if sequence_column not in identity:
        identity = identity + [sequence_column]

    fragments = []
    for _, group in df.groupby(identity, sort=False, dropna=False):
        row = group.iloc[0]
        sequence = row.get(sequence_column)
        if not is_stated(sequence):
            continue

        counts = {}
        provenance = {}
        method = row.get("read_count_method")
        supporting_method = row.get("supporting_read_count_method")
        for count_field in _COUNT_FIELDS:
            value = row.get(count_field)
            if value is None or pd.isna(value):
                continue
            counts[count_field] = int(value)
            # n_overlapping_reads is a direct count wherever a reader
            # reports it; only the split carries the derivation.
            applicable = (
                None if count_field == "n_overlapping_reads"
                else (supporting_method if count_field.endswith("_sequence")
                      else method)
            )
            resolved = provenance_for_method(applicable)
            if resolved is not None:
                provenance[count_field] = resolved

        expression_method = row.get("variant_allele_expression_method")
        if is_stated(expression_method):
            provenance["transcript_expression"] = provenance_for_method(
                expression_method
            )

        annotations = {}
        for key in ("sequence_source", "read_count_method",
                    "read_count_subject", "supporting_read_count_method",
                    "variant_allele_expression",
                    "variant_allele_expression_method"):
            value = row.get(key)
            if is_stated(value) and not (
                isinstance(value, float) and pd.isna(value)
            ):
                annotations[key] = value

        fragments.append(ProteinFragment(
            fragment_id=make_fragment_id(
                prefix=str(row.get("variant") or row.get(
                    "source_sequence_name") or "fragment"),
                sequence=str(sequence),
            ),
            source_type=_optional(row.get("source_type")),
            sequence=str(sequence),
            target_intervals=None,
            variant=_optional(row.get("variant")),
            gene=_optional(row.get("gene")),
            gene_id=_optional(row.get("gene_id")),
            transcript_id=_optional(row.get("transcript_id")),
            gene_expression=_optional_float(row.get("gene_expression")),
            transcript_expression=_optional_float(
                row.get("transcript_expression")
            ),
            field_provenance=provenance,
            annotations=annotations,
            **counts,
        ))
    return fragments


def _optional(value):
    """The value as a string, or ``None`` when the source said nothing."""
    from .ranking import is_stated
    return str(value) if is_stated(value) else None


def _optional_float(value):
    from .ranking import is_stated
    if not is_stated(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
