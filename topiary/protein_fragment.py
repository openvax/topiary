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
    annotations : dict
        Tool-specific signals that don't fit the above fields.
        Serialized as JSON in TSV IO; carried through prediction.
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
        known = {
            "fragment_id", "source_type", "sequence",
            "reference_sequence", "germline_sequence", "target_intervals",
            "variant", "effect", "effect_type",
            "gene", "gene_id", "transcript_id", "transcript_name",
            "gene_expression", "transcript_expression", "annotations",
        }
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
