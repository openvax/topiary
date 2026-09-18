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
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .serialization import normalize_python_types


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


class _DefaultFactory:
    """Signature sentinel for fields whose default comes from a factory."""

    __slots__ = ()

    def __repr__(self):
        return "<factory>"


_DEFAULT_FACTORY = _DefaultFactory()


def _fragment_field_renames(known_fields: set) -> dict:
    """Legacy-to-current renames that belong to ProteinFragment."""
    from .evidence import RENAMED_COLUMNS

    return {
        old: new for old, new in RENAMED_COLUMNS.items()
        if new in known_fields
    }


def _migrate_fragment_dict(values: dict, known_fields: set) -> dict:
    """Return *values* with legacy ProteinFragment field names migrated.

    The public rename table also contains reader-frame columns. Only
    renames whose destination is a ProteinFragment field apply here.
    Provenance keys travel with their values so a legacy serialized
    fragment remains internally consistent after migration.
    """
    migrated = dict(values)

    def _missing(value):
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        return isinstance(value, float) and value != value

    def _move(mapping, old, new):
        old_value = mapping.pop(old)
        if new not in mapping or _missing(mapping[new]):
            mapping[new] = old_value
        elif not _missing(old_value) and mapping[new] != old_value:
            raise ValueError(
                f"Conflicting ProteinFragment fields {old!r} and "
                f"{new!r}: {old_value!r} != {mapping[new]!r}."
            )

    fragment_renames = _fragment_field_renames(known_fields)
    for old, new in fragment_renames.items():
        if old in migrated:
            _move(migrated, old, new)

    provenance = migrated.get("field_provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        for old, new in fragment_renames.items():
            if old in provenance:
                _move(provenance, old, new)
        migrated["field_provenance"] = provenance
    return migrated


def _current_fragment_field(name: str) -> str:
    """Canonical field name for a current or legacy fragment field."""
    known = {field.name for field in dataclasses.fields(ProteinFragment)}
    renamed = _fragment_field_renames(known).get(name, name)
    return renamed if renamed in known else name


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
    n_rna_overlapping_reads, n_rna_alt_reads, n_rna_ref_reads, n_rna_other_reads, \
n_rna_alt_reads_supporting_protein_sequence : int, optional
        RNA evidence counted in **reads**.
    n_rna_overlapping_fragments, n_rna_alt_fragments, n_rna_ref_fragments, n_rna_other_fragments, \
n_rna_alt_fragments_supporting_protein_sequence : int, optional
        The same evidence counted in **fragments**. A paired-end
        fragment is one molecule read twice, so it is one piece of
        evidence and two reads — which is why both are carried rather
        than one being converted to the other. isovar reports both;
        sources that estimate from depth report only reads.

        Prefer the :attr:`n_rna_alt` family over either: it takes the
        better of the two and says which it took.

        RNA read-level evidence.  Not derivable from the aggregate
        expression fields above, and separately useful: a consumer that
        weights a candidate by depth of support needs the counts, not a
        TPM.  ``n_rna_alt_reads_supporting_protein_sequence`` is deliberately
        distinct from ``n_rna_alt_reads`` — it counts reads supporting *this
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
        NumPy boolean, integer, real and string scalars are normalized to
        native Python values, including in nested containers, on construction
        and serialization. The caller's input containers are not mutated.
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

    n_rna_overlapping_reads: Optional[int] = None
    n_rna_alt_reads: Optional[int] = None
    n_rna_ref_reads: Optional[int] = None
    n_rna_other_reads: Optional[int] = None
    n_rna_alt_reads_supporting_protein_sequence: Optional[int] = None

    n_rna_overlapping_fragments: Optional[int] = None
    n_rna_alt_fragments: Optional[int] = None
    n_rna_ref_fragments: Optional[int] = None
    n_rna_other_fragments: Optional[int] = None
    n_rna_alt_fragments_supporting_protein_sequence: Optional[int] = None

    field_provenance: dict = field(default_factory=dict)

    annotations: dict = field(default_factory=dict)

    def __init__(
        self,
        fragment_id: str,
        source_type: Optional[str] = None,
        sequence: str = "",
        reference_sequence: Optional[str] = None,
        germline_sequence: Optional[str] = None,
        target_intervals: Optional[list] = None,
        variant: Optional[str] = None,
        effect: Optional[str] = None,
        effect_type: Optional[str] = None,
        gene: Optional[str] = None,
        gene_id: Optional[str] = None,
        transcript_id: Optional[str] = None,
        transcript_name: Optional[str] = None,
        gene_expression: Optional[float] = None,
        transcript_expression: Optional[float] = None,
        n_rna_overlapping_reads: Optional[int] = None,
        n_rna_alt_reads: Optional[int] = None,
        n_rna_ref_reads: Optional[int] = None,
        n_rna_other_reads: Optional[int] = None,
        n_rna_alt_reads_supporting_protein_sequence: Optional[int] = None,
        n_rna_overlapping_fragments: Optional[int] = None,
        n_rna_alt_fragments: Optional[int] = None,
        n_rna_ref_fragments: Optional[int] = None,
        n_rna_other_fragments: Optional[int] = None,
        n_rna_alt_fragments_supporting_protein_sequence: Optional[int] = None,
        field_provenance: dict = _DEFAULT_FACTORY,
        annotations: dict = _DEFAULT_FACTORY,
        **legacy_fields,
    ) -> None:
        """Initialize a fragment, accepting legacy evidence names as keywords.

        Current fields retain their dataclass order and positional behavior.
        Evidence names from Topiary 5.47 and earlier are accepted only as
        additional keywords and migrate through the same implementation used
        by serialized input.
        """
        arguments = locals()
        # Bind exactly the fields advertised by this constructor. A subclass
        # may declare additional dataclass fields without making the inherited
        # base initializer look for arguments it never accepted.
        fragment_fields = dataclasses.fields(ProteinFragment)
        values = {}
        for fragment_field in fragment_fields:
            value = arguments[fragment_field.name]
            if value is _DEFAULT_FACTORY:
                value = fragment_field.default_factory()
            values[fragment_field.name] = value

        values.update(legacy_fields)
        known = {fragment_field.name for fragment_field in fragment_fields}
        values = _migrate_fragment_dict(normalize_python_types(values), known)
        unknown = set(values) - known
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(
                "ProteinFragment.__init__() got an unexpected keyword "
                f"argument {name!r}"
            )

        for fragment_field in fragment_fields:
            object.__setattr__(
                self, fragment_field.name, values[fragment_field.name],
            )
        self.__post_init__()

    # Legacy keywords are a 5.x compatibility input, not fields in the
    # current data model. Keep the public class signature identical to the
    # dataclass field surface while leaving the implementation's
    # ``**legacy_fields`` visible on ``ProteinFragment.__init__`` itself.
    __signature__ = inspect.signature(__init__).replace(
        parameters=tuple(inspect.signature(__init__).parameters.values())[1:-1],
        return_annotation=None,
    )

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
        known = {f.name for f in dataclasses.fields(self)}
        migrated = _migrate_fragment_dict(
            {"field_provenance": self.field_provenance}, known,
        )["field_provenance"]
        if migrated != self.field_provenance:
            object.__setattr__(self, "field_provenance", migrated)
        if not self.field_provenance:
            return
        if not isinstance(self.field_provenance, dict):
            raise TypeError(
                f"field_provenance must be a dict of field name -> "
                f"provenance, got {type(self.field_provenance).__name__}"
            )
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
    # Topiary 5.47 evidence-name compatibility. These are ordinary,
    # read-only class properties so compatibility is visible to
    # introspection and does not mutate the class after its definition.
    # ------------------------------------------------------------------

    @property
    def n_alt_reads(self):
        """Compatibility alias for :attr:`n_rna_alt_reads`."""
        return self.n_rna_alt_reads

    @property
    def n_alt_fragments(self):
        """Compatibility alias for :attr:`n_rna_alt_fragments`."""
        return self.n_rna_alt_fragments

    @property
    def n_ref_reads(self):
        """Compatibility alias for :attr:`n_rna_ref_reads`."""
        return self.n_rna_ref_reads

    @property
    def n_ref_fragments(self):
        """Compatibility alias for :attr:`n_rna_ref_fragments`."""
        return self.n_rna_ref_fragments

    @property
    def n_other_reads(self):
        """Compatibility alias for :attr:`n_rna_other_reads`."""
        return self.n_rna_other_reads

    @property
    def n_other_fragments(self):
        """Compatibility alias for :attr:`n_rna_other_fragments`."""
        return self.n_rna_other_fragments

    @property
    def n_overlapping_reads(self):
        """Compatibility alias for :attr:`n_rna_overlapping_reads`."""
        return self.n_rna_overlapping_reads

    @property
    def n_overlapping_fragments(self):
        """Compatibility alias for :attr:`n_rna_overlapping_fragments`."""
        return self.n_rna_overlapping_fragments

    @property
    def n_alt_reads_supporting_protein_sequence(self):
        """Compatibility alias for the corresponding RNA read count."""
        return self.n_rna_alt_reads_supporting_protein_sequence

    @property
    def n_alt_fragments_supporting_protein_sequence(self):
        """Compatibility alias for the corresponding RNA fragment count."""
        return self.n_rna_alt_fragments_supporting_protein_sequence

    # ------------------------------------------------------------------
    # Knownness
    # ------------------------------------------------------------------

    def provenance_of(self, name: str) -> Optional[str]:
        """How real *name*'s value is, or ``None`` if unqualified."""
        return self.field_provenance.get(_current_fragment_field(name))

    def is_known(self, name: str) -> bool:
        """Whether *name* carries a value at all.

        The distinction this exists for: ``n_rna_alt_reads == 0`` means the
        source looked and found no support; ``n_rna_alt_reads is None``
        means the source cannot answer. Both are legitimate and they are
        not the same claim.
        """
        name = _current_fragment_field(name)
        if name not in {f.name for f in dataclasses.fields(self)}:
            raise ValueError(
                f"{name!r} is not a ProteinFragment field."
            )
        return getattr(self, name) is not None

    def is_approximate(self, name: str) -> bool:
        """Whether *name*'s value was derived rather than observed."""
        return self.provenance_of(name) == APPROXIMATED

    # ------------------------------------------------------------------
    # RNA evidence: ask for the evidence, not for a unit
    # ------------------------------------------------------------------

    @property
    def n_rna_alt(self) -> Optional[int]:
        """RNA evidence supporting the variant allele.

        Fragments when the source counted them, reads otherwise.
        :meth:`rna_evidence_subject` says which you got.

        Prefer this to reading :attr:`n_rna_alt_reads` or
        :attr:`n_rna_alt_fragments` directly. A paired-end fragment is one
        molecule read twice, so it is *one* piece of evidence and *two*
        reads — fragments are the better count where a source has them,
        and reads are what you get where it does not.
        """
        return self._rna_evidence("alt")[0]

    @property
    def n_rna_ref(self) -> Optional[int]:
        """RNA evidence supporting the reference allele."""
        return self._rna_evidence("ref")[0]

    @property
    def n_rna_other(self) -> Optional[int]:
        """Support for neither the reference nor the alt allele, or ``None``.

        A third allele at the locus, a sequencing error, or a nearby
        indel. ``None`` where the source counted only alt and depth, in
        which case the reference count already absorbs these.
        """
        return self._rna_evidence("other")[0]

    @property
    def n_rna_overlapping(self) -> Optional[int]:
        """RNA evidence covering the variant position."""
        return self._rna_evidence("overlapping")[0]

    @property
    def n_rna_supporting_protein_sequence(self) -> Optional[int]:
        """RNA evidence supporting *this assembled protein sequence*.

        Distinct from :attr:`n_rna_alt`: that counts support for the
        variant allele, this counts support for the whole assembled
        sequence, which only an assembler can report.
        """
        return self._rna_evidence("supporting")[0]

    def rna_evidence_subject(self) -> Optional[str]:
        """What the ``n_rna_*`` values are counted in.

        ``"fragments"``, ``"reads"``, or ``None`` when this fragment
        carries no RNA evidence at all. Report it alongside a count that
        travels — five fragments and five reads are different bars. Raises
        when different canonical quantities would use different units;
        use the unit-specific fields for such a fragment.
        """
        subjects = set()
        for name in ("alt", "overlapping", "ref", "other", "supporting"):
            value, subject = self._rna_evidence(name)
            if value is not None:
                subjects.add(subject)
        if len(subjects) > 1:
            raise ValueError(
                f"ProteinFragment {self.fragment_id!r} mixes RNA evidence "
                f"units {sorted(subjects)}. Use the unit-specific fields "
                f"instead of treating its n_rna_* values as one unit."
            )
        return next(iter(subjects), None)

    _RNA_FIELDS = {
        "alt": ("n_rna_alt_fragments", "n_rna_alt_reads"),
        "ref": ("n_rna_ref_fragments", "n_rna_ref_reads"),
        "other": ("n_rna_other_fragments", "n_rna_other_reads"),
        "overlapping": ("n_rna_overlapping_fragments", "n_rna_overlapping_reads"),
        "supporting": (
            "n_rna_alt_fragments_supporting_protein_sequence",
            "n_rna_alt_reads_supporting_protein_sequence",
        ),
    }

    def _rna_evidence(self, name):
        """``(value, subject)`` — fragments if present, else reads."""
        from .evidence import FRAGMENTS, READS

        fragment_field, read_field = self._RNA_FIELDS[name]
        value = getattr(self, fragment_field, None)
        if value is not None:
            return value, FRAGMENTS
        value = getattr(self, read_field, None)
        if value is not None:
            return value, READS
        return None, None

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
        """Plain dict with native scalars and target-interval tuples as lists.

        Nested NumPy scalars are normalized even when annotations were
        changed after construction. Unsupported custom objects remain
        unchanged and will be rejected by the JSON encoder.
        """
        d = normalize_python_types(self, dataclasses_as_dict=True)
        if d["target_intervals"] is not None:
            d["target_intervals"] = [list(p) for p in d["target_intervals"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ProteinFragment":
        """Construct from a plain dict (e.g. parsed JSON or a row-dict).

        Missing optional fields fall back to ``None`` / empty
        annotations. Evidence names from 5.47 and earlier are migrated,
        including keys inside ``field_provenance``. Other unknown keys
        are rejected to catch typos — pass them through ``annotations``
        instead.
        """
        # Derived, not restated: a hand-maintained copy of the field
        # list silently rejects every field added after it was written.
        known = {f.name for f in dataclasses.fields(cls)}
        d = _migrate_fragment_dict(d, known)
        unknown = set(d.keys()) - known
        if unknown:
            raise ValueError(
                f"Unknown ProteinFragment field(s): {sorted(unknown)}. "
                f"Move them to the annotations dict."
            )
        values = dict(d)
        ti = values.get("target_intervals")
        if ti is not None:
            ti = [tuple(pair) for pair in ti]
        values["target_intervals"] = ti
        values["field_provenance"] = dict(
            values.get("field_provenance") or {}
        )
        values["annotations"] = dict(values.get("annotations") or {})
        return cls(**values)

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
    qualifiers: Iterable[Optional[str]] = (),
    hash_length: int = 8,
) -> str:
    """Build a stable, human-readable fragment id.

    Format: ``{prefix}__{short_hash}``.  Prefix is sanitized to
    ``[A-Za-z0-9._:-]``; runs of other characters collapse to ``_``.
    Empty prefix yields just ``__{short_hash}``.

    The hash portion is a SHA-1 prefix over ``sequence``, ``variant``
    (when provided) and ``qualifiers``, so it is deterministic for the
    same inputs and differs when any of them does.

    Parameters
    ----------
    prefix : str
        Readable label, usually the variant or source name.
    sequence : str
        The fragment's amino-acid sequence.
    variant : str, optional
        Variant identifier, hashed exactly (the prefix is sanitized, so
        two variants can share one).
    qualifiers : iterable of str or None, optional
        Anything else that tells this record apart from another with the
        same sequence and variant, such as the peptide a table row was
        reported for. A producer passes every value it groups records
        by, so two records share an ID only when they describe the same
        observation. ``None`` entries mean "not stated"; order matters.
        Empty (the default) reproduces the IDs of earlier releases.
    hash_length : int
        Hex characters of the hash to keep.
    """
    prefix = _sanitize_prefix(prefix or "")
    hasher = hashlib.sha1()
    hasher.update(sequence.encode("utf-8"))
    if variant:
        hasher.update(b"\x00")
        hasher.update(variant.encode("utf-8"))
    qualifiers = [None if q is None else str(q) for q in qualifiers]
    if qualifiers:
        # JSON keeps None, "" and the boundaries between values distinct.
        hasher.update(b"\x00")
        hasher.update(json.dumps(qualifiers).encode("utf-8"))
    short = hasher.hexdigest()[:hash_length]
    return f"{prefix}__{short}"


def fragments_for_sample(fragments: Iterable[ProteinFragment], sample_name: str) -> list:
    """Label fragments as one sample's observations.

    The same peptide seen in two samples is two observations with two
    sets of evidence. Prediction keys its rows on ``fragment_id``, so the
    observations need different IDs to both survive, and a sample name to
    be told apart afterwards. This gives them both: each returned
    fragment's ID becomes ``"{sample_name}:{fragment_id}"`` and its
    annotations record ``sample_name``, which prediction writes to the
    ``sample_name`` column.

    Every fragment producer that takes a ``sample_name`` routes through
    this function, so a label means the same thing whichever source the
    fragment came from.

    Parameters
    ----------
    fragments : iterable of ProteinFragment
        Fragments from any source.
    sample_name : str
        The observation's label: a sample, or any separately analysed view
        of one (a sequencing library, an alignment-placement policy) whose
        evidence must not be merged with another's. Characters outside
        ``[A-Za-z0-9._:-]`` become ``_`` in the ID; the annotation keeps
        the label exactly.

    Returns
    -------
    list of ProteinFragment
        New records in input order; inputs are not mutated. A fragment
        already labelled with this sample is returned unchanged, so
        labelling twice is harmless.

    Raises
    ------
    ValueError
        *sample_name* is blank or missing, or a fragment is already
        labelled with a different sample. Relabelling would claim one
        sample's evidence for another.

    Notes
    -----
    Because the ID names the observation, rows from two samples do not
    share a ``fragment_id``. To pool evidence for one candidate across
    samples, pass :func:`~topiary.aggregate_evidence_across_samples`
    group keys that identify the candidate without it, for example
    ``["variant", "peptide", "peptide_offset", "allele"]``.
    """
    from .ranking import is_stated

    if not isinstance(sample_name, str) or not is_stated(sample_name):
        raise ValueError(f"sample_name must be a non-blank string; got {sample_name!r}.")
    namespace = _sanitize_prefix(sample_name)
    if not namespace:
        raise ValueError(f"sample_name {sample_name!r} has no characters usable in an ID.")
    labelled = []
    for fragment in fragments:
        existing = fragment.annotations.get("sample_name")
        if existing == sample_name:
            labelled.append(fragment)
            continue
        if existing is not None and is_stated(existing):
            raise ValueError(
                f"Fragment {fragment.fragment_id!r} is already labelled with sample "
                f"{existing!r}; refusing to relabel it as {sample_name!r}."
            )
        labelled.append(dataclasses.replace(
            fragment,
            fragment_id=f"{namespace}:{fragment.fragment_id}",
            annotations={**fragment.annotations, "sample_name": sample_name},
        ))
    return labelled


# =============================================================================
# Iteration helpers
# =============================================================================


def unique_fragments(fragments: Iterable[ProteinFragment]) -> list:
    """Coalesce identical records and reject contradictory fragment identities.

    Parameters
    ----------
    fragments : iterable of ProteinFragment
        Records to validate. Empty iterables return an empty list; ``None``
        is not an iterable. Single-occurrence IDs need no serialization.

    Returns
    -------
    list of ProteinFragment
        The first record for each ID, in input order. Repeated IDs must hold
        the same content in every field, including evidence, provenance and
        annotations, compared the way fragment IO stores it: ``5`` and
        ``5.0`` agree, NaN and ``None`` both mean "not stated", tuples and
        lists agree, and mapping keys compare as strings. So a record and
        its own :func:`~topiary.write_fragments` round trip coalesce. This
        deliberately does not use fragment equality, which compares only
        IDs. Inputs are not mutated.

    Raises
    ------
    ValueError
        A repeated ID has conflicting content (the message names the fields
        that differ), or holds a value fragment IO cannot store, so agreement
        cannot be established. The same object repeated is always accepted.
        Give distinct observations distinct IDs even when sequences match —
        :func:`fragments_for_sample` does this for samples, libraries and
        analysis policies. Silently picking one would discard the other
        observation's evidence.
    """
    first = {}
    for fragment in fragments:
        seen = first.get(fragment.fragment_id)
        if seen is None:
            # Content is only needed once an ID repeats.
            first[fragment.fragment_id] = [fragment, None]
            continue
        if fragment is seen[0]:
            continue
        if seen[1] is None:
            seen[1] = _stored_content(seen[0])
        content = _stored_content(fragment)
        differing = sorted(name for name in content if content[name] != seen[1][name])
        if differing:
            raise ValueError(
                f"Conflicting records for fragment_id {fragment.fragment_id!r}: "
                f"they differ in {', '.join(differing)}. Use distinct IDs for "
                "distinct observations (see fragments_for_sample)."
            )
    return [fragment for fragment, _ in first.values()]


def _stored_content(fragment: ProteinFragment) -> dict:
    """Each field of *fragment* as canonical JSON text, as fragment IO stores it."""
    content = {}
    for name, value in fragment.to_dict().items():
        try:
            content[name] = json.dumps(_as_stored(value), sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot compare repeated fragment_id {fragment.fragment_id!r}: "
                f"its {name} holds a value fragment IO cannot store ({error})."
            ) from error
    return content


def _as_stored(value):
    """*value* in the form it takes after a fragment IO round trip."""
    if isinstance(value, float):
        if value != value:
            return None
        return int(value) if value.is_integer() else value
    if isinstance(value, dict):
        return {str(key): _as_stored(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_stored(item) for item in value]
    return value


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
    "n_rna_overlapping_reads", "n_rna_alt_reads", "n_rna_ref_reads",
    "n_rna_other_reads", "n_rna_alt_reads_supporting_protein_sequence",
    "n_rna_overlapping_fragments", "n_rna_alt_fragments",
    "n_rna_ref_fragments", "n_rna_other_fragments",
    "n_rna_alt_fragments_supporting_protein_sequence",
)

#: Reader-frame columns that tell one fragment apart from another. Rows
#: sharing all of these and the sequence describe one fragment; rows that
#: differ in any of them are different fragments with different IDs. A
#: context reported for several peptides is therefore several fragments,
#: because readers such as LENS report evidence per peptide.
_FRAGMENT_IDENTITY = ("sample_name", "source_sequence_name", "variant", "peptide")

#: Frame columns copied, as text, onto the fragment field of the same name.
_FRAME_TEXT_FIELDS = ("source_type", "gene", "gene_id", "transcript_id")

#: Frame columns copied, as numbers, onto the fragment field of the same name.
_FRAME_NUMBER_FIELDS = ("gene_expression", "transcript_expression")

#: Frame columns copied into annotations under the same name.
_FRAME_ANNOTATIONS = (
    "sequence_source", "rna_evidence_method", "rna_evidence_subject",
    "rna_alt_expression", "rna_alt_expression_method",
)

#: Frame column → the fragment field it fills, per unit.
#:
#: The frame carries one name per quantity (``n_rna_alt``) plus the unit
#: it is in; the fragment carries a field per unit. This is where the
#: two meet, so the fragment never holds a fragment count under a name
#: for reads.
_FRAME_COUNTS = {
    "n_rna_alt": ("n_rna_alt_reads", "n_rna_alt_fragments"),
    "n_rna_ref": ("n_rna_ref_reads", "n_rna_ref_fragments"),
    "n_rna_overlapping": ("n_rna_overlapping_reads", "n_rna_overlapping_fragments"),
    "n_rna_other": ("n_rna_other_reads", "n_rna_other_fragments"),
}


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
    :mod:`topiary.evidence`, so a frame and a fragment cannot
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
        One per distinct sample, source, variant, reported peptide and
        sequence, in frame order. Each fragment's ID is derived from all
        of them, so distinct rows never share an ID. A context reported for
        several peptides becomes one fragment per peptide, each carrying
        that peptide's evidence and naming it in
        ``annotations["reported_peptide"]``. A stated ``sample_name``
        labels the fragment through :func:`fragments_for_sample`. Rows with
        no sequence are skipped: a fragment with nothing to present is not
        a fragment.

    Raises
    ------
    ValueError
        Rows describing the same fragment disagree about a fragment field
        (keeping the first would silently discard the rest), or a count or
        expression cell is stated but is not a number. Counts must be whole
        and non-negative.

    Notes
    -----
    Every cell is read once, under topiary's one rule for absence
    (:func:`~topiary.stated_values`): an unstated cell becomes ``None``
    before anything else looks at it. ``NaN`` therefore never reaches an
    ID or a field as the text ``"nan"``.
    """
    from .evidence import provenance_for_method
    from .ranking import stated_values

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

    identity = [c for c in _FRAGMENT_IDENTITY if c in df.columns and c != sequence_column]
    columns = [
        column for column in dict.fromkeys([
            *identity, sequence_column, *_FRAME_TEXT_FIELDS,
            *_FRAME_NUMBER_FIELDS, *_FRAME_COUNTS, *_FRAME_ANNOTATIONS,
        ])
        if column in df.columns
    ]
    frame = df[columns].astype(object)
    frame = frame.where(frame.apply(stated_values), None)
    frame = frame[frame[sequence_column].notna()]
    for column in _FRAME_NUMBER_FIELDS:
        if column in frame.columns:
            frame[column] = _stated_numbers(frame[column], column)
    for column in _FRAME_COUNTS:
        if column in frame.columns:
            frame[column] = _stated_numbers(frame[column], column, counts=True)
    # Qualifiers hash every identity value the prefix and variant do not
    # already carry, so the ID distinguishes exactly what the rows do.
    qualifiers = [c for c in identity if c not in ("sample_name", "variant")]

    fragments = []
    for record in frame.drop_duplicates().to_dict("records"):
        sequence = str(record[sequence_column])
        variant = _text(record.get("variant"))
        subject = _text(record.get("rna_evidence_subject"))
        in_fragments = subject is not None and subject.strip() == "fragments"
        method = provenance_for_method(record.get("rna_evidence_method"))
        counts = {
            fragment_field if in_fragments else read_field: record[column]
            for column, (read_field, fragment_field) in _FRAME_COUNTS.items()
            if record.get(column) is not None
        }
        provenance = {name: method for name in counts} if method else {}
        expression_method = provenance_for_method(record.get("rna_alt_expression_method"))
        if expression_method:
            provenance["transcript_expression"] = expression_method
        annotations = {
            key: record[key] for key in _FRAME_ANNOTATIONS
            if record.get(key) is not None
        }
        if record.get("peptide") is not None and sequence_column != "peptide":
            annotations["reported_peptide"] = str(record["peptide"])
        fragment = ProteinFragment(
            fragment_id=make_fragment_id(
                variant or _text(record.get("source_sequence_name")) or "fragment",
                sequence,
                variant=variant,
                qualifiers=[_text(record.get(c)) for c in qualifiers],
            ),
            source_type=_text(record.get("source_type")),
            sequence=sequence,
            target_intervals=None,
            variant=variant,
            gene=_text(record.get("gene")),
            gene_id=_text(record.get("gene_id")),
            transcript_id=_text(record.get("transcript_id")),
            gene_expression=record.get("gene_expression"),
            transcript_expression=record.get("transcript_expression"),
            field_provenance=provenance,
            annotations=annotations,
            **counts,
        )
        if record.get("sample_name") is not None:
            fragment, = fragments_for_sample([fragment], str(record["sample_name"]))
        fragments.append(fragment)
    try:
        return unique_fragments(fragments)
    except ValueError as error:
        raise ValueError(
            f"Rows sharing {[*identity, sequence_column]} must describe one "
            f"fragment. {error}"
        ) from error


def _text(value):
    """A normalized frame cell as text, or ``None`` when it was not stated."""
    return None if value is None else str(value)


def _stated_numbers(values, column, *, counts=False):
    """A normalized frame column as Python numbers, ``None`` where unstated.

    A stated cell that is not a number — or, for *counts*, not a whole
    non-negative number — raises rather than being dropped or truncated.
    """
    import pandas as pd

    numbers = pd.to_numeric(values, errors="coerce")
    valid = numbers.notna()
    if counts:
        valid &= numbers.ge(0) & numbers.mod(1).eq(0)
    invalid = values.notna() & ~valid
    if invalid.any():
        expected = "whole non-negative counts" if counts else "numbers"
        raise ValueError(
            f"Column {column!r} must hold {expected}; got "
            f"{values[invalid].unique()[:5].tolist()}."
        )
    convert = int if counts else float
    return pd.Series(
        [None if value is None else convert(number) for value, number in zip(values, numbers)],
        index=values.index, dtype=object,
    )
