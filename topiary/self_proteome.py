"""SelfProteome — reference protein corpus for cross-reactivity analysis.

Holds a species-tagged, scope-filtered protein set indexed by peptide
length.  Answers per-query nearest-neighbor lookups: "given this mutant
peptide, what's the most similar peptide in healthy human self?"

Scopes
------
- ``"all"``: no filter, whole Ensembl proteome (any pyensembl-supported species).
- ``"non_cta"`` (default for human): remove cancer-testis-antigen genes via
  pirlygenes.  Non-human species require an explicit ``cta_source=`` set
  or callable — pirlygenes is human-only today.
- callable: user-supplied ``gene → bool`` filter.  Works for any species.

Additional scopes (``"protected_tissues"`` with HPA/GTEx expression-based
filtering, 1aa-indel candidates, the ``self_nearest_by_binding`` and
``self_strongest_nearby`` axes, and the full candidate-set structured
column) are queued for a follow-up PR.  This module currently ships the
core architecture and one nearest-by-sequence axis so downstream code can
start consuming ``self_nearest_*`` columns in a real predictor run.

Algorithm
---------
SIMD-vectorized Hamming distance against int8-encoded reference arrays.
Only substitutions (same-length matches) in this PR; indels land in the
follow-up alongside seed-and-extend for larger reference corpora.  See
#124 for the benchmark plan driving the eventual algorithm choice.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Amino-acid encoding
# ---------------------------------------------------------------------------


_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_TO_INT = {aa: i for i, aa in enumerate(_AA_ALPHABET)}
# Non-standard residues (B/J/O/U/X/Z/*) all map to one sentinel so they
# always count as a mismatch against canonical residues.
_UNKNOWN_AA = len(_AA_ALPHABET)


def _encode_peptides(peptides: List[str], length: int) -> np.ndarray:
    """Encode peptides as an ``(N, length)`` int8 array.  Peptides
    shorter than ``length`` are padded with the unknown sentinel;
    peptides longer than ``length`` are truncated."""
    arr = np.full((len(peptides), length), _UNKNOWN_AA, dtype=np.int8)
    for i, pep in enumerate(peptides):
        for j, aa in enumerate(pep[:length]):
            arr[i, j] = _AA_TO_INT.get(aa.upper(), _UNKNOWN_AA)
    return arr


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


# Species → per-scope defaults.  Only human is populated today; other
# species raise a clear error when asked to default rather than silently
# returning unfiltered data.
_SPECIES_DEFAULTS: Dict[str, Dict[str, str]] = {
    "human": {
        "cta_source": "pirlygenes",
    },
}


def _resolve_cta_gene_ids(
    species: str, cta_source,
) -> Union[Set[str], Callable]:
    """Produce either a set of CTA gene IDs or a callable filter.

    Accepts ``"pirlygenes"``, ``"tsarina"``, a set of gene IDs, or a
    ``Callable[[gene_id], bool]``.  When ``cta_source`` is ``None`` the
    species default is consulted; unregistered species raise."""
    if cta_source is None:
        defaults = _SPECIES_DEFAULTS.get(species, {})
        cta_source = defaults.get("cta_source")
        if cta_source is None:
            raise ValueError(
                f"scope='non_cta' needs a CTA source for "
                f"species={species!r}; no default registered.  Pass "
                f"cta_source=<set or callable> explicitly, or use "
                f"scope='all'."
            )

    if callable(cta_source):
        return cta_source
    if isinstance(cta_source, (set, frozenset)):
        return set(cta_source)
    if cta_source == "pirlygenes":
        if species != "human":
            raise ValueError(
                f"cta_source='pirlygenes' is human-only; got "
                f"species={species!r}.  Pass a species-appropriate "
                f"set or callable."
            )
        from topiary.sources import _pirlygenes_cta_gene_ids
        return set(_pirlygenes_cta_gene_ids())
    if cta_source == "tsarina":
        raise NotImplementedError(
            "cta_source='tsarina' is reserved for a follow-up PR."
        )
    raise ValueError(
        f"Unsupported cta_source: {cta_source!r}.  Use 'pirlygenes', "
        f"a set of gene IDs, a callable, or None to use the species "
        f"default."
    )


# ---------------------------------------------------------------------------
# SelfProteome
# ---------------------------------------------------------------------------


class SelfProteome:
    """Reference proteome for cross-reactivity / nearest-self lookups.

    Constructed via :meth:`from_fasta`, :meth:`from_ensembl`, or
    :meth:`from_peptides` (test helper).  Holds per-length reference
    arrays + a provenance index, and answers :meth:`nearest` queries.
    """

    def __init__(
        self,
        *,
        species: str,
        release: Optional[str],
        scope_label: str,
        reference_arrays: Dict[int, np.ndarray],
        reference_peptides: Dict[int, List[str]],
        provenance: Dict[str, List[Tuple[str, str, int]]],
    ):
        self.species = species
        self.release = release
        self.scope_label = scope_label
        self._reference_arrays = reference_arrays
        self._reference_peptides = reference_peptides
        self._provenance = provenance

    # --- metadata ---

    @property
    def peptide_lengths(self) -> List[int]:
        return sorted(self._reference_arrays.keys())

    @property
    def n_reference_peptides(self) -> int:
        return sum(len(v) for v in self._reference_peptides.values())

    @property
    def reference_version(self) -> str:
        """Composite version string for reproducibility.

        Stamped on every row of :meth:`nearest`'s output; two runs with
        matching strings produce interchangeable ``self_nearest_peptide``
        values.
        """
        release_part = f"-{self.release}" if self.release is not None else ""
        return (
            f"ensembl-{self.species}{release_part}+scope-{self.scope_label}"
        )

    # --- lookup ---

    def nearest(self, peptides: Iterable[str]) -> pd.DataFrame:
        """For each query peptide, return the closest reference peptide
        at the same length (Hamming distance, substitutions only).

        Returns a DataFrame with one row per input peptide, preserving
        input order, containing ``peptide``, ``self_nearest_peptide``,
        ``self_nearest_peptide_length``, ``self_nearest_edit_distance``,
        ``self_nearest_gene_id``, ``self_nearest_transcript_id``,
        ``self_nearest_reference_offset``, and
        ``self_nearest_reference_version``.  Rows where no reference
        exists at the query's length have ``None`` / ``NaN`` for the
        nearest-peptide columns.

        Tie-breaking: when multiple reference peptides share the
        minimum Hamming distance to the query, the first one in the
        internal reference array (construction / insertion order) is
        returned.  Deterministic but implementation-dependent — don't
        rely on which specific peptide wins unless you also control
        the reference-construction order.  The upcoming
        ``self_nearest_candidates`` structured column (see #124, part B)
        exposes the full tied set for callers who care.
        """
        peptides = [str(p) for p in peptides]
        # Partition queries by length to batch SIMD calls.
        by_length: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for idx, pep in enumerate(peptides):
            by_length[len(pep)].append((idx, pep))

        results: List[Optional[dict]] = [None] * len(peptides)
        for L, items in by_length.items():
            if L not in self._reference_arrays:
                for idx, pep in items:
                    results[idx] = self._empty_row(pep)
                continue
            self._resolve_length(L, items, results)

        return pd.DataFrame(results)

    def _empty_row(self, peptide: str) -> dict:
        return {
            "peptide": peptide,
            "self_nearest_peptide": None,
            "self_nearest_peptide_length": None,
            "self_nearest_edit_distance": None,
            "self_nearest_gene_id": None,
            "self_nearest_transcript_id": None,
            "self_nearest_reference_offset": None,
            "self_nearest_reference_version": self.reference_version,
        }

    def _resolve_length(
        self,
        L: int,
        items: List[Tuple[int, str]],
        results: List[Optional[dict]],
        chunk_size: int = 1000,
    ) -> None:
        ref_arr = self._reference_arrays[L]
        ref_peps = self._reference_peptides[L]
        query_peps = [pep for _, pep in items]
        query_arr = _encode_peptides(query_peps, L)

        # Chunk to bound working memory: (chunk, M, L) diff tensor.
        for start in range(0, len(query_arr), chunk_size):
            end = min(start + chunk_size, len(query_arr))
            q_chunk = query_arr[start:end]
            # (chunk, M, L) != broadcasted → sum over last axis → (chunk, M).
            diffs = (
                q_chunk[:, None, :] != ref_arr[None, :, :]
            ).sum(axis=2)
            best_idx = diffs.argmin(axis=1)
            best_dist = diffs[np.arange(len(q_chunk)), best_idx]

            for k, (orig_idx, orig_pep) in enumerate(items[start:end]):
                ref_pep = ref_peps[int(best_idx[k])]
                prov = self._provenance.get(ref_pep)
                if prov:
                    gene_id, transcript_id, offset = prov[0]
                else:
                    gene_id, transcript_id, offset = None, None, None
                results[orig_idx] = {
                    "peptide": orig_pep,
                    "self_nearest_peptide": ref_pep,
                    "self_nearest_peptide_length": L,
                    "self_nearest_edit_distance": int(best_dist[k]),
                    "self_nearest_gene_id": gene_id,
                    "self_nearest_transcript_id": transcript_id,
                    "self_nearest_reference_offset": offset,
                    "self_nearest_reference_version": self.reference_version,
                }

    # --- constructors ---

    @classmethod
    def from_peptides(
        cls,
        peptides_by_source: Dict[str, str],
        *,
        peptide_lengths: Iterable[int] = (8, 9, 10, 11),
        species: str = "synthetic",
        release: Optional[str] = None,
        scope_label: str = "all",
    ) -> "SelfProteome":
        """Build from an in-memory ``{source_id: amino_acid_sequence}``
        mapping.  Primarily a test helper; also useful for small
        programmatic reference sets.

        ``source_id`` is recorded as both ``gene_id`` and
        ``transcript_id`` in the provenance index since this constructor
        doesn't distinguish between them.
        """
        reference_arrays, reference_peptides, provenance = _build_index(
            (
                (source_id, source_id, source_id, seq)
                for source_id, seq in peptides_by_source.items()
            ),
            peptide_lengths,
        )
        return cls(
            species=species,
            release=release,
            scope_label=scope_label,
            reference_arrays=reference_arrays,
            reference_peptides=reference_peptides,
            provenance=provenance,
        )

    @classmethod
    def from_fasta(
        cls,
        path,
        *,
        peptide_lengths: Iterable[int] = (8, 9, 10, 11),
        species: str = "fasta",
        release: Optional[str] = None,
        scope: Union[str, Callable] = "all",
    ) -> "SelfProteome":
        """Build from a FASTA file.  Each record's ID is used as both
        gene_id and transcript_id in provenance (FASTA doesn't carry
        the distinction).

        ``scope`` accepts ``"all"`` or a callable ``(source_id) -> bool``;
        ``"non_cta"`` and ``"protected_tissues"`` require gene-metadata
        that FASTA doesn't provide, so use :meth:`from_ensembl` for
        those scopes.
        """
        records = list(_parse_fasta(path))
        scope_label, records = _apply_fasta_scope(scope, records)
        reference_arrays, reference_peptides, provenance = _build_index(
            records, peptide_lengths,
        )
        return cls(
            species=species,
            release=release,
            scope_label=scope_label,
            reference_arrays=reference_arrays,
            reference_peptides=reference_peptides,
            provenance=provenance,
        )

    @classmethod
    def from_ensembl(
        cls,
        species: str = "human",
        release: Optional[int] = None,
        *,
        peptide_lengths: Iterable[int] = (8, 9, 10, 11),
        scope: Union[str, Callable] = "non_cta",
        cta_source=None,
    ) -> "SelfProteome":
        """Build from a pyensembl EnsemblRelease.

        ``scope`` accepts ``"all"``, ``"non_cta"``, or a callable
        ``(gene_id) -> bool``.  For ``scope="non_cta"`` with
        ``species="human"``, the default ``cta_source="pirlygenes"``
        needs no configuration.  Non-human species must pass
        ``cta_source=<set or callable>`` explicitly.

        ``scope="protected_tissues"`` lands in a follow-up PR.
        """
        try:
            from pyensembl import EnsemblRelease
        except ImportError as e:
            raise ImportError(
                "pyensembl is required for SelfProteome.from_ensembl"
            ) from e

        genome = EnsemblRelease(release=release, species=species)
        scope_label, gene_filter = _resolve_ensembl_scope(
            scope, species, cta_source,
        )
        records = list(_iter_ensembl_proteins(genome, gene_filter))
        reference_arrays, reference_peptides, provenance = _build_index(
            records, peptide_lengths,
        )
        return cls(
            species=species,
            release=str(release) if release is not None else None,
            scope_label=scope_label,
            reference_arrays=reference_arrays,
            reference_peptides=reference_peptides,
            provenance=provenance,
        )


# ---------------------------------------------------------------------------
# FASTA + Ensembl helpers
# ---------------------------------------------------------------------------


def _parse_fasta(path):
    """Yield ``(source_id, source_id, source_id, sequence)`` tuples."""
    current_id = None
    buf: List[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    yield current_id, current_id, current_id, "".join(buf)
                header = line[1:].split()[0]
                current_id = header
                buf = []
            else:
                buf.append(line)
    if current_id is not None:
        yield current_id, current_id, current_id, "".join(buf)


def _apply_fasta_scope(scope, records):
    """Filter FASTA records by scope.  Returns (scope_label, filtered_records)."""
    if scope == "all":
        return "all", records
    if callable(scope):
        label = (
            "callable-"
            + hashlib.sha256(repr(scope).encode()).hexdigest()[:12]
        )
        return label, [r for r in records if scope(r[0])]
    raise ValueError(
        f"scope={scope!r} isn't available for from_fasta (FASTA has no "
        f"gene/tissue metadata).  Use 'all', a callable, or switch to "
        f"from_ensembl."
    )


def _resolve_ensembl_scope(scope, species, cta_source):
    """Return (scope_label, gene_filter) where gene_filter takes a gene_id
    and returns True to keep."""
    if callable(scope):
        label = (
            "callable-"
            + hashlib.sha256(repr(scope).encode()).hexdigest()[:12]
        )
        return label, scope
    if scope == "all":
        return "all", lambda _gene_id: True
    if scope == "non_cta":
        cta = _resolve_cta_gene_ids(species, cta_source)
        if callable(cta):
            return "non_cta-callable", lambda g: not cta(g)
        label = _cta_label(species, cta_source, cta)
        cta_set = cta  # set of gene IDs
        return label, lambda gene_id: gene_id not in cta_set
    if scope == "protected_tissues":
        raise NotImplementedError(
            "scope='protected_tissues' lands in a follow-up PR."
        )
    raise ValueError(f"Unknown scope: {scope!r}.")


def _cta_label(species, cta_source, cta_set):
    """Compose a human-readable label for the non_cta scope."""
    if cta_source is None and species == "human":
        # Using pirlygenes default.
        try:
            import pirlygenes
            return f"non_cta+cta-pirlygenes-{pirlygenes.__version__}"
        except (ImportError, AttributeError):
            return "non_cta+cta-pirlygenes"
    if isinstance(cta_source, str):
        return f"non_cta+cta-{cta_source}"
    # Custom set — hash the gene IDs for reproducibility.
    digest = hashlib.sha256(
        "\n".join(sorted(cta_set)).encode()
    ).hexdigest()[:12]
    return f"non_cta+cta-sha256:{digest}"


def _iter_ensembl_proteins(genome, gene_filter):
    """Yield ``(gene_id, transcript_id, protein_id, sequence)`` tuples for
    every protein whose gene passes ``gene_filter``."""
    for protein_id in genome.protein_ids():
        try:
            gene_id = genome.gene_id_of_protein_id(protein_id)
        except ValueError:
            continue
        if not gene_filter(gene_id):
            continue
        try:
            transcript_id = genome.transcript_id_of_protein_id(protein_id)
        except (ValueError, AttributeError):
            transcript_id = protein_id
        seq = genome.protein_sequence(protein_id)
        if not seq:
            continue
        yield gene_id, transcript_id, protein_id, seq


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _build_index(records, peptide_lengths):
    """Build per-length reference arrays and a provenance index.

    ``records`` yields ``(gene_id, transcript_id, protein_id, sequence)``.

    Returns (reference_arrays, reference_peptides, provenance) where:
    - ``reference_arrays[L]`` is an ``(M_L, L)`` int8 NumPy array.
    - ``reference_peptides[L]`` is a list of peptide strings aligned to
      the rows of ``reference_arrays[L]``.
    - ``provenance[peptide]`` is a list of
      ``(gene_id, transcript_id, offset)`` tuples — one entry per
      occurrence, so paralogs / repeats all contribute.
    """
    peptide_lengths = sorted(set(peptide_lengths))
    # Dedupe peptides per length while accumulating provenance.
    peptides_by_length: Dict[int, Dict[str, None]] = {
        L: {} for L in peptide_lengths
    }
    provenance: Dict[str, List[Tuple[str, str, int]]] = defaultdict(list)

    for gene_id, transcript_id, _protein_id, seq in records:
        for L in peptide_lengths:
            if L > len(seq):
                continue
            for offset in range(len(seq) - L + 1):
                pep = seq[offset:offset + L]
                peptides_by_length[L][pep] = None
                provenance[pep].append((gene_id, transcript_id, offset))

    reference_peptides: Dict[int, List[str]] = {}
    reference_arrays: Dict[int, np.ndarray] = {}
    for L, pep_dict in peptides_by_length.items():
        peps = list(pep_dict.keys())
        reference_peptides[L] = peps
        reference_arrays[L] = _encode_peptides(peps, L) if peps else (
            np.empty((0, L), dtype=np.int8)
        )

    logging.info(
        "SelfProteome index: %d distinct peptides across lengths %s",
        sum(len(v) for v in reference_peptides.values()),
        peptide_lengths,
    )
    return reference_arrays, reference_peptides, dict(provenance)
