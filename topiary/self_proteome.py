"""SelfProteome — reference protein corpus for cross-reactivity analysis.

Holds a species-tagged, ``include=``-filtered protein set indexed by
peptide length.  Answers per-query nearest-neighbor lookups: "given
this mutant peptide, what's the most similar peptide in healthy self?"

Include modes
-------------
- ``"all"``: no filter, whole Ensembl proteome (any pyensembl-supported
  species).
- ``"non_cta"`` (default for human): remove cancer-testis-antigen genes
  via pirlygenes.  Non-human species require an explicit ``cta_source=``
  set or callable — pirlygenes is human-only today.
- ``"protected_tissues"``: keep only genes expressed in named tissues
  (pirlygenes/HPA for human; user-supplied ``tissue_gene_ids=`` for
  any species).
- callable: user-supplied ``gene → bool`` filter.  Works for any species.

Distance metrics
----------------
- ``"blosum62"`` (default): BLOSUM62-weighted substitution distance.
  Conservative substitutions (I↔L) produce lower distances than
  non-conservative (I↔W).  Matrix is inlined as a constant — no
  runtime dependency on Biopython.
- ``"hamming"``: count mismatched positions (all mismatches equal).

Indels
------
``nearest(include_indels=True)`` (default) checks 1-deletion (L-1) and
1-insertion (L+1) neighbors via hash-set lookup.  An indel match at
edit_distance=1 beats a same-length substitution match at
edit_distance≥2.

Binding-aware axes (``self_mimic_*``, ``self_strongest_nearby_*``,
``self_nearest_candidates``) are tracked for a follow-up PR — they
require MHC prediction on candidate peptides, which is
architecturally separate from the sequence-only ``nearest()`` method.

Algorithm
---------
SIMD-vectorized distance against int8-encoded reference arrays. Indels
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


# Canonical BLOSUM62 substitution scores for the 20 standard amino
# acids.  Rows / columns are ordered by ``_AA_ALPHABET``
# ("ACDEFGHIKLMNPQRSTVWY").  Values are the integer scores from the
# standard NCBI BLOSUM62 matrix and match what
# ``Bio.Align.substitution_matrices.load("BLOSUM62")`` returns for
# these 20 residues.  We inline the table to avoid a biopython
# dependency for a 400-integer constant.
_BLOSUM62_STANDARD = np.array([
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    [  4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2],  # A
    [  0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],  # C
    [ -2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3],  # D
    [ -1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2],  # E
    [ -2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3],  # F
    [  0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3],  # G
    [ -2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2],  # H
    [ -1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1],  # I
    [ -1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2],  # K
    [ -1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1],  # L
    [ -1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1],  # M
    [ -2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2],  # N
    [ -1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3],  # P
    [ -1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1],  # Q
    [ -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2],  # R
    [  1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2],  # S
    [  0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2],  # T
    [  0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1],  # V
    [ -3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2],  # W
    [ -2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7],  # Y
], dtype=np.int8)


def _load_blosum62() -> np.ndarray:
    """Return the BLOSUM62 lookup as a (21, 21) int8 table indexed by
    ``_AA_TO_INT`` encoding.

    Row / column 20 (the sentinel for unknown residues B/J/O/U/X/Z/*)
    scores -4 against everything so unknowns always rank as
    large-distance mismatches.
    """
    n = len(_AA_ALPHABET) + 1  # +1 for sentinel
    table = np.full((n, n), -4, dtype=np.int8)
    table[:20, :20] = _BLOSUM62_STANDARD
    return table


# Lazy-loaded on first use — kept behind a cache so the module import
# path stays allocation-free.
_BLOSUM62: Optional[np.ndarray] = None


def _get_blosum62() -> np.ndarray:
    global _BLOSUM62
    if _BLOSUM62 is None:
        _BLOSUM62 = _load_blosum62()
    return _BLOSUM62


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
                f"include='non_cta' needs a CTA source for "
                f"species={species!r}; no default registered.  Pass "
                f"cta_source=<set or callable> explicitly, or use "
                f"include='all'."
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
        include_label: str,
        reference_arrays: Dict[int, np.ndarray],
        reference_peptides: Dict[int, List[str]],
        provenance: Dict[str, List[Tuple[str, str, int]]],
    ):
        self.species = species
        self.release = release
        self.include_label = include_label
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
            f"ensembl-{self.species}{release_part}+include-{self.include_label}"
        )

    # --- lookup ---

    def nearest(
        self,
        peptides: Iterable[str],
        metric: str = "blosum62",
        include_indels: bool = True,
    ) -> pd.DataFrame:
        """For each query peptide, return the closest reference peptide
        at the same length.

        Parameters
        ----------
        peptides : iterable of str
        metric : ``"blosum62"`` (default) or ``"hamming"``
            ``"blosum62"`` uses the BLOSUM62 substitution matrix to
            score each position — conservative substitutions (I↔L)
            contribute less distance than non-conservative ones (W↔A).

            ``"hamming"`` counts the number of mismatched positions
            (all mismatches weighted equally).

        Returns a DataFrame with one row per input peptide, preserving
        input order.  Columns: ``peptide``, ``self_nearest_peptide``,
        ``self_nearest_peptide_length``, ``self_nearest_edit_distance``
        (Hamming), ``self_nearest_blosum_distance`` (BLOSUM62-based,
        only when ``metric="blosum62"``), ``self_nearest_gene_id``,
        ``self_nearest_transcript_id``, ``self_nearest_reference_offset``,
        ``self_nearest_reference_version``.

        Tie-breaking: when multiple reference peptides share the
        minimum distance, the first in the internal reference array
        (construction / insertion order) is returned.

        Indel matching: when ``include_indels=True`` and the best
        same-length match has ``edit_distance ≥ 2``, the method also
        checks 1-deletion (L-1) and 1-insertion (L+1) neighbors via
        hash-set lookup.  The **first** indel hit found (deletions
        checked before insertions, in position order) is returned —
        not necessarily the best among all indel candidates.  This is
        fast (~200 lookups per 9-mer) but not exhaustive; the upcoming
        ``self_nearest_candidates`` structured column will expose the
        full candidate set.
        """
        if metric not in ("blosum62", "hamming"):
            raise ValueError(
                f"metric must be 'blosum62' or 'hamming'; got {metric!r}."
            )
        peptides = [str(p) for p in peptides]
        by_length: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for idx, pep in enumerate(peptides):
            by_length[len(pep)].append((idx, pep))

        results: List[Optional[dict]] = [None] * len(peptides)
        for L, items in by_length.items():
            if L not in self._reference_arrays:
                for idx, pep in items:
                    results[idx] = self._empty_row(pep, metric)
                continue
            self._resolve_length(L, items, results, metric=metric)

        # Check 1aa indel neighbors when enabled.  An indel match
        # at edit_distance=1 beats a same-length match at edit_distance≥2.
        if include_indels:
            for idx, pep in enumerate(peptides):
                current = results[idx]
                if current is None:
                    continue
                best_edit = current.get("self_nearest_edit_distance")
                if best_edit is not None and best_edit <= 1:
                    continue
                indel_hit = self._check_indel_neighbors(pep, metric)
                if indel_hit is not None:
                    results[idx] = indel_hit

        return pd.DataFrame(results)

    def _reference_set(self, L: int) -> set:
        """Lazy-built set of reference peptide strings at length L."""
        if not hasattr(self, "_reference_sets_cache"):
            self._reference_sets_cache: Dict[int, set] = {}
        if L not in self._reference_sets_cache:
            peps = self._reference_peptides.get(L, [])
            self._reference_sets_cache[L] = set(peps)
        return self._reference_sets_cache[L]

    def _check_indel_neighbors(
        self, peptide: str, metric: str,
    ) -> Optional[dict]:
        """Check if any 1-insertion or 1-deletion neighbor of ``peptide``
        exists in the reference at lengths L±1.  Returns a result row
        for the first hit found (edit_distance=1), or ``None``.

        Deletion: remove one character at each position → L-1 length.
        Insertion: insert one of 20 AAs at each position → L+1 length.

        ~L + L×20 ≈ 200 hash-set lookups for a 9-mer.  Fast.
        """
        L = len(peptide)
        # Try deletions first (cheaper: L lookups vs L×20 for insertions)
        ref_set_del = self._reference_set(L - 1) if L > 1 else set()
        for i in range(L):
            candidate = peptide[:i] + peptide[i + 1:]
            if candidate in ref_set_del:
                return self._indel_row(
                    peptide, candidate, L - 1, "deletion", metric,
                )
        # Try insertions
        ref_set_ins = self._reference_set(L + 1)
        for i in range(L + 1):
            for aa in _AA_ALPHABET:
                candidate = peptide[:i] + aa + peptide[i:]
                if candidate in ref_set_ins:
                    return self._indel_row(
                        peptide, candidate, L + 1, "insertion", metric,
                    )
        return None

    def _indel_row(
        self, peptide: str, match: str, match_len: int,
        edit_type: str, metric: str,
    ) -> dict:
        prov = self._provenance.get(match)
        if prov:
            gene_id, transcript_id, offset = prov[0]
        else:
            gene_id, transcript_id, offset = None, None, None
        row = {
            "peptide": peptide,
            "self_nearest_peptide": match,
            "self_nearest_peptide_length": match_len,
            "self_nearest_edit_distance": 1,
            "self_nearest_edit_type": edit_type,
            "self_nearest_gene_id": gene_id,
            "self_nearest_transcript_id": transcript_id,
            "self_nearest_reference_offset": offset,
            "self_nearest_reference_version": self.reference_version,
        }
        if metric == "blosum62":
            row["self_nearest_blosum_distance"] = None
        return row

    def _empty_row(self, peptide: str, metric: str) -> dict:
        row = {
            "peptide": peptide,
            "self_nearest_peptide": None,
            "self_nearest_peptide_length": None,
            "self_nearest_edit_distance": None,
            "self_nearest_gene_id": None,
            "self_nearest_transcript_id": None,
            "self_nearest_reference_offset": None,
            "self_nearest_reference_version": self.reference_version,
        }
        if metric == "blosum62":
            row["self_nearest_blosum_distance"] = None
        return row

    def _resolve_length(
        self,
        L: int,
        items: List[Tuple[int, str]],
        results: List[Optional[dict]],
        metric: str = "blosum62",
        chunk_size: int = 1000,
    ) -> None:
        ref_arr = self._reference_arrays[L]
        ref_peps = self._reference_peptides[L]
        query_peps = [pep for _, pep in items]
        query_arr = _encode_peptides(query_peps, L)

        use_blosum = metric == "blosum62"
        blosum = _get_blosum62() if use_blosum else None

        for start in range(0, len(query_arr), chunk_size):
            end = min(start + chunk_size, len(query_arr))
            q_chunk = query_arr[start:end]

            if use_blosum:
                # BLOSUM62 distance: sum of (self_score - pair_score)
                # per position.  Lower = more similar.
                pair_scores = blosum[
                    q_chunk[:, None, :], ref_arr[None, :, :]
                ].sum(axis=2)
                self_scores = blosum[
                    q_chunk, q_chunk
                ].sum(axis=1)
                dists = self_scores[:, None] - pair_scores
            else:
                # Hamming: count mismatched positions.
                dists = (
                    q_chunk[:, None, :] != ref_arr[None, :, :]
                ).sum(axis=2)

            best_idx = dists.argmin(axis=1)
            best_dist = dists[np.arange(len(q_chunk)), best_idx]

            # Also compute Hamming for the edit_distance column
            # regardless of metric (it's always useful).
            if use_blosum:
                hamming = (
                    q_chunk[:, None, :] != ref_arr[None, :, :]
                ).sum(axis=2)
                best_hamming = hamming[np.arange(len(q_chunk)), best_idx]
            else:
                best_hamming = best_dist

            for k, (orig_idx, orig_pep) in enumerate(items[start:end]):
                ref_pep = ref_peps[int(best_idx[k])]
                prov = self._provenance.get(ref_pep)
                if prov:
                    gene_id, transcript_id, offset = prov[0]
                else:
                    gene_id, transcript_id, offset = None, None, None
                row = {
                    "peptide": orig_pep,
                    "self_nearest_peptide": ref_pep,
                    "self_nearest_peptide_length": L,
                    "self_nearest_edit_distance": int(best_hamming[k]),
                    "self_nearest_gene_id": gene_id,
                    "self_nearest_transcript_id": transcript_id,
                    "self_nearest_reference_offset": offset,
                    "self_nearest_reference_version": self.reference_version,
                }
                if use_blosum:
                    row["self_nearest_blosum_distance"] = int(best_dist[k])
                results[orig_idx] = row

    # --- constructors ---

    @classmethod
    def from_peptides(
        cls,
        peptides_by_source: Dict[str, str],
        *,
        peptide_lengths: Iterable[int] = (8, 9, 10, 11),
        species: str = "synthetic",
        release: Optional[str] = None,
        include_label: str = "all",
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
            include_label=include_label,
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
        include: Union[str, Callable] = "all",
    ) -> "SelfProteome":
        """Build from a FASTA file.  Each record's ID is used as both
        gene_id and transcript_id in provenance (FASTA doesn't carry
        the distinction).

        ``include`` accepts ``"all"`` or a callable ``(source_id) -> bool``;
        ``"non_cta"`` and ``"protected_tissues"`` require gene-metadata
        that FASTA doesn't provide, so use :meth:`from_ensembl` for
        those.
        """
        records = list(_parse_fasta(path))
        include_label, records = _apply_fasta_scope(include, records)
        reference_arrays, reference_peptides, provenance = _build_index(
            records, peptide_lengths,
        )
        return cls(
            species=species,
            release=release,
            include_label=include_label,
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
        include: Union[str, Callable] = "non_cta",
        cta_source=None,
        tissues: Optional[List[str]] = None,
        tissue_gene_ids: Optional[Set[str]] = None,
        min_tissue_ntpm: float = 1.0,
    ) -> "SelfProteome":
        """Build from a pyensembl EnsemblRelease.

        ``include`` accepts ``"all"``, ``"non_cta"``,
        ``"protected_tissues"``, or a callable ``(gene_id) -> bool``.

        For ``include="non_cta"`` with ``species="human"``, the default
        ``cta_source="pirlygenes"`` needs no configuration.  Non-human
        species must pass ``cta_source=<set or callable>`` explicitly.

        For ``include="protected_tissues"``:

        - **Human (default)**: filters to genes expressed in the
          ``tissues`` list via pirlygenes' HPA expression data.
          ``tissues`` defaults to a curated vital-organ set
          (``["heart_muscle", "lung", "liver", "kidney",
          "cerebral_cortex"]``).  ``min_tissue_ntpm`` sets the
          expression threshold (normalized TPM).
        - **Any species with explicit data**: pass
          ``tissue_gene_ids=<set of gene IDs>`` to supply the gene set
          directly.  ``tissues`` and ``min_tissue_ntpm`` are ignored
          when ``tissue_gene_ids`` is provided.  This is the path for
          mouse (e.g. Tabula Muris data), dog, or any non-human
          species.
        - **Non-human without ``tissue_gene_ids``**: raises a
          ``ValueError`` explaining that pirlygenes tissue data is
          human-only.
        """
        try:
            from pyensembl import EnsemblRelease
        except ImportError as e:
            raise ImportError(
                "pyensembl is required for SelfProteome.from_ensembl"
            ) from e

        genome = EnsemblRelease(release=release, species=species)
        include_label, gene_filter = _resolve_ensembl_scope(
            include, species, cta_source,
            tissues=tissues,
            tissue_gene_ids=tissue_gene_ids,
            min_tissue_ntpm=min_tissue_ntpm,
        )
        records = list(_iter_ensembl_proteins(genome, gene_filter))
        reference_arrays, reference_peptides, provenance = _build_index(
            records, peptide_lengths,
        )
        return cls(
            species=species,
            release=str(release) if release is not None else None,
            include_label=include_label,
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


def _apply_fasta_scope(include, records):
    """Filter FASTA records by include mode.  Returns (include_label, filtered)."""
    if include == "all":
        return "all", records
    if callable(include):
        label = (
            "callable-"
            + hashlib.sha256(repr(include).encode()).hexdigest()[:12]
        )
        return label, [r for r in records if include(r[0])]
    raise ValueError(
        f"include={include!r} isn't available for from_fasta (FASTA has no "
        f"gene/tissue metadata).  Use 'all', a callable, or switch to "
        f"from_ensembl."
    )


_DEFAULT_PROTECTED_TISSUES = [
    "heart_muscle", "lung", "liver", "kidney", "cerebral_cortex",
]


def _resolve_ensembl_scope(
    include, species, cta_source, *,
    tissues=None, tissue_gene_ids=None, min_tissue_ntpm=1.0,
):
    """Return (include_label, gene_filter) where gene_filter takes a gene_id
    and returns True to keep."""
    if callable(include):
        label = (
            "callable-"
            + hashlib.sha256(repr(include).encode()).hexdigest()[:12]
        )
        return label, include
    if include == "all":
        return "all", lambda _gene_id: True
    if include == "non_cta":
        cta = _resolve_cta_gene_ids(species, cta_source)
        if callable(cta):
            return "non_cta-callable", lambda g: not cta(g)
        label = _cta_label(species, cta_source, cta)
        cta_set = cta  # set of gene IDs
        return label, lambda gene_id: gene_id not in cta_set
    if include == "protected_tissues":
        keep_ids, label = _resolve_protected_tissues(
            species, tissues, tissue_gene_ids, min_tissue_ntpm,
        )
        return label, lambda gene_id: gene_id in keep_ids
    raise ValueError(f"Unknown include value: {include!r}.")


def _resolve_protected_tissues(species, tissues, tissue_gene_ids, min_ntpm):
    """Resolve the gene set for ``include="protected_tissues"``.

    Returns ``(keep_gene_ids: set[str], include_label: str)``.

    Three paths:

    1. **Explicit gene set** (``tissue_gene_ids`` provided): used
       as-is, any species.  ``tissues`` / ``min_ntpm`` ignored.
    2. **Human default** (``tissue_gene_ids`` is None, species is
       human): queries pirlygenes via
       ``topiary.sources.tissue_expressed_gene_ids`` with the
       requested tissue list (or the curated vital-organ default)
       and ``min_ntpm`` threshold.
    3. **Non-human without explicit gene set**: raises a clear error
       telling the user to supply ``tissue_gene_ids=``.
    """
    if tissue_gene_ids is not None:
        keep = set(tissue_gene_ids)
        digest = hashlib.sha256(
            "\n".join(sorted(keep)).encode()
        ).hexdigest()[:12]
        return keep, f"protected_tissues+gene_ids-sha256:{digest}"

    # Default path — pirlygenes tissue data (human only).
    if species != "human":
        raise ValueError(
            f"include='protected_tissues' without tissue_gene_ids= "
            f"defaults to pirlygenes/HPA expression data, which is "
            f"human-only; got species={species!r}.  For non-human "
            f"species, pass tissue_gene_ids=<set of gene IDs> with "
            f"the gene IDs expressed in your protected tissues "
            f"(e.g. from Tabula Muris for mouse, or your own "
            f"RNA-seq data)."
        )

    if tissues is None:
        tissues = list(_DEFAULT_PROTECTED_TISSUES)

    from topiary.sources import tissue_expressed_gene_ids
    keep = tissue_expressed_gene_ids(tissues, min_ntpm=min_ntpm)
    tissue_str = "+".join(sorted(tissues))
    return keep, (
        f"protected_tissues+tissues-{tissue_str}+min_ntpm-{min_ntpm}"
    )


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
