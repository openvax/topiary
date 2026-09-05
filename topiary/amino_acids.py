"""Amino-acid encoding and BLOSUM62 score/distance data."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np


# Canonical amino acids retain their historical indices so existing encoded
# data and canonical distance calculations do not change. NCBI supplies
# BLOSUM62 rows for the ambiguity symbols B, J, and Z. Other explicitly named
# residues are distinguished by the encoding but use Topiary's unknown-distance
# policy because BLOSUM62 does not provide substitution evidence for them.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
BLOSUM62_AMINO_ACIDS = AMINO_ACIDS + "BJZ"
ENCODED_AMINO_ACIDS = BLOSUM62_AMINO_ACIDS + "OUX*"
AMINO_ACID_INDEX = MappingProxyType({
    amino_acid: index for index, amino_acid in enumerate(ENCODED_AMINO_ACIDS)
})
UNKNOWN_AMINO_ACID_INDEX = len(ENCODED_AMINO_ACIDS)


# Canonical BLOSUM62 substitution scores for the 20 standard amino acids in
# AMINO_ACIDS order. This immutable tuple is transcribed from NCBI's BLOSUM62
# matrix: https://ftp.ncbi.nlm.nih.gov/blast/matrices/BLOSUM62
_BLOSUM62_STANDARD_SCORES = (
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y
    ( 4,  0, -2, -1, -2,  0, -2, -1, -1, -1, -1, -2, -1, -1, -1,  1,  0,  0, -3, -2),  # A
    ( 0,  9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2),  # C
    (-2, -3,  6,  2, -3, -1, -1, -3, -1, -4, -3,  1, -1,  0, -2,  0, -1, -3, -4, -3),  # D
    (-1, -4,  2,  5, -3, -2,  0, -3,  1, -3, -2,  0, -1,  2,  0,  0, -1, -2, -3, -2),  # E
    (-2, -2, -3, -3,  6, -3, -1,  0, -3,  0,  0, -3, -4, -3, -3, -2, -2, -1,  1,  3),  # F
    ( 0, -3, -1, -2, -3,  6, -2, -4, -2, -4, -3,  0, -2, -2, -2,  0, -2, -3, -2, -3),  # G
    (-2, -3, -1,  0, -1, -2,  8, -3, -1, -3, -2,  1, -2,  0,  0, -1, -2, -3, -2,  2),  # H
    (-1, -1, -3, -3,  0, -4, -3,  4, -3,  2,  1, -3, -3, -3, -3, -2, -1,  3, -3, -1),  # I
    (-1, -3, -1,  1, -3, -2, -1, -3,  5, -2, -1,  0, -1,  1,  2,  0, -1, -2, -3, -2),  # K
    (-1, -1, -4, -3,  0, -4, -3,  2, -2,  4,  2, -3, -3, -2, -2, -2, -1,  1, -2, -1),  # L
    (-1, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5, -2, -2,  0, -1, -1, -1,  1, -1, -1),  # M
    (-2, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6, -2,  0,  0,  1,  0, -3, -4, -2),  # N
    (-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7, -1, -2, -1, -1, -2, -4, -3),  # P
    (-1, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,  1,  0, -1, -2, -2, -1),  # Q
    (-1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5, -1, -1, -3, -3, -2),  # R
    ( 1, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,  1, -2, -3, -2),  # S
    ( 0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,  0, -2, -2),  # T
    ( 0, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4, -3, -1),  # V
    (-3, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,  2),  # W
    (-2, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2,  7),  # Y
)


# NCBI's BLOSUM62 rows for ambiguity symbols, reordered to
# BLOSUM62_AMINO_ACIDS. The source matrix is symmetric, so each tuple supplies
# both the corresponding row and column.
_BLOSUM62_AMBIGUOUS_SCORES = {
    # A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y   B   J   Z
    "B": (-2, -3,  4,  1, -3, -1,  0, -3,  0, -4, -3,  4, -2,  0, -1,  0, -1, -3, -4, -3,  4, -3,  0),
    "J": (-1, -1, -3, -3,  0, -4, -3,  3, -3,  3,  2, -3, -3, -2, -2, -2, -1,  2, -2, -1, -3,  3, -3),
    "Z": (-1, -3,  1,  4, -3, -2,  0, -3,  1, -3, -1,  0, -1,  4,  0,  0, -1, -2, -2, -2,  0, -3,  4),
}


def blosum62_matrix():
    """Return Topiary's read-only BLOSUM62 lookup matrix.

    Returns
    -------
    numpy.ndarray
        A fresh int8 array indexed by :data:`AMINO_ACID_INDEX`, plus the
        catch-all :data:`UNKNOWN_AMINO_ACID_INDEX`. B/J/Z use the ambiguity
        rows published with NCBI's BLOSUM62 matrix. O/U/X/* and the catch-all
        index have substitution score -4. The array is read-only, and no
        mutable storage is shared between calls.

    Notes
    -----
    This function exposes substitution scores. For nearest-self distance
    calculations, use :func:`blosum62_distance_matrix`, which defines explicit
    symmetric behavior for residues without published BLOSUM62 scores.
    """
    size = len(ENCODED_AMINO_ACIDS) + 1
    matrix = np.full((size, size), -4, dtype=np.int8)
    matrix[:len(AMINO_ACIDS), :len(AMINO_ACIDS)] = _BLOSUM62_STANDARD_SCORES
    scored_size = len(BLOSUM62_AMINO_ACIDS)
    for amino_acid, scores in _BLOSUM62_AMBIGUOUS_SCORES.items():
        index = AMINO_ACID_INDEX[amino_acid]
        matrix[index, :scored_size] = scores
        matrix[:scored_size, index] = scores
    matrix.flags.writeable = False
    return matrix


def blosum62_distance_matrix():
    """Return Topiary's read-only BLOSUM62 distance lookup matrix.

    Returns
    -------
    numpy.ndarray
        A fresh int8 array aligned with :func:`blosum62_matrix`. Canonical
        amino-acid pairs preserve Topiary's historical query-directed
        ``self score - pair score`` distance. Pairs involving B/J/Z use the
        symmetric, conservative ``max(self scores) - pair score`` distance.
        Pairs involving O/U/X/* or an unrecognized character receive 15, the
        maximum canonical substitution distance. The array is read-only, and
        no mutable storage is shared between calls.

    Notes
    -----
    B/J/Z have published substitution scores in NCBI's BLOSUM62 matrix. The
    fixed distance for other residues prevents missing substitution evidence
    from appearing to be an exact match. Canonical-only calculations are
    unchanged for compatibility.
    """
    scores = blosum62_matrix()
    canonical_size = len(AMINO_ACIDS)
    scored_size = len(BLOSUM62_AMINO_ACIDS)

    canonical_scores = scores[:canonical_size, :canonical_size]
    canonical_distances = np.diag(canonical_scores)[:, None] - canonical_scores
    unknown_distance = canonical_distances.max()
    distances = np.full(scores.shape, unknown_distance, dtype=np.int8)

    scored_scores = scores[:scored_size, :scored_size]
    scored_self = np.diag(scored_scores)
    symmetric_distances = (
        np.maximum(scored_self[:, None], scored_self[None, :]) - scored_scores
    )
    distances[:scored_size, :scored_size] = symmetric_distances
    distances[:canonical_size, :canonical_size] = canonical_distances
    distances.flags.writeable = False
    return distances


def encode_amino_acids(sequences, length):
    """Encode amino-acid sequences as compact integer rows.

    Parameters
    ----------
    sequences : iterable of str
        Amino-acid sequences. Matching is case-insensitive.
    length : int
        Width of the returned rows. Short sequences are padded with the
        unknown index and long sequences are truncated.

    Returns
    -------
    numpy.ndarray
        ``(N, length)`` int8 array using :data:`AMINO_ACID_INDEX` and
        :data:`UNKNOWN_AMINO_ACID_INDEX`. B/J/Z retain their published
        BLOSUM62 identities; O/U/X/* are distinguished so Hamming distance can
        compare them faithfully. Any other character uses the catch-all
        unknown index.
    """
    sequences = list(sequences)
    encoded = np.full(
        (len(sequences), length), UNKNOWN_AMINO_ACID_INDEX, dtype=np.int8,
    )
    for row, sequence in enumerate(sequences):
        for column, amino_acid in enumerate(sequence[:length]):
            encoded[row, column] = AMINO_ACID_INDEX.get(
                amino_acid.upper(), UNKNOWN_AMINO_ACID_INDEX,
            )
    return encoded


__all__ = [
    "AMINO_ACIDS",
    "AMINO_ACID_INDEX",
    "BLOSUM62_AMINO_ACIDS",
    "ENCODED_AMINO_ACIDS",
    "UNKNOWN_AMINO_ACID_INDEX",
    "blosum62_distance_matrix",
    "blosum62_matrix",
    "encode_amino_acids",
]
