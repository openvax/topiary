"""Canonical amino-acid encoding and substitution-score data."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np


# The order used by Topiary's compact integer encoding. It differs from the
# display order of NCBI's BLOSUM62 file, but the scores below have been
# reordered without changing their values.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_INDEX = MappingProxyType({
    amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)
})
UNKNOWN_AMINO_ACID_INDEX = len(AMINO_ACIDS)


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


def blosum62_matrix():
    """Return Topiary's read-only BLOSUM62 lookup matrix.

    Returns
    -------
    numpy.ndarray
        A fresh ``(21, 21)`` int8 array. Indices 0–19 follow
        :data:`AMINO_ACIDS`; index :data:`UNKNOWN_AMINO_ACID_INDEX` represents
        every non-standard residue. Its row and column retain Topiary's
        historical score of -4, including unknown-versus-unknown. The array is
        read-only, and no mutable storage is shared between calls.

    Notes
    -----
    The canonical 20×20 scores come from NCBI's BLOSUM62 matrix. Collapsing
    B/J/O/U/X/Z/* into one sentinel is a Topiary compatibility policy, not part
    of the canonical matrix. Its distance semantics are tracked in issue #268.
    """
    size = len(AMINO_ACIDS) + 1
    matrix = np.full((size, size), -4, dtype=np.int8)
    matrix[:len(AMINO_ACIDS), :len(AMINO_ACIDS)] = _BLOSUM62_STANDARD_SCORES
    matrix.flags.writeable = False
    return matrix


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
        :data:`UNKNOWN_AMINO_ACID_INDEX`. B/J/O/U/X/Z/* and any other
        non-standard character all use the unknown index.
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
    "UNKNOWN_AMINO_ACID_INDEX",
    "blosum62_matrix",
    "encode_amino_acids",
]
