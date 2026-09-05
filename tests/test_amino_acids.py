"""Scientific and mutability checks for the public amino-acid data API."""

from pathlib import Path

import numpy as np
import pytest

import topiary
from topiary import (
    AMINO_ACIDS,
    AMINO_ACID_INDEX,
    BLOSUM62_AMINO_ACIDS,
    ENCODED_AMINO_ACIDS,
    UNKNOWN_AMINO_ACID_INDEX,
    blosum62_distance_matrix,
    blosum62_matrix,
    encode_amino_acids,
)


def _ncbi_blosum62():
    """Read the unmodified NCBI table checked into tests as provenance."""
    lines = [
        line.split()
        for line in Path("tests/data/BLOSUM62.ncbi").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    order = lines[0]
    rows = {tokens[0]: [int(value) for value in tokens[1:]] for tokens in lines[1:]}
    return order, rows


def test_every_scored_blosum62_cell_matches_ncbi():
    order, rows = _ncbi_blosum62()
    expected = np.array(
        [
            [rows[row][order.index(column)] for column in BLOSUM62_AMINO_ACIDS]
            for row in BLOSUM62_AMINO_ACIDS
        ],
        dtype=np.int8,
    )

    size = len(BLOSUM62_AMINO_ACIDS)
    observed = blosum62_matrix()[:size, :size]

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed, observed.T)


def test_unscored_lookup_policy_is_explicit():
    matrix = blosum62_matrix()

    size = len(ENCODED_AMINO_ACIDS) + 1
    assert matrix.shape == (size, size)
    assert matrix.dtype == np.int8
    for amino_acid in "OUX*":
        index = AMINO_ACID_INDEX[amino_acid]
        assert np.all(matrix[index, :] == -4)
        assert np.all(matrix[:, index] == -4)
    assert np.all(matrix[UNKNOWN_AMINO_ACID_INDEX] == -4)
    assert np.all(matrix[:, UNKNOWN_AMINO_ACID_INDEX] == -4)


def test_canonical_distance_policy_is_unchanged():
    scores = blosum62_matrix()[: len(AMINO_ACIDS), : len(AMINO_ACIDS)]
    expected = np.diag(scores)[:, None] - scores

    observed = blosum62_distance_matrix()[: len(AMINO_ACIDS), : len(AMINO_ACIDS)]

    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    ("ambiguous", "canonical", "expected"),
    [
        ("B", "D", 2),
        ("B", "N", 2),
        ("J", "I", 1),
        ("J", "L", 1),
        ("Z", "E", 1),
        ("Z", "Q", 1),
    ],
)
def test_published_ambiguity_distances_are_symmetric(
    ambiguous,
    canonical,
    expected,
):
    distances = blosum62_distance_matrix()
    ambiguous_index = AMINO_ACID_INDEX[ambiguous]
    canonical_index = AMINO_ACID_INDEX[canonical]

    assert distances[ambiguous_index, canonical_index] == expected
    assert distances[canonical_index, ambiguous_index] == expected


def test_every_published_ambiguity_distance_is_symmetric():
    distances = blosum62_distance_matrix()
    scored_indices = [AMINO_ACID_INDEX[aa] for aa in BLOSUM62_AMINO_ACIDS]
    ambiguous_indices = [AMINO_ACID_INDEX[aa] for aa in "BJZ"]

    for ambiguous_index in ambiguous_indices:
        np.testing.assert_array_equal(
            distances[ambiguous_index, scored_indices],
            distances[scored_indices, ambiguous_index],
        )


def test_unscored_residues_have_symmetric_worst_case_distance():
    distances = blosum62_distance_matrix()
    unscored_indices = [AMINO_ACID_INDEX[aa] for aa in "OUX*"]
    unscored_indices.append(UNKNOWN_AMINO_ACID_INDEX)

    for index in unscored_indices:
        assert np.all(distances[index, :] == 15)
        assert np.all(distances[:, index] == 15)


@pytest.mark.parametrize(
    "matrix_factory",
    [
        blosum62_matrix,
        blosum62_distance_matrix,
    ],
)
def test_matrix_calls_do_not_share_mutable_state(matrix_factory):
    first = matrix_factory()
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 99

    # A determined caller can make its own array writable, but a later caller
    # still receives canonical data because no mutable storage is shared.
    first.setflags(write=True)
    first[0, 0] = 99
    second = matrix_factory()
    assert second[0, 0] != 99
    assert not second.flags.writeable


def test_amino_acid_index_is_immutable():
    with pytest.raises(TypeError):
        AMINO_ACID_INDEX["A"] = 99
    assert AMINO_ACID_INDEX["A"] == 0


def test_encoding_is_case_insensitive_and_accepts_iterables():
    sequences = (sequence for sequence in ["aCd", "WyX"])

    observed = encode_amino_acids(sequences, length=3)

    expected = np.array(
        [
            [AMINO_ACID_INDEX["A"], AMINO_ACID_INDEX["C"], AMINO_ACID_INDEX["D"]],
            [AMINO_ACID_INDEX["W"], AMINO_ACID_INDEX["Y"], AMINO_ACID_INDEX["X"]],
        ],
        dtype=np.int8,
    )
    np.testing.assert_array_equal(observed, expected)


def test_encoding_pads_truncates_and_distinguishes_named_residues():
    observed = encode_amino_acids(["AC", "ACDE", "BJOUXZ*?"], length=8)

    assert observed[0].tolist() == [
        AMINO_ACID_INDEX["A"],
        AMINO_ACID_INDEX["C"],
        *([UNKNOWN_AMINO_ACID_INDEX] * 6),
    ]
    assert observed[1, :3].tolist() == [
        AMINO_ACID_INDEX["A"],
        AMINO_ACID_INDEX["C"],
        AMINO_ACID_INDEX["D"],
    ]
    assert observed[2].tolist() == [
        AMINO_ACID_INDEX[amino_acid] for amino_acid in "BJOUXZ*"
    ] + [UNKNOWN_AMINO_ACID_INDEX]


@pytest.mark.parametrize(
    "name",
    [
        "AMINO_ACIDS",
        "AMINO_ACID_INDEX",
        "BLOSUM62_AMINO_ACIDS",
        "ENCODED_AMINO_ACIDS",
        "UNKNOWN_AMINO_ACID_INDEX",
        "blosum62_distance_matrix",
        "blosum62_matrix",
        "encode_amino_acids",
    ],
)
def test_amino_acid_api_is_exported(name):
    assert name in topiary.__all__
    assert getattr(topiary, name) is not None
