"""Scientific and mutability checks for the public amino-acid data API."""

from pathlib import Path

import numpy as np
import pytest

import topiary
from topiary import (
    AMINO_ACIDS,
    AMINO_ACID_INDEX,
    UNKNOWN_AMINO_ACID_INDEX,
    blosum62_matrix,
    encode_amino_acids,
)


def _ncbi_blosum62():
    """Read the unmodified NCBI table checked into tests as provenance."""
    lines = [
        line.split() for line in
        Path("tests/data/BLOSUM62.ncbi").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    order = lines[0]
    rows = {tokens[0]: [int(value) for value in tokens[1:]]
            for tokens in lines[1:]}
    return order, rows


def test_every_canonical_blosum62_cell_matches_ncbi():
    order, rows = _ncbi_blosum62()
    expected = np.array([
        [rows[row][order.index(column)] for column in AMINO_ACIDS]
        for row in AMINO_ACIDS
    ], dtype=np.int8)

    observed = blosum62_matrix()[:len(AMINO_ACIDS), :len(AMINO_ACIDS)]

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(observed, observed.T)


def test_unknown_lookup_policy_is_explicit_and_preserved():
    matrix = blosum62_matrix()

    assert matrix.shape == (21, 21)
    assert matrix.dtype == np.int8
    assert np.all(matrix[UNKNOWN_AMINO_ACID_INDEX, :] == -4)
    assert np.all(matrix[:, UNKNOWN_AMINO_ACID_INDEX] == -4)


def test_matrix_calls_do_not_share_mutable_state():
    first = blosum62_matrix()
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0, 0] = 99

    # A determined caller can make its own array writable, but a later caller
    # still receives canonical data because no mutable storage is shared.
    first.setflags(write=True)
    first[0, 0] = 99
    second = blosum62_matrix()
    assert second[0, 0] == 4
    assert not second.flags.writeable


def test_amino_acid_index_is_immutable():
    with pytest.raises(TypeError):
        AMINO_ACID_INDEX["A"] = 99
    assert AMINO_ACID_INDEX["A"] == 0


def test_encoding_is_case_insensitive_and_accepts_iterables():
    sequences = (sequence for sequence in ["aCd", "WYX"])

    observed = encode_amino_acids(sequences, length=3)

    expected = np.array([
        [AMINO_ACID_INDEX["A"], AMINO_ACID_INDEX["C"],
         AMINO_ACID_INDEX["D"]],
        [AMINO_ACID_INDEX["W"], AMINO_ACID_INDEX["Y"],
         UNKNOWN_AMINO_ACID_INDEX],
    ], dtype=np.int8)
    np.testing.assert_array_equal(observed, expected)


def test_encoding_pads_truncates_and_collapses_nonstandard_residues():
    observed = encode_amino_acids(["AC", "ACDE", "BJOUXZ*?"], length=8)

    assert observed[0].tolist() == [
        AMINO_ACID_INDEX["A"], AMINO_ACID_INDEX["C"],
        *([UNKNOWN_AMINO_ACID_INDEX] * 6),
    ]
    assert observed[1, :3].tolist() == [
        AMINO_ACID_INDEX["A"], AMINO_ACID_INDEX["C"],
        AMINO_ACID_INDEX["D"],
    ]
    assert np.all(observed[2] == UNKNOWN_AMINO_ACID_INDEX)


@pytest.mark.parametrize(
    "name",
    [
        "AMINO_ACIDS",
        "AMINO_ACID_INDEX",
        "UNKNOWN_AMINO_ACID_INDEX",
        "blosum62_matrix",
        "encode_amino_acids",
    ],
)
def test_amino_acid_api_is_exported(name):
    assert name in topiary.__all__
    assert getattr(topiary, name) is not None
