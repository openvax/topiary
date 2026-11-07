"""Direct tests for public protein/peptide sequence geometry helpers."""

import logging

import numpy as np
import pytest

from topiary import peptide_occurrence_flanks, scannable_segments, scorable_peptide
from topiary.sequence_helpers import protein_subsequences_around_mutations


def test_peptide_occurrence_flanks_returns_exact_source_context():
    assert peptide_occurrence_flanks("AASIINFEKLKK", "SIINFEKL", 2) == ("AA", "KK")
    assert peptide_occurrence_flanks("SIINFEKL", "SIINFEKL", np.int64(0)) == ("", "")


@pytest.mark.parametrize("source,peptide,offset,error", [
    ("AASIINFEKL", "siinfekl", 2, ValueError),
    ("AASIINFEKL", "SIINFEKL", 3, ValueError),
    ("AASIINFEKL", "SIINFEKL", -1, ValueError),
    ("", "SIINFEKL", 0, ValueError),
    ("SIINFEKL", "", 0, ValueError),
    (None, "SIINFEKL", 0, TypeError),
    ("SIINFEKL", "SIINFEKL", 0.0, TypeError),
    ("SIINFEKL", "SIINFEKL", True, TypeError),
])
def test_peptide_occurrence_flanks_refuses_ambiguous_or_invalid_geometry(
        source, peptide, offset, error):
    with pytest.raises(error):
        peptide_occurrence_flanks(source, peptide, offset)


class _MockEffect:
    """Hashable mock effect with the fields the function accesses."""
    def __init__(self, protein_seq, start, end):
        self.mutant_protein_sequence = protein_seq
        self.aa_mutation_start_offset = start
        self.aa_mutation_end_offset = end


def _mock_effect(protein_seq, start, end):
    return _MockEffect(protein_seq, start, end)


def test_single_substitution_middle():
    protein = "A" * 100
    effect = _mock_effect(protein, 50, 51)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert offsets[effect] == 41  # max(0, 50-9)
    assert subseqs[effect] == protein[41:60]  # min(100, 51+9) = 60


def test_mutation_near_start():
    protein = "M" * 50
    effect = _mock_effect(protein, 2, 3)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert offsets[effect] == 0  # max(0, 2-9) clamped
    assert subseqs[effect] == protein[0:12]  # min(50, 3+9) = 12


def test_mutation_near_end():
    protein = "G" * 20
    effect = _mock_effect(protein, 18, 19)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert offsets[effect] == 9  # max(0, 18-9)
    assert subseqs[effect] == protein[9:20]  # min(20, 19+9) = 20, clamped


def test_stop_codon_trimming():
    protein = "AAAAAAAAAAMAAAA*AAAA"  # stop at position 15
    effect = _mock_effect(protein, 10, 11)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    # End is min(first_stop=15, 11+9=20) = 15
    assert subseqs[effect] == protein[1:15]
    assert "*" not in subseqs[effect]


def test_silent_effect_skipped():
    """Effects with mutant_protein_sequence=None are skipped."""
    effect = _mock_effect(None, 10, 11)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert len(subseqs) == 0
    assert len(offsets) == 0


def test_multiple_effects():
    protein_a = "A" * 50
    protein_b = "B" * 50
    e1 = _mock_effect(protein_a, 25, 26)
    e2 = _mock_effect(protein_b, 10, 11)
    subseqs, offsets = protein_subsequences_around_mutations([e1, e2], padding_around_mutation=9)
    assert len(subseqs) == 2
    assert e1 in subseqs
    assert e2 in subseqs


def test_deletion_zero_width():
    """Deletion: start_offset == end_offset."""
    protein = "A" * 40
    effect = _mock_effect(protein, 20, 20)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert offsets[effect] == 11  # max(0, 20-9)
    assert subseqs[effect] == protein[11:29]  # min(40, 20+9)


def test_empty_string_protein_skipped():
    """Empty string protein sequence is falsy, should be skipped."""
    effect = _mock_effect("", 0, 0)
    subseqs, offsets = protein_subsequences_around_mutations([effect], padding_around_mutation=9)
    assert len(subseqs) == 0


# ---------------------------------------------------------------------------
# Unscorable residues (#573)
#
# A few CTA proteins carry X where Ensembl does not know the residue. Every
# predictor refuses a window covering one, which used to end the whole run:
# `topiary --cta` produced nothing at all, under a message that printed a
# pandas array repr.
# ---------------------------------------------------------------------------


def test_scorable_peptide_accepts_only_the_standard_residues():
    assert scorable_peptide("SIINFEKL")
    assert not scorable_peptide("XIINFEKL")
    assert not scorable_peptide("SIINFEKU")
    assert not scorable_peptide("siinfekl")
    assert not scorable_peptide("")


def test_scannable_segments_pass_whole_proteins_through():
    sequences = {"a": "MASIINFEKLGGG", "b": "QRSTVWYACDEFGH"}

    assert scannable_segments(sequences) == {
        "a": ("a", 0, "MASIINFEKLGGG"),
        "b": ("b", 0, "QRSTVWYACDEFGH"),
    }


def test_scannable_segments_split_at_unscorable_residues():
    segments = scannable_segments({"p": "XMTHCGHLGXAAKL", "empty": ""})

    assert segments == {
        "p@1": ("p", 1, "MTHCGHLG"),
        "p@10": ("p", 10, "AAKL"),
        "empty": ("empty", 0, ""),
    }
    # Every segment is located where the source protein really holds it.
    for source, offset, segment in segments.values():
        assert {"p": "XMTHCGHLGXAAKL", "empty": ""}[source][
            offset:offset + len(segment)
        ] == segment


def test_scannable_segments_do_not_collide_with_a_real_name():
    segments = scannable_segments({"p": "XSIINFEKL", "p@1": "SIINFEKLA"})

    assert segments["p@1"] == ("p@1", 0, "SIINFEKLA")
    assert segments["p@1#2"] == ("p", 1, "SIINFEKL")


def test_scannable_segments_warn_once_about_what_was_dropped(caplog):
    with caplog.at_level(logging.WARNING):
        scannable_segments({"a": "SIINFEKL", "b": "XSIINFEKL", "c": "SIXNFEKL"})

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    # Count and example, in plain text: the old message ended the run and
    # printed <ArrowStringArray> rows instead.
    assert "2 of 3" in messages[0]
    assert "2 residues in all" in messages[0]
    assert "b" in messages[0] and "ArrowStringArray" not in messages[0]


def test_protein_scan_skips_only_the_windows_covering_an_unknown_residue():
    """A protein with an X still contributes every scorable window (#573)."""
    from mhctools import RandomBindingPredictor

    from topiary import TopiaryPredictor

    sequences = {"clean": "MASIINFEKLGG", "unknown": "XMTHCGHLGAAAKL"}
    predictor = TopiaryPredictor(models=RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
    ))

    df = predictor.predict_from_named_sequences(sequences)

    assert set(df.source_sequence_name) == {"clean", "unknown"}
    # Coordinates are the caller's, not the scanned segment's.
    for row in df.to_dict("records"):
        source = sequences[row["source_sequence_name"]]
        offset = row["peptide_offset"]
        assert source[offset:offset + len(row["peptide"])] == row["peptide"]
    scored = set(df.loc[df.source_sequence_name == "unknown", "peptide"])
    assert scored == {
        sequences["unknown"][offset:offset + 9] for offset in range(1, 6)
    }
    assert not any("X" in peptide for peptide in df.peptide)
