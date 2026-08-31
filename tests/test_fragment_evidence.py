"""Read-level evidence and per-field knownness on ProteinFragment (topiary #102).

Consumer requirements from vaxrank, for the multi-source fragment abstraction.
Two of the three are here; isovar integration is deliberately not (see the PR).

The property under test throughout is the one that makes a multi-source
abstraction usable at all: **every source produces the same shape, differing
only in which fields are populated** — so a consumer never branches on
`source_type`, and can always tell a field that is real from one that merely
has a value.
"""

import pathlib

import pytest

from topiary import (
    APPROXIMATED,
    MEASURED,
    PROVENANCE_VALUES,
    SYNTHESIZED,
    ProteinFragment,
    read_fragments,
    write_fragments,
)


def _fragment(**kwargs):
    kwargs.setdefault("fragment_id", "f1")
    kwargs.setdefault("sequence", "SIINFEKLA")
    return ProteinFragment(**kwargs)


# ---------------------------------------------------------------------------
# Read-level evidence is not derivable from aggregate expression
# ---------------------------------------------------------------------------


def test_read_counts_are_carried():
    f = _fragment(
        n_overlapping_reads=40, n_alt_reads=12, n_ref_reads=28,
        n_alt_reads_supporting_protein_sequence=9,
    )

    assert f.n_overlapping_reads == 40
    assert f.n_alt_reads == 12
    assert f.n_ref_reads == 28
    assert f.n_alt_reads_supporting_protein_sequence == 9


def test_reads_supporting_the_protein_sequence_are_separate_from_alt_reads():
    """Reads supporting *this assembled sequence*, not merely the allele."""
    f = _fragment(n_alt_reads=12, n_alt_reads_supporting_protein_sequence=9)

    assert f.n_alt_reads != f.n_alt_reads_supporting_protein_sequence


def test_read_counts_default_to_unknown():
    """A source with no read data says nothing rather than saying zero."""
    f = _fragment()

    assert f.n_alt_reads is None
    assert not f.is_known("n_alt_reads")


# ---------------------------------------------------------------------------
# Absent is not zero — the requirement that matters most
# ---------------------------------------------------------------------------


def test_zero_and_unknown_are_different():
    """"No RNA support" and "this source cannot answer" are not one claim."""
    looked = _fragment(n_alt_reads=0)
    cannot_answer = _fragment(n_alt_reads=None)

    assert looked.n_alt_reads == 0
    assert looked.is_known("n_alt_reads")
    assert cannot_answer.n_alt_reads is None
    assert not cannot_answer.is_known("n_alt_reads")


def test_zero_survives_a_file_round_trip_as_zero(tmp_path):
    """The distinction has to survive serialization, or it is decorative."""
    path = tmp_path / "frags.tsv"
    write_fragments(
        [_fragment(fragment_id="zero", n_alt_reads=0),
         _fragment(fragment_id="unknown", n_alt_reads=None)],
        path,
    )

    back = {f.fragment_id: f for f in read_fragments(path)}

    assert back["zero"].n_alt_reads == 0
    assert back["unknown"].n_alt_reads is None


def test_is_known_rejects_a_field_that_does_not_exist():
    with pytest.raises(ValueError, match="not a ProteinFragment field"):
        _fragment().is_known("n_alt_readz")


# ---------------------------------------------------------------------------
# Per-field provenance: populated is not the same as trustworthy
# ---------------------------------------------------------------------------


def test_an_unqualified_field_has_no_provenance():
    f = _fragment(n_alt_reads=12)

    assert f.provenance_of("n_alt_reads") is None
    assert f.is_usable_as_biology("n_alt_reads")


def test_an_approximated_count_is_marked():
    """LENS and pVACseq estimate read counts differently; both estimate."""
    f = _fragment(n_alt_reads=12, field_provenance={"n_alt_reads": APPROXIMATED})

    assert f.is_approximate("n_alt_reads")
    assert f.is_usable_as_biology("n_alt_reads")   # an estimate is still data


def test_a_synthesized_field_must_not_be_read_as_biology():
    """vaxrank's placeholder_alleles, generalized: refuse rather than compute."""
    f = _fragment(variant="chr1:100:N>N",
                  field_provenance={"variant": SYNTHESIZED})

    assert f.variant is not None            # it has a value
    assert f.is_known("variant")            # ... and the value is present
    assert not f.is_usable_as_biology("variant")   # ... but means nothing


def test_a_measured_field_is_usable():
    f = _fragment(n_alt_reads=12, field_provenance={"n_alt_reads": MEASURED})

    assert f.is_usable_as_biology("n_alt_reads")
    assert not f.is_approximate("n_alt_reads")


def test_an_absent_field_is_not_usable_as_biology():
    assert not _fragment().is_usable_as_biology("n_alt_reads")


def test_provenance_survives_a_file_round_trip(tmp_path):
    path = tmp_path / "frags.tsv"
    write_fragments(
        [_fragment(variant="chr1:100:N>N", n_alt_reads=12,
                   field_provenance={"variant": SYNTHESIZED,
                                     "n_alt_reads": APPROXIMATED})],
        path,
    )

    back = list(read_fragments(path))[0]

    assert not back.is_usable_as_biology("variant")
    assert back.is_approximate("n_alt_reads")


# ---------------------------------------------------------------------------
# A provenance claim that cannot mean anything is refused
# ---------------------------------------------------------------------------


def test_an_unknown_field_name_is_refused():
    """A typo would sit inert and stop protecting the field it names."""
    with pytest.raises(ValueError, match="not a ProteinFragment field"):
        _fragment(field_provenance={"n_alt_readz": MEASURED})


def test_an_unknown_provenance_value_is_refused():
    with pytest.raises(ValueError, match="use one of"):
        _fragment(field_provenance={"n_alt_reads": "probably fine"})


def test_the_vocabulary_is_exactly_three_values():
    assert PROVENANCE_VALUES == {MEASURED, APPROXIMATED, SYNTHESIZED}


# ---------------------------------------------------------------------------
# Shape conformance across sources
# ---------------------------------------------------------------------------


def test_from_dict_accepts_every_field_the_dataclass_has():
    """It hardcoded its field list, so new fields were rejected on load."""
    full = _fragment(
        n_overlapping_reads=40, n_alt_reads=12, n_ref_reads=28,
        n_alt_reads_supporting_protein_sequence=9,
        field_provenance={"n_alt_reads": APPROXIMATED},
    )

    assert ProteinFragment.from_dict(full.to_dict()) == full
    assert ProteinFragment.from_dict(full.to_dict()).n_alt_reads == 12


def test_the_degenerate_fragment_flows_through_unchanged(tmp_path):
    """pVACseq's shape: no source context at all.

    vaxrank proposed this as the conformance test for the abstraction — a
    peptide with no surrounding sequence, an interval spanning all of it,
    no read data and no reference. If that cannot round-trip, the shape is
    wrong.
    """
    degenerate = ProteinFragment(
        fragment_id="pvacseq__deadbeef",
        source_type="variant:snv",
        sequence="SIINFEKLA",
        target_intervals=[(0, 9)],
    )

    path = tmp_path / "degenerate.tsv"
    write_fragments([degenerate], path)
    back = list(read_fragments(path))[0]

    assert back.sequence == degenerate.sequence
    assert back.target_intervals == [(0, 9)]
    assert back.reference_sequence is None
    # Every evidence field says "unknown", and says it the same way a
    # richly populated fragment would.
    for name in ("n_alt_reads", "n_ref_reads", "n_overlapping_reads",
                 "n_alt_reads_supporting_protein_sequence",
                 "gene_expression", "transcript_expression"):
        assert not back.is_known(name)
        assert not back.is_usable_as_biology(name)


def test_a_consumer_need_not_branch_on_source_type():
    """The whole point: one code path reads every source."""
    isovar_like = _fragment(fragment_id="a", n_alt_reads=12)
    lens_like = _fragment(
        fragment_id="b", n_alt_reads=8,
        field_provenance={"n_alt_reads": APPROXIMATED},
    )
    varcode_like = _fragment(fragment_id="c")

    def support(fragment):
        if not fragment.is_usable_as_biology("n_alt_reads"):
            return None
        return fragment.n_alt_reads

    assert [support(f) for f in (isovar_like, lens_like, varcode_like)] == [
        12, 8, None,
    ]
