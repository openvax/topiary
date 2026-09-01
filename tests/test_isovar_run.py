"""Running isovar for real, interchangeably with reference translation.

The point of `fragments_from_variants`: give it an alignment file and the
protein context around each mutation is **assembled from RNA reads**; leave
it out and the same variants are **translated from the reference**. Either
way the result is `ProteinFragment`s with the same core, so a pipeline does
not change shape when the RNA does or does not exist.

The assembled sequence is deliberately longer than one peptide — a fragment
is scanned by a sliding window downstream, so it has to contain every
peptide that could cover the mutation.
"""

import warnings

import pytest

from topiary import (
    DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    ProteinFragment,
    fragments_from_variants,
)
from topiary.io_isovar import _reference_fragments


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILEDAAWQ"
    mutation_start_idx = 10
    mutation_end_idx = 12
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27


class _Result:
    def __init__(self, supported=True, variant="chr7 g.140453136 A>T"):
        self.top_protein_sequence = _ProteinSequence() if supported else None
        self.variant = variant
        self.num_total_fragments = 61
        self.num_alt_fragments = 30
        self.num_ref_fragments = 31


class _FakeIsovar:
    """Stands in for the isovar module, recording how it was called."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def run_isovar(self, **kwargs):
        self.calls.append(kwargs)
        return self._results


@pytest.fixture
def fake_isovar(monkeypatch):
    module = _FakeIsovar([_Result()])
    monkeypatch.setattr("topiary.io_isovar._check_isovar", lambda: module)
    return module


# ---------------------------------------------------------------------------
# The RNA arm
# ---------------------------------------------------------------------------


def test_an_alignment_file_assembles_from_rna(fake_isovar):
    fragments = fragments_from_variants(["v"], alignment_file=object())

    assert len(fragments) == 1
    assert fragments[0].annotations["sequence_source"] == "isovar_assembly"


def test_the_assembled_context_is_longer_than_a_peptide(fake_isovar):
    """A sliding window has to fit; that is why this is a sequence length."""
    fragments = fragments_from_variants(["v"], alignment_file=object())

    assert len(fragments[0].sequence) > 9


def test_the_requested_context_length_reaches_isovar(fake_isovar):
    fragments_from_variants(
        ["v"], alignment_file=object(), protein_sequence_length=31,
    )

    creator = fake_isovar.calls[0]["protein_sequence_creator"]
    assert creator.protein_sequence_length == 31


def test_the_default_context_length_is_used_when_unspecified(fake_isovar):
    fragments_from_variants(["v"], alignment_file=object())

    creator = fake_isovar.calls[0]["protein_sequence_creator"]
    assert creator.protein_sequence_length == DEFAULT_PROTEIN_SEQUENCE_LENGTH


def test_isovar_knobs_are_passed_through(fake_isovar):
    fragments_from_variants(
        ["v"], alignment_file=object(),
        transcript_id_whitelist={"ENST1"},
        filter_thresholds={"min_alt_rna_reads": 3},
        min_shared_fragments_for_phasing=4,
    )

    call = fake_isovar.calls[0]
    assert call["transcript_id_whitelist"] == {"ENST1"}
    assert call["filter_thresholds"] == {"min_alt_rna_reads": 3}
    assert call["min_shared_fragments_for_phasing"] == 4


def test_a_caller_supplied_creator_wins(fake_isovar):
    """A caller who has configured isovar keeps their configuration."""
    from isovar.protein_sequence_creator import ProteinSequenceCreator

    mine = ProteinSequenceCreator(protein_sequence_length=45)
    fragments_from_variants(
        ["v"], alignment_file=object(), protein_sequence_creator=mine,
    )

    assert fake_isovar.calls[0]["protein_sequence_creator"] is mine


def test_the_read_counts_come_through_as_measured(fake_isovar):
    fragment = fragments_from_variants(["v"], alignment_file=object())[0]

    assert fragment.n_alt_reads == 30
    assert not fragment.is_approximate("n_alt_reads")


# ---------------------------------------------------------------------------
# Variants isovar cannot support
# ---------------------------------------------------------------------------


def test_an_unsupported_variant_is_dropped_by_default(monkeypatch):
    module = _FakeIsovar([_Result(supported=False)])
    monkeypatch.setattr("topiary.io_isovar._check_isovar", lambda: module)

    assert fragments_from_variants(["v"], alignment_file=object()) == []


def test_the_fallback_translates_it_instead(monkeypatch):
    module = _FakeIsovar([_Result(supported=False, variant="v")])
    monkeypatch.setattr("topiary.io_isovar._check_isovar", lambda: module)
    translated = [ProteinFragment(fragment_id="f", sequence="MKTV")]
    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: translated,
    )

    fragments = fragments_from_variants(
        ["v"], alignment_file=object(), allow_reference_fallback=True,
    )

    assert fragments == translated


def test_supported_and_fallback_fragments_are_distinguishable(monkeypatch):
    """An RNA-backed candidate and an inferred one must not blend."""
    module = _FakeIsovar([_Result(), _Result(supported=False, variant="w")])
    monkeypatch.setattr("topiary.io_isovar._check_isovar", lambda: module)
    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: [ProteinFragment(
            fragment_id="ref", sequence="MKTV",
            annotations={"sequence_source": "varcode_translation"},
        )],
    )

    fragments = fragments_from_variants(
        ["v", "w"], alignment_file=object(), allow_reference_fallback=True,
    )

    sources = {f.annotations.get("sequence_source") for f in fragments}
    assert sources == {"isovar_assembly", "varcode_translation"}


# ---------------------------------------------------------------------------
# The reference arm, and interchangeability
# ---------------------------------------------------------------------------


def test_no_alignment_file_means_no_isovar(monkeypatch):
    """A caller without RNA must never reach the optional dependency."""
    def explode():
        raise AssertionError("isovar must not be needed without a BAM")

    monkeypatch.setattr("topiary.io_isovar._check_isovar", explode)
    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: [],
    )

    assert fragments_from_variants(["v"]) == []


def test_the_padding_defaults_to_half_the_context(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: seen.setdefault("padding", padding) and [],
    )

    fragments_from_variants(["v"], protein_sequence_length=21)

    assert seen["padding"] == 10


def test_both_arms_return_the_same_type(fake_isovar, monkeypatch):
    """Interchangeable means the caller's next line does not change."""
    rna = fragments_from_variants(["v"], alignment_file=object())

    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: [ProteinFragment(
            fragment_id="ref", sequence="MKTVRQERLKSIVRILE",
        )],
    )
    reference = fragments_from_variants(["v"])

    assert all(isinstance(f, ProteinFragment) for f in rna + reference)
    for fragment in rna + reference:
        assert fragment.sequence
        assert fragment.is_known("sequence") if hasattr(
            fragment, "is_known"
        ) else True


def test_a_consumer_reads_both_arms_the_same_way(fake_isovar, monkeypatch):
    def support(fragment):
        if not fragment.is_usable_as_biology("n_alt_reads"):
            return None
        return fragment.n_alt_reads

    rna = fragments_from_variants(["v"], alignment_file=object())[0]
    monkeypatch.setattr(
        "topiary.io_isovar._reference_fragments",
        lambda variants, padding: [ProteinFragment(
            fragment_id="ref", sequence="MKTV",
        )],
    )
    reference = fragments_from_variants(["v"])[0]

    assert support(rna) == 30
    assert support(reference) is None      # no RNA, and it says so
