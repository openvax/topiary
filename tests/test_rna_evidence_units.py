"""RNA evidence, in whichever unit the source counted (topiary).

A paired-end fragment is one molecule read twice: **one** piece of evidence
and **two** reads. isovar reports both units for every count; sources that
estimate from depth x VAF report only reads.

So both are carried under names that say what they hold, and `n_rna_*` takes
the better one — fragments where a source has them, reads where it does not —
with `rna_evidence_subject` recording which. One threshold spans every
source, and a number that travels can still name its unit.

This replaces an earlier design that stored isovar's *fragment* counts in
fields named `n_alt_reads` and offered `count_in(name, subject)` to ask for a
unit. That was built on a false premise: isovar exposes `num_alt_reads`
alongside `num_alt_fragments`, so "the source cannot give you reads" was
never true. Carrying both is simpler and does not need the accessor.
"""

import warnings

import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    FRAGMENTS,
    READS,
    ProteinFragment,
    TopiaryPredictor,
    evaluate_scores,
    fragment_from_isovar_result,
    read_lens,
    read_pvacseq,
)
from topiary.ranking import parse


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILEDAAWQ"
    mutation_start_idx = 10
    mutation_end_idx = 12
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27
    num_supporting_reads = 52


class _IsovarResult:
    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_total_reads = 118
    num_alt_fragments = 30
    num_alt_reads = 58
    num_ref_fragments = 31
    num_ref_reads = 60


@pytest.fixture(scope="module")
def isovar_fragment():
    return fragment_from_isovar_result(_IsovarResult())


@pytest.fixture(scope="module")
def isovar_frame(isovar_fragment):
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return predictor.predict_from_fragments([isovar_fragment])


def _reader_frame(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return reader(path).df


# ---------------------------------------------------------------------------
# Both units, each under its own name
# ---------------------------------------------------------------------------


def test_isovar_reports_reads_and_fragments(isovar_fragment):
    """It exposes both, so topiary carries both rather than choosing."""
    assert isovar_fragment.n_alt_reads == 58
    assert isovar_fragment.n_alt_fragments == 30
    assert isovar_fragment.n_ref_reads == 60
    assert isovar_fragment.n_ref_fragments == 31


def test_a_field_holds_what_its_name_says(isovar_fragment):
    """The earlier bug: a fragment count in a field named for reads."""
    assert isovar_fragment.n_alt_reads > isovar_fragment.n_alt_fragments


def test_a_depth_source_reports_reads_only():
    fragment = ProteinFragment(
        fragment_id="f", sequence="SIINFEKLA", n_alt_reads=429,
    )

    assert fragment.n_alt_fragments is None
    assert fragment.n_rna_alt == 429


# ---------------------------------------------------------------------------
# n_rna_* prefers fragments, and says so
# ---------------------------------------------------------------------------


def test_fragments_are_preferred_when_present(isovar_fragment):
    """One molecule read twice is one piece of evidence, not two."""
    assert isovar_fragment.n_rna_alt == 30
    assert isovar_fragment.rna_evidence_subject() == FRAGMENTS


def test_reads_are_used_when_that_is_all_there_is():
    fragment = ProteinFragment(
        fragment_id="f", sequence="SIINFEKLA", n_alt_reads=429,
    )

    assert fragment.n_rna_alt == 429
    assert fragment.rna_evidence_subject() == READS


def test_no_evidence_says_nothing_rather_than_zero():
    fragment = ProteinFragment(fragment_id="f", sequence="SIINFEKLA")

    assert fragment.n_rna_alt is None
    assert fragment.rna_evidence_subject() is None


@pytest.mark.parametrize("accessor,fragments,reads", [
    ("n_rna_alt", 30, 58),
    ("n_rna_ref", 31, 60),
    ("n_rna_overlapping", 61, 118),
    ("n_rna_supporting_protein_sequence", 27, 52),
])
def test_every_accessor_prefers_fragments(isovar_fragment, accessor,
                                          fragments, reads):
    assert getattr(isovar_fragment, accessor) == fragments


def test_supporting_is_not_the_same_question_as_alt(isovar_fragment):
    """One counts support for the variant allele, the other for the whole
    assembled sequence — only an assembler can report the second."""
    assert (
        isovar_fragment.n_rna_supporting_protein_sequence
        != isovar_fragment.n_rna_alt
    )


# ---------------------------------------------------------------------------
# One threshold, every source
# ---------------------------------------------------------------------------


def test_the_frame_carries_the_columns_a_threshold_is_written_against(
    isovar_frame,
):
    for column in ("n_rna_alt", "n_rna_ref", "n_rna_overlapping",
                   "rna_evidence_subject"):
        assert column in isovar_frame.columns


@pytest.mark.parametrize("reader,path", [
    (read_lens, "tests/data/lens/sample_v1_4.tsv"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_all_epitopes.tsv"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_aggregated.tsv"),
], ids=["lens", "pvacseq-all-epitopes", "pvacseq-aggregated"])
def test_every_reader_frame_carries_them_too(reader, path):
    df = _reader_frame(reader, path)

    assert "n_rna_alt" in df.columns
    assert "rna_evidence_subject" in df.columns


def test_one_threshold_spans_a_fragment_frame_and_a_read_frame(isovar_frame):
    lens = _reader_frame(read_lens, "tests/data/lens/sample_v1_4.tsv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        on_fragments = evaluate_scores(isovar_frame, parse("n_rna_alt > 5"))
        on_reads = evaluate_scores(lens, parse("n_rna_alt > 5"))

    assert on_fragments.notna().any()
    assert on_reads.notna().any()


def test_each_frame_states_its_own_unit(isovar_frame):
    lens = _reader_frame(read_lens, "tests/data/lens/sample_v1_4.tsv")

    assert set(isovar_frame["rna_evidence_subject"].dropna()) == {FRAGMENTS}
    assert set(lens["rna_evidence_subject"].dropna()) == {READS}


def test_the_underlying_columns_are_still_there(isovar_frame):
    """A caller who needs one unit specifically names it."""
    assert "n_alt_reads" in isovar_frame.columns
    assert "n_alt_fragments" in isovar_frame.columns


# ---------------------------------------------------------------------------
# Where the number came from, alongside what unit it is in
# ---------------------------------------------------------------------------
#
# Two independent questions, and the answers must not imply each other:
# `read_count_method` says where a number came from, `rna_evidence_subject`
# says what it counts. The method was called `rna_reads` while explicitly
# fixing no subject — misleading once the value it labels can be fragments.


def test_an_alignment_derived_count_names_its_source_not_a_unit(
    isovar_fragment,
):
    from topiary import RNA_ALIGNMENT

    assert isovar_fragment.annotations["read_count_method"] == RNA_ALIGNMENT
    assert isovar_fragment.rna_evidence_subject() == FRAGMENTS


def test_the_old_method_name_still_resolves():
    from topiary import RNA_ALIGNMENT, RNA_READS

    assert RNA_READS == RNA_ALIGNMENT


@pytest.mark.parametrize("reader,path,method", [
    (read_pvacseq, "tests/data/pvacseq/mhc_i_all_epitopes.tsv",
     "rna_depth_x_vaf"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_aggregated.tsv",
     "rna_depth_x_vaf"),
], ids=["pvacseq-all-epitopes", "pvacseq-aggregated"])
def test_an_indirect_count_says_how_it_was_computed(reader, path, method):
    """Where a source gives only depth and VAF, the alt count is
    depth x VAF — computed, and labelled as computed."""
    df = _reader_frame(reader, path)

    assert set(df["read_count_method"].dropna()) == {method}


def test_a_computed_count_is_not_called_measured():
    from topiary import provenance_for_method

    assert provenance_for_method("rna_depth_x_vaf") == "approximated"
    assert provenance_for_method("rna_alignment") == "measured"


def test_every_populated_count_can_say_where_it_came_from(isovar_frame):
    """The pair a number needs to travel: its unit and its origin."""
    populated = isovar_frame["n_rna_alt"].notna()

    assert populated.any()
    assert isovar_frame.loc[populated, "rna_evidence_subject"].notna().all()
    assert isovar_frame.loc[populated, "read_count_method"].notna().all()
