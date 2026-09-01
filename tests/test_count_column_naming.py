"""A count column is named for what it counts.

Follow-on from the read-subject work, found by checking whether the subject
reached the place thresholds are actually written — the DSL — rather than
only the fragment and the reader frame.

Two things were wrong:

1. Read counts did not survive fragment -> prediction frame at all. The
   frame carried `read_count_method` and `read_count_subject`, which arrive
   as annotations, describing a count that was not there — so it said how a
   number was obtained and what it counted while omitting the number.

2. Once they did survive, `n_alt_reads > 5` on an isovar-derived frame was
   answered by a *fragment* count. Both are integers and both are
   plausible, so nothing failed.

Naming the column for its subject is what makes the wrong reference fail:
the DSL raises "Column not found" and suggests the right one.
"""

import warnings

import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    FRAGMENTS,
    READS,
    TopiaryPredictor,
    evaluate_scores,
    fragment_from_isovar_result,
    read_lens,
    read_pvacseq,
)
from topiary.ranking import parse
from topiary.rna_evidence import count_column_for_subject


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILEDAAWQ"
    mutation_start_idx = 10
    mutation_end_idx = 12
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27


class _IsovarResult:
    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_alt_fragments = 8
    num_ref_fragments = 31


@pytest.fixture(scope="module")
def isovar_frame():
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return predictor.predict_from_fragments(
            [fragment_from_isovar_result(_IsovarResult())]
        )


# ---------------------------------------------------------------------------
# The counts reach the frame at all
# ---------------------------------------------------------------------------


def test_read_counts_survive_onto_the_prediction_frame(isovar_frame):
    """They were dropped, leaving method and subject columns describing a
    count that was not there."""
    counts = [c for c in isovar_frame.columns if c.startswith("n_alt")]

    assert counts
    assert isovar_frame[counts[0]].notna().any()


def test_the_subject_and_the_count_arrive_together(isovar_frame):
    assert "read_count_subject" in isovar_frame.columns
    assert "n_alt_fragments" in isovar_frame.columns
    assert set(isovar_frame["read_count_subject"].dropna()) == {FRAGMENTS}


# ---------------------------------------------------------------------------
# The name says the subject
# ---------------------------------------------------------------------------


def test_a_fragment_frame_uses_fragment_names(isovar_frame):
    assert "n_alt_fragments" in isovar_frame.columns
    assert "n_alt_reads" not in isovar_frame.columns


@pytest.mark.parametrize("reader,path", [
    (read_lens, "tests/data/lens/sample_v1_4.tsv"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_all_epitopes.tsv"),
], ids=["lens", "pvacseq"])
def test_a_read_frame_keeps_the_read_names(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = reader(path).df

    assert "n_alt_reads" in df.columns
    assert "n_alt_fragments" not in df.columns


def test_a_threshold_for_reads_is_refused_on_a_fragment_frame(isovar_frame):
    """The harm, as an assertion: 8 fragments would clear a bar written
    for 5 reads, and both are plausible integers."""
    with pytest.raises(ValueError, match="not found"):
        evaluate_scores(isovar_frame, parse("n_alt_reads > 5"))


def test_the_error_suggests_the_right_column(isovar_frame):
    with pytest.raises(ValueError) as excinfo:
        evaluate_scores(isovar_frame, parse("n_alt_reads > 5"))

    assert "n_alt_fragments" in str(excinfo.value)


def test_the_subject_qualified_threshold_works(isovar_frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(isovar_frame, parse("n_alt_fragments > 5"))

    assert scores.sum() == len(scores)


# ---------------------------------------------------------------------------
# The naming rule itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,fragment_name", [
    ("n_alt_reads", "n_alt_fragments"),
    ("n_ref_reads", "n_ref_fragments"),
    ("n_overlapping_reads", "n_overlapping_fragments"),
    ("n_alt_reads_supporting_protein_sequence",
     "n_alt_fragments_supporting_protein_sequence"),
])
def test_every_count_field_has_a_fragment_spelling(field, fragment_name):
    assert count_column_for_subject(field, FRAGMENTS) == fragment_name
    assert count_column_for_subject(field, READS) == field


@pytest.mark.parametrize("subject", [None, "", "nan"])
def test_an_unstated_subject_keeps_the_read_spelling(subject):
    """That is what a source which does not say counts."""
    assert count_column_for_subject("n_alt_reads", subject) == "n_alt_reads"


def test_an_unknown_field_is_passed_through():
    assert count_column_for_subject("something_else", FRAGMENTS) == (
        "something_else"
    )
