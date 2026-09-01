"""A read count names what it counts (topiary, read-subject axis).

isovar counts **fragments** — `num_alt_fragments`, not `num_alt_reads`. A
depth x VAF estimate is inherently about **reads**, because depth is a read
depth. Both landed in `n_alt_reads`, so the field was honest about *how* a
number was obtained and silent about *what it counts*: the same shape as the
CDS-overlap column, a real count of an adjacent thing.

Within one run the unit is internally consistent, so a ranking does not
change. The harm is in what travels — a documented `n_alt_reads > 5`, a
config copied between projects, a number in a paper. Five fragments and five
reads are different bars and nothing said which was cleared.

Perfect cross-path comparability is not attainable and is not the goal:
converting a read estimate to fragments needs library information no source
carries. The goal is that every path names its subject, and a source asked
for the other one says so rather than substituting.
"""

import warnings

import pytest

from topiary import (
    FRAGMENTS,
    READ_SUBJECTS,
    READS,
    fragment_from_isovar_result,
    fragments_from_dataframe,
    read_lens,
    read_pvacseq,
    subject_for_method,
)
from topiary.rna_evidence import (
    CDS_OVERLAP_READS,
    RNA_DEPTH_X_VAF,
    RNA_READS,
)


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILE"
    mutation_start_idx = 4
    mutation_end_idx = 6
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27


class _IsovarResult:
    top_protein_sequence = _ProteinSequence()
    variant = "chr7 g.140453136 A>T"
    num_total_fragments = 61
    num_alt_fragments = 30
    num_ref_fragments = 31


def _frame_fragment(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fragments_from_dataframe(reader(path).df)[0]


@pytest.fixture(scope="module")
def isovar_fragment():
    return fragment_from_isovar_result(_IsovarResult())


@pytest.fixture(scope="module")
def pvacseq_fragment():
    return _frame_fragment(
        read_pvacseq, "tests/data/pvacseq/mhc_i_all_epitopes.tsv",
    )


# ---------------------------------------------------------------------------
# Every path names its subject
# ---------------------------------------------------------------------------


def test_isovar_counts_fragments(isovar_fragment):
    assert isovar_fragment.read_count_subject() == FRAGMENTS


def test_a_depth_estimate_counts_reads(pvacseq_fragment):
    assert pvacseq_fragment.read_count_subject() == READS


@pytest.mark.parametrize("reader,path", [
    (read_lens, "tests/data/lens/sample_v1_4.tsv"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_all_epitopes.tsv"),
    (read_pvacseq, "tests/data/pvacseq/mhc_i_aggregated.tsv"),
], ids=["lens", "pvacseq-all-epitopes", "pvacseq-aggregated"])
def test_every_reader_states_a_subject(reader, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = reader(path).df

    assert "read_count_subject" in df.columns


def test_a_derivation_that_fixes_the_subject_says_so():
    assert subject_for_method(RNA_DEPTH_X_VAF) == READS
    assert subject_for_method(CDS_OVERLAP_READS) == READS


def test_a_derivation_that_does_not_fix_the_subject_says_nothing():
    """`rna_reads` says a count came from an alignment, not whether the
    aligner counted reads or fragments — so the producer states it."""
    assert subject_for_method(RNA_READS) is None


def test_an_unstated_method_has_no_subject():
    assert subject_for_method(None) is None
    assert subject_for_method("nan") is None


# ---------------------------------------------------------------------------
# Asked for the other subject, a source says so
# ---------------------------------------------------------------------------


def test_a_fragment_count_is_not_offered_as_a_read_count(isovar_fragment):
    """The substitution the vocabulary exists to prevent."""
    assert isovar_fragment.count_in("n_alt_reads", FRAGMENTS) == 30
    assert isovar_fragment.count_in("n_alt_reads", READS) is None


def test_a_read_estimate_is_not_offered_as_a_fragment_count(pvacseq_fragment):
    assert pvacseq_fragment.count_in("n_alt_reads", READS) > 0
    assert pvacseq_fragment.count_in("n_alt_reads", FRAGMENTS) is None


def test_an_absent_count_is_none_for_either_subject():
    from topiary import ProteinFragment

    fragment = ProteinFragment(fragment_id="f", sequence="SIINFEKLA")

    assert fragment.count_in("n_alt_reads", READS) is None
    assert fragment.count_in("n_alt_reads", FRAGMENTS) is None


def test_an_unknown_subject_is_refused(isovar_fragment):
    with pytest.raises(ValueError, match="must be one of"):
        isovar_fragment.count_in("n_alt_reads", "molecules")


def test_the_vocabulary_is_exactly_two_subjects():
    assert READ_SUBJECTS == {FRAGMENTS, READS}


# ---------------------------------------------------------------------------
# What this is for
# ---------------------------------------------------------------------------


def test_a_threshold_knows_which_bar_it_cleared(isovar_fragment,
                                                pvacseq_fragment):
    """A documented `n_alt_reads > 5` travelling between projects.

    Both fragments carry a count above 5. Only one of them cleared the
    bar the threshold was written against, and now that is answerable.
    """
    def clears(fragment, minimum, subject):
        count = fragment.count_in("n_alt_reads", subject)
        return None if count is None else count > minimum

    assert clears(isovar_fragment, 5, FRAGMENTS) is True
    assert clears(pvacseq_fragment, 5, FRAGMENTS) is None   # cannot answer
    assert clears(pvacseq_fragment, 5, READS) is True


def test_a_confidence_weight_can_demand_fragments(isovar_fragment,
                                                  pvacseq_fragment):
    """Two mates of one fragment are not independent evidence, so a
    sqrt() confidence weight over reads inherits a paired-end inflation a
    confidence transform should not. A score that wants fragments can now
    ask for them and get nothing rather than something wrong."""
    import math

    def weight(fragment):
        count = fragment.count_in("n_alt_reads", FRAGMENTS)
        return None if count is None else math.sqrt(count)

    assert weight(isovar_fragment) == pytest.approx(math.sqrt(30))
    assert weight(pvacseq_fragment) is None
