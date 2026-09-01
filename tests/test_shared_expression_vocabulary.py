"""Both readers name expression the same way (topiary #238).

A consumer writing one filter should not have to know which reader produced
the frame. Filed with three problems; two of them were fixed by 5.37.0
before the issue was read, and this closes the third.

Fixed in 5.37.0, asserted here so they stay fixed:
  - both frames carry `rna_alt_expression` (there was never an
    `allele_expression`, so no two names for one quantity)
  - both label the derivation, on the expression *and* the read axes

Still open, and fixed here:
  - gene-level abundance was `gene_tpm` on LENS and `gene_expression` on
    pVACseq, so a filter naming either matched nothing on the other frame
    rather than failing.
"""

import warnings

import pytest

from topiary import evaluate_scores, read_lens, read_pvacseq
from topiary.ranking import parse

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


def _frames():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return {
            "lens": read_lens(LENS).to_long().df,
            "pvacseq": read_pvacseq(PVACSEQ).df,
        }


@pytest.fixture(scope="module")
def frames():
    return _frames()


# ---------------------------------------------------------------------------
# One name for gene-level abundance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
def test_gene_expression_exists_on_both(frames, reader):
    assert "gene_expression" in frames[reader].columns


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
def test_one_filter_spans_both_readers(frames, reader):
    """The property the issue is about: the consumer does not branch."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(frames[reader], parse("gene_expression > 1"))

    assert len(scores) == len(frames[reader])
    assert scores.notna().any()


def test_the_lens_native_spelling_still_works(frames):
    """gene_tpm is what LENS calls it; existing expressions keep working."""
    lens = frames["lens"]

    assert "gene_tpm" in lens.columns
    assert "gene_tpm_raw" in lens.columns
    assert lens["gene_expression"].equals(lens["gene_tpm"])


def test_the_raw_string_is_still_kept_separately(frames):
    """LENS writes fusion rows as composite strings, so the numeric column
    is NaN for them and the original is preserved — that is why `tpm` was
    never a blind rename."""
    lens = frames["lens"]

    assert lens["gene_tpm_raw"].notna().any()


# ---------------------------------------------------------------------------
# Already fixed in 5.37.0 — pinned so they stay fixed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
def test_both_readers_name_the_variant_allele_estimate_the_same(frames, reader):
    assert "rna_alt_expression" in frames[reader].columns


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
def test_neither_reader_uses_the_other_spelling(frames, reader):
    """`allele_expression` and `rna_alt_expression` would be the
    same quantity under two names."""
    assert "allele_expression" not in frames[reader].columns


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
@pytest.mark.parametrize("column", [
    "rna_alt_expression_method",
    "rna_evidence_method",
])
def test_both_readers_label_every_derivation(frames, reader, column):
    """The reason for labelling one applies identically to the other: the
    estimate assumes both alleles are transcribed equally, so a variant on
    a silenced allele looks expressed. A bare number cannot say that."""
    assert column in frames[reader].columns


def test_a_consumer_can_ask_how_a_number_was_obtained_on_either_frame(frames):
    from topiary import describe_read_evidence

    for reader in ("lens", "pvacseq"):
        described = describe_read_evidence(frames[reader])
        assert all(isinstance(v, str) for v in described.values())


# ---------------------------------------------------------------------------
# One reader, one vocabulary — whichever flavour of its format it was given
# ---------------------------------------------------------------------------
#
# The sharper bug behind #238, and the one I missed: read_pvacseq took two
# branches. Its aggregated report supplies pVACseq's own `Allele Expr` and
# `RNA Expr`, which were passed through under names the all_epitopes path
# never emits — and that path never ran attach_read_evidence at all, so it
# had no method columns.
#
# Both of us mis-verified this by checking one branch: I read the
# all_epitopes fixture and concluded about the reader; the consumer grepped
# for "express" against headers abbreviated "Expr" and concluded the
# opposite. A function with two branches needs both exercised.

AGGREGATED = "tests/data/pvacseq/mhc_i_aggregated.tsv"
ALL_EPITOPES = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


def _pvacseq(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_pvacseq(path).df


@pytest.mark.parametrize("path", [AGGREGATED, ALL_EPITOPES],
                         ids=["aggregated", "all-epitopes"])
def test_both_pvacseq_flavours_name_expression_the_same(path):
    df = _pvacseq(path)

    assert "rna_alt_expression" in df.columns
    assert "transcript_expression" in df.columns


@pytest.mark.parametrize("path", [AGGREGATED, ALL_EPITOPES],
                         ids=["aggregated", "all-epitopes"])
def test_neither_flavour_uses_the_old_spellings(path):
    """`allele_expression` and `rna_transcript_expression` existed on one
    branch only, so a filter naming them worked on one pVACseq file and
    silently matched nothing on the other."""
    df = _pvacseq(path)

    assert "allele_expression" not in df.columns
    assert "rna_transcript_expression" not in df.columns


@pytest.mark.parametrize("path", [AGGREGATED, ALL_EPITOPES],
                         ids=["aggregated", "all-epitopes"])
def test_both_flavours_label_their_derivations(path):
    df = _pvacseq(path)

    for column in ("rna_evidence_method",
                   "rna_alt_expression_method"):
        assert column in df.columns


def test_a_source_supplied_estimate_says_so():
    """pVACseq computed `Allele Expr` itself. Keeping it and calling it
    ours would claim a derivation nobody can check; recomputing it would
    discard the number the source stands behind."""
    df = _pvacseq(AGGREGATED)

    methods = set(df["rna_alt_expression_method"].dropna())
    assert methods == {"source_reported"}


def test_a_derived_estimate_says_that_instead():
    df = _pvacseq(ALL_EPITOPES)

    methods = set(df["rna_alt_expression_method"].dropna())
    assert methods == {"tpm_x_dna_vaf"}


def test_source_reported_is_not_measured():
    """The source stands behind it but did not say how it got there."""
    from topiary import provenance_for_method

    assert provenance_for_method("source_reported") == "approximated"


@pytest.mark.parametrize("path", [AGGREGATED, ALL_EPITOPES],
                         ids=["aggregated", "all-epitopes"])
def test_one_filter_spans_both_pvacseq_flavours(path):
    df = _pvacseq(path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = evaluate_scores(df, parse("rna_alt_expression > 0"))

    assert len(scores) == len(df)
