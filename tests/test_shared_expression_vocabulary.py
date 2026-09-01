"""Both readers name expression the same way (topiary #238).

A consumer writing one filter should not have to know which reader produced
the frame. Filed with three problems; two of them were fixed by 5.37.0
before the issue was read, and this closes the third.

Fixed in 5.37.0, asserted here so they stay fixed:
  - both frames carry `variant_allele_expression` (there was never an
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
    assert "variant_allele_expression" in frames[reader].columns


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
def test_neither_reader_uses_the_other_spelling(frames, reader):
    """`allele_expression` and `variant_allele_expression` would be the
    same quantity under two names."""
    assert "allele_expression" not in frames[reader].columns


@pytest.mark.parametrize("reader", ["lens", "pvacseq"])
@pytest.mark.parametrize("column", [
    "variant_allele_expression_method",
    "read_count_method",
    "supporting_read_count_method",
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
