"""RNA read-level evidence, and naming each derivation (topiary #102).

The content half of the multi-source fragment work. 5.32.0 added the fields;
nothing populated them. The readers do now, and every derived number carries
the name of its derivation — because "12 reads counted" and "12 reads implied
by depth x VAF" are different claims and a consumer weighting a candidate by
depth of support needs to tell them apart.

`None` throughout means the source could not answer, which is not zero.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from topiary import (
    CDS_OVERLAP_READS,
    RNA_DEPTH_X_VAF,
    TPM_X_DNA_VAF,
    attach_read_evidence,
    attach_sequence_source,
    describe_read_evidence,
    read_lens,
    read_pvacseq,
    split_reads_by_vaf,
)

LENS = "tests/data/lens/sample_v1_4.tsv"
PVACSEQ = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


def _read(fn, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(path)


# ---------------------------------------------------------------------------
# The arithmetic, on its own
# ---------------------------------------------------------------------------


def test_a_depth_and_a_fraction_split_into_alt_and_ref():
    alt, ref = split_reads_by_vaf(pd.Series([1000]), pd.Series([0.25]))

    assert alt.tolist() == [250]
    assert ref.tolist() == [750]


@pytest.mark.parametrize("depth,vaf", [
    (None, 0.25), (1000, None), (None, None),
], ids=["no-depth", "no-vaf", "neither"])
def test_a_missing_half_yields_no_estimate(depth, vaf):
    """An estimate needs both halves; inventing one is how a missing value
    becomes a number nobody measured."""
    alt, ref = split_reads_by_vaf(pd.Series([depth]), pd.Series([vaf]))

    assert alt.isna().all() and ref.isna().all()


@pytest.mark.parametrize("vaf", [-0.1, 1.5, "abc"])
def test_a_value_that_is_not_a_fraction_is_not_a_vaf(vaf):
    alt, _ = split_reads_by_vaf(pd.Series([1000]), pd.Series([vaf]))

    assert alt.isna().all()


def test_zero_support_is_a_real_answer():
    """VAF 0 means the source looked and found none — not "cannot answer"."""
    alt, ref = split_reads_by_vaf(pd.Series([1000]), pd.Series([0.0]))

    assert alt.tolist() == [0]
    assert ref.tolist() == [1000]


# ---------------------------------------------------------------------------
# pVACseq: a depth is counted, the split is arithmetic
# ---------------------------------------------------------------------------


def test_pvacseq_populates_the_read_counts():
    df = _read(read_pvacseq, PVACSEQ).df

    row = df.iloc[0]
    assert row["n_overlapping_reads"] > 0
    assert row["n_alt_reads"] + row["n_ref_reads"] == row["n_overlapping_reads"]


def test_pvacseq_names_the_split_as_derived():
    """Not counted — depth x VAF, and the frame says so."""
    df = _read(read_pvacseq, PVACSEQ).df

    assert set(df["read_count_method"].dropna()) == {RNA_DEPTH_X_VAF}


def test_pvacseq_estimates_variant_allele_expression():
    df = _read(read_pvacseq, PVACSEQ).df

    assert df["variant_allele_expression"].notna().any()
    assert set(df["variant_allele_expression_method"].dropna()) == {
        TPM_X_DNA_VAF,
    }


def test_the_expression_estimate_is_abundance_times_fraction():
    df = _read(read_pvacseq, PVACSEQ).df
    row = df.dropna(subset=["variant_allele_expression"]).iloc[0]

    expected = row["transcript_expression"] * row["tumor_dna_vaf"]
    assert row["variant_allele_expression"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LENS: two real counts, one of something adjacent
# ---------------------------------------------------------------------------


def test_lens_populates_the_counted_columns():
    df = _read(read_lens, LENS).df

    assert df["n_overlapping_reads"].notna().any()
    assert df["n_alt_reads_supporting_protein_sequence"].notna().any()


def test_lens_rows_without_a_vaf_get_no_split_and_no_method():
    """Absent stays absent, and says so — it does not become zero."""
    df = _read(read_lens, LENS).df
    unstated = df[df["vaf"].isna()]

    assert unstated["n_alt_reads"].isna().all()
    assert unstated["read_count_method"].isna().all()


def test_lens_rows_with_a_vaf_do_get_a_split():
    df = _read(read_lens, LENS).df
    stated = df[df["vaf"].notna()]

    assert len(stated) > 0
    assert stated["n_alt_reads"].notna().all()
    assert set(stated["read_count_method"]) == {RNA_DEPTH_X_VAF}


def test_a_cds_overlap_count_is_not_called_a_read_count():
    """It is a genuine count — of reads overlapping the peptide's CDS, not
    of reads supporting the variant. The distinction is the point."""
    frame = attach_read_evidence(
        pd.DataFrame({"x": [1]}),
        supporting=pd.Series([45]),
        supporting_method=CDS_OVERLAP_READS,
    )

    assert frame["n_alt_reads_supporting_protein_sequence"].tolist() == [45]


def test_an_unnamed_supporting_derivation_is_refused():
    with pytest.raises(ValueError, match="must name a derivation"):
        attach_read_evidence(
            pd.DataFrame({"x": [1]}), supporting=pd.Series([45]),
        )


# ---------------------------------------------------------------------------
# A source with nothing to say says nothing
# ---------------------------------------------------------------------------


def test_a_frame_with_no_rna_columns_gets_absent_not_zero():
    frame = attach_read_evidence(pd.DataFrame({"x": [1, 2, 3]}))

    for column in ("n_overlapping_reads", "n_alt_reads", "n_ref_reads",
                   "n_alt_reads_supporting_protein_sequence"):
        assert frame[column].isna().all()
    assert frame["read_count_method"].isna().all()


def test_the_columns_exist_even_when_unpopulated():
    """Same shape from every source; only which fields are filled differs."""
    frame = attach_read_evidence(pd.DataFrame({"x": [1]}))

    assert "n_alt_reads" in frame.columns
    assert "read_count_method" in frame.columns


# ---------------------------------------------------------------------------
# describe_read_evidence
# ---------------------------------------------------------------------------


def test_describe_reports_how_each_number_was_obtained():
    df = _read(read_pvacseq, PVACSEQ).df

    assert describe_read_evidence(df) == {
        "n_alt_reads": RNA_DEPTH_X_VAF,
        "n_ref_reads": RNA_DEPTH_X_VAF,
        "variant_allele_expression": TPM_X_DNA_VAF,
    }


def test_describe_says_nothing_about_columns_nothing_populated():
    assert describe_read_evidence(
        attach_read_evidence(pd.DataFrame({"x": [1]}))
    ) == {}


# ---------------------------------------------------------------------------
# Where the sequence came from
# ---------------------------------------------------------------------------


def test_each_reader_records_its_own_sequence_source():
    assert set(_read(read_lens, LENS).df["sequence_source"]) == {
        "lens_pep_context",
    }
    assert set(_read(read_pvacseq, PVACSEQ).df["sequence_source"]) == {
        "pvacseq_epitope",
    }


def test_an_unknown_sequence_source_is_refused():
    with pytest.raises(ValueError, match="must be one of"):
        attach_sequence_source(pd.DataFrame({"x": [1]}), "vibes")


def test_the_source_answers_a_question_source_type_cannot():
    """source_type is biology ("variant:snv"); this is method."""
    df = _read(read_pvacseq, PVACSEQ).df

    assert "sequence_source" in df.columns
    assert df["sequence_source"].nunique() == 1
