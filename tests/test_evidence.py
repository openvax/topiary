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
    RNA_DEPTH_X_SOURCE_VAF,
    RNA_DEPTH_X_VAF,
    TPM_X_DNA_VAF,
    attach_rna_evidence,
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
    assert row["n_rna_overlapping"] > 0
    assert row["n_rna_alt"] + row["n_rna_ref"] == row["n_rna_overlapping"]


def test_pvacseq_names_the_split_as_derived():
    """Not counted — depth x VAF, and the frame says so."""
    df = _read(read_pvacseq, PVACSEQ).df

    assert set(df["rna_evidence_method"].dropna()) == {RNA_DEPTH_X_VAF}


def test_pvacseq_estimates_rna_alt_expression():
    df = _read(read_pvacseq, PVACSEQ).df

    assert df["rna_alt_expression"].notna().any()
    assert set(df["rna_alt_expression_method"].dropna()) == {
        TPM_X_DNA_VAF,
    }


def test_the_expression_estimate_is_abundance_times_fraction():
    df = _read(read_pvacseq, PVACSEQ).df
    row = df.dropna(subset=["rna_alt_expression"]).iloc[0]

    expected = row["transcript_expression"] * row["pvacseq_tumor_dna_vaf"]
    assert row["rna_alt_expression"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# LENS: two real counts, one of something adjacent
# ---------------------------------------------------------------------------


def test_lens_populates_the_counted_columns():
    df = _read(read_lens, LENS).df

    assert df["n_rna_overlapping"].notna().any()
    assert df["n_rna_alt"].notna().any()


def test_lens_rows_without_a_vaf_get_no_split_and_no_method():
    """Absent stays absent, and says so — it does not become zero."""
    df = _read(read_lens, LENS).df
    unstated = df[df["lens_vaf"].isna()]

    assert unstated["n_rna_alt"].isna().all()
    assert unstated["rna_evidence_method"].isna().all()


def test_lens_rows_with_a_vaf_do_get_a_split():
    df = _read(read_lens, LENS).df
    stated = df[df["lens_vaf"].notna()]

    assert len(stated) > 0
    assert stated["n_rna_alt"].notna().all()
    # LENS's `vaf` carries no assay qualifier, so the method says the
    # depth was multiplied by *the source's* fraction rather than
    # asserting it was an RNA one.
    assert set(stated["rna_evidence_method"]) == {RNA_DEPTH_X_SOURCE_VAF}


def test_a_cds_overlap_count_keeps_its_own_name():
    """LENS counts reads overlapping the peptide's CDS, which is not a
    count of reads supporting the assembled protein sequence. Emitting it
    under that name would overstate what the reader has, so it passes
    through under the source's own column instead."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = read_lens(LENS).df

    assert "lens_rna_reads_covering_genomic_origin_with_peptide_cds" in df.columns
    assert "n_rna_alt_reads_supporting_protein_sequence" not in df.columns


# ---------------------------------------------------------------------------
# A source with nothing to say says nothing
# ---------------------------------------------------------------------------


def test_a_source_with_no_rna_gets_no_rna_columns():
    """Absent, not zero -- and absent at the column level, not the value.

    This pair used to assert the opposite ("same shape from every source;
    only which fields are filled differs"), which is the policy the rest
    of the package abandoned: `available_evidence_columns` is only a
    useful signal if a present column means the source could answer.
    A column of nulls says it looked and found nothing.

    The DNA side had this from the start; the RNA side kept writing six
    null columns until the twin-conformance harness compared them.
    """
    frame = attach_rna_evidence(pd.DataFrame({"x": [1, 2, 3]}))

    assert list(frame.columns) == ["x"]


def test_a_partially_answerable_source_gets_the_columns_it_can_fill():
    """Column presence is a frame-level claim, NA is a row-level one.

    A source that reports depth but no fraction can say how much
    coverage there was and cannot say how it split, so it gets
    `n_rna_overlapping` and no `n_rna_alt`.
    """
    frame = attach_rna_evidence(
        pd.DataFrame({"x": [1]}), overlapping=pd.Series([50]),
    )

    assert frame["n_rna_overlapping"].iloc[0] == 50
    assert "n_rna_alt" not in frame.columns
    assert "rna_evidence_method" not in frame.columns


def test_all_null_rna_inputs_are_omitted_like_absent_inputs():
    frame = attach_rna_evidence(
        pd.DataFrame({"x": [1, 2]}),
        overlapping=pd.Series([None, None]),
        vaf=pd.Series([None, None]),
        reported_rna_alt_expression=pd.Series([None, None]),
    )

    assert list(frame.columns) == ["x"]


# ---------------------------------------------------------------------------
# describe_read_evidence
# ---------------------------------------------------------------------------


def test_describe_reports_how_each_number_was_obtained():
    df = _read(read_pvacseq, PVACSEQ).df

    assert describe_read_evidence(df) == {
        "n_rna_alt": RNA_DEPTH_X_VAF,
        "n_rna_ref": RNA_DEPTH_X_VAF,
        "n_rna_overlapping": RNA_DEPTH_X_VAF,
        "rna_alt_expression": TPM_X_DNA_VAF,
    }


def test_describe_says_nothing_about_columns_nothing_populated():
    assert describe_read_evidence(
        attach_rna_evidence(pd.DataFrame({"x": [1]}))
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
