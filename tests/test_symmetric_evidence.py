"""DNA and RNA evidence are the same shape, and sources stay attributable.

The defect class these guard: a caller who has written a filter against
RNA depth should be able to write the DNA one by changing three letters,
and a frame holding two tools' numbers should never answer "what was the
VAF" with whichever tool filled the column first.
"""

import warnings

import pandas as pd
import pytest

from topiary import (
    DNA_EVIDENCE_COLUMNS,
    PREDICTION_KEY_COLUMNS,
    RNA_EVIDENCE_COLUMNS,
    SOURCE_PREFIXES,
    ProteinFragment,
    TopiaryResult,
    attach_dna_evidence,
    other_allele_count,
    read_lens,
    read_pvacseq,
    source_column,
    source_columns,
    stack_results,
)
from topiary.evidence import DNA_DEPTH_X_VAF, attach_rna_evidence

LENS = "tests/data/lens/sample_v1_4.tsv"
PVAC_ALL = "tests/data/pvacseq/mhc_i_all_epitopes.tsv"


def _read(fn, path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(path).df


# ---------------------------------------------------------------------------
# Symmetry: the same question, asked of either assay
# ---------------------------------------------------------------------------


def test_the_dna_columns_mirror_the_rna_columns_name_for_name():
    """The point of the symmetry: s/rna/dna/ names a real column."""
    rna_shape = {c.replace("rna", "", 1) for c in RNA_EVIDENCE_COLUMNS}
    dna_shape = {c.replace("dna", "", 1) for c in DNA_EVIDENCE_COLUMNS}
    # Expression is transcript-only and has no DNA meaning; everything
    # else must exist on both sides.
    expression_only = {"_alt_expression", "_alt_expression_method",
                       "gene_expression"}
    assert rna_shape - expression_only == dna_shape


def test_pvacseq_derives_dna_counts_from_depth_and_vaf():
    """The asymmetry this closes: DNA depth+VAF were raw passthroughs."""
    df = _read(read_pvacseq, PVAC_ALL)
    row = df[df["n_dna_alt"].notna()].iloc[0]

    depth = row["pvacseq_tumor_dna_depth"]
    vaf = row["pvacseq_tumor_dna_vaf"]
    assert row["n_dna_alt"] == round(depth * vaf)
    assert row["n_dna_ref"] == depth - row["n_dna_alt"]
    assert row["n_dna_overlapping"] == depth
    assert row["dna_evidence_method"] == DNA_DEPTH_X_VAF
    assert row["dna_evidence_subject"] == "reads"


def test_the_dna_subject_is_inferred_not_asserted():
    """It was hardcoded to READS, which made it a literal, not data.

    A depth x VAF split really is about reads, because depth is a read
    depth — that one is inferable. A direct count's unit is known only
    to whoever counted it.
    """
    split = attach_dna_evidence(
        pd.DataFrame({"x": [1]}), depth=pd.Series([100]), vaf=pd.Series([0.4]),
    )
    assert split["dna_evidence_subject"].iloc[0] == "reads"

    counted = attach_dna_evidence(
        pd.DataFrame({"x": [1]}), alt=pd.Series([40]), ref=pd.Series([60]),
    )
    assert "dna_evidence_subject" not in counted.columns

    stated = attach_dna_evidence(
        pd.DataFrame({"x": [1]}), alt=pd.Series([40]), ref=pd.Series([60]),
        subject="fragments",
    )
    assert stated["dna_evidence_subject"].iloc[0] == "fragments"


def test_a_nonsense_dna_subject_is_refused():
    with pytest.raises(ValueError, match="subject must be"):
        attach_dna_evidence(
            pd.DataFrame({"x": [1]}), alt=pd.Series([40]), subject="molecules",
        )


def test_dna_counts_come_from_the_dna_columns_not_the_rna_ones():
    """The fixture cannot prove this: its DNA and RNA depths are equal.

    tests/data/pvacseq/mhc_i_all_epitopes.tsv has Tumor DNA Depth ==
    Tumor RNA Depth on every row, so a reader wired to the wrong column
    passes every assertion made against it. Build the case the fixture
    cannot: distinct depths, distinct fractions, and check each assay
    landed on its own numbers.
    """
    df = pd.DataFrame({"x": [1]})
    out = attach_rna_evidence(
        df, overlapping=pd.Series([200]), vaf=pd.Series([0.5]),
    )
    out = attach_dna_evidence(
        out, depth=pd.Series([100]), vaf=pd.Series([0.25]),
    )
    assert out["n_rna_overlapping"].iloc[0] == 200
    assert out["n_dna_overlapping"].iloc[0] == 100
    assert out["n_rna_alt"].iloc[0] == 100
    assert out["n_dna_alt"].iloc[0] == 25
    assert out["rna_vaf"].iloc[0] == pytest.approx(0.5)
    assert out["dna_vaf"].iloc[0] == pytest.approx(0.25)


def test_a_depth_without_a_fraction_yields_no_split_rather_than_zero():
    out = attach_dna_evidence(
        pd.DataFrame({"x": [1]}), depth=pd.Series([50]), vaf=pd.Series([None]),
    )
    assert pd.isna(out["n_dna_alt"].iloc[0])
    assert pd.isna(out["dna_evidence_method"].iloc[0])
    # Depth is still known: the source did cover the locus.
    assert out["n_dna_overlapping"].iloc[0] == 50


def test_a_source_that_states_only_a_fraction_gets_only_the_fraction_column():
    """The defect: 6 all-null DNA columns claiming the source answered.

    pVACseq's aggregated report has a DNA VAF but no DNA depth, so no
    alt/ref split exists. Emitting `n_dna_alt` full of nulls would make
    available_evidence_columns() report a capability the source lacks.
    """
    agg = _read(read_pvacseq, "tests/data/pvacseq/mhc_i_aggregated.tsv")
    dna = [c for c in agg.columns if c.startswith(("n_dna_", "dna_"))]
    assert dna == ["dna_vaf"]


def test_no_dna_inputs_at_all_writes_no_dna_columns():
    out = attach_dna_evidence(pd.DataFrame({"x": [1]}))
    assert list(out.columns) == ["x"]


def test_other_allele_column_is_omitted_not_nulled_when_ref_was_derived():
    """A column of nulls would claim the locus was checked for third alleles."""
    out = attach_rna_evidence(
        pd.DataFrame({"x": [1]}),
        overlapping=pd.Series([100]), vaf=pd.Series([0.4]),
    )
    assert "n_rna_other" not in out.columns


def test_lens_gets_no_dna_columns_because_it_never_names_the_assay():
    """LENS's single `vaf` could be either assay; guessing would be wrong."""
    df = _read(read_lens, LENS)
    assert not [c for c in df.columns if c.startswith(("n_dna_", "dna_"))]


# ---------------------------------------------------------------------------
# Other-allele support, and why it is usually absent
# ---------------------------------------------------------------------------


def test_other_allele_support_is_real_when_ref_was_counted():
    out = attach_dna_evidence(
        pd.DataFrame({"x": [1]}),
        depth=pd.Series([100]), alt=pd.Series([40]), ref=pd.Series([50]),
    )
    assert out["n_dna_other"].iloc[0] == 10


def test_other_allele_support_is_absent_when_ref_was_derived_from_depth():
    """A derived ref already absorbs third alleles; 0 would be a lie."""
    out = attach_dna_evidence(
        pd.DataFrame({"x": [1]}), depth=pd.Series([100]), vaf=pd.Series([0.4]),
    )
    assert out["n_dna_ref"].iloc[0] == 60
    assert "n_dna_other" not in out.columns


def test_counts_that_do_not_add_up_clip_at_zero_rather_than_go_negative():
    got = other_allele_count(
        pd.Series([100]), pd.Series([80]), pd.Series([70]),
    )
    assert got.iloc[0] == 0


def test_a_fragment_reports_other_allele_support_in_its_preferred_unit():
    both = ProteinFragment(
        fragment_id="f", sequence="MKTVRQ",
        n_rna_other_fragments=10, n_rna_other_reads=19,
    )
    assert both.n_rna_other == 10

    reads_only = ProteinFragment(
        fragment_id="f", sequence="MKTVRQ", n_rna_other_reads=4,
    )
    assert reads_only.n_rna_other == 4

    assert ProteinFragment(
        fragment_id="f", sequence="MKTVRQ",
    ).n_rna_other is None


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------


def test_each_tools_own_numbers_carry_that_tools_name():
    lens = _read(read_lens, LENS)
    pvac = _read(read_pvacseq, PVAC_ALL)

    assert "lens_vaf" in lens.columns and "vaf" not in lens.columns
    assert "pvacseq_tumor_dna_vaf" in pvac.columns
    assert "tumor_dna_vaf" not in pvac.columns


def test_two_tools_vafs_coexist_and_stay_attributable_in_one_frame():
    """The defect this prevents: an unattributable bare `vaf` column."""
    lens = _read(read_lens, LENS).head(2)
    pvac = _read(read_pvacseq, PVAC_ALL).head(2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        merged = stack_results(
            [TopiaryResult(lens), TopiaryResult(pvac)],
        ).df

    present = source_columns(merged)
    assert "lens_vaf" in present
    assert "pvacseq_tumor_rna_vaf" in present
    # And each is answerable per tool.
    assert source_columns(merged, "lens") == tuple(
        c for c in present if c.startswith("lens_")
    )


def test_an_unknown_source_is_refused_rather_than_given_a_guessed_prefix():
    with pytest.raises(ValueError, match="unknown source"):
        source_column("nonesuch", "vaf")
    with pytest.raises(ValueError, match="unknown source"):
        source_columns(pd.DataFrame(), "nonesuch")


def test_every_known_prefix_ends_in_an_underscore():
    """Otherwise `lensvaf` and a real column could collide."""
    for name, prefix in SOURCE_PREFIXES.items():
        assert prefix == f"{name}_"


# ---------------------------------------------------------------------------
# Canonical VAF
# ---------------------------------------------------------------------------


def test_the_canonical_vaf_prefers_the_number_the_source_stated():
    out = attach_rna_evidence(
        pd.DataFrame({"x": [1]}),
        overlapping=pd.Series([100]), vaf=pd.Series([0.37]),
    )
    assert out["rna_vaf"].iloc[0] == pytest.approx(0.37)
    # Not the rounded round-trip through the integer counts.
    assert out["n_rna_alt"].iloc[0] == 37


def test_the_canonical_vaf_is_absent_when_nothing_supports_it():
    out = attach_rna_evidence(pd.DataFrame({"x": [1]}))
    assert pd.isna(out["rna_vaf"].iloc[0])


# ---------------------------------------------------------------------------
# The cache key, now public
# ---------------------------------------------------------------------------


def test_the_cache_key_is_public_and_names_the_genotype_column():
    assert "allele_set" in PREDICTION_KEY_COLUMNS
    assert "sample_name" not in PREDICTION_KEY_COLUMNS


def test_the_concat_overlap_error_names_the_key_it_actually_used():
    """It used to name a stale 4-tuple, so the message misled."""
    from topiary import CachedPredictor

    row = pd.DataFrame([dict(
        peptide="SIINFEKLA", peptide_length=9, allele="HLA-A*02:01",
        kind="pMHC_affinity", value=0.5, score=0.5, percentile_rank=1.0,
        prediction_method_name="netmhcpan", predictor_version="4.1",
    )])
    with pytest.raises(ValueError) as excinfo:
        CachedPredictor.concat(
            [CachedPredictor(row), CachedPredictor(row)],
        )
    for column in PREDICTION_KEY_COLUMNS:
        assert column in str(excinfo.value)
