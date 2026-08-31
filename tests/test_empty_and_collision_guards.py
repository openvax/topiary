"""Two silent-failure guards (topiary #216, #217).

Both are the same species as the bugs this area keeps producing: a case
where *absent* and *empty* were treated as different things, and a case
where a frame silently ended up holding data it could not read back.
"""

import pathlib

import pandas as pd
import pytest

from topiary import SelfProteome, from_wide, read_lens, to_wide

LENS_FIXTURE = pathlib.Path(__file__).parent / "data" / "lens" / "sample_v1_4.tsv"


def _long(extra=None):
    row = dict(
        source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
        allele="HLA-A*02:01", kind="pMHC_affinity", value=75.0, score=0.5,
        percentile_rank=1.0, prediction_method_name="netmhcpan",
        predictor_version="1",
    )
    if extra:
        row.update(extra)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# #216: an empty length bucket answers like an absent one
# ---------------------------------------------------------------------------
#
# nearest() guarded on key presence, not emptiness, and _build_index created
# a bucket for every requested peptide length whether or not any source was
# long enough to fill it. So a proteome of short sequences reached
# `dists.argmin(axis=1)` over a zero-width axis and raised
# "attempt to get argmin of an empty sequence" — while the same fact reached
# by a different route (a query length the proteome has no bucket for)
# already returned a clean empty row.


def test_a_proteome_of_short_sequences_reports_no_match():
    proteome = SelfProteome.from_peptides({"a": "SIIN", "b": "KLQA"})

    result = proteome.nearest(["SIINFEKLA"]).to_dict("records")[0]

    assert result["self_nearest_peptide"] is None
    assert result["self_nearest_edit_distance"] is None


def test_the_same_holds_for_a_fasta_proteome(tmp_path):
    """from_fasta is the caller-supplied path, so it is the one that bites."""
    path = tmp_path / "short.fasta"
    path.write_text(">a\nSIIN\n>b\nKLQA\n")

    result = SelfProteome.from_fasta(path).nearest(["SIINFEKLA"])

    assert result.to_dict("records")[0]["self_nearest_peptide"] is None


def test_peptide_lengths_does_not_claim_coverage_it_lacks():
    """An empty bucket made the proteome advertise lengths it had none of."""
    proteome = SelfProteome.from_peptides({"a": "SIIN"})

    assert proteome.peptide_lengths == []
    assert proteome.n_reference_peptides == 0


def test_an_absent_length_and_an_empty_one_agree():
    """The two routes to "nothing to compare against" give one answer."""
    populated = SelfProteome.from_peptides({"src": "SIINFEKLBKLQAAMAVM"})
    empty = SelfProteome.from_peptides({"a": "SIIN"})

    # A query longer than any bucket the populated proteome has.
    absent = populated.nearest(["SIINFEKLAKQW"]).to_dict("records")[0]
    unfilled = empty.nearest(["SIINFEKLA"]).to_dict("records")[0]

    assert absent["self_nearest_peptide"] is None
    assert unfilled["self_nearest_peptide"] is None


def test_a_populated_proteome_is_unaffected():
    proteome = SelfProteome.from_peptides({"src": "SIINFEKLBKLQAAMAVM"})

    result = proteome.nearest(["SIINFEKLA"]).to_dict("records")[0]

    assert result["self_nearest_peptide"] == "SIINFEKLB"
    assert result["self_nearest_edit_distance"] == 1
    assert proteome.peptide_lengths == [8, 9, 10, 11]


def test_a_partially_fillable_proteome_keeps_the_lengths_it_has():
    """Only the unfillable lengths drop out, not the whole index."""
    proteome = SelfProteome.from_peptides(
        {"src": "SIINFEKLB"}, peptide_lengths=(9, 10, 11),
    )

    assert proteome.peptide_lengths == [9]
    assert proteome.nearest(
        ["SIINFEKLA"]
    ).to_dict("records")[0]["self_nearest_peptide"] == "SIINFEKLB"


# ---------------------------------------------------------------------------
# #217: to_wide must not silently produce _x / _y columns
# ---------------------------------------------------------------------------
#
# The annotation columns were merged against the generated prediction
# columns with pandas' default suffixes, so a name on both sides became
# _x / _y: the canonical name ceased to exist, and from_wide returned NaN
# for a prediction that was in the input.


def test_a_colliding_annotation_column_is_refused():
    with pytest.raises(ValueError, match="would collide"):
        to_wide(_long({"netmhcpan_affinity_value": 999.0}))


def test_the_error_names_the_offending_column():
    with pytest.raises(ValueError) as excinfo:
        to_wide(_long({"netmhcpan_affinity_value": 999.0}))

    assert "netmhcpan_affinity_value" in str(excinfo.value)


def test_no_x_y_columns_can_reach_the_frame():
    """_x / _y are pandas merge artifacts, not part of the wide schema."""
    with pytest.raises(ValueError):
        to_wide(_long({"netmhcpan_affinity_value": 999.0}))

    wide = to_wide(_long({"tpm": 12.0}))
    assert not any(
        c.endswith("_x") or c.endswith("_y") for c in wide.columns
    )


def test_an_unrelated_annotation_column_still_passes_through():
    wide = to_wide(_long({"tpm": 12.0}))

    assert "tpm" in wide.columns
    assert from_wide(wide)["value"].tolist() == [75.0]


def test_the_lens_round_trip_is_unaffected():
    """The realistic pipeline: it consumed the columns, so it never collided."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = read_lens(LENS_FIXTURE)

    wide = to_wide(result.to_long().df)

    assert wide.columns.is_unique
    assert "netmhcpan_affinity_value" in wide.columns
