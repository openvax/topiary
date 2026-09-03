"""Cross-sample aggregation of canonical evidence columns."""

import pandas as pd
import pytest

from topiary import aggregate_evidence_across_samples


GROUP_KEYS = ["fragment_id", "peptide", "peptide_offset", "allele"]


def _row(sample_name, **overrides):
    row = {
        "fragment_id": "fragment-1",
        "peptide": "SIINFEKL",
        "peptide_offset": 3,
        "allele": "HLA-A*02:01",
        "sample_name": sample_name,
        "prediction_method_name": "netmhcpan",
        "kind": "pMHC_affinity",
        "n_rna_alt": 20,
        "n_rna_ref": 80,
        "n_rna_overlapping": 100,
        "rna_vaf": 0.2,
        "rna_evidence_subject": "reads",
        "rna_evidence_method": "rna_alignment",
        "rna_alt_expression": 7.5,
        "n_dna_alt": 3,
        "n_dna_overlapping": 10,
        "dna_vaf": 0.3,
        "dna_evidence_subject": "reads",
        "dna_evidence_method": "dna_alignment",
    }
    row.update(overrides)
    return row


def test_counts_are_summed_once_per_sample_and_vafs_are_recomputed():
    rows = []
    for sample_name, values in (
        ("pre", {"n_rna_alt": 20, "n_rna_ref": 80,
                 "n_rna_overlapping": 100, "rna_vaf": 0.2,
                 "n_dna_alt": 3, "n_dna_overlapping": 10,
                 "dna_vaf": 0.3}),
        ("post", {"n_rna_alt": 20, "n_rna_ref": 30,
                  "n_rna_overlapping": 50, "rna_vaf": 0.4,
                  "n_dna_alt": 1, "n_dna_overlapping": 10,
                  "dna_vaf": 0.1}),
    ):
        for method, kind in (
            ("netmhcpan", "pMHC_affinity"),
            ("mhcflurry", "pMHC_presentation"),
        ):
            rows.append(_row(
                sample_name,
                prediction_method_name=method,
                kind=kind,
                **values,
            ))
    df = pd.DataFrame(rows)
    original = df.copy(deep=True)

    pooled = aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)

    pd.testing.assert_frame_equal(df, original)
    assert len(pooled) == 1
    row = pooled.iloc[0]
    assert row["n_samples"] == 2
    assert row["n_rna_alt"] == 40
    assert row["n_rna_ref"] == 110
    assert row["n_rna_overlapping"] == 150
    assert row["rna_vaf"] == pytest.approx(40 / 150)
    assert row["rna_vaf"] != pytest.approx((0.2 + 0.4) / 2)
    assert row["n_dna_alt"] == 4
    assert row["n_dna_overlapping"] == 20
    assert row["dna_vaf"] == pytest.approx(4 / 20)
    assert row["rna_evidence_subject"] == "reads"
    assert row["rna_evidence_method"] == "rna_alignment"
    assert row["dna_evidence_subject"] == "reads"
    assert row["dna_evidence_method"] == "dna_alignment"
    assert "sample_name" not in pooled.columns
    assert "rna_alt_expression" not in pooled.columns


def test_partial_counts_stay_absent_instead_of_becoming_zero():
    df = pd.DataFrame([
        _row("pre"),
        _row("post", n_rna_alt=pd.NA, n_rna_ref=30,
             n_rna_overlapping=50),
    ])

    pooled = aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)

    assert "n_rna_alt" not in pooled.columns
    assert "rna_vaf" not in pooled.columns
    assert pooled.loc[0, "n_rna_ref"] == 110
    assert pooled.loc[0, "n_rna_overlapping"] == 150


@pytest.mark.parametrize(
    ("column", "second_value", "message"),
    [
        ("rna_evidence_subject", "fragments", "different.*subject"),
        ("rna_evidence_method", "rna_depth_x_vaf", "different.*method"),
    ],
)
def test_incompatible_evidence_is_not_flattened(column, second_value, message):
    df = pd.DataFrame([_row("pre"), _row("post", **{column: second_value})])

    with pytest.raises(ValueError, match=message):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


def test_repeated_prediction_rows_must_agree_within_a_sample():
    df = pd.DataFrame([
        _row("pre"),
        _row("pre", prediction_method_name="mhcflurry", n_rna_alt=21),
    ])

    with pytest.raises(ValueError, match="Contradictory 'n_rna_alt'.*pre"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


@pytest.mark.parametrize(
    "values",
    [(1, True), (True, 1), (0, False), (False, 0)],
)
def test_boolean_counts_cannot_hide_as_duplicate_integers(values):
    df = pd.DataFrame([
        _row("pre", n_rna_alt=value) for value in values
    ])

    with pytest.raises(ValueError, match="boolean values are not counts"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, float("inf"), True, 1 + 2j, "many"],
)
def test_invalid_counts_are_rejected(value):
    df = pd.DataFrame([_row("pre", n_rna_alt=value)])

    with pytest.raises(ValueError, match="n_rna_alt"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


@pytest.mark.parametrize(
    "value",
    [-1, 1.5, float("inf"), True, 1 + 2j, "many"],
)
def test_invalid_stated_counts_are_rejected_even_when_pool_is_incomplete(value):
    df = pd.DataFrame([
        _row("pre", n_rna_alt=value),
        _row("post", n_rna_alt=pd.NA),
    ])

    with pytest.raises(ValueError, match="n_rna_alt"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


@pytest.mark.parametrize("sample_name", [None, "", " ", "nan"])
def test_every_row_needs_a_real_sample_name(sample_name):
    df = pd.DataFrame([_row(sample_name)])

    with pytest.raises(ValueError, match="every row.*sample_name"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


def test_missing_assay_metadata_is_rejected_when_a_count_can_be_pooled():
    df = pd.DataFrame([_row("pre")]).drop(columns="rna_evidence_method")

    with pytest.raises(ValueError, match="must state 'rna_evidence_method'"):
        aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)


def test_zero_depth_does_not_invent_a_vaf():
    df = pd.DataFrame([
        _row("pre", n_rna_alt=0, n_rna_ref=0, n_rna_overlapping=0),
        _row("post", n_rna_alt=0, n_rna_ref=0, n_rna_overlapping=0),
    ])

    pooled = aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)

    assert pooled.loc[0, "n_rna_overlapping"] == 0
    assert "rna_vaf" not in pooled.columns


def test_group_keys_are_validated_and_must_exclude_sample_name():
    df = pd.DataFrame([_row("pre")])

    with pytest.raises(ValueError, match="exclude 'sample_name'"):
        aggregate_evidence_across_samples(
            df,
            group_keys=[*GROUP_KEYS, "sample_name"],
        )
    with pytest.raises(ValueError, match="missing_key"):
        aggregate_evidence_across_samples(df, group_keys=["missing_key"])


def test_one_tuple_valued_group_key_remains_one_identity_value():
    variant = ("chr1", 101, "A", "T")
    df = pd.DataFrame([
        _row("pre", variant=variant),
        _row("post", variant=variant),
    ])

    pooled = aggregate_evidence_across_samples(df, group_keys=["variant"])

    assert pooled.loc[0, "variant"] == variant
    assert pooled.loc[0, "n_rna_alt"] == 40


def test_equivalent_missing_identity_spellings_pool_as_one_group():
    df = pd.DataFrame([
        _row("pre", allele=None),
        _row("post", allele="nan"),
    ])

    pooled = aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)

    assert len(pooled) == 1
    assert pd.isna(pooled.loc[0, "allele"])
    assert pooled.loc[0, "n_samples"] == 2
    assert pooled.loc[0, "n_rna_alt"] == 40


def test_evidence_column_used_as_group_key_appears_once():
    df = pd.DataFrame([
        _row("pre", sequence_source="rna"),
        _row("post", sequence_source="rna"),
    ])

    pooled = aggregate_evidence_across_samples(
        df,
        group_keys=[*GROUP_KEYS, "sequence_source"],
    )

    assert pooled.columns.is_unique
    assert pooled.loc[0, "sequence_source"] == "rna"
    assert pooled.loc[0, "n_rna_alt"] == 40


@pytest.mark.parametrize(
    "column",
    ["n_rna_alt", "n_rna_overlapping", "rna_vaf", "dna_vaf"],
)
def test_aggregate_values_cannot_be_used_as_group_keys(column):
    df = pd.DataFrame([_row("pre"), _row("post")])

    with pytest.raises(ValueError, match="not canonical counts or VAFs"):
        aggregate_evidence_across_samples(
            df,
            group_keys=[*GROUP_KEYS, column],
        )


def test_large_integer_counts_are_summed_exactly():
    count = 2 ** 60 + 1
    df = pd.DataFrame([
        _row("pre", n_rna_alt=count),
        _row("post", n_rna_alt=count),
    ])

    pooled = aggregate_evidence_across_samples(df, group_keys=GROUP_KEYS)

    assert pooled.loc[0, "n_rna_alt"] == 2 * count
