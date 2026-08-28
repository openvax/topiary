"""``allele_set``: storing what a genotype-level prediction was scored against.

MHCflurry's presentation predictor in haplotype mode scores a peptide against
a sample's whole allele list and reports the allele it deconvolved as the
likeliest presenter. mhctools puts that one allele in the row, which reads
exactly like a per-allele prediction — so the frame loses the fact that the
score is about the set (topiary #168).

``allele_set`` records the set. ``allele`` keeps the best allele, so nothing
that reads it breaks and the attribution isn't thrown away.
"""

import warnings

import pandas as pd
import pytest

from topiary import from_wide, read_csv, to_csv, to_wide
from topiary.ranking import (
    Column,
    EvalContext,
    apply_filter,
    evaluate_scores,
    format_allele_set,
    parse,
    peptide_view,
    split_allele_set,
)

GENOTYPE = ["HLA-A*02:01", "HLA-B*07:02"]
GENOTYPE_SET = "HLA-A*02:01,HLA-B*07:02"


def _row(allele, kind, score, value=None, allele_set="", peptide="SIINFEKLA"):
    return {
        "source_sequence_name": "s", "peptide": peptide, "peptide_offset": 0,
        "allele": allele, "allele_set": allele_set, "kind": kind,
        "value": value, "score": score, "percentile_rank": 1.0,
        "prediction_method_name": "mhcflurry",
    }


def _genotype_df(presentation_score=0.88):
    """Per-allele affinity plus one genotype-level presentation row."""
    return pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 0.9, value=50.0),
        _row("HLA-B*07:02", "pMHC_affinity", 0.2, value=4000.0),
        _row("HLA-A*02:01", "pMHC_presentation", presentation_score,
             allele_set=GENOTYPE_SET),
    ])


# ---------------------------------------------------------------------------
# The stored format
# ---------------------------------------------------------------------------


def test_format_is_sorted_so_the_same_set_compares_equal():
    assert format_allele_set(GENOTYPE) == format_allele_set(reversed(GENOTYPE))
    assert format_allele_set(GENOTYPE) == GENOTYPE_SET


def test_format_drops_blanks_and_duplicates():
    assert format_allele_set(
        ["HLA-A*02:01", "", "  ", "HLA-A*02:01", "HLA-B*07:02"]
    ) == GENOTYPE_SET


@pytest.mark.parametrize("cell", ["", "   ", None, float("nan")])
def test_split_reads_an_absent_set_as_empty(cell):
    assert split_allele_set(cell) == []


def test_membership_is_by_whole_token_never_substring():
    """`HLA-A*02:01` is a textual prefix of `HLA-A*02:010`, a real allele."""
    members = split_allele_set("HLA-A*02:010,HLA-B*07:02")

    assert "HLA-A*02:01" not in members
    assert "HLA-A*02:010" in members


# ---------------------------------------------------------------------------
# Grouping: the genotype row must not sit in one allele's group
# ---------------------------------------------------------------------------


def test_allele_set_joins_the_group_key_when_populated():
    ctx = EvalContext(_genotype_df())

    assert ctx.group_keys == [
        "source_sequence_name", "peptide", "peptide_offset",
        "allele", "allele_set",
    ]


def test_frames_without_genotype_rows_keep_the_narrower_key():
    df = _genotype_df().drop(columns=["allele_set"])

    assert "allele_set" not in EvalContext(df).group_keys


def test_blank_allele_set_column_is_not_a_group_key():
    df = _genotype_df()
    df["allele_set"] = ""

    assert "allele_set" not in EvalContext(df).group_keys


def test_genotype_row_gets_its_own_group():
    """Without this it would share A*02:01's group and be read as its score."""
    ctx = EvalContext(_genotype_df())

    assert len(ctx.group_index) == 3
    per_allele = parse("affinity.score").eval(ctx).reindex(ctx.group_index)
    assert per_allele.tolist()[:2] == [0.9, 0.2]
    assert pd.isna(per_allele.tolist()[2])


# ---------------------------------------------------------------------------
# The set makes the frame self-describing — no kind_support needed
# ---------------------------------------------------------------------------


def test_genotype_score_reaches_the_whole_genotype_without_metadata():
    df = _genotype_df()

    with pytest.warns(UserWarning, match="whole genotype"):
        bare = evaluate_scores(df, parse("presentation.score"))

    assert bare.tolist() == [0.88, 0.88, 0.88]


def test_explicit_peptide_view_reads_the_same_and_stays_silent():
    df = _genotype_df()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scores = evaluate_scores(df, peptide_view(parse("presentation.score")))

    assert scores.tolist() == [0.88, 0.88, 0.88]


def test_dependence_is_read_from_the_set():
    from mhctools import Kind

    from topiary.ranking.nodes import (
        _filter_kind_method_version, _resolve_mhc_dependence,
    )

    ctx = EvalContext(_genotype_df())
    for kind, expected in (
        (Kind.pMHC_presentation, "haplotype"),
        (Kind.pMHC_affinity, "single_allele"),
    ):
        sub = _filter_kind_method_version(ctx, kind, None, None)
        assert _resolve_mhc_dependence(ctx, kind, sub) == expected


def test_an_allele_scoped_filter_keeps_the_genotype_row():
    """Same peptide-level rule as allele-free evidence (#183)."""
    df = _genotype_df()

    kept = apply_filter(df, parse("affinity.value <= 500"))

    assert sorted(kept["kind"].unique()) == [
        "pMHC_affinity", "pMHC_presentation",
    ]


def test_a_peptide_filtered_out_entirely_takes_its_genotype_row():
    df = _genotype_df()

    assert len(apply_filter(df, parse("affinity.value <= 10"))) == 0


# ---------------------------------------------------------------------------
# eq() stays equality; includes() asks the set question
# ---------------------------------------------------------------------------


def test_eq_on_allele_stays_literal():
    df = _genotype_df()

    kept = apply_filter(df, Column("allele").eq("HLA-B*07:02"))

    # The genotype row's allele is A*02:01, whatever its set covers.
    assert kept["kind"].tolist() == ["pMHC_affinity"]


def test_includes_asks_whether_the_set_covers_an_allele():
    df = _genotype_df()

    kept = apply_filter(df, Column("allele_set").includes("HLA-B*07:02"))

    assert kept["kind"].tolist() == ["pMHC_presentation"]


def test_includes_is_false_for_rows_with_no_set():
    df = _genotype_df()

    kept = apply_filter(df, Column("allele_set").includes("HLA-A*02:01"))

    # Literal membership: a per-allele row's set is empty, not {its allele}.
    assert kept["kind"].tolist() == ["pMHC_presentation"]


def test_includes_rejects_a_prefix_of_a_member():
    df = pd.DataFrame([
        _row("HLA-A*02:010", "pMHC_presentation", 0.5,
             allele_set="HLA-A*02:010,HLA-B*07:02"),
    ])

    assert len(apply_filter(df, Column("allele_set").includes("HLA-A*02:01"))) == 0
    assert len(apply_filter(df, Column("allele_set").includes("HLA-A*02:010"))) == 1


def test_includes_negates():
    df = _genotype_df()

    kept = apply_filter(df, ~Column("allele_set").includes("HLA-B*07:02"))

    assert kept["kind"].tolist() == ["pMHC_affinity", "pMHC_affinity"]


def test_includes_takes_one_name():
    with pytest.raises(TypeError, match="one member name as a string"):
        Column("allele_set").includes(["HLA-A*02:01"])


def test_includes_parses_and_round_trips():
    for text in (
        "column(allele_set).includes('HLA-B*07:02')",
        "~column(allele_set).includes('HLA-B*07:02')",
        "affinity <= 500 & column(allele_set).includes('HLA-A*02:01')",
    ):
        node = parse(text)
        assert parse(node.to_expr_string()).to_ast_string() == node.to_ast_string()


def test_includes_on_a_non_column_is_rejected():
    with pytest.raises(ValueError, match="applies to column"):
        parse("affinity.score.includes('x')")


# ---------------------------------------------------------------------------
# Round trips — the reason the column exists
# ---------------------------------------------------------------------------


def test_csv_round_trip_preserves_the_set(tmp_path):
    df = _genotype_df()
    path = tmp_path / "predictions.csv"

    to_csv(df, str(path))
    loaded = read_csv(str(path))
    loaded_df = loaded.df if hasattr(loaded, "df") else loaded

    assert loaded_df["allele_set"].fillna("").tolist() == ["", "", GENOTYPE_SET]


def test_a_loaded_frame_still_knows_it_is_genotype_level(tmp_path):
    """kind_support cannot survive a file; the column can."""
    path = tmp_path / "predictions.csv"
    to_csv(_genotype_df(), str(path))
    loaded = read_csv(str(path))
    loaded_df = loaded.df if hasattr(loaded, "df") else loaded

    with pytest.warns(UserWarning, match="whole genotype"):
        scores = evaluate_scores(loaded_df, parse("presentation.score"))

    assert scores.tolist() == [0.88, 0.88, 0.88]


def test_wide_treats_the_set_as_identity_not_a_prediction():
    wide = to_wide(_genotype_df())

    assert "allele_set" in wide.columns
    assert sorted(wide["allele_set"].fillna("").unique()) == ["", GENOTYPE_SET]


def test_wide_round_trip_preserves_the_set():
    restored = from_wide(to_wide(_genotype_df()))

    assert sorted(restored["allele_set"].fillna("").unique()) == [
        "", GENOTYPE_SET,
    ]


def test_two_genotypes_for_one_peptide_stay_distinct():
    """Different sets are different predictions, not a duplicate."""
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_presentation", 0.8, allele_set=GENOTYPE_SET),
        _row("HLA-A*02:01", "pMHC_presentation", 0.3,
             allele_set="HLA-A*02:01,HLA-C*06:02"),
    ])

    ctx = EvalContext(df)

    assert len(ctx.group_index) == 2


# ---------------------------------------------------------------------------
# The producer side
# ---------------------------------------------------------------------------


class _FakeHaplotypeModel:
    """Reports presentation per genotype and affinity per allele."""

    prediction_method_name = "fake-flurry"
    predictor_version = "0.0.0"
    default_peptide_lengths = [9]

    def __init__(self, alleles):
        self.alleles = list(alleles)

    def kind_support(self):
        return {
            "pMHC_presentation": {
                "mhc_dependence": "haplotype", "mhc_class": "I",
            },
            "pMHC_affinity": {
                "mhc_dependence": "single_allele", "mhc_class": "I",
            },
        }


def test_predictor_stamps_the_set_on_genotype_level_kinds_only():
    from topiary import TopiaryPredictor

    predictor = TopiaryPredictor(models=[_FakeHaplotypeModel(GENOTYPE)])
    raw = pd.DataFrame([
        {"peptide": "SIINFEKLA", "allele": "HLA-A*02:01",
         "kind": "pMHC_affinity", "score": 0.9},
        {"peptide": "SIINFEKLA", "allele": "HLA-A*02:01",
         "kind": "pMHC_presentation", "score": 0.8},
        {"peptide": "SIINFEKLA", "allele": "",
         "kind": "antigen_processing", "score": 0.4},
    ])

    stamped = predictor._attach_allele_set(raw, _FakeHaplotypeModel(GENOTYPE))

    by_kind = dict(zip(stamped["kind"], stamped["allele_set"]))
    assert by_kind["pMHC_presentation"] == GENOTYPE_SET
    assert by_kind["pMHC_affinity"] == ""
    assert by_kind["antigen_processing"] == ""
    # The best allele is kept, not replaced.
    assert stamped["allele"].tolist()[1] == "HLA-A*02:01"


def test_models_that_report_no_genotype_kinds_add_no_column():
    from mhctools import RandomBindingPredictor

    from topiary import TopiaryPredictor

    predictor = TopiaryPredictor(
        models=[RandomBindingPredictor(["HLA-A*02:01"])],
    )
    raw = pd.DataFrame([
        {"peptide": "SIINFEKLA", "allele": "HLA-A*02:01",
         "kind": "pMHC_affinity", "score": 0.9},
    ])

    stamped = predictor._attach_allele_set(
        raw, RandomBindingPredictor(["HLA-A*02:01"]),
    )

    assert "allele_set" not in stamped.columns


def test_combine_predictions_keeps_genotypes_apart():
    """Two genotypes are different predictions, not a duplicate to reject."""
    from topiary import TopiaryResult, combine_predictions

    def result(allele_set, score):
        return TopiaryResult(pd.DataFrame([
            _row("HLA-A*02:01", "pMHC_presentation", score,
                 allele_set=allele_set),
        ]))

    combined = combine_predictions([
        result(GENOTYPE_SET, 0.8),
        result("HLA-A*02:01,HLA-C*06:02", 0.3),
    ], coverage="partial")

    assert sorted(combined.df["allele_set"]) == [
        "HLA-A*02:01,HLA-B*07:02", "HLA-A*02:01,HLA-C*06:02",
    ]
