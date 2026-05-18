"""Tests for combining separately-run predictor outputs."""

import pandas as pd
import pytest

from topiary import (
    Affinity,
    EvalContext,
    Presentation,
    TopiaryPredictor,
    TopiaryResult,
    combine_predictions,
    read_csv,
    read_tsv,
    to_wide,
)
from topiary.io import Metadata


class ToyAffinityPredictor:
    default_peptide_lengths = [9]
    supported_kinds = ("pMHC_affinity",)

    def __init__(self, name, version, alleles, offset):
        self.prediction_method_name = name
        self.predictor_version = version
        self.alleles = alleles
        self.offset = offset

    def predict_dataframe(self, peptides):
        rows = []
        for peptide_i, peptide in enumerate(peptides):
            for allele_i, allele in enumerate(self.alleles):
                affinity = self.offset + 10 * peptide_i + allele_i
                rows.append({
                    "peptide": peptide,
                    "allele": allele,
                    "kind": "pMHC_affinity",
                    "value": float(affinity),
                    "score": 1.0 / affinity,
                    "percentile_rank": affinity / 100.0,
                    "predictor_name": self.prediction_method_name,
                    "predictor_version": self.predictor_version,
                })
        return pd.DataFrame(rows)


class ToyGridPredictor:
    default_peptide_lengths = [9, 10]
    supported_kinds = ("pMHC_affinity", "pMHC_presentation")

    def __init__(
        self, name, version, alleles, offset, peptide_lengths=None,
        kinds=None,
    ):
        self.prediction_method_name = name
        self.predictor_version = version
        self.alleles = alleles
        self.offset = offset
        self.peptide_lengths = set(peptide_lengths) if peptide_lengths else None
        self.kinds = tuple(kinds) if kinds else self.supported_kinds

    def predict_dataframe(self, peptides):
        rows = []
        for peptide in peptides:
            peptide_length = len(peptide)
            if self.peptide_lengths and peptide_length not in self.peptide_lengths:
                continue
            for allele_i, allele in enumerate(self.alleles):
                allele_rank = {
                    "HLA-A*02:01": 0,
                    "HLA-B*07:02": 1,
                }.get(allele, allele_i)
                base = self.offset + peptide_length * 10 + allele_rank
                if "pMHC_affinity" in self.kinds:
                    affinity = 100.0 + base + allele_rank * 100.0
                    rows.append({
                        "peptide": peptide,
                        "allele": allele,
                        "kind": "pMHC_affinity",
                        "value": affinity,
                        "score": 1.0 / affinity,
                        "percentile_rank": affinity / 100.0,
                        "predictor_name": self.prediction_method_name,
                        "predictor_version": self.predictor_version,
                    })
                if "pMHC_presentation" in self.kinds:
                    score = 0.1 + allele_rank * 0.5 + peptide_length / 1000.0
                    rows.append({
                        "peptide": peptide,
                        "allele": allele,
                        "kind": "pMHC_presentation",
                        "value": score,
                        "score": score,
                        "percentile_rank": 100.0 - score,
                        "predictor_name": self.prediction_method_name,
                        "predictor_version": self.predictor_version,
                    })
        return pd.DataFrame(rows)


class ToyHaplotypeMHCflurryPredictor:
    default_peptide_lengths = [9, 10]
    supported_kinds = ("pMHC_affinity", "pMHC_presentation")

    def __init__(self, alleles):
        self.prediction_method_name = "mhcflurry"
        self.predictor_version = "2.1.1"
        self.alleles = alleles

    def predict_dataframe(self, peptides):
        rows = []
        best_allele = self.alleles[-1]
        for peptide in peptides:
            peptide_length = len(peptide)
            for allele_i, allele in enumerate(self.alleles):
                affinity = 50.0 + peptide_length * 10 + allele_i * 100
                rows.append({
                    "peptide": peptide,
                    "allele": allele,
                    "kind": "pMHC_affinity",
                    "value": affinity,
                    "score": 1.0 / affinity,
                    "percentile_rank": affinity / 100.0,
                    "predictor_name": self.prediction_method_name,
                    "predictor_version": self.predictor_version,
                })
            rows.append({
                "peptide": peptide,
                "allele": best_allele,
                "kind": "pMHC_presentation",
                "value": 0.9,
                "score": 0.9,
                "percentile_rank": 0.5,
                "predictor_name": self.prediction_method_name,
                "predictor_version": self.predictor_version,
            })
        return pd.DataFrame(rows)


def _sort_predictions(df):
    cols = [
        "source_sequence_name", "peptide", "allele", "kind",
        "prediction_method_name", "predictor_version",
    ]
    return (
        df.sort_values(cols)
        .reset_index(drop=True)
        .loc[:, sorted(df.columns)]
    )


def _sort_wide(df):
    cols = [
        c for c in ["source_sequence_name", "peptide", "allele"]
        if c in df.columns
    ]
    return (
        df.sort_values(cols)
        .reset_index(drop=True)
        .loc[:, sorted(df.columns)]
    )


def _without_run_name(df):
    return df.drop(columns=["prediction_run_name"], errors="ignore")


def _simple_result(method="netmhcpan", peptide="SIINFEKLA", allele="HLA-A*02:01"):
    df = pd.DataFrame([{
        "source_sequence_name": "pep1",
        "peptide": peptide,
        "peptide_offset": 0,
        "peptide_length": len(peptide),
        "allele": allele,
        "kind": "pMHC_affinity",
        "value": 100.0,
        "score": 0.9,
        "percentile_rank": 1.0,
        "affinity": 100.0,
        "prediction_method_name": method,
        "predictor_version": "1.0",
    }])
    return TopiaryResult(
        df,
        Metadata(
            form="long",
            models={method: "1.0"},
            sources=[method],
        ),
    )


_CONTEXT_MISMATCH_VALUES = {
    "sample_name": ("sample-a", "sample-b"),
    "source_sequence_name": ("pep1", "pep1-copy"),
    "peptide_offset": (0, 1),
    "peptide_length": (9, 10),
    "n_flank": ("AAA", "BBB"),
    "c_flank": ("CCC", "DDD"),
}


def _mismatched_context_results(column, right_has_column=True):
    left, right = _CONTEXT_MISMATCH_VALUES[column]
    r1 = _simple_result("netmhcpan")
    r2 = _simple_result("mhcflurry")
    r1.df[column] = left
    if right_has_column:
        r2.df[column] = right
    elif column in r2.df.columns:
        r2.df = r2.df.drop(columns=[column])
    return r1, r2


def _input_pair(tmp_path, input_type, r1, r2):
    if input_type == "dataframe":
        return [r1.df, r2.df]
    if input_type == "result":
        return [r1, r2]
    if input_type == "roundtrip":
        r1_path = tmp_path / "netmhcpan.tsv"
        r2_path = tmp_path / "mhcflurry.csv"
        r1.to_tsv(r1_path)
        r2.to_csv(r2_path)
        return [read_tsv(r1_path), read_csv(r2_path)]
    raise ValueError(f"unknown input type: {input_type}")


def _grid_peptides():
    return {"pep9": "SIINFEKLA", "pep10": "SIINFEKLAA"}


def _grid_alleles():
    return ["HLA-A*02:01", "HLA-B*07:02"]


def test_combine_separate_predictor_runs_matches_combined_run():
    peptides = {"pep1": "SIINFEKLA", "pep2": "ELAGIGILT"}
    alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    netmhcpan = ToyAffinityPredictor("netmhcpan", "4.1b", alleles, offset=100)
    mhcflurry = ToyAffinityPredictor("mhcflurry", "2.1.1", alleles, offset=200)

    direct = TopiaryPredictor(
        models=[netmhcpan, mhcflurry]
    ).predict_from_named_peptides(peptides)
    net_only = TopiaryPredictor(models=netmhcpan).predict_from_named_peptides(peptides)
    flurry_only = TopiaryPredictor(models=mhcflurry).predict_from_named_peptides(peptides)

    combined = combine_predictions([net_only, flurry_only])

    pd.testing.assert_frame_equal(
        _sort_predictions(combined.df),
        _sort_predictions(direct),
    )
    pd.testing.assert_frame_equal(
        _sort_wide(to_wide(combined.df)),
        _sort_wide(to_wide(direct)),
    )
    assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
    assert "kind_support" not in combined.extra


def test_combine_same_method_split_by_allele_and_length_matches_direct_run():
    peptides = _grid_peptides()
    alleles = _grid_alleles()
    direct = TopiaryPredictor(
        models=ToyGridPredictor("netmhcpan", "4.1b", alleles, offset=10)
    ).predict_from_named_peptides(peptides)
    split_results = []
    for allele in alleles:
        for peptide_length in [8, 9, 10]:
            split_results.append(
                TopiaryPredictor(
                    models=ToyGridPredictor(
                        "netmhcpan", "4.1b", [allele], offset=10,
                        peptide_lengths=[peptide_length],
                    ),
                    name=f"netmhcpan_{allele}_len{peptide_length}",
                ).predict_from_named_peptides(peptides)
            )

    combined = combine_predictions(split_results)

    pd.testing.assert_frame_equal(
        _sort_predictions(_without_run_name(combined.df)),
        _sort_predictions(direct),
    )
    pd.testing.assert_frame_equal(
        _sort_wide(to_wide(combined.df)),
        _sort_wide(to_wide(direct)),
    )
    assert combined.models == {"netmhcpan": "4.1b"}
    assert set(combined.df["prediction_run_name"]) == {
        f"netmhcpan_{allele}_len{peptide_length}"
        for allele in alleles
        for peptide_length in [9, 10]
    }


def test_combine_multi_method_inputs_split_by_allele_match_direct_run():
    peptides = _grid_peptides()
    alleles = _grid_alleles()
    direct_models = [
        ToyGridPredictor("netmhcpan", "4.1b", alleles, offset=10),
        ToyGridPredictor("mhcflurry", "2.1.1", alleles, offset=20),
        ToyGridPredictor(
            "pepsickle", "0.2", alleles, offset=30,
            kinds=["pMHC_presentation"],
        ),
    ]
    direct = TopiaryPredictor(models=direct_models).predict_from_named_peptides(
        peptides
    )
    split_results = []
    for allele in alleles:
        split_results.append(
            TopiaryPredictor(models=[
                ToyGridPredictor("netmhcpan", "4.1b", [allele], offset=10),
                ToyGridPredictor("mhcflurry", "2.1.1", [allele], offset=20),
                ToyGridPredictor(
                    "pepsickle", "0.2", [allele], offset=30,
                    kinds=["pMHC_presentation"],
                ),
            ]).predict_from_named_peptides(peptides)
        )

    combined = combine_predictions(split_results)

    pd.testing.assert_frame_equal(
        _sort_predictions(combined.df),
        _sort_predictions(direct),
    )
    assert combined.models == {
        "netmhcpan": "4.1b",
        "mhcflurry": "2.1.1",
        "pepsickle": "0.2",
    }


def test_combined_split_grid_supports_best_ba_and_el_allele_aggregation():
    peptides = _grid_peptides()
    alleles = _grid_alleles()
    split_results = [
        TopiaryPredictor(
            models=ToyGridPredictor(
                "netmhcpan", "4.1b", [allele], offset=10,
                peptide_lengths=[peptide_length],
            ),
            name=f"netmhcpan_{allele}_len{peptide_length}",
        ).predict_from_named_peptides(peptides)
        for allele in alleles
        for peptide_length in [8, 9, 10]
    ]
    combined = combine_predictions(split_results)
    ctx = EvalContext(combined.df)

    best_ba_allele = Affinity["netmhcpan"].best_value_allele.eval(ctx)
    best_el_allele = Presentation["netmhcpan"].best_score_allele.eval(ctx)

    for source_name, peptide in peptides.items():
        for allele in alleles:
            key = (source_name, peptide, 0, allele)
            assert best_ba_allele.loc[key] == "HLA-A*02:01"
            assert best_el_allele.loc[key] == "HLA-B*07:02"


def test_combine_haplotype_style_presentation_uses_partial_coverage():
    peptides = _grid_peptides()
    alleles = _grid_alleles()
    netmhcpan_rows = TopiaryPredictor(
        models=ToyGridPredictor("netmhcpan", "4.1b", alleles, offset=10),
        name="netmhcpan_all",
    ).predict_from_named_peptides(peptides)
    mhcflurry_rows = TopiaryPredictor(
        models=ToyHaplotypeMHCflurryPredictor(alleles),
        name="mhcflurry_haplotype",
    ).predict_from_named_peptides(peptides)

    with pytest.raises(ValueError, match="coverage='complete'"):
        combine_predictions([netmhcpan_rows, mhcflurry_rows])

    combined = combine_predictions(
        [netmhcpan_rows, mhcflurry_rows],
        coverage="partial",
    )
    ctx = EvalContext(combined.df)
    best_flurry_allele = Presentation["mhcflurry"].best_score_allele.eval(ctx)

    for source_name, peptide in peptides.items():
        for allele in alleles:
            key = (source_name, peptide, 0, allele)
            assert best_flurry_allele.loc[key] == "HLA-B*07:02"

    wide = to_wide(combined.df)
    assert len(wide) == len(peptides) * len(alleles)
    assert not wide["mhcflurry_affinity_value"].isna().any()
    assert wide.loc[
        wide["allele"] == "HLA-A*02:01",
        "mhcflurry_presentation_score",
    ].isna().all()
    assert not wide.loc[
        wide["allele"] == "HLA-B*07:02",
        "mhcflurry_presentation_score",
    ].isna().any()


def test_topiary_predictor_name_adds_run_provenance(tmp_path):
    peptides = {"pep1": "SIINFEKLA"}
    run_name = "netmhcpan_A0201_len9"
    predictor = TopiaryPredictor(
        models=ToyGridPredictor(
            "netmhcpan", "4.1b", ["HLA-A*02:01"], offset=10,
            peptide_lengths=[9],
        ),
        name=run_name,
    )

    df = predictor.predict_from_named_peptides(peptides)

    assert set(df["prediction_run_name"]) == {run_name}
    assert set(df["prediction_method_name"]) == {"netmhcpan"}

    path = tmp_path / "named-run.tsv"
    TopiaryResult(df).to_tsv(path)
    roundtripped = read_tsv(path)

    assert set(roundtripped.df["prediction_run_name"]) == {run_name}
    assert roundtripped.models == {"netmhcpan": "4.1b"}


def test_prediction_run_name_does_not_split_wide_rows():
    peptides = _grid_peptides()
    alleles = _grid_alleles()
    split_results = [
        TopiaryPredictor(
            models=ToyGridPredictor(
                "netmhcpan", "4.1b", [allele], offset=10,
                peptide_lengths=[len(peptide)],
            ),
            name=f"netmhcpan_{allele}_{source_name}",
        ).predict_from_named_peptides({source_name: peptide})
        for source_name, peptide in peptides.items()
        for allele in alleles
    ]

    combined = combine_predictions(split_results)
    wide = to_wide(combined.df)

    assert "prediction_run_name" not in wide.columns
    assert len(wide) == len(peptides) * len(alleles)
    assert not wide["netmhcpan_affinity_value"].isna().any()
    assert not wide["netmhcpan_presentation_score"].isna().any()


def test_combine_rejects_overlapping_named_shards():
    peptides = {"pep1": "SIINFEKLA"}
    shard_a = TopiaryPredictor(
        models=ToyGridPredictor(
            "netmhcpan", "4.1b", ["HLA-A*02:01"], offset=10,
        ),
        name="netmhcpan_A0201_first",
    ).predict_from_named_peptides(peptides)
    shard_b = TopiaryPredictor(
        models=ToyGridPredictor(
            "netmhcpan", "4.1b", ["HLA-A*02:01"], offset=20,
        ),
        name="netmhcpan_A0201_second",
    ).predict_from_named_peptides(peptides)

    with pytest.raises(ValueError, match="duplicate predictions"):
        combine_predictions([shard_a, shard_b])


def test_combine_roundtripped_topiary_results(tmp_path):
    peptides = {"pep1": "SIINFEKLA", "pep2": "ELAGIGILT"}
    alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    netmhcpan = ToyAffinityPredictor("netmhcpan", "4.1b", alleles, offset=100)
    mhcflurry = ToyAffinityPredictor("mhcflurry", "2.1.1", alleles, offset=200)

    direct = TopiaryPredictor(
        models=[netmhcpan, mhcflurry]
    ).predict_from_named_peptides(peptides)
    net_only = TopiaryResult(
        TopiaryPredictor(models=netmhcpan).predict_from_named_peptides(peptides)
    )
    flurry_only = TopiaryResult(
        TopiaryPredictor(models=mhcflurry).predict_from_named_peptides(peptides)
    )

    net_path = tmp_path / "netmhcpan.tsv"
    flurry_path = tmp_path / "mhcflurry.csv"
    net_only.to_tsv(net_path)
    flurry_only.to_csv(flurry_path)

    combined = combine_predictions([
        read_tsv(net_path),
        read_csv(flurry_path),
    ])

    assert "source" not in combined.df.columns
    assert combined.sources == ["netmhcpan.tsv", "mhcflurry.csv"]
    pd.testing.assert_frame_equal(
        _sort_predictions(combined.df),
        _sort_predictions(direct),
    )
    pd.testing.assert_frame_equal(
        _sort_wide(to_wide(combined.df)),
        _sort_wide(to_wide(direct)),
    )
    assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
    assert "kind_support" not in combined.extra


def test_combine_accepts_wide_topiary_results():
    peptides = {"pep1": "SIINFEKLA", "pep2": "ELAGIGILT"}
    alleles = ["HLA-A*02:01", "HLA-B*07:02"]
    netmhcpan = ToyAffinityPredictor("netmhcpan", "4.1b", alleles, offset=100)
    mhcflurry = ToyAffinityPredictor("mhcflurry", "2.1.1", alleles, offset=200)

    direct = TopiaryPredictor(
        models=[netmhcpan, mhcflurry]
    ).predict_from_named_peptides(peptides)
    net_wide = TopiaryResult(
        TopiaryPredictor(models=netmhcpan).predict_from_named_peptides(peptides)
    ).to_wide()
    flurry_wide = TopiaryResult(
        TopiaryPredictor(models=mhcflurry).predict_from_named_peptides(peptides)
    ).to_wide()

    combined = combine_predictions([net_wide, flurry_wide])

    assert combined.form == "long"
    pd.testing.assert_frame_equal(
        _sort_predictions(combined.df),
        _sort_predictions(direct),
    )


def test_combine_ignores_empty_bare_dataframes():
    result = _simple_result("netmhcpan")

    combined = combine_predictions([pd.DataFrame(), result])

    assert len(combined) == 1
    assert combined.models == {"netmhcpan": "1.0"}


def test_combine_recomputes_models_from_combined_rows():
    peptides = {"pep1": "SIINFEKLA"}
    alleles = ["HLA-A*02:01"]
    netmhcpan = ToyAffinityPredictor("netmhcpan", "4.1b", alleles, offset=100)
    mhcflurry = ToyAffinityPredictor("mhcflurry", "2.1.1", alleles, offset=200)
    direct = TopiaryPredictor(
        models=[netmhcpan, mhcflurry]
    ).predict_from_named_peptides(peptides)

    stale_models = {
        "netmhcpan": "4.1b",
        "mhcflurry": "2.1.1",
        "old_model": "0.1",
    }
    net_only = TopiaryResult(
        direct[direct["prediction_method_name"] == "netmhcpan"],
        models=stale_models,
    )
    flurry_only = TopiaryResult(
        direct[direct["prediction_method_name"] == "mhcflurry"],
        models=stale_models,
    )

    combined = combine_predictions([net_only, flurry_only])

    assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
    assert combined.metadata.models == combined.models


def test_combine_fills_missing_row_versions_from_observed_metadata():
    net_only = _simple_result("netmhcpan")
    net_df = net_only.df.drop(columns=["predictor_version"])
    net_df.attrs["topiary_models"] = {
        "netmhcpan": "4.1b",
        "old_model": "0.1",
    }

    flurry_only = _simple_result("mhcflurry")
    flurry_df = flurry_only.df.assign(predictor_version="")
    flurry_df.attrs["topiary_models"] = {"mhcflurry": "2.1.1"}

    assert TopiaryResult(flurry_df).models == {"mhcflurry": "2.1.1"}

    combined = combine_predictions([net_df, flurry_df])

    assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}


def test_combine_rejects_different_identity_sets():
    r1 = _simple_result("netmhcpan", peptide="SIINFEKLA")
    r2 = _simple_result("mhcflurry", peptide="ELAGIGILT")

    with pytest.raises(ValueError, match="coverage='complete'"):
        combine_predictions([r1, r2])


def test_combine_rejects_incomplete_method_coverage_within_one_input():
    method_a = pd.concat(
        [
            _simple_result("netmhcpan", peptide="SIINFEKLA").df,
        ],
        ignore_index=True,
    )
    method_b = pd.concat(
        [
            _simple_result("mhcflurry", peptide="SIINFEKLA").df,
            _simple_result("mhcflurry", peptide="ELAGIGILT").df,
        ],
        ignore_index=True,
    )
    result = TopiaryResult(pd.concat([method_a, method_b], ignore_index=True))

    with pytest.raises(ValueError, match="coverage='complete'"):
        combine_predictions([result])


def test_combine_partial_coverage_allows_sparse_union():
    r1 = _simple_result("netmhcpan", peptide="SIINFEKLA")
    r2 = _simple_result("mhcflurry", peptide="ELAGIGILT")

    combined = combine_predictions([r1, r2], coverage="partial")

    assert len(combined) == 2


def test_combine_rejects_unknown_coverage_mode():
    with pytest.raises(ValueError, match="coverage"):
        combine_predictions([_simple_result("netmhcpan")], coverage="loose")


@pytest.mark.parametrize("column", sorted(_CONTEXT_MISMATCH_VALUES))
@pytest.mark.parametrize("input_type", ["dataframe", "result", "roundtrip"])
@pytest.mark.parametrize("right_has_column", [True, False])
def test_combine_rejects_mismatched_context_columns(
    tmp_path, column, input_type, right_has_column,
):
    r1, r2 = _mismatched_context_results(column, right_has_column)
    inputs = _input_pair(tmp_path, input_type, r1, r2)

    with pytest.raises(ValueError, match=column):
        combine_predictions(inputs)


def test_combine_treats_null_identity_keys_as_equal():
    r1 = _simple_result("netmhcpan", allele=pd.NA)
    r2 = _simple_result("mhcflurry", allele=pd.NA)

    combined = combine_predictions([r1, r2])

    assert len(combined) == 2


def test_combine_rejects_duplicate_prediction_methods():
    r1 = _simple_result("netmhcpan")
    r2 = _simple_result("netmhcpan")

    with pytest.raises(ValueError, match="duplicate predictions"):
        combine_predictions([r1, r2])


def test_combine_allows_same_method_across_samples():
    sample_a = _simple_result("netmhcpan")
    sample_a.df["sample_name"] = "sample-a"
    sample_b = _simple_result("netmhcpan")
    sample_b.df["sample_name"] = "sample-b"
    sample_b.df["value"] = 1000.0
    sample_b.df["affinity"] = 1000.0

    combined = combine_predictions([sample_a, sample_b])
    ctx = EvalContext(combined.df)
    scores = Affinity["netmhcpan"].value.eval(ctx)
    filtered = combined.filter_by("affinity <= 500")
    wide = to_wide(combined.df)

    sample_a_key = (
        "sample-a", "pep1", "SIINFEKLA", 0, "HLA-A*02:01",
    )
    sample_b_key = (
        "sample-b", "pep1", "SIINFEKLA", 0, "HLA-A*02:01",
    )

    assert len(combined) == 2
    assert ctx.group_keys == [
        "sample_name", "source_sequence_name", "peptide",
        "peptide_offset", "allele",
    ]
    assert scores.loc[sample_a_key] == 100.0
    assert scores.loc[sample_b_key] == 1000.0
    assert filtered.df["sample_name"].tolist() == ["sample-a"]
    assert len(wide) == 2
    assert set(wide["sample_name"]) == {"sample-a", "sample-b"}
    assert not wide["netmhcpan_affinity_value"].isna().any()


def test_result_combine_predictions_convenience():
    net_only = _simple_result("netmhcpan")
    flurry_only = _simple_result("mhcflurry")

    combined = net_only.combine_predictions(flurry_only)

    assert set(combined.df["prediction_method_name"]) == {
        "netmhcpan", "mhcflurry",
    }


def test_null_sample_name_is_not_a_ranking_group_key():
    result = _simple_result("netmhcpan")
    result.df["sample_name"] = pd.NA

    ctx = EvalContext(result.df)
    filtered = result.filter_by("affinity <= 500")

    assert "sample_name" not in ctx.group_keys
    assert len(filtered.df) == 1


def test_combine_ignores_legacy_kind_support_metadata():
    r1 = _simple_result("netmhcpan")
    r1.extra["kind_support"] = "not a mapping"
    r2 = _simple_result("mhcflurry")

    combined = combine_predictions([r1, r2])

    assert "kind_support" not in combined.extra


def test_topiary_result_reads_predictor_model_attrs():
    peptides = {"pep1": "SIINFEKLA"}
    predictor = ToyAffinityPredictor(
        "netmhcpan", "4.1b", ["HLA-A*02:01"], offset=100,
    )
    df = TopiaryPredictor(models=predictor).predict_from_named_peptides(peptides)

    result = TopiaryResult(df)

    assert result.models == {"netmhcpan": "4.1b"}
    assert result.extra == {}


def test_topiary_result_model_attrs_filtered_to_observed_rows():
    peptides = {"pep1": "SIINFEKLA"}
    alleles = ["HLA-A*02:01"]
    df = TopiaryPredictor(
        models=[
            ToyAffinityPredictor("netmhcpan", "4.1b", alleles, offset=100),
            ToyAffinityPredictor("mhcflurry", "2.1.1", alleles, offset=200),
        ]
    ).predict_from_named_peptides(peptides)
    filtered = df[df["prediction_method_name"] == "netmhcpan"]

    result = TopiaryResult(filtered)

    assert result.models == {"netmhcpan": "4.1b"}
