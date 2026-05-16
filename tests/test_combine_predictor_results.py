"""Tests for combining separately-run predictor outputs."""

import pandas as pd
import pytest

from topiary import (
    TopiaryPredictor,
    TopiaryResult,
    combine_predictor_results,
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

    combined = combine_predictor_results([net_only, flurry_only])

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

    combined = combine_predictor_results([
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

    combined = combine_predictor_results([net_only, flurry_only])

    assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
    assert combined.metadata.models == combined.models


def test_combine_rejects_different_identity_sets():
    r1 = _simple_result("netmhcpan", peptide="SIINFEKLA")
    r2 = _simple_result("mhcflurry", peptide="ELAGIGILT")

    with pytest.raises(ValueError, match="same .* keys"):
        combine_predictor_results([r1, r2])


def test_combine_treats_null_identity_keys_as_equal():
    r1 = _simple_result("netmhcpan", allele=pd.NA)
    r2 = _simple_result("mhcflurry", allele=pd.NA)

    combined = combine_predictor_results([r1, r2])

    assert len(combined) == 2


def test_combine_rejects_duplicate_prediction_methods():
    r1 = _simple_result("netmhcpan")
    r2 = _simple_result("netmhcpan")

    with pytest.raises(ValueError, match="duplicate prediction method"):
        combine_predictor_results([r1, r2])


def test_combine_ignores_legacy_kind_support_metadata():
    r1 = _simple_result("netmhcpan")
    r1.extra["kind_support"] = "not a mapping"
    r2 = _simple_result("mhcflurry")

    combined = combine_predictor_results([r1, r2])

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
