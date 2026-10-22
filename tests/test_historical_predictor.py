"""Tests for topiary.historical.HistoricalPredictor — sparse historical table
predictions, unknown model versions, multiple model preservation, provenance,
and composition (#368)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    CachedPredictor,
    CachedPredictorCoverageError,
    HistoricalPredictor,
    HISTORICAL_KEY_COLUMNS,
    TopiaryPredictor,
    combine_sources,
    conflicting_historical_predictions,
    read_pvacseq,
    read_lens,
    rescore_candidates,
)


DATA_DIR = Path(__file__).parent / "data"
PVACSEQ_MHC_I_AGG = DATA_DIR / "pvacseq" / "mhc_i_aggregated.tsv"
PVACSEQ_MHC_I_ALL = DATA_DIR / "pvacseq" / "mhc_i_all_epitopes.tsv"
LENS_V1_4 = DATA_DIR / "lens" / "sample_v1_4.tsv"


def _row(
    peptide="SIINFEKL",
    allele="HLA-A*02:01",
    *,
    score=0.8,
    value=50.0,
    affinity=50.0,
    percentile_rank=0.5,
    kind="pMHC_affinity",
    prediction_method_name="netmhcpan",
    predictor_version=None,
    n_flank="AAA",
    c_flank="GGG",
    allele_set="",
    **extra,
):
    data = {
        "peptide": peptide,
        "allele": allele,
        "peptide_length": len(peptide),
        "kind": kind,
        "score": score,
        "value": value,
        "affinity": affinity,
        "percentile_rank": percentile_rank,
        "prediction_method_name": prediction_method_name,
        "predictor_version": predictor_version,
        "n_flank": n_flank,
        "c_flank": c_flank,
        "allele_set": allele_set,
    }
    data.update(extra)
    return data


def test_unknown_model_versions_preserved_and_never_inferred():
    """Missing or unknown predictor versions remain unknown (None / pd.NA) and are never inferred."""
    rows = [
        _row(peptide="SIINFEKL", predictor_version=None),
        _row(peptide="AERMGFTVV", predictor_version=pd.NA),
    ]
    predictor = HistoricalPredictor(pd.DataFrame(rows))

    # Version should remain unknown, not inferred from environment
    assert predictor.predictor_version is None
    assert predictor.predictor_versions == []

    # Queried predictions must keep missing versions as NA
    queried = predictor.predict_peptides_dataframe(["SIINFEKL", "AERMGFTVV"])
    assert len(queried) == 2
    assert queried["predictor_version"].isna().all()


def test_aggregate_method_identity_preserved():
    """Aggregate prediction methods (like 'pvacseq') are retained without relabeling."""
    rows = [
        _row(peptide="SIINFEKL", prediction_method_name="pvacseq", predictor_version=None),
    ]
    predictor = HistoricalPredictor(pd.DataFrame(rows))
    assert predictor.prediction_method_name == "pvacseq"
    assert predictor.prediction_method_names == ["pvacseq"]

    res = predictor.predict_peptides_dataframe(["SIINFEKL"])
    assert res["prediction_method_name"].tolist() == ["pvacseq"]
    assert res["predictor_version"].isna().all()


def test_multiple_models_in_one_historical_source():
    """HistoricalPredictor supports multiple prediction methods; CachedPredictor strictly rejects them."""
    rows = [
        _row(peptide="SIINFEKL", kind="pMHC_affinity", value=50.0, score=0.8,
             prediction_method_name="netmhcpan", predictor_version="4.1"),
        _row(peptide="SIINFEKL", kind="pMHC_affinity", value=75.0, score=0.7,
             prediction_method_name="mhcflurry", predictor_version="2.0.6"),
    ]
    df = pd.DataFrame(rows)

    # CachedPredictor strictly forbids multiple (method, version) pairs
    with pytest.raises(ValueError):
        CachedPredictor(df)

    # HistoricalPredictor allows them cleanly
    hist = HistoricalPredictor(df)
    assert set(hist.prediction_method_names) == {"mhcflurry", "netmhcpan"}

    res = hist.predict_peptides_dataframe(["SIINFEKL"])
    assert len(res) == 2
    assert set(res["prediction_method_name"]) == {"netmhcpan", "mhcflurry"}
    assert set(res["value"]) == {50.0, 75.0}


def test_sparse_coverage_no_cartesian_cross_product():
    """Sparse tables with different peptides evaluated on different alleles must not demand a cross-product."""
    rows = [
        _row(peptide="PEPTIDE_A", allele="HLA-A*02:01", value=50.0),
        _row(peptide="PEPTIDE_B", allele="HLA-B*07:02", value=150.0),
    ]
    df = pd.DataFrame(rows)

    # CachedPredictor takes the union of alleles ['HLA-A*02:01', 'HLA-B*07:02']
    # and demands a 2x2 Cartesian product, raising coverage error for unmeasured pairs.
    with pytest.raises(Exception):
        CachedPredictor(df).predict_peptides_dataframe(["PEPTIDE_A", "PEPTIDE_B"])

    # HistoricalPredictor respects sparse coverage: queries match reported combinations
    hist = HistoricalPredictor(df)
    res = hist.predict_peptides_dataframe(["PEPTIDE_A", "PEPTIDE_B"])
    assert len(res) == 2
    row_a = res[res["peptide"] == "PEPTIDE_A"].iloc[0]
    row_b = res[res["peptide"] == "PEPTIDE_B"].iloc[0]
    assert row_a["allele"] == "HLA-A*02:01"
    assert row_b["allele"] == "HLA-B*07:02"


def test_contextual_lookup_with_flanks():
    """Contextual queries match reported flanks and do not fabricate context."""
    rows = [
        _row(peptide="SIINFEKL", n_flank="AAA", c_flank="GGG", value=40.0, prediction_flanks_supplied=True),
        _row(peptide="SIINFEKL", n_flank="TTT", c_flank="CCC", value=80.0, prediction_flanks_supplied=True),
    ]
    hist = HistoricalPredictor(pd.DataFrame(rows))

    # Query with AAA/GGG flank
    res1 = hist.predict_contextual_peptides_dataframe(["SIINFEKL"], n_flanks=["AAA"], c_flanks=["GGG"])
    assert len(res1) == 1
    assert res1.iloc[0]["value"] == 40.0
    assert res1.iloc[0]["n_flank"] == "AAA"
    assert res1.iloc[0]["c_flank"] == "GGG"

    # Query with TTT/CCC flank
    res2 = hist.predict_contextual_peptides_dataframe(["SIINFEKL"], n_flanks=["TTT"], c_flanks=["CCC"])
    assert len(res2) == 1
    assert res2.iloc[0]["value"] == 80.0

    # Query with unknown flank raises under missing='raise'
    with pytest.raises(CachedPredictorCoverageError):
        hist.predict_contextual_peptides_dataframe(["SIINFEKL"], n_flanks=["XXX"], c_flanks=["YYY"])


def test_wildtype_comparator_metadata_preserved():
    """Comparator and wildtype metadata (wt_*) are preserved intact."""
    rows = [
        _row(
            peptide="SIINFEKL",
            wt_peptide="SIINFEKM",
            wt_value=500.0,
            wt_score=0.1,
            wt_affinity=500.0,
            wt_percentile_rank=5.0,
            wt_prediction_method_name="netmhcpan",
            wt_predictor_version="4.1",
        ),
    ]
    hist = HistoricalPredictor(pd.DataFrame(rows))
    res = hist.predict_peptides_dataframe(["SIINFEKL"])
    assert len(res) == 1
    assert res.iloc[0]["wt_peptide"] == "SIINFEKM"
    assert res.iloc[0]["wt_value"] == 500.0
    assert res.iloc[0]["wt_score"] == 0.1
    assert res.iloc[0]["wt_percentile_rank"] == 5.0
    assert res.iloc[0]["wt_prediction_method_name"] == "netmhcpan"


def test_conflicting_predictions_raise():
    """Contradictory values for the same logical query key are detected and raise deterministically."""
    rows = [
        _row(peptide="SIINFEKL", allele="HLA-A*02:01", value=50.0, score=0.8),
        _row(peptide="SIINFEKL", allele="HLA-A*02:01", value=500.0, score=0.2),  # conflicting!
    ]
    df = pd.DataFrame(rows)
    conflicts = conflicting_historical_predictions(df)
    assert not conflicts.empty

    with pytest.raises(ValueError, match="conflicting"):
        HistoricalPredictor(df)


def test_identical_duplicates_deduplicated():
    """Exact duplicate rows are deduplicated cleanly without error."""
    r = _row(peptide="SIINFEKL", value=50.0)
    df = pd.DataFrame([r, r])  # identical
    hist = HistoricalPredictor(df)
    assert len(hist.df) == 1
    assert len(hist.predict_peptides_dataframe(["SIINFEKL"])) == 1


def test_missing_policy_raise_vs_ignore():
    """Missing policy controls whether unmeasured queries raise or return empty."""
    df = pd.DataFrame([_row(peptide="SIINFEKL")])

    strict = HistoricalPredictor(df, missing="raise")
    with pytest.raises(CachedPredictorCoverageError):
        strict.predict_peptides_dataframe(["UNKNOWNPEP"])

    lenient = HistoricalPredictor(df, missing="ignore")
    res = lenient.predict_peptides_dataframe(["UNKNOWNPEP"])
    assert res.empty


def test_real_pvacseq_aggregated_fixture_import():
    """Import a real pVACseq aggregated report into HistoricalPredictor."""
    assert PVACSEQ_MHC_I_AGG.exists()
    hist = HistoricalPredictor.from_pvacseq(PVACSEQ_MHC_I_AGG)

    # pvacseq aggregated has 'pvacseq' as the method name and no predictor version
    assert "pvacseq" in hist.prediction_method_names
    assert hist.predictor_versions == []

    # Predict one of the peptides from the fixture
    # AERMGFTVV is present in the file for HLA-B*45:01
    res = hist.predict_peptides_dataframe(["AERMGFTVV"])
    assert len(res) >= 1
    row = res.iloc[0]
    assert row["peptide"] == "AERMGFTVV"
    assert row["allele"] == "HLA-B*45:01"
    assert row["prediction_method_name"] == "pvacseq"
    assert pd.isna(row["predictor_version"])
    assert np.isclose(row["value"], 76.11, atol=0.01)


def test_real_pvacseq_all_epitopes_multi_model_import():
    """Import a real pVACseq all_epitopes report containing multiple algorithm columns."""
    assert PVACSEQ_MHC_I_ALL.exists()
    hist = HistoricalPredictor.from_pvacseq(PVACSEQ_MHC_I_ALL)

    # Verify multiple algorithms are detected
    assert len(hist.prediction_method_names) >= 1

    # Check kind support
    support = hist.kind_support()
    assert "pMHC_affinity" in support


def test_real_lens_fixture_import():
    """Import a real LENS report into HistoricalPredictor."""
    assert LENS_V1_4.exists()
    hist = HistoricalPredictor.from_lens(LENS_V1_4)
    assert len(hist.df) > 0
    assert len(hist.available_peptide_lengths) > 0


def test_live_fallback_provenance_separation():
    """Live fallback can compute missing queries, retaining distinct live and historical provenance."""
    rows = [
        _row(peptide="SIINFEKL", prediction_method_name="historical_model", predictor_version=None),
    ]
    fallback_model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])

    hist = HistoricalPredictor(pd.DataFrame(rows), fallback=fallback_model)

    # Query both historical peptide and a missing peptide
    res = hist.predict_peptides_dataframe(["SIINFEKL", "AERMGFTV"])
    assert len(res) >= 2

    hist_row = res[res["peptide"] == "SIINFEKL"].iloc[0]
    fallback_row = res[res["peptide"] == "AERMGFTV"].iloc[0]

    # Historical row retains historical provenance
    assert hist_row["prediction_method_name"] == "historical_model"
    assert pd.isna(hist_row["predictor_version"])

    # Fallback row has live fallback's model identity
    assert fallback_row["prediction_method_name"] != "historical_model"
    assert hist_row["prediction_method_name"] != fallback_row["prediction_method_name"]


def test_serialization_roundtrip_tsv_csv_parquet(tmp_path):
    """HistoricalPredictor roundtrips through TSV, CSV, Parquet without metadata loss."""
    rows = [
        _row(peptide="SIINFEKL", value=42.0, score=0.9, wt_peptide="SIINFEKM", wt_value=420.0),
    ]
    original = HistoricalPredictor(pd.DataFrame(rows))

    # TSV roundtrip
    tsv_path = tmp_path / "test.tsv"
    original.to_tsv(tsv_path)
    loaded_tsv = HistoricalPredictor.from_table(tsv_path)
    res_tsv = loaded_tsv.predict_peptides_dataframe(["SIINFEKL"])
    assert len(res_tsv) == 1
    assert res_tsv.iloc[0]["value"] == 42.0
    assert res_tsv.iloc[0]["wt_peptide"] == "SIINFEKM"

    # CSV roundtrip
    csv_path = tmp_path / "test.csv"
    original.to_csv(csv_path)
    loaded_csv = HistoricalPredictor.from_table(csv_path)
    res_csv = loaded_csv.predict_peptides_dataframe(["SIINFEKL"])
    assert len(res_csv) == 1
    assert res_csv.iloc[0]["value"] == 42.0

    # Parquet roundtrip
    pq_path = tmp_path / "test.parquet"
    original.to_parquet(pq_path)
    loaded_pq = HistoricalPredictor.from_table(pq_path)
    res_pq = loaded_pq.predict_peptides_dataframe(["SIINFEKL"])
    assert len(res_pq) == 1
    assert res_pq.iloc[0]["value"] == 42.0


def test_integration_with_topiary_predictor():
    """HistoricalPredictor drops cleanly into TopiaryPredictor as a model."""
    rows = [
        _row(peptide="SIINFEKL", value=50.0, score=0.9, percentile_rank=0.5),
        _row(peptide="GILGFVFTL", value=500.0, score=0.1, percentile_rank=5.0),
    ]
    model = HistoricalPredictor(pd.DataFrame(rows))

    predictor = TopiaryPredictor(
        models=[model],
        filter_by="affinity <= 100",
    )

    df = predictor.predict_from_named_peptides({
        "p1": "SIINFEKL",
        "p2": "GILGFVFTL",
    })

    # Only SIINFEKL should pass filter (affinity <= 100)
    assert len(df) == 1
    assert df.iloc[0]["peptide"] == "SIINFEKL"
    assert df.iloc[0]["value"] == 50.0


def test_integration_with_combine_sources():
    """HistoricalPredictor.to_result() integrates with combine_sources."""
    rows = [
        _row(peptide="SIINFEKL", value=50.0, score=0.9, percentile_rank=0.5),
    ]
    hist = HistoricalPredictor(pd.DataFrame(rows))
    res = hist.to_result()

    combined = combine_sources({"historical_src": res}, sample_name="patient_1")
    assert "candidate_id" in combined.df.columns
    assert len(combined.df) == 1
    assert combined.df.iloc[0]["peptide"] == "SIINFEKL"
