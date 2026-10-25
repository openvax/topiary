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
    res1 = hist.lookup_occurrences([dict(peptide="SIINFEKL", n_flank="AAA", c_flank="GGG")])
    assert len(res1) == 1
    assert res1.iloc[0]["value"] == 40.0
    assert res1.iloc[0]["n_flank"] == "AAA"
    assert res1.iloc[0]["c_flank"] == "GGG"

    # Query with TTT/CCC flank
    res2 = hist.lookup_occurrences([dict(peptide="SIINFEKL", n_flank="TTT", c_flank="CCC")])
    assert len(res2) == 1
    assert res2.iloc[0]["value"] == 80.0

    # Query with unknown flank raises under missing='raise'
    with pytest.raises(CachedPredictorCoverageError):
        hist.lookup_occurrences([dict(peptide="SIINFEKL", n_flank="XXX", c_flank="YYY")])


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


def test_historical_source_does_not_accept_unvalidated_live_fallback():
    model = RandomBindingPredictor(alleles=["HLA-A*02:01"], default_peptide_lengths=[8])
    with pytest.raises(TypeError, match="fallback"):
        HistoricalPredictor(pd.DataFrame([_row()]), fallback=model)


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


def test_historical_source_uses_result_policies_instead_of_live_inference():
    rows = [_row(value=50., affinity=50.), _row(peptide="GILGFVFTL", value=500., affinity=500.)]
    source = HistoricalPredictor(pd.DataFrame(rows))
    with pytest.raises(TypeError, match="sparse source"):
        TopiaryPredictor(models=[source])
    filtered = source.to_result().filter_by("affinity <= 100")
    assert filtered.df.peptide.tolist() == ["SIINFEKL"]
    assert filtered.df.value.tolist() == [50.]


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


def test_provenance_and_comparators_are_not_coalesced():
    rows = [_row(source_sequence_name='p1', wt_value=20.),
            _row(source_sequence_name='p2', wt_value=30.)]
    source = HistoricalPredictor(pd.DataFrame(rows))
    assert source.df.source_sequence_name.tolist() == ['p1', 'p2']
    assert source.df.wt_value.tolist() == [20., 30.]
    with pytest.raises(ValueError, match='conflicting'):
        HistoricalPredictor(pd.DataFrame([rows[0], dict(rows[0], wt_value=99.)]))


@pytest.mark.parametrize('suffix', ['.tsv', '.csv', '.tsv.gz', '.csv.gz', '.parquet'])
def test_typed_records_and_result_metadata_survive_files(tmp_path, suffix):
    from topiary import TopiaryResult
    record = _row(predictor_version='001', n_flank='', c_flank=None,
                  measurement_context={'policy': '001', 'nested': [{'units': ['a', 'b']}]},
                  annotations={'labels': ['x', 'y']})
    original = HistoricalPredictor(TopiaryResult(pd.DataFrame([record]),
                                  sources=['original-report'], extra={'audit': {'revision': 1}}))
    path = tmp_path / ('archive' + suffix)
    writer = original.to_parquet if suffix == '.parquet' else original.to_tsv if '.tsv' in suffix else original.to_csv
    writer(path)
    restored = HistoricalPredictor.from_table(path)
    row = restored.df.iloc[0]
    assert row.predictor_version == '001'
    assert row.measurement_context == record['measurement_context']
    assert row.annotations == record['annotations']
    assert row.n_flank == '' and pd.isna(row.c_flank)
    assert row.historical_source_id == original.df.iloc[0].historical_source_id
    assert restored.to_result().sources[0] == 'original-report'
    assert restored.to_result().extra == {'audit': {'revision': 1}}


def test_exact_occurrences_do_not_cross_join_repeated_peptides():
    rows = [_row(source_sequence_name='p1', peptide_offset=3, n_flank='AAA', c_flank='GGG', value=10., prediction_flanks_supplied=True),
            _row(source_sequence_name='p2', peptide_offset=3, n_flank='TTT', c_flank='CCC', value=20., prediction_flanks_supplied=True)]
    source = HistoricalPredictor(pd.DataFrame(rows), missing='omit')
    queries = [dict(rows[0]), dict(rows[1]), dict(rows[0], source_sequence_name='other')]
    result = source.lookup_occurrences(queries)
    assert result.historical_query_index.tolist() == [0, 1]
    assert result.source_sequence_name.tolist() == ['p1', 'p2']
    assert result.value.tolist() == [10., 20.]
    repeated = source.lookup_occurrences([queries[0], queries[0]])
    assert repeated.historical_query_index.tolist() == [0, 1]


@pytest.mark.parametrize('change', [dict(sample_name='other'), dict(peptide_offset=4),
                                   dict(source_sequence_name='other'), dict(n_flank='CCC'),
                                   dict(allele_set='HLA-A*02:01')])
def test_scoped_queries_refuse_other_occurrences_and_genotypes(change):
    row = _row(sample_name='patient', source_sequence_name='protein', peptide_offset=3,
               allele_set='HLA-A*02:01,HLA-B*07:02', prediction_mhc_dependence='haplotype',
               prediction_flanks_supplied=True)
    source = HistoricalPredictor(pd.DataFrame([row]))
    assert len(source.lookup_occurrences([row])) == 1
    with pytest.raises(CachedPredictorCoverageError):
        source.lookup_occurrences([dict(row, **change)])


def test_selection_keeps_allele_free_measurements_and_exact_genotypes():
    rows = [_row(n_flank=None, c_flank=None),
            _row(allele='', kind='antigen_processing', n_flank=None, c_flank=None, prediction_mhc_dependence='none'),
            _row(kind='pMHC_presentation', allele_set='HLA-A*02:01,HLA-B*07:02', prediction_mhc_dependence='haplotype', n_flank=None, c_flank=None)]
    source = HistoricalPredictor(pd.DataFrame(rows), alleles=['HLA-A*02:01'])
    assert set(source.predict_peptides_dataframe(['SIINFEKL']).kind) == {'pMHC_affinity', 'antigen_processing'}
    source.alleles = ['HLA-A*02:01', 'HLA-B*07:02']
    assert len(source.predict_peptides_dataframe(['SIINFEKL'])) == 3


def test_nested_input_and_returned_cells_cannot_mutate_the_source():
    context = {'policy': {'values': [1, 2]}}
    frame = pd.DataFrame([_row(measurement_context=context)])
    source = HistoricalPredictor(frame)
    context['policy']['values'].append(3)
    copy = source.df
    copy.iloc[0].measurement_context['policy']['values'].append(4)
    assert source.df.iloc[0].measurement_context == {'policy': {'values': [1, 2]}}


def test_conflicts_compare_exact_values_missing_cells_and_duplicate_indices():
    rows = [_row(value=1.), _row(value=1. + 1e-10)]
    frame = pd.DataFrame(rows, index=[7, 7])
    assert len(conflicting_historical_predictions(frame)) == 2
    with pytest.raises(ValueError, match='conflicting'):
        HistoricalPredictor(frame)
    with pytest.raises(ValueError, match='conflicting'):
        HistoricalPredictor(pd.DataFrame([_row(wt_value=None), _row(wt_value=1.)]))


def test_mixed_known_unknown_versions_do_not_advertise_a_single_version():
    source = HistoricalPredictor(pd.DataFrame([_row(), _row(peptide='AERMGFTVV', predictor_version='1')]))
    assert source.predictor_version is None
    assert source.predictor_versions == ['1']


@pytest.mark.parametrize('column', ['source_row', 'source_label', 'gene_id', 'transcript_id'])
def test_explicit_source_identifiers_scope_occurrence_lookup(column):
    row = _row(**{column: 7 if column == 'source_row' else 'original'})
    source = HistoricalPredictor(pd.DataFrame([row]))
    assert len(source.lookup_occurrences([row])) == 1
    with pytest.raises(CachedPredictorCoverageError):
        source.lookup_occurrences([dict(row, **{column: 8 if column == 'source_row' else 'other'})])
    query = dict(row)
    del query[column]
    with pytest.raises(CachedPredictorCoverageError):
        source.lookup_occurrences([query])


@pytest.mark.parametrize('suffix', ['.tsv', '.csv', '.parquet'])
def test_explicit_source_id_cannot_relabel_saved_records(tmp_path, suffix):
    source = HistoricalPredictor(pd.DataFrame([_row()]), source_id='original')
    path = tmp_path / ('source' + suffix)
    getattr(source, 'to_' + suffix[1:])(path)
    assert HistoricalPredictor(path, source_id='original').df.historical_source_id.eq('original').all()
    with pytest.raises(ValueError, match='cannot relabel'):
        HistoricalPredictor(path, source_id='other')


def test_empty_source_and_empty_queries_keep_schema():
    source = HistoricalPredictor(missing='omit')
    assert source.to_dataframe(selected=True).empty
    assert source.lookup_occurrences([]).empty
    assert 'historical_query_index' in source.lookup_occurrences([])
    assert source.lookup_occurrences([{'peptide': 'SIINFEKL'}]).empty
    assert source.predict_peptides_dataframe(['SIINFEKL']).empty
    assert source.predictor_version is None
    assert source.alleles == []
    with pytest.raises(CachedPredictorCoverageError):
        HistoricalPredictor().lookup_occurrences([{'peptide': 'SIINFEKL'}])


@pytest.mark.parametrize('column', ['value', 'wt_score', 'self_nearest_value'])
@pytest.mark.parametrize('bad', ['50', 'invalid', True])
def test_measurements_are_numeric_without_silent_coercion(column, bad):
    with pytest.raises(ValueError, match='numeric measurements'):
        HistoricalPredictor(pd.DataFrame([_row(**{column: bad})]))


def test_selected_export_is_detached_and_does_not_change_backing_source():
    source = HistoricalPredictor(pd.DataFrame([_row(), _row(allele='HLA-B*07:02')]), alleles=['HLA-A*02:01'])
    selected = source.to_dataframe(selected=True)
    assert selected.allele.tolist() == ['HLA-A*02:01']
    selected['value'] = 1.
    assert source.df.value.tolist() == [50., 50.]
    assert len(source.to_result().df) == 2


@pytest.mark.parametrize('contexts', [(False, 0), (True, 1), ({'scale': False}, {'scale': 0}), ([True], [1])])
def test_typed_context_conflicts_cannot_collapse(contexts):
    rows = pd.DataFrame([_row(measurement_context=context) for context in contexts])
    assert len(conflicting_historical_predictions(rows)) == 2
    with pytest.raises(ValueError, match='conflicting'):
        HistoricalPredictor(rows)


@pytest.mark.parametrize('annotations', [(False, 0), (True, 1), ({'flag': False}, {'flag': 0}), ([True], [1])])
def test_typed_annotation_identity_survives_construction_and_lookup(annotations):
    rows = [_row(annotation=value, value=10. + index) for index, value in enumerate(annotations)]
    source = HistoricalPredictor(pd.DataFrame(rows))
    assert len(source.df) == 2
    queried = source.lookup_occurrences(rows)
    assert queried.historical_query_index.tolist() == [0, 1]
    assert queried.value.tolist() == [10., 11.]
