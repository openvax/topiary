"""Scanner declarations belong to measurements, not a different source row."""

import pandas as pd
import pytest

from topiary import (
    CachedPredictor, PROTEIN_SCAN_METADATA_COLUMNS, TopiaryPredictor,
    combine_sources, from_wide, to_wide,
)
from .test_protein_fallback import ProteinModel, PEPTIDE
from .test_twin_conformance import PROTEIN_SCAN_REPLAY_TWINS, DELIMITED_IO_TWINS


def mixed_scan():
    models = []
    for width in (2, 5):
        model = ProteinModel()
        model.flank_length, model.method = width, f"window{width}"
        models.append(model)
    proteins = {"original": "MA" + PEPTIDE + "GG", "extended": "WWMA" + PEPTIDE + "GG"}
    return TopiaryPredictor(models=models).predict_from_named_sequences(proteins), proteins


@pytest.mark.parametrize("replay", PROTEIN_SCAN_REPLAY_TWINS)
def test_native_scan_metadata_replays_without_fabricated_model_rows(replay):
    frame, proteins = mixed_scan()
    restored = replay(frame)
    keys = ["source_sequence_name", "peptide_offset", "peptide", "allele", "prediction_method_name", "kind"]
    columns = ["value", "score", "n_flank", "c_flank", *PROTEIN_SCAN_METADATA_COLUMNS]
    pd.testing.assert_frame_equal(frame.set_index(keys)[columns].sort_index(),
                                  restored.set_index(keys)[columns].sort_index(), check_dtype=False)
    for method, rows in restored.groupby("prediction_method_name"):
        cached = CachedPredictor(rows).predict_proteins_dataframe(proteins)
        assert len(cached) == len(rows)
        expected = frame[frame.prediction_method_name == method]
        assert sorted(cached.value.tolist()) == sorted(expected.value.tolist())


@pytest.mark.parametrize("suffix,writer,method,reader", DELIMITED_IO_TWINS)
def test_file_replay_retains_each_models_flags_and_limits(tmp_path, suffix, writer, method, reader):
    frame, _ = mixed_scan()
    wide = to_wide(frame)
    target = wide[(wide.peptide == PEPTIDE) & (wide.source_sequence_name == "original")]
    assert len(target) == 1
    assert target.window2_affinity_n_flank_limit.tolist() == [2]
    assert target.window5_affinity_n_flank_limit.tolist() == [5]
    path = tmp_path / f"scan.{suffix}"
    writer(wide, path)
    restored = reader(path).long_df
    assert len(restored) == len(frame)
    for name, rows in restored.groupby("prediction_method_name"):
        assert rows.prediction_n_flank_limit.eq(int(name[-1])).all()
        assert rows.prediction_c_flank_limit.eq(int(name[-1])).all()
        assert rows.prediction_flanks_supplied.eq(True).all()


@pytest.mark.parametrize("flag,limit", [(True, 0), (False, None), (None, None)])
def test_boolean_zero_and_unknown_metadata_survive_without_scores(flag, limit):
    frame, _ = mixed_scan()
    row = frame.iloc[:1].assign(value=None, score=None, affinity=None, percentile_rank=None, n_flank="", c_flank="",
        prediction_flanks_supplied=flag, prediction_n_flank_limit=limit, prediction_c_flank_limit=limit)
    restored = from_wide(to_wide(row))
    assert len(restored) == 1
    assert restored.value.isna().all()
    if flag is None:
        assert restored.prediction_flanks_supplied.isna().all()
    else:
        assert restored.prediction_flanks_supplied.tolist() == [flag]
    if limit is None:
        assert restored.prediction_n_flank_limit.isna().all()
    else:
        assert restored.prediction_n_flank_limit.tolist() == [0]


def test_legacy_flat_metadata_is_not_overwritten_by_missing_model_columns():
    frame, _ = mixed_scan()
    row = frame.iloc[:1]
    wide = to_wide(row.drop(columns=list(PROTEIN_SCAN_METADATA_COLUMNS)))
    for column in PROTEIN_SCAN_METADATA_COLUMNS:
        wide[column] = row[column].iloc[0]
    restored = from_wide(wide)
    for column in PROTEIN_SCAN_METADATA_COLUMNS:
        assert restored[column].tolist() == row[column].tolist()


def test_conflicting_policies_for_one_model_are_not_silently_collapsed():
    frame, _ = mixed_scan()
    target = frame[(frame.peptide == PEPTIDE) & (frame.source_sequence_name == "original")]
    same_model = target.assign(prediction_method_name="one-model")
    with pytest.raises(ValueError, match="Conflicting scanner input policies"):
        to_wide(same_model)
    with pytest.raises(ValueError, match="conflicting predictions"):
        combine_sources({"one-run": same_model}, sample_name="sample")


def test_flat_and_model_specific_metadata_cannot_silently_disagree():
    frame, _ = mixed_scan()
    wide = to_wide(frame.iloc[:1]).assign(prediction_n_flank_limit=999)
    with pytest.raises(ValueError, match="Both flat and model-specific"):
        from_wide(wide)
