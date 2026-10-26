"""A boundary-shortened input is not a declaration of the scanner's window."""

import pandas as pd
import pytest

from topiary import (
    CachedPredictor, CachedPredictorCoverageError, PREDICTION_KEY_COLUMNS,
    TopiaryPredictor, annotate_protein_prediction_context, from_predictions,
    prediction_flank_limits, prediction_matches_protein_context,
    predict_peptide_occurrences,
)
from .test_protein_fallback import PEPTIDE, ProteinModel, filled_cache
from .test_twin_conformance import PROTEIN_SCAN_PRODUCER_TWINS

LIMITS = ("prediction_n_flank_limit", "prediction_c_flank_limit")


@pytest.mark.parametrize("produce", PROTEIN_SCAN_PRODUCER_TWINS)
@pytest.mark.parametrize("n_limit,c_limit", [(5, 5), (0, 5), (5, 0), (0, 0)])
def test_producer_doors_record_limits_and_recompute_boundary_extensions(produce, n_limit, c_limit):
    model = ProteinModel()
    model.n_flank_length, model.c_flank_length = n_limit, c_limit
    original = {"original": "MA" + PEPTIDE + "GG"}
    cache = CachedPredictor(produce(model, original), fallback=model)
    before = cache.to_dataframe()
    assert before[LIMITS[0]].eq(n_limit).all()
    assert before[LIMITS[1]].eq(c_limit).all()
    inputs = dict(original, n="WWMA" + PEPTIDE + "GG", c="MA" + PEPTIDE + "GGWW")
    result = cache.predict_proteins_dataframe(inputs)
    direct = TopiaryPredictor(models=model).predict_from_named_sequences(inputs)
    keys = ["source_sequence_name", "peptide", "allele"]
    pd.testing.assert_frame_equal(
        result.set_index(keys)[["value", "score"]].sort_index(),
        direct.set_index(keys)[["value", "score"]].sort_index())
    target = result[result.peptide == PEPTIDE].set_index("source_sequence_name").value
    assert (target.n != target.original) == bool(n_limit)
    assert (target.c != target.original) == bool(c_limit)
    pd.testing.assert_frame_equal(before, cache.to_dataframe().iloc[:len(before)][before.columns])
    model.scans.clear()
    pd.testing.assert_frame_equal(result, cache.predict_proteins_dataframe(inputs))
    assert not model.scans


@pytest.mark.parametrize("width", [0, 1, 2, 5])
def test_full_window_matches_at_boundary_and_further_inside_protein(width):
    n, c = "WWGMA"[-width:] if width else "", "GGTWW"[:width]
    row = dict(peptide=PEPTIDE, n_flank=n, c_flank=c, prediction_flanks_supplied=True,
               prediction_n_flank_limit=width, prediction_c_flank_limit=width)
    for left, right in ((n, c), ("TT" + n, c + "TT")):
        assert prediction_matches_protein_context(row, left + PEPTIDE + right, len(left))


def test_unknown_limits_do_not_clip_or_invent_boundaries():
    row = dict(peptide=PEPTIDE, n_flank="MA", c_flank="GG", prediction_flanks_supplied=True)
    assert prediction_flank_limits(row) == (None, None)
    assert prediction_matches_protein_context(row, "MA" + PEPTIDE + "GG", 2)
    assert not prediction_matches_protein_context(row, "WWMA" + PEPTIDE + "GG", 4)
    assert not prediction_matches_protein_context(row, "MA" + PEPTIDE + "GGWW", 2)
    # A legacy blank annotation remains unknown, not an asserted terminus.
    row.update(n_flank="", c_flank="", prediction_flanks_supplied=None)
    assert prediction_matches_protein_context(row, "WWMA" + PEPTIDE + "GGWW", 4)


@pytest.mark.parametrize("suffix", ["tsv", "csv", "csv.gz", "parquet"])
def test_window_limits_survive_persistence_and_remain_part_of_identity(tmp_path, suffix):
    model = ProteinModel()
    model.flank_length = 5
    cache, _ = filled_cache(model)
    original = cache.to_dataframe()
    # Same strings with a different declared input policy are distinct keys.
    target = original[original.peptide == PEPTIDE].assign(**dict.fromkeys(LIMITS, 2))
    combined = CachedPredictor(pd.concat([original, target], ignore_index=True))
    assert len(combined.to_dataframe()) == len(original) + 1
    assert set(LIMITS) <= set(PREDICTION_KEY_COLUMNS)
    assert len(set(combined._row_keys_from_frame(combined.to_dataframe()))) == len(original) + 1
    path = tmp_path / f"windows.{suffix}"
    combined.save(path)
    restored = CachedPredictor.from_topiary_output(path)
    pd.testing.assert_frame_equal(restored.to_dataframe(), combined.to_dataframe(), check_dtype=False)
    assert set(map(tuple, restored.to_dataframe()[list(LIMITS)].to_numpy())) == {(5, 5), (2, 2)}


def test_legacy_cache_miss_is_filled_without_rewriting_old_metadata():
    model = ProteinModel()
    model.flank_length = 5
    cache, _ = filled_cache(model)
    legacy = cache.to_dataframe().drop(columns=list(LIMITS))
    offline = CachedPredictor(legacy)
    query = {"extended": "WWMA" + PEPTIDE + "GG"}
    with pytest.raises(CachedPredictorCoverageError, match="Unknown window limits"):
        offline.predict_proteins_dataframe(query)
    online = CachedPredictor(legacy, fallback=model)
    before = online.to_dataframe()
    result = online.predict_proteins_dataframe(query)
    assert result.loc[result.peptide == PEPTIDE, "n_flank"].tolist() == ["WWMA"]
    pd.testing.assert_frame_equal(before, online.to_dataframe().iloc[:len(before)])
    pd.testing.assert_frame_equal(result, online.predict_proteins_dataframe(query))


@pytest.mark.parametrize("bad", [-1, 1.5, True, "False", "inf", "bad"])
def test_invalid_limits_are_rejected_at_public_and_cache_doors(bad):
    row = dict(prediction_n_flank_limit=bad)
    with pytest.raises(ValueError, match="nonnegative integer"):
        prediction_flank_limits(row)
    cache, _ = filled_cache()
    with pytest.raises(ValueError, match="nonnegative integer"):
        CachedPredictor(cache.to_dataframe().assign(**row))


@pytest.mark.parametrize("values", [dict(prediction_n_flank_limit=1), dict(prediction_flanks_supplied=False)])
def test_contradictory_cached_limits_are_rejected(values):
    cache, _ = filled_cache()
    with pytest.raises(ValueError, match="limit|window"):
        CachedPredictor(cache.to_dataframe().assign(**values))


def test_wrong_producer_window_is_rejected_without_cache_mutation():
    cache, model = filled_cache()
    before = cache.to_dataframe()
    model.corrupt = lambda frame: frame.assign(prediction_n_flank_limit=5)
    with pytest.raises(ValueError, match="configured scanner"):
        cache.predict_proteins_dataframe({"p": "WW" + PEPTIDE + "CC"})
    pd.testing.assert_frame_equal(before, cache.to_dataframe())


def test_nested_legacy_cache_does_not_gain_invented_limits():
    model = ProteinModel()
    proteins = {"p": "WWMA" + PEPTIDE + "GGWW"}
    inner = CachedPredictor(from_predictions(model.predict_proteins_dataframe(proteins)))
    outer = CachedPredictor(fallback=inner)
    with pytest.raises(CachedPredictorCoverageError):
        outer.predict_proteins_dataframe(proteins)
    assert inner.to_dataframe()[list(LIMITS)].isna().all().all()


def test_exact_peptide_inputs_are_not_clipped_to_scan_window():
    model = ProteinModel()
    cache = CachedPredictor(fallback=model)
    result = cache.predict_contextual_peptides_dataframe([PEPTIDE], n_flanks=["WWMA"], c_flanks=["GGWW"])
    assert result.n_flank.tolist() == ["WWMA"]
    assert result.c_flank.tolist() == ["GGWW"]
    assert cache.to_dataframe()[list(LIMITS)].isna().all().all()
    assert result.value.tolist() == [732.]
    for column in LIMITS:
        with pytest.raises(ValueError, match="prediction columns"):
            predict_peptide_occurrences(pd.DataFrame([dict(prediction_id="p", peptide=PEPTIDE,
                n_flank="MA", c_flank="GG", **{column: 2})]), model)


def test_annotation_handles_empty_frames_without_mutating_input():
    frame = pd.DataFrame()
    result = annotate_protein_prediction_context(frame, ProteinModel())
    assert result.empty and set(LIMITS) <= set(result)
    assert not len(frame.columns)
