"""Producer hypotheses stay comparison evidence rather than new candidates."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from topiary import combine_sources, protein_evidence_view, read_isovar_hypotheses
from .test_twin_conformance import ISOVAR_HYPOTHESIS_INPUT_TWINS


FIXTURE = Path(__file__).parent / "data" / "isovar_hypotheses.json"


def hypothesis_export():
    return json.loads(FIXTURE.read_text())


def read_both(export, tmp_path):
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps(export))
    return [reader(export, path) for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS]


def test_reader_twins_keep_synonymous_alternative_and_partial_observations(tmp_path):
    export = hypothesis_export()
    original = deepcopy(export)
    outputs = read_both(export, tmp_path)
    for result in outputs:
        frame = result.df
        assert frame.protein_hypothesis_sequence.tolist() == ["MAQG", "MAQG", "MAQD", "AQG"]
        assert frame.isovar_rank.tolist() == [1, 1, 2, 3]
        assert frame.translation_id.nunique() == 4
        assert frame.nucleotide_sequence_id.nunique() == 4
        assert frame.isovar_protein_sequence_id.nunique() == 3
        assert frame.protein_sequence.iloc[:3].tolist() == ["MAQG", "MAQG", "MAQD"]
        assert pd.isna(frame.protein_sequence.iloc[3])
        assert frame.protein_segments.tolist() == [4, 4, 2, 1]
        assert sorted(frame.translation_segments.iloc[:2]) == [1, 3]
        assert result.extra["isovar_hypotheses"] == export
        combined = combine_sources({"isovar": result})
        assert combined.df.candidate_id.isna().all()
        assert len(protein_evidence_view(combined)) == 2
        assert combined.df.iloc[-1].protein_sequence_id is None
    pd.testing.assert_frame_equal(outputs[0].df, outputs[1].df)
    outputs[0].extra["isovar_hypotheses"]["events"].clear()
    assert outputs[1].extra["isovar_hypotheses"] == original
    assert export == original


@pytest.mark.parametrize("passing", [True, False, None])
@pytest.mark.parametrize("complete", [True, False, None])
def test_filter_and_cap_outcomes_are_preserved_without_selecting(tmp_path, passing, complete):
    export = hypothesis_export()
    event = export["events"][0]
    event["filters"] = {"values": {"support": passing}, "passes_all_filters": passing}
    event["protein_hypotheses_complete"] = complete
    event["protein_sequence_limit"] = 3 if complete is False else None
    for result in read_both(export, tmp_path):
        assert len(result) == 4
        assert result.df.passes_all_filters.tolist() == [passing] * 4
        assert result.df.protein_hypotheses_complete.tolist() == [complete] * 4
        assert result.extra["isovar_hypotheses"] == export


@pytest.mark.parametrize("empty_events", [False, True])
def test_no_hypotheses_retain_event_outcomes_in_metadata(tmp_path, empty_events):
    export = hypothesis_export()
    export["events"][0]["protein_hypotheses"] = []
    if empty_events:
        export["events"] = []
    for result in read_both(export, tmp_path):
        assert result.df.empty
        assert result.extra["isovar_hypotheses"] == export
        assert combine_sources({"empty": result}).df.empty


def test_missing_translation_or_read_id_never_fabricates_identity(tmp_path):
    export = hypothesis_export()
    proteins = export["events"][0]["protein_hypotheses"]
    proteins[0]["translations"] = []
    proteins[1]["translations"][0]["rna_support"]["evidence_set_id"] = None
    for result in read_both(export, tmp_path):
        assert len(result) == 3
        assert pd.isna(result.df.translation_id.iloc[0])
        assert pd.isna(result.df.protein_sequence.iloc[0])
        assert pd.isna(result.df.translation_evidence_set_id.iloc[1])
        assert result.df.translation_segments.iloc[1] == 2


@pytest.mark.parametrize("fault,match", [
    ("schema", "Expected isovar.protein_hypotheses.v1"),
    ("interval", "zero_based_half_open"), ("scope", "evidence_scope"),
    ("event", "Duplicate Isovar event"), ("hypothesis", "Duplicate Isovar hypothesis"),
    ("translation", "Duplicate Isovar translation"), ("rank", "positive integer"),
    ("mutation", "mutation_interval"), ("reference", "Unknown Isovar evidence_set_id"),
    ("evidence_scope", "identity or scope"), ("count", "count"),
])
def test_reader_twins_refuse_ambiguous_or_inconsistent_exports(tmp_path, fault, match):
    export = hypothesis_export()
    event = export["events"][0]
    protein = event["protein_hypotheses"][0]
    if fault == "schema":
        export["schema"] = "isovar.protein_hypotheses.v99"
    elif fault == "interval":
        export["interval_convention"] = "one_based_closed"
    elif fault == "scope":
        export["evidence_scope"] = ["another patient", export["source"]]
    elif fault == "event":
        export["events"].append(deepcopy(event))
    elif fault == "hypothesis":
        event["protein_hypotheses"].append(deepcopy(protein))
    elif fault == "translation":
        protein["translations"].append(deepcopy(protein["translations"][0]))
    elif fault == "rank":
        protein["isovar_rank"] = True
    elif fault == "mutation":
        protein["mutation_interval"] = [0, 100]
    elif fault == "reference":
        export["evidence_sets"].clear()
    else:
        evidence = export["evidence_sets"][protein["rna_support"]["evidence_set_id"]]
        if fault == "evidence_scope":
            evidence["evidence_scope"] = [export["sample_id"], "other reads"]
        else:
            evidence["segments"] = 999
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError, match=match):
            reader(export, path)


def test_reader_does_not_import_isovar_or_invoke_reconstruction(monkeypatch):
    import builtins

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] != "isovar", "comparison import invoked Isovar"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert len(read_isovar_hypotheses(FIXTURE)) == 4


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_unknown_read_identity_does_not_admit_invalid_counts(tmp_path, count):
    export = hypothesis_export()
    support = export["events"][0]["protein_hypotheses"][0]["rna_support"]
    support.update(evidence_set_id=None, segments=count)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError, match="nonnegative integer"):
            reader(export, path)
