"""Producer hypotheses stay comparison evidence rather than new candidates."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from topiary import combine_sources, protein_evidence_view, read_isovar_hypotheses
from .test_twin_conformance import ISOVAR_HYPOTHESIS_INPUT_TWINS


FIXTURE = Path(__file__).parent / "data" / "isovar_hypotheses.json"


def hypothesis_export(name=None):
    path = FIXTURE if name is None else FIXTURE.parent / "isovar_exports" / (name + ".json")
    return json.loads(path.read_text())


def read_both(export, tmp_path):
    path = tmp_path / "hypotheses.json"
    path.write_text(json.dumps(export))
    return [reader(export, path) for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS]


@pytest.mark.parametrize("name", [None, "protein-v2", "protein-v2-labelled"])
def test_reader_twins_keep_synonymous_alternative_and_partial_observations(tmp_path, name):
    export = hypothesis_export(name)
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
        assert frame.protein_reads.tolist() == frame.protein_segments.tolist()
        assert frame.translation_reads.tolist() == frame.translation_segments.tolist()
        assert sorted(frame.translation_segments.iloc[:2]) == [1, 3]
        assert result.extra["isovar_hypotheses"] == export
        combined = combine_sources({"isovar": result})
        assert combined.df.candidate_id.isna().all()
        assert len(protein_evidence_view(combined)) == 2
        assert pd.isna(combined.df.iloc[-1].protein_sequence_id)
    pd.testing.assert_frame_equal(outputs[0].df, outputs[1].df)
    outputs[0].extra["isovar_hypotheses"]["events"].clear()
    assert outputs[1].extra["isovar_hypotheses"] == original
    assert export == original


@pytest.mark.parametrize("name", ["protein-v2", "protein-v2-labelled"])
def test_new_support_measurements_keep_unknown_and_incomplete_distinct(tmp_path, name):
    export = hypothesis_export(name)
    for result in read_both(export, tmp_path):
        for _, row in result.df.iterrows():
            proteins = export["events"][0]["protein_hypotheses"]
            protein = next(p for p in proteins if p["hypothesis_id"] == row.hypothesis_id)
            translation = next(t for t in protein["translations"] if t["translation_id"] == row.translation_id)
            for prefix, support in (("protein", protein["rna_support"]),
                                    ("translation", translation["rna_support"])):
                for field in ("reads", "fragments", "umis", "cells", "umis_complete", "cells_complete",
                              "unlabeled_reads", "unknown_library_reads"):
                    actual = row[prefix + "_" + field]
                    expected = support[field]
                    assert pd.isna(actual) if expected is None else actual == expected
        if name.endswith("labelled"):
            assert result.df.protein_umis.tolist() == [2, 2, 1, 1]
            assert result.df.protein_umis_complete.tolist() == [False, False, False, True]
            assert result.df.protein_cells_complete.tolist() == [True, True, False, True]


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
    ("translation", "Conflicting Isovar translation"), ("rank", "positive integer"),
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
        conflicting = deepcopy(protein["translations"][0])
        conflicting["nucleotide_sequence"] += "A"
        protein["translations"].append(conflicting)
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


@pytest.mark.parametrize("name", [None, "protein-v2", "protein-v2-labelled"])
def test_reader_twins_coalesce_exact_repeats_without_changing_support(tmp_path, name):
    export = hypothesis_export(name)
    expected = read_isovar_hypotheses(export).df
    proteins = export["events"][0]["protein_hypotheses"]
    translations = proteins[0]["translations"]
    # Nonadjacent repeats, with a different mapping insertion order, still
    # represent the same records. The synonymous distinct ID must survive.
    translations.extend(dict(reversed(list(deepcopy(t).items()))) for t in translations[:])
    translations.append(deepcopy(translations[0]))
    original = deepcopy(export)
    for result in read_both(export, tmp_path):
        pd.testing.assert_frame_equal(result.df, expected)
        assert result.df.translation_id.nunique() == 4
        assert result.extra["isovar_hypotheses"] == original
        assert len(result.extra["isovar_hypotheses"]["events"][0]
                   ["protein_hypotheses"][0]["translations"]) == 5
    assert export == original


@pytest.mark.parametrize("name", [None, "protein-v2", "protein-v2-labelled"])
@pytest.mark.parametrize("field", ["sequence", "support", "reference", "edits", "flag_type"])
def test_reader_twins_reject_conflicts_in_entire_repeated_record(tmp_path, name, field):
    export = hypothesis_export(name)
    translations = export["events"][0]["protein_hypotheses"][0]["translations"]
    conflict = deepcopy(translations[0])
    if field == "sequence":
        conflict["nucleotide_sequence"] += "A"
    elif field == "support":
        conflict["rna_support"]["fragments"] += 1
    elif field == "reference":
        conflict["reference_context"]["transcript_names"] = ["different"]
    elif field == "edits":
        conflict["observed_edits"].append({"origin": "unexplained"})
    else:
        # Python mapping equality alone considers these values equal.
        conflict["starts_at_annotated_start_codon"] = int(conflict["starts_at_annotated_start_codon"])
    translations.append(conflict)
    path = tmp_path / "conflict.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError, match="Conflicting Isovar translation_id"):
            reader(export, path)


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


@pytest.mark.parametrize("bad", [None, {}, [], False, 0, ""])
def test_nonempty_translation_list_requires_real_translations(tmp_path, bad):
    export = hypothesis_export()
    export["events"][0]["protein_hypotheses"][0]["translations"] = [bad]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError, match="translation"):
            reader(export, path)


@pytest.mark.parametrize("fault", [
    "nucleotide_sequence_id", "nucleotide_sequence", "translated_interval", "variant_cdna_interval",
    "protein_hypotheses", "evidence_set", "support", "starts_at_annotated_start_codon",
])
def test_reader_rejects_malformed_comparison_identity_and_evidence(tmp_path, fault):
    export = hypothesis_export()
    event = export["events"][0]
    protein = event["protein_hypotheses"][0]
    translation = protein["translations"][0]
    if fault in {"nucleotide_sequence_id", "nucleotide_sequence"}:
        del translation[fault]
    elif fault in {"translated_interval", "variant_cdna_interval"}:
        translation[fault] = [0, len(translation["nucleotide_sequence"]) + 1]
    elif fault == "protein_hypotheses":
        event[fault] = {}
    elif fault == "evidence_set":
        export["evidence_sets"][protein["rna_support"]["evidence_set_id"]] = []
    elif fault == "support":
        protein["rna_support"] = False
    else:
        translation[fault] = "false"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError):
            reader(export, path)


@pytest.mark.parametrize("field", [
    "protein_hypotheses_complete", "passes_all_filters", "representative", "ends_with_stop_codon",
])
def test_reader_rejects_text_instead_of_boolean_outcomes(tmp_path, field):
    export = hypothesis_export()
    event = export["events"][0]
    target = (event if field == "protein_hypotheses_complete" else
              event["filters"] if field == "passes_all_filters" else event["protein_hypotheses"][0])
    target[field] = "false"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(export))
    for _, reader in ISOVAR_HYPOTHESIS_INPUT_TWINS:
        with pytest.raises(ValueError, match="boolean or null"):
            reader(export, path)
