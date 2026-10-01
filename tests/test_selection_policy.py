"""Saved policy definitions replay the public candidate ranker unchanged."""

from dataclasses import FrozenInstanceError, replace
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from topiary import (
    SelectionPolicy, combine_sources, read_selection_policy, write_selection_policy,
    resolve_selection_policy, rank_with_policy,
)
from .test_candidate_tables import source
from .test_twin_conformance import SELECTION_POLICY_TWINS, POLICY_MAPPING_TWINS


def test_definition_is_immutable_and_file_preserves_every_setting(tmp_path):
    methods = {"affinity": "original"}
    versions = {("affinity", "original"): "1"}
    strata = ["source_label", "candidate_mhc_class"]
    profile = SelectionPolicy(
        "openvax-v1", "1 / affinity.value", filter_by="n_rna_alt >= 5",
        ascending=True, duplicates="worst", strata=strata,
        default_methods=methods, default_versions=versions,
    )
    before = profile.to_dict()
    methods.clear()
    versions.clear()
    strata.clear()
    assert profile.to_dict() == before
    with pytest.raises(FrozenInstanceError):
        profile.score_by = "0"
    with pytest.raises(TypeError):
        profile.default_methods["affinity"] = "other"
    path = tmp_path / "openvax-v1.json"
    write_selection_policy(profile, path)
    restored = read_selection_policy(path)
    assert restored == profile
    assert restored.sha256 == profile.sha256
    assert json.loads(path.read_text()) == before
    with pytest.raises(FileExistsError):
        write_selection_policy(replace(profile, score_by="affinity.value"), path)
    assert read_selection_policy(path) == profile
    assert replace(profile, score_by="affinity.value").sha256 != profile.sha256


@pytest.mark.parametrize("settings", [
    {"name": ""}, {"score_by": ""}, {"score_by": 1}, {"score_by": "(??"},
    {"filter_by": False}, {"ascending": "false"}, {"duplicates": []},
    {"strata": "source_label"}, {"strata": ["x", "x"]}, {"strata": [None]},
    {"default_methods": []}, {"default_methods": {"affinity": ""}},
    {"default_versions": {"affinity": "1"}},
])
def test_invalid_definition_rejected(settings):
    with pytest.raises((ValueError, SyntaxError)):
        SelectionPolicy(**dict({"name": "test-v1", "score_by": "affinity.value"}, **settings))


@pytest.mark.parametrize("change", [
    {"schema_version": 3}, {"schema_version": True}, {"unexpected": "setting"},
    {"default_versions": {}},
    {"default_versions": [{"kind": "affinity", "method": "original"}]},
    {"default_versions": [{"kind": "affinity", "method": "original", "version": "1"}] * 2},
])
def test_unknown_or_ambiguous_schema_rejected(change):
    definition = SelectionPolicy("test-v1", "affinity.value").to_dict()
    with pytest.raises(ValueError):
        SelectionPolicy.from_dict(dict(definition, **change))
    del definition["duplicates"]
    with pytest.raises(ValueError):
        SelectionPolicy.from_dict(definition)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"name": "first", "name": "second"}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_selection_policy(path)


def test_yaml_composition_resolves_once_and_preserves_null_vs_inheritance(tmp_path):
    base = yaml.safe_load('''
name: openvax-v1
score_by: 1 / affinity.value
filter_by: n_rna_alt >= 10
strata: [source_label, candidate_mhc_class]
default_methods:
  affinity: original
  proteasome_cleavage: cleavage_fixture
default_versions:
  - {kind: affinity, method: original, version: '1'}
''')
    override = yaml.safe_load('''
score_by: affinity.value
ascending: true
strata: []
default_methods:
  affinity: alternative
default_versions: []
''')
    # Match Vaxrank's authoring rule, which belongs to the consumer. Topiary
    # receives the composed subtree, never its YAML files or merge algorithm.
    composed = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(composed.get(key), dict):
            composed[key].update(value)
        else:
            composed[key] = value
    policy = resolve_selection_policy(composed)
    assert policy.name == "openvax-v1"
    assert policy.filter_by == base["filter_by"]  # omitted override inherits
    assert policy.strata == ()  # lists replace, never concatenate
    assert policy.default_versions == {}
    assert dict(policy.default_methods) == {"affinity": "alternative", "proteasome_cleavage": "cleavage_fixture"}
    assert policy.sha256 != resolve_selection_policy(base).sha256
    changed_filter = resolve_selection_policy(dict(composed, filter_by="n_rna_alt >= 1"))
    assert changed_filter.filter_by == "n_rna_alt >= 1"  # replace, never AND
    no_filter = resolve_selection_policy(dict(composed, filter_by=None))
    assert no_filter.filter_by is None
    evidence = combine_sources({"one": source()}, sample_name="patient")
    assert len(rank_with_policy(evidence, policy).df) == 1
    assert len(rank_with_policy(evidence, no_filter).df) == 2
    path = tmp_path / "policy.json"
    write_selection_policy(policy, path)
    for decode in (json.loads, yaml.safe_load):
        definition = decode(path.read_text())
        for resolve in POLICY_MAPPING_TWINS:
            restored = resolve(definition)
            assert restored.to_dict() == policy.to_dict()
            assert restored.sha256 == read_selection_policy(path).sha256
            pd.testing.assert_frame_equal(rank_with_policy(evidence, restored).df,
                                          rank_with_policy(evidence, policy).df)


def test_saved_complete_defaults_survive_changed_constructor_defaults(monkeypatch):
    saved = SelectionPolicy("frozen-v1", "affinity.value").to_dict()
    monkeypatch.setattr(SelectionPolicy.__init__, "__defaults__", (None, True, "best", (), None, None, None, None, (), (), "exclude"))
    assert SelectionPolicy("new-defaults", "affinity.value").ascending is True
    for resolve in POLICY_MAPPING_TWINS:
        assert resolve(saved).to_dict() == saved


def test_derivation_and_execution_provenance_are_distinct_and_detached(tmp_path):
    from topiary import __version__
    from .test_twin_conformance import DELIMITED_IO_TWINS

    original = source(predictor_version=None)
    original.topiary_version = "historical"
    evidence = combine_sources({"one": original}, sample_name="patient")
    policy = SelectionPolicy("openvax-v1", "affinity.value")
    provenance = {"base_sha256": "base-digest", "overrides": [{"file": "overrides.yaml", "sha256": "file-digest"}]}
    ranked = rank_with_policy(evidence, policy, provenance=provenance)
    record = deepcopy(ranked.extra["selection_policy"])
    provenance["overrides"].clear()
    assert record["provenance"]["overrides"]
    assert record["sha256"] == policy.sha256
    assert record["definition"] == policy.to_dict()
    assert record["execution"]["topiary_version"] == __version__
    assert record["execution"]["prediction_inventory"][0]["predictor_version"] is None
    assert "execution" not in policy.to_dict()
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("ranked." + suffix)
        writer(ranked, path)
        assert reader(path).extra["selection_policy"] == record
    assert "selection_policy" not in evidence.extra


@pytest.mark.parametrize("configuration,field", [
    ({"score_by": "1"}, "name"), ({"name": "test"}, "score_by"),
    ({"name": "test", "score_by": "1", "typo": 1}, "typo"),
    ({"name": "test", "score_by": "??"}, "score_by"),
    ({"name": "test", "score_by": "1", "filter_by": "??"}, "filter_by"),
])
def test_authoring_errors_identify_the_field(configuration, field):
    with pytest.raises(ValueError, match=field):
        resolve_selection_policy(configuration)


@pytest.mark.parametrize("settings", [
    {}, {"ascending": True}, {"filter_by": "n_rna_alt >= 10"},
    {"filter_by": "n_rna_alt > 100"}, {"duplicates": "best"},
    {"duplicates": "worst"}, {"strata": ()},
    {"strata": ("source_label", "candidate_mhc_class")},
    {"default_methods": {"affinity": "original"},
     "default_versions": {("affinity", "original"): "1"}},
])
@pytest.mark.parametrize("case", ["ordinary", "missing", "conflicting", "empty"])
def test_profile_and_direct_ranker_agree(case, settings):
    inputs = {"one": source(values=(50., np.nan) if case == "missing" else (50., 500.))}
    if case == "conflicting":
        inputs["two"] = source(values=(60., 600.))
    if case == "empty":
        inputs = {}
    evidence = combine_sources(inputs, sample_name="patient")
    original = evidence.df.copy(deep=True)
    profile = SelectionPolicy("test-v1", "affinity.value", **settings)
    direct, replay = SELECTION_POLICY_TWINS
    calls = (lambda: direct(evidence, profile.score_by, **settings),
             lambda: replay(evidence, profile).df)
    outputs, errors = [], []
    for call in calls:
        try:
            outputs.append(call())
            errors.append(None)
        except ValueError as error:
            errors.append((type(error), str(error)))
    assert errors[0] == errors[1]
    if outputs:
        pd.testing.assert_frame_equal(*outputs)
    pd.testing.assert_frame_equal(evidence.df, original)


def test_profile_model_and_version_selections_change_results():
    from topiary import TopiaryResult

    rows = pd.concat([
        source(values=(50., 500.)).df,
        source(values=(900., 10.), predictor_version="2").df,
        source(values=(800., 20.), prediction_method_name="alternative").df,
    ], ignore_index=True)
    evidence = combine_sources({"one": TopiaryResult(rows)}, sample_name="patient")
    direct, replay = SELECTION_POLICY_TWINS
    orders = []
    for method, version in (("original", "1"), ("original", "2"), ("alternative", "1")):
        settings = dict(ascending=True, default_methods={"affinity": method},
                        default_versions={("affinity", method): version})
        profile = SelectionPolicy("selection", "affinity.value", **settings)
        result = replay(evidence, profile)
        pd.testing.assert_frame_equal(result.df, direct(evidence, profile.score_by, **settings))
        orders.append(result.df.peptide.tolist())
    assert orders == [["SIINFEKL", "GILGFVFTL"], ["GILGFVFTL", "SIINFEKL"], ["GILGFVFTL", "SIINFEKL"]]
