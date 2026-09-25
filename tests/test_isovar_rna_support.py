"""Both reader families preserve and validate the same RNA measurements."""

from copy import deepcopy

import pytest

from topiary import normalize_isovar_rna_support
from .test_isovar_hypotheses import hypothesis_export
from .test_twin_conformance import ISOVAR_RNA_SUPPORT_TWINS


def support_record():
    export = hypothesis_export("protein-v2-labelled")
    support = deepcopy(export["events"][0]["protein_hypotheses"][0]["rna_support"])
    support.update(export["evidence_sets"][support["evidence_set_id"]])
    return support


@pytest.mark.parametrize("fault", [
    "negative", "float", "boolean_count", "text_flag", "count", "duplicate", "null_ids", "scope",
    "alias", "labels", "too_many_umis", "complete_unlabelled", "complete_unknown_library",
    "missing_complete_count", "empty_complete", "not_mapping",
])
def test_reader_pair_rejects_the_same_invalid_support(fault):
    support = support_record()
    if fault in ("negative", "float", "boolean_count"):
        support["umis"] = {"negative": -1, "float": 1.5, "boolean_count": True}[fault]
    elif fault == "text_flag":
        support["cells_complete"] = "false"
    elif fault == "count":
        support["reads"] += 1
    elif fault == "duplicate":
        support["read_ids"].append(support["read_ids"][0])
    elif fault == "null_ids":
        support["fragment_ids"] = None
    elif fault == "scope":
        support["evidence_scope"] = ["other", "source"]
    elif fault == "alias":
        support["segments"] = 99
    elif fault == "labels":
        support["label_statuses"] = {"resolved_label": 99}
    elif fault == "too_many_umis":
        support["umis"] = 99
    elif fault == "complete_unlabelled":
        support["umis_complete"] = True
    elif fault == "complete_unknown_library":
        support["unknown_library_reads"] = 1
    elif fault == "missing_complete_count":
        support["cells"] = None
    elif fault == "empty_complete":
        support.update(reads=0, fragments=0, umis=0, cells=0, unlabeled_reads=0,
                       unknown_library_reads=0, read_ids=[], fragment_ids=[], label_statuses={})
    elif fault == "not_mapping":
        support = False
    for reader in ISOVAR_RNA_SUPPORT_TWINS:
        with pytest.raises(ValueError):
            reader(support)


@pytest.mark.parametrize("legacy", [False, True])
def test_reader_pair_accepts_the_same_incomplete_label_measurement(legacy):
    support = support_record()
    if legacy:
        support["segments"] = support.pop("reads")
        support["segment_ids"] = support.pop("read_ids")
    before = deepcopy(support)
    protein, sv = [reader(support) for reader in ISOVAR_RNA_SUPPORT_TWINS]
    row = protein.df.iloc[0]
    observed = sv["protein_hypotheses"][0]["source_observations"][0]["rna_support"]
    for field in ("reads", "fragments", "umis", "cells", "umis_complete", "cells_complete",
                  "unlabeled_reads", "unknown_library_reads"):
        assert row["protein_" + field] == observed[field]
    assert observed["label_statuses"] == support["label_statuses"]
    assert support == before


def test_normalization_preserves_empty_sets_and_unknown_measurements():
    for value in (None, {}):
        result = normalize_isovar_rna_support(value)
        assert result["reads"] is result["umis"] is result["umis_complete"] is None
    empty = dict(reads=0, fragments=0, read_ids=[], fragment_ids=[], evidence_set_id="empty",
                 evidence_scope=["s", "rna"], umis=0, cells=0, umis_complete=False,
                 cells_complete=False, unlabeled_reads=0, unknown_library_reads=0, label_statuses={})
    result = normalize_isovar_rna_support(empty, evidence_scope=["s", "rna"])
    assert result == empty
    result["read_ids"].append("modified")
    assert empty["read_ids"] == []


@pytest.mark.parametrize("scope", [None, [], ["sample"], ["sample", ""], [1, "source"]])
def test_normalization_rejects_an_explicit_unusable_scope(scope):
    support = support_record()
    support["evidence_scope"] = scope
    with pytest.raises(ValueError, match="evidence_scope"):
        normalize_isovar_rna_support(support)
