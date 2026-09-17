"""Annotation joins cannot mutate or silently select historical observations."""

from copy import deepcopy

import pandas as pd
import pytest

from topiary import TopiaryResult, join_annotations, read_tsv


PROVENANCE = {"source": "test RNA", "unit": "TPM", "sample": "T2"}


def result():
    return TopiaryResult(
        pd.DataFrame({"variant": ["b", "a", "b", float("nan")], "value": [2., 1., 3., 4.],
                      "source": ["historical"] * 4},
                     index=pd.Index([9, 2, 9, 5], name="original")),
        form="unknown", extra={"nested": {"unchanged": True}},
        sources=["historical"], filter_by_str="value > 0")


def test_annotation_join_preserves_all_original_values_index_order_and_metadata():
    original = result()
    before = deepcopy(original)
    annotation = pd.DataFrame({"variant": ["a", "b"], "tpm": [0., 7.]})
    out = join_annotations(original, annotation, on="variant", prefix="rna_t2", provenance=PROVENANCE)
    pd.testing.assert_frame_equal(out.df[original.columns], original.df)
    pd.testing.assert_frame_equal(original.df, before.df)
    assert out.df.rna_t2_tpm.iloc[:3].tolist() == [7., 0., 7.]
    assert pd.isna(out.df.rna_t2_tpm.iloc[3])
    assert out.sources == original.sources
    assert out.filter_by_str == original.filter_by_str
    assert "annotation_overlays" not in original.extra
    out.extra["nested"]["unchanged"] = False
    assert original.extra["nested"]["unchanged"] is True
    out.extra["annotation_overlays"][0]["provenance"]["sample"] = "changed"
    assert PROVENANCE["sample"] == "T2"


@pytest.mark.parametrize("empty", ["predictions", "annotations", "both"])
@pytest.mark.parametrize("value", [0., False, "measured"])
def test_empty_annotations_and_predictions_do_not_create_zero_measurements(empty, value):
    original = result()
    annotation = pd.DataFrame({"variant": ["a"], "tpm": [value]})
    if empty in ("predictions", "both"):
        original.df = original.df.iloc[:0]
    if empty in ("annotations", "both"):
        annotation = annotation.iloc[:0]
    out = join_annotations(original, annotation, on="variant", prefix="rna", provenance=PROVENANCE)
    pd.testing.assert_frame_equal(out.df[original.columns], original.df)
    assert out.df.rna_tpm.isna().all()


@pytest.mark.parametrize("keys", [["a", "a"], [None], ["a", None], [""], ["<NA>"]])
def test_annotation_keys_must_be_unique_and_present(keys):
    annotation = pd.DataFrame({"variant": keys, "tpm": [1.] * len(keys)})
    with pytest.raises(ValueError, match="join keys"):
        join_annotations(result(), annotation, on="variant", prefix="rna", provenance=PROVENANCE)


@pytest.mark.parametrize("kwargs,match", [
    ({"on": []}, "distinct"),
    ({"on": ["variant", "variant"]}, "distinct"),
    ({"on": "missing"}, "lacks join columns"),
    ({"prefix": ""}, "prefix"),
    ({"prefix": "rna-bad"}, "prefix"),
    ({"provenance": {}}, "provenance"),
])
def test_annotation_join_rejects_invalid_contract(kwargs, match):
    options = dict(on="variant", prefix="rna", provenance=PROVENANCE)
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        join_annotations(result(), pd.DataFrame({"variant": ["a"], "tpm": [1]}), **options)


def test_annotation_join_refuses_to_overwrite_an_existing_overlay():
    annotation = pd.DataFrame({"variant": ["a"], "tpm": [1]})
    out = join_annotations(result(), annotation, on="variant", prefix="rna", provenance=PROVENANCE)
    with pytest.raises(ValueError, match="already exist"):
        join_annotations(out, annotation, on="variant", prefix="rna", provenance=PROVENANCE)


def test_multikey_annotation_join_keeps_samples_distinct():
    original = TopiaryResult(pd.DataFrame({"variant": ["a", "a"], "sample": ["T1", "T2"]}))
    annotation = pd.DataFrame({"variant": ["a", "a"], "sample": ["T1", "T2"], "alt": [0, 8]})
    out = join_annotations(original, annotation, on=["variant", "sample"],
                           prefix="rna", provenance=PROVENANCE)
    assert out.df.rna_alt.tolist() == [0, 8]
    with pytest.raises(ValueError, match="duplicate join keys"):
        join_annotations(original, annotation, on="variant", prefix="rna", provenance=PROVENANCE)


def test_annotation_join_and_provenance_survive_tsv(tmp_path):
    out = join_annotations(result(), pd.DataFrame({"variant": ["a"], "tpm": [0.]}),
                           on="variant", prefix="rna", provenance=PROVENANCE)
    path = tmp_path / "annotated.tsv"
    out.to_tsv(path)
    restored = read_tsv(path)
    pd.testing.assert_frame_equal(restored.df, out.df.reset_index(drop=True), check_dtype=False)
    assert restored.extra == out.extra
    assert restored.sources == out.sources + [path.name]
    assert restored.filter_by_str == out.filter_by_str
