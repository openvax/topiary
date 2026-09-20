"""Tests for topiary.io — read/write with comment-block metadata."""

import numpy as np
import pandas as pd
import pytest

from topiary.io import (
    Metadata,
    _format_comment_block,
    _parse_comment_block,
    read_csv,
    read_tsv,
    to_csv,
    to_tsv,
)
from topiary.wide import to_wide
from topiary import TopiaryResult
from .test_twin_conformance import DELIMITED_IO_TWINS


# ---------------------------------------------------------------------------
# Comment block parsing
# ---------------------------------------------------------------------------


class TestParseCommentBlock:
    def test_well_formed(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#form=long\n",
            "#model:netmhcpan=4.1b\n",
            "#model:mhcflurry=2.1.1\n",
            "peptide\tallele\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 4
        assert meta.topiary_version == "4.11.0"
        assert meta.form == "long"
        assert meta.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}

    def test_unknown_keys_in_extra(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#custom_key=custom_value\n",
            "#other_key=other_value\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 3
        assert meta.extra == {"custom_key": "custom_value", "other_key": "other_value"}

    def test_kind_support_legacy_literal_extra(self):
        lines = [
            "#kind_support={'netmhcpan': {'pMHC_affinity': "
            "{'mhc_dependence': 'single_allele', 'mhc_class': 'I'}}}\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 1
        assert meta.extra["kind_support"] == {
            "netmhcpan": {
                "pMHC_affinity": {
                    "mhc_dependence": "single_allele",
                    "mhc_class": "I",
                },
            },
        }

    def test_source_lines(self):
        lines = [
            "#source=patient01.tsv\n",
            "#source=patient02.tsv\n",
            "peptide\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 2
        assert meta.sources == ["patient01.tsv", "patient02.tsv"]

    def test_no_comments(self):
        lines = ["peptide\tallele\n", "SIINFEKL\tHLA-A*02:01\n"]
        meta, n = _parse_comment_block(lines)
        assert n == 0
        assert meta.topiary_version is None
        assert meta.models == {}

    def test_empty_lines(self):
        meta, n = _parse_comment_block([])
        assert n == 0

    def test_malformed_comment_skipped(self):
        lines = [
            "#topiary_version=4.11.0\n",
            "#no-equals-sign\n",
            "#form=wide\n",
            "data\n",
        ]
        meta, n = _parse_comment_block(lines)
        assert n == 3
        assert meta.topiary_version == "4.11.0"
        assert meta.form == "wide"

    def test_model_entries(self):
        lines = [
            "#model:netmhcpan=4.1b\n",
            "#model:mhcflurry=2.1.1\n",
            "#model:netmhcstabpan=1.0\n",
            "data\n",
        ]
        meta, _ = _parse_comment_block(lines)
        assert len(meta.models) == 3
        assert meta.models["netmhcstabpan"] == "1.0"


# ---------------------------------------------------------------------------
# Comment block formatting
# ---------------------------------------------------------------------------


class TestFormatCommentBlock:
    def test_full_metadata(self):
        meta = Metadata(
            topiary_version="4.11.0",
            form="long",
            models={"netmhcpan": "4.1b"},
        )
        block = _format_comment_block(meta)
        assert "#topiary_version=4.11.0" in block
        assert "#form=long" in block
        assert "#model:netmhcpan=4.1b" in block

    def test_empty_metadata(self):
        meta = Metadata()
        block = _format_comment_block(meta)
        assert block == ""

    def test_extra_keys_preserved(self):
        meta = Metadata(extra={"custom_key": "custom_value"})
        block = _format_comment_block(meta)
        assert "#custom_key=custom_value" in block

    def test_structured_extra_roundtrip(self):
        kind_support = {
            "netmhcpan": {
                "pMHC_affinity": {
                    "mhc_dependence": "single_allele",
                    "mhc_class": "I",
                },
            },
        }
        meta = Metadata(extra={"kind_support": kind_support})
        block = _format_comment_block(meta)
        assert "#kind_support=json:" in block

        lines = [line + "\n" for line in block.split("\n")] + ["data\n"]
        parsed, _ = _parse_comment_block(lines)

        assert parsed.extra["kind_support"] == kind_support

    def test_sources_formatted(self):
        meta = Metadata(sources=["patient01.tsv", "patient02.tsv"])
        block = _format_comment_block(meta)
        assert "#source=patient01.tsv" in block
        assert "#source=patient02.tsv" in block

    def test_format_parse_roundtrip(self):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            sources=["lens-v1.9.tsv"],
            extra={"custom_key": "custom_value"},
        )
        block = _format_comment_block(meta)
        lines = [line + "\n" for line in block.split("\n")] + ["data\n"]
        parsed, _ = _parse_comment_block(lines)
        assert parsed.topiary_version == meta.topiary_version
        assert parsed.form == meta.form
        assert dict(parsed.models) == dict(meta.models)
        assert parsed.sources == meta.sources
        assert dict(parsed.extra) == dict(meta.extra)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_long_df():
    return pd.DataFrame([
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=3, peptide_length=8,
            kind="pMHC_affinity", score=0.85, value=120.0,
            percentile_rank=0.5, affinity=120.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
        dict(
            peptide="ELAGIGILT", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=10, peptide_length=9,
            kind="pMHC_affinity", score=0.3, value=5000.0,
            percentile_rank=15.0, affinity=5000.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
    ])


def _sample_long_df_with_version_state(version_state):
    df = _sample_long_df().iloc[[0]].copy()
    if version_state == "missing":
        return df.drop(columns=["predictor_version"])
    if version_state == "blank":
        df["predictor_version"] = ""
        return df
    if version_state == "na":
        df["predictor_version"] = pd.NA
        return df
    raise ValueError(f"unknown version state: {version_state}")


def _write_extra_case(writer, method, path, extra, call_style):
    meta = Metadata(extra=extra)
    result = TopiaryResult(_sample_long_df())
    # Exercise late mutation as well as explicit Metadata: construction-time
    # validation alone cannot protect the public mutable extra mapping.
    result.extra.update(extra)
    if call_style == "dataframe":
        writer(result.df, path, metadata=meta)
    elif call_style == "result-function":
        writer(result, path)
    elif call_style == "result-method":
        method(result, path)
    else:
        result.extra.clear()
        writer(result, path, metadata=meta)


@pytest.mark.parametrize("call_style", ["dataframe", "result-function", "result-method", "override"])
@pytest.mark.parametrize("key", [
    "topiary_version", "form", "source", "filter_by", "sort_by", "model:", "model:netmhcpan",
    " source", "form\t", "", " ", "custom=source", "custom\n#source", "custom\r#form", 1, None,
])
def test_metadata_extra_keys_fail_before_touching_output(tmp_path, key, call_style):
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("predictions." + suffix)
        original = b"existing output must survive\n"
        path.write_bytes(original)
        with pytest.raises(ValueError, match="Metadata.extra key"):
            _write_extra_case(writer, method, path, {key: "custom value"}, call_style)
        assert path.read_bytes() == original
        path.unlink()
        with pytest.raises(ValueError, match="Metadata.extra key"):
            _write_extra_case(writer, method, path, {key: "custom value"}, call_style)
        assert not path.exists()


@pytest.mark.parametrize("call_style", ["dataframe", "result-function", "result-method", "override"])
def test_metadata_extra_twins_preserve_nested_reserved_names(tmp_path, call_style):
    extra = {
        "dataset": {"source": "a" * 64, "form": "source reads", "topiary_version": "producer",
                    "filter_by": {"passed": True, "missing": None}, "sort_by": [3, 1],
                    "model:netmhcpan": {"version": "producer model"}},
        "settings": [False, {"text": "line 1\n#source=still nested data", "threshold": 0.5}],
        # These names are not built-in comment keys and must remain accepted.
        "model": "custom", "models": "custom models", "sources": "custom sources",
        "source_digest": "sha256", "custom:source": "custom namespace",
    }
    frames = []
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("predictions." + suffix)
        _write_extra_case(writer, method, path, extra, call_style)
        restored = reader(path, tag="output")
        assert restored.extra == extra
        assert restored.sources == ["output"]
        assert restored.form == "long"
        assert restored.models == {"netmhcpan": "4.1b"}
        assert restored.filter_by_str is None and restored.sort_by_str is None
        frames.append(restored.df)
    pd.testing.assert_frame_equal(*frames)


class _ExtraText:
    """Exercise the existing fallback for objects without a JSON encoder."""

    def __init__(self, text):
        self.text = text

    def __str__(self):
        return self.text


class _UnprintableExtra:
    def __str__(self):
        raise ValueError("extra has no text representation")


@pytest.mark.parametrize("call_style", ["dataframe", "result-function", "result-method", "override"])
@pytest.mark.parametrize("key,value", [
    ("notes", "note\n#source=unexpected-source"),
    ("notes", "note\r#source=unexpected-source"),
    ("notes", "note\r\n#source=unexpected-source"),
    ("notes", "note\n#topiary_version=wrong\n#form=wide\n#model:netmhcpan=wrong\n"
              "#model:unexpected\n#filter_by=wrong\n#sort_by=wrong\n#injected=wrong"),
    ("notes", "note\nnot-a-comment\nextra,data,row"),
    ("notes", 'json:{"flag":true}'),
    ("notes", "json:[1,true,null]"),
    ("notes", 'json:"literal"'),
    ("notes", "json:true"),
    ("notes", "json:null"),
    ("notes", "json:12.5"),
    ("notes", "json:NaN"),
    ("notes", "json:not-json"),
    ("notes", "json:"),
    ("notes", " leading and trailing \t"),
    ("notes", "\u00a0Unicode whitespace\u00a0"),
    ("notes", "\t \r\n"),
    ("notes", ""),
    ("notes", r"literal\n#source=still-text"),
    ("notes", 'quotes " and \\ and café\n#source=still-text'),
    ("kind_support", '{}'),
    ("kind_support", '{"fixture":{"enabled":true}}'),
    ("kind_support", "{'fixture': {'enabled': True}}"),
    ("kind_support", "unresolved"),
    ("notes", _ExtraText("fallback\n#source=unexpected-source")),
    ("notes", _ExtraText('json:{"flag":true}')),
    ("kind_support", _ExtraText("{'fixture': {'enabled': True}}")),
])
def test_metadata_extra_text_twins_preserve_values_and_builtins(tmp_path, key, value, call_style):
    from topiary import __version__

    frames = []
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("text." + suffix)
        _write_extra_case(writer, method, path, {key: value}, call_style)
        restored = reader(path, tag="output")
        assert restored.extra == {key: str(value)}
        assert type(restored.extra[key]) is str
        assert restored.topiary_version == __version__
        assert restored.sources == ["output"]
        assert restored.form == "long"
        assert restored.models == {"netmhcpan": "4.1b"}
        assert restored.filter_by_str is None and restored.sort_by_str is None
        pd.testing.assert_frame_equal(restored.df.drop(columns="source"), _sample_long_df())
        frames.append(restored.df)
        # Rewriting must not add another marker or decode the literal twice.
        second = tmp_path / ("second." + suffix)
        method(restored, second)
        repeated = reader(second, tag="output")
        assert repeated.metadata == restored.metadata
        pd.testing.assert_frame_equal(repeated.df, restored.df)
    pd.testing.assert_frame_equal(*frames)


@pytest.mark.parametrize("call_style", ["dataframe", "result-function", "result-method", "override"])
def test_metadata_extra_text_conversion_failure_preserves_output(tmp_path, call_style):
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("failed." + suffix)
        for existing in (False, True):
            if existing:
                path.write_bytes(b"keep original output\n")
            with pytest.raises(ValueError, match="extra has no text representation"):
                _write_extra_case(writer, method, path, {"notes": _UnprintableExtra()}, call_style)
            if existing:
                assert path.read_bytes() == b"keep original output\n"
            else:
                assert not path.exists()


def test_legacy_comment_metadata_keeps_builtins_and_custom_values(tmp_path):
    comments = (
        "#topiary_version=4.11.0\n#form=long\n#source=input.tsv\n"
        "#model:fixture=1\n#model:unversioned\n"
        "#filter_by=review_score <= 20\n#sort_by=review_score\n"
        "#patient=PT01\n#settings=json:{\"source\":\"dataset\",\"enabled\":true}\n"
        "#kind_support={'fixture': {'custom': {'mhc_dependence': 'independent'}}}\n"
    )
    for suffix, writer, method, reader in DELIMITED_IO_TWINS:
        path = tmp_path / ("legacy." + suffix)
        separator = "\t" if suffix == "tsv" else ","
        path.write_text(comments + separator.join(["peptide", "kind", "value"]) + "\n"
                        + separator.join(["SIINFEKL", "custom", "1"]) + "\n")
        restored = reader(path, tag="legacy")
        assert restored.topiary_version == "4.11.0"
        assert restored.form == "long"
        assert restored.sources == ["input.tsv", "legacy"]
        assert restored.models == {"fixture": "1", "unversioned": ""}
        assert restored.filter_by_str == "review_score <= 20"
        assert restored.sort_by_str == "review_score"
        assert restored.extra == {
            "patient": "PT01", "settings": {"source": "dataset", "enabled": True},
            "kind_support": {"fixture": {"custom": {"mhc_dependence": "independent"}}},
        }


# ---------------------------------------------------------------------------
# Read/write round-trip tests
# ---------------------------------------------------------------------------


class TestReadWriteTSV:
    def test_long_form_roundtrip(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.tsv"
        to_tsv(df, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "long"
        assert result.df is result.long_df
        assert "netmhcpan_affinity_value" in result.wide_df.columns
        assert meta.topiary_version is not None
        assert meta.models.get("netmhcpan") == "4.1b"
        assert len(df2) == len(df)
        assert list(df2["peptide"]) == list(df["peptide"])
        assert df2.iloc[0]["value"] == pytest.approx(120.0)

    def test_wide_form_roundtrip(self, tmp_path):
        df = _sample_long_df()
        wide = to_wide(df)
        path = tmp_path / "out.wide.tsv"
        to_tsv(wide, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "wide"
        assert result.df is result.wide_df
        assert "kind" in result.long_df.columns
        assert "netmhcpan_affinity_value" in df2.columns
        assert len(df2) == len(wide)

    def test_metadata_preserved(self, tmp_path):
        df = _sample_long_df()
        meta = Metadata(
            sources=["test_cohort"],
            extra={"patient": "PT01"},
        )
        path = tmp_path / "out.tsv"
        to_tsv(df, path, metadata=meta)
        meta2 = read_tsv(path).metadata
        assert "test_cohort" in meta2.sources
        assert meta2.extra.get("patient") == "PT01"

    @pytest.mark.parametrize("writer,reader", [(to_tsv, read_tsv), (to_csv, read_csv)])
    def test_numpy_scalars_in_structured_metadata(self, tmp_path, writer, reader):
        settings = {"enabled": np.bool_(True), "disabled": np.bool_(False),
                    "limits": [np.int64(2**60 + 1), np.float32(0.5)]}
        meta = Metadata(extra={"settings": settings})
        path = tmp_path / "out.txt"
        writer(_sample_long_df(), path, metadata=meta)
        restored = reader(path).metadata.extra["settings"]
        assert restored == {"enabled": True, "disabled": False, "limits": [2**60 + 1, 0.5]}
        assert restored["enabled"] is True
        assert restored["disabled"] is False
        assert type(restored["limits"][0]) is int
        assert type(restored["limits"][1]) is float
        assert type(settings["enabled"]) is np.bool_

    @pytest.mark.parametrize("writer,reader", [(to_tsv, read_tsv), (to_csv, read_csv)])
    @pytest.mark.parametrize("dtype", [np.datetime64, np.timedelta64])
    @pytest.mark.parametrize("unit", ["s", "ns"])
    @pytest.mark.parametrize("nested", [False, True])
    def test_temporal_metadata_keeps_its_existing_text_format(
        self, tmp_path, writer, reader, dtype, unit, nested,
    ):
        value = dtype(1, unit)
        original = {"value": value} if nested else value
        path = tmp_path / "temporal.txt"
        writer(_sample_long_df(), path, metadata=Metadata(extra={"temporal": original}))
        # Non-JSON metadata has historically used a text representation.
        # Do not introduce a new encoding or silently drop the temporal unit.
        assert reader(path).metadata.extra["temporal"] == str(original)

    def test_model_versions_auto_extracted(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.tsv"
        to_tsv(df, path)
        meta = read_tsv(path).metadata
        assert meta.models.get("netmhcpan") == "4.1b"

    def test_model_attrs_filtered_to_observed_long_rows(self, tmp_path):
        df = _sample_long_df()
        df.loc[1, "prediction_method_name"] = "mhcflurry"
        df.loc[1, "predictor_version"] = "2.1.1"
        df.attrs["topiary_models"] = {
            "netmhcpan": "4.1b",
            "mhcflurry": "2.1.1",
        }
        filtered = df[df["prediction_method_name"] == "netmhcpan"]

        path = tmp_path / "filtered.tsv"
        to_tsv(filtered, path)
        meta = read_tsv(path).metadata

        assert meta.models == {"netmhcpan": "4.1b"}

    def test_result_metadata_filtered_to_observed_long_rows(self, tmp_path):
        from topiary import TopiaryResult

        df = _sample_long_df()
        df.loc[1, "prediction_method_name"] = "mhcflurry"
        df.loc[1, "predictor_version"] = "2.1.1"
        result = TopiaryResult(
            df,
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
        )
        filtered = result[result["prediction_method_name"] == "netmhcpan"]

        path = tmp_path / "filtered_result.tsv"
        to_tsv(filtered, path)
        meta = read_tsv(path).metadata

        assert meta.models == {"netmhcpan": "4.1b"}

    @pytest.mark.parametrize("version_state", ["missing", "blank", "na"])
    @pytest.mark.parametrize(
        "writer,reader,suffix",
        [(to_tsv, read_tsv, "tsv"), (to_csv, read_csv, "csv")],
    )
    def test_dataframe_writer_fills_blank_row_versions_from_attrs(
        self, tmp_path, version_state, writer, reader, suffix,
    ):
        df = _sample_long_df_with_version_state(version_state)
        df.attrs["topiary_models"] = {
            "netmhcpan": "4.1b",
            "old_model": "0.1",
        }

        path = tmp_path / f"attrs.{suffix}"
        writer(df, path)

        meta = reader(path).metadata
        assert meta.models == {"netmhcpan": "4.1b"}

    @pytest.mark.parametrize("version_state", ["missing", "blank", "na"])
    @pytest.mark.parametrize(
        "writer,method_name,reader,suffix",
        [
            (to_tsv, "to_tsv", read_tsv, "tsv"),
            (to_csv, "to_csv", read_csv, "csv"),
        ],
    )
    @pytest.mark.parametrize("call_style", ["function", "method"])
    def test_result_writer_fills_blank_row_versions_from_metadata(
        self, tmp_path, version_state, writer, method_name, reader, suffix,
        call_style,
    ):
        from topiary import TopiaryResult

        df = _sample_long_df_with_version_state(version_state)
        result = TopiaryResult(
            df,
            models={"netmhcpan": "4.1b", "old_model": "0.1"},
        )

        path = tmp_path / f"result.{suffix}"
        if call_style == "function":
            writer(result, path)
        else:
            getattr(result, method_name)(path)

        meta = reader(path).metadata
        assert meta.models == {"netmhcpan": "4.1b"}

    @pytest.mark.parametrize(
        "writer,method_name,reader,suffix",
        [
            (to_tsv, "to_tsv", read_tsv, "tsv"),
            (to_csv, "to_csv", read_csv, "csv"),
        ],
    )
    @pytest.mark.parametrize("call_style", ["function", "method"])
    def test_result_writer_preserves_empty_result_model_metadata(
        self, tmp_path, writer, method_name, reader, suffix, call_style,
    ):
        from topiary import TopiaryResult

        df = _sample_long_df().iloc[0:0].copy()
        result = TopiaryResult(
            df,
            models={"netmhcpan": "4.1b"},
            sources=["empty-run"],
        )

        path = tmp_path / f"empty_result.{suffix}"
        if call_style == "function":
            writer(result, path)
        else:
            getattr(result, method_name)(path)

        reloaded = reader(path)
        assert reloaded.form == "long"
        assert reloaded.models == {"netmhcpan": "4.1b"}
        assert reloaded.sources[:1] == ["empty-run"]
        assert reloaded.long_df.empty
        assert reloaded.wide_df.empty
        assert "peptide" in reloaded.wide_df.columns


class TestReadWriteCSV:
    def test_csv_roundtrip(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "out.csv"
        to_csv(df, path)
        result = read_csv(path)
        df2, meta = result.df, result.metadata
        assert meta.form == "long"
        assert result.df is result.long_df
        assert "netmhcpan_affinity_value" in result.wide_df.columns
        assert len(df2) == len(df)
        assert df2.iloc[0]["value"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_file_without_comments(self, tmp_path):
        path = tmp_path / "plain.tsv"
        df = pd.DataFrame({"peptide": ["SIINFEKL"], "allele": ["A"]})
        df.to_csv(path, sep="\t", index=False)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert meta.topiary_version is None
        assert meta.models == {}
        assert len(df2) == 1

    def test_empty_df_roundtrip(self, tmp_path):
        df = pd.DataFrame(columns=["peptide", "allele", "kind"])
        path = tmp_path / "empty.tsv"
        to_tsv(df, path)
        result = read_tsv(path)
        df2, meta = result.df, result.metadata
        assert len(df2) == 0
        assert meta.form == "long"

    def test_header_only_file(self, tmp_path):
        path = tmp_path / "header_only.tsv"
        with open(path, "w") as f:
            f.write("#topiary_version=4.11.0\n")
            f.write("#form=long\n")
            f.write("peptide\tallele\tkind\n")
        result = read_tsv(path)
        df, meta = result.df, result.metadata
        assert len(df) == 0
        assert meta.topiary_version == "4.11.0"

    def test_all_metadata_fields(self, tmp_path):
        meta = Metadata(
            topiary_version="4.11.0",
            form="wide",
            models={"netmhcpan": "4.1b", "mhcflurry": "2.1.1"},
            sources=["lens-v1.9.tsv"],
            extra={"patient": "PT01"},
        )
        df = pd.DataFrame({"peptide": ["A"], "netmhcpan_affinity_value": [100]})
        path = tmp_path / "full.tsv"
        to_tsv(df, path, metadata=meta)
        result = read_tsv(path)
        meta2 = result.metadata
        assert meta2.topiary_version == "4.11.0"
        assert meta2.form == "wide"
        assert meta2.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}
        # The read function appends the filename to sources too
        assert "lens-v1.9.tsv" in meta2.sources
        assert meta2.extra == {"patient": "PT01"}

    def test_pandas_read_csv_with_comment_hash(self, tmp_path):
        """Standard pandas can still read our files (losing metadata)."""
        df = _sample_long_df()
        path = tmp_path / "compat.tsv"
        to_tsv(df, path)
        df2 = pd.read_csv(path, sep="\t", comment="#")
        assert len(df2) == 2
        assert "peptide" in df2.columns

    def test_write_index_false_by_default(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "no_idx.tsv"
        to_tsv(df, path)
        with open(path) as f:
            lines = f.readlines()
        # First non-comment line is the header; should not start with ","
        for line in lines:
            if not line.startswith("#"):
                assert not line.startswith(",")
                assert not line.startswith("\t,")
                break


# ---------------------------------------------------------------------------
# Full round-trip integration tests with real predictor
# ---------------------------------------------------------------------------


class TestRoundTripIntegration:
    """End-to-end: predict → to_wide → to_tsv → read_tsv → from_wide → verify."""

    def test_predict_to_wide_tsv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor
        from topiary.wide import to_wide, from_wide

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGGLLLAAA"}
        )
        assert len(long_orig) > 0

        # Long → wide
        wide = to_wide(long_orig)
        assert "kind" not in wide.columns

        # Wide → TSV → read back
        path = tmp_path / "roundtrip.wide.tsv"
        to_tsv(wide, path)
        read_result = read_tsv(path)
        wide_read, meta = read_result.df, read_result.metadata
        assert meta.form == "wide"
        assert len(wide_read) == len(wide)

        # Read back → long
        long_back = from_wide(wide_read, metadata=meta)
        assert "kind" in long_back.columns
        assert len(long_back) == len(long_orig)

        # Values round-trip correctly
        orig_values = sorted(long_orig["value"].dropna().tolist())
        back_values = sorted(long_back["value"].dropna().tolist())
        assert orig_values == pytest.approx(back_values)

    def test_predict_long_tsv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"prot": "MASIINFEKLGGGLLLAAA"}
        )

        # Long → TSV → read back
        path = tmp_path / "roundtrip.tsv"
        to_tsv(long_orig, path)
        read_result = read_tsv(path)
        long_read, meta = read_result.df, read_result.metadata
        assert meta.form == "long"
        assert len(long_read) == len(long_orig)
        assert list(long_read["peptide"]) == list(long_orig["peptide"])

    def test_predict_wide_csv_roundtrip(self, tmp_path):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor
        from topiary.wide import to_wide, from_wide

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201", "B0702"],
        )
        long_orig = predictor.predict_from_named_sequences(
            {"braf": "MASIINFEKLGGG", "tp53": "MRKKLLQQREEY"}
        )
        wide = to_wide(long_orig)

        path = tmp_path / "roundtrip.wide.csv"
        to_csv(wide, path)
        read_result = read_csv(path)
        wide_read, meta = read_result.df, read_result.metadata
        long_back = from_wide(wide_read, metadata=meta)

        assert set(long_back["kind"].unique()) == set(long_orig["kind"].unique())
        assert len(long_back) == len(long_orig)

    def test_from_wide_to_wide_roundtrip(self, tmp_path):
        """Reverse direction: from_wide then to_wide should be identity."""
        from topiary.wide import to_wide, from_wide

        wide_orig = pd.DataFrame({
            "peptide": ["SIINFEKL", "ELAGIGILT"],
            "allele": ["HLA-A*02:01", "HLA-A*02:01"],
            "source_sequence_name": ["prot1", "prot1"],
            "netmhcpan_affinity_value": [120.0, 5000.0],
            "netmhcpan_affinity_score": [0.85, 0.3],
            "netmhcpan_affinity_rank": [0.5, 15.0],
        })
        long = from_wide(wide_orig)
        wide_back = to_wide(long)

        assert "netmhcpan_affinity_value" in wide_back.columns
        assert len(wide_back) == 2
        vals = sorted(wide_back["netmhcpan_affinity_value"].tolist())
        assert vals == pytest.approx([120.0, 5000.0])

    def test_sample_name_survives_roundtrip(self):
        """sample_name column from mhctools should survive wide/long."""
        from topiary.wide import to_wide, from_wide

        df = pd.DataFrame([
            dict(
                peptide="SIINFEKL", allele="HLA-A*02:01",
                sample_name="patient_01",
                kind="pMHC_affinity", score=0.85, value=120.0,
                percentile_rank=0.5, affinity=120.0,
                prediction_method_name="netmhcpan", predictor_version="4.1b",
            ),
        ])
        wide = to_wide(df)
        assert "sample_name" in wide.columns
        long = from_wide(wide)
        assert "sample_name" in long.columns
        assert long.iloc[0]["sample_name"] == "patient_01"

    def test_model_versions_in_metadata_after_write(self, tmp_path):
        """Model versions auto-extracted on write, available on read."""
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
        path = tmp_path / "models.tsv"
        to_tsv(df, path)
        meta = read_tsv(path).metadata
        assert len(meta.models) > 0
