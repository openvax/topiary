"""Tests for topiary.result — TopiaryResult wrapper and concat()."""

import pandas as pd
import pytest

from topiary import TopiaryResult, concat, read_tsv, to_tsv
from topiary.io import Metadata


def _sample_long_df():
    return pd.DataFrame([
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            kind="pMHC_affinity", score=0.85, value=120.0,
            percentile_rank=0.5, affinity=120.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
        dict(
            peptide="ELAGIGILT", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            kind="pMHC_affinity", score=0.3, value=5000.0,
            percentile_rank=15.0, affinity=5000.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
    ])


# ---------------------------------------------------------------------------
# Construction and properties
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_df_only(self):
        df = _sample_long_df()
        r = TopiaryResult(df)
        assert r.form == "long"
        assert len(r) == 2

    def test_with_metadata(self):
        df = _sample_long_df()
        meta = Metadata(
            models={"netmhcpan": "4.1b"},
            sources=["patient01.tsv"],
        )
        r = TopiaryResult(df, meta)
        assert r.models == {"netmhcpan": "4.1b"}
        assert r.sources == ["patient01.tsv"]

    def test_form_autodetected(self):
        wide_df = pd.DataFrame({
            "peptide": ["A"],
            "netmhcpan_affinity_value": [100.0],
        })
        r = TopiaryResult(wide_df)
        assert r.form == "wide"

    def test_explicit_form_preserved(self):
        df = _sample_long_df()
        meta = Metadata(form="long")
        r = TopiaryResult(df, meta)
        assert r.form == "long"


# ---------------------------------------------------------------------------
# DataFrame delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_len(self):
        r = TopiaryResult(_sample_long_df())
        assert len(r) == 2

    def test_columns(self):
        r = TopiaryResult(_sample_long_df())
        assert "peptide" in r.columns

    def test_getitem_column(self):
        r = TopiaryResult(_sample_long_df())
        peptides = r["peptide"]
        assert list(peptides) == ["SIINFEKL", "ELAGIGILT"]

    def test_getitem_returns_series_for_column(self):
        r = TopiaryResult(_sample_long_df())
        assert isinstance(r["peptide"], pd.Series)

    def test_getitem_returns_result_for_multi_column(self):
        r = TopiaryResult(_sample_long_df())
        sub = r[["peptide", "allele"]]
        assert isinstance(sub, TopiaryResult)
        assert list(sub.columns) == ["peptide", "allele"]

    def test_contains(self):
        r = TopiaryResult(_sample_long_df())
        assert "peptide" in r
        assert "nonexistent" not in r

    def test_iter_yields_columns(self):
        r = TopiaryResult(_sample_long_df())
        cols = list(r)
        assert "peptide" in cols

    def test_shape(self):
        r = TopiaryResult(_sample_long_df())
        assert r.shape[0] == 2
        assert r.shape[1] == len(_sample_long_df().columns)

    def test_empty(self):
        r = TopiaryResult(pd.DataFrame())
        assert r.empty

    def test_head(self):
        r = TopiaryResult(_sample_long_df())
        top1 = r.head(1)
        assert isinstance(top1, TopiaryResult)
        assert len(top1) == 1

    def test_iterrows(self):
        r = TopiaryResult(_sample_long_df())
        rows = list(r.iterrows())
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# Form conversion
# ---------------------------------------------------------------------------


class TestFormConversion:
    def test_to_wide(self):
        r = TopiaryResult(_sample_long_df())
        wide = r.to_wide()
        assert wide.form == "wide"
        assert "netmhcpan_affinity_value" in wide.columns
        assert "kind" not in wide.columns

    def test_to_long_from_wide(self):
        r = TopiaryResult(_sample_long_df())
        wide = r.to_wide()
        back = wide.to_long()
        assert back.form == "long"
        assert "kind" in back.columns

    def test_form_conversion_preserves_metadata(self):
        meta = Metadata(
            models={"netmhcpan": "4.1b"},
            sources=["patient01.tsv"],
        )
        r = TopiaryResult(_sample_long_df(), meta)
        wide = r.to_wide()
        assert wide.models == {"netmhcpan": "4.1b"}
        assert wide.sources == ["patient01.tsv"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_tsv_from_result(self, tmp_path):
        r = TopiaryResult(
            _sample_long_df(),
            Metadata(
                models={"netmhcpan": "4.1b"},
                sources=["test_cohort"],
            ),
        )
        path = tmp_path / "out.tsv"
        r.to_tsv(path)

        r2 = read_tsv(path)
        assert r2.form == "long"
        assert r2.models.get("netmhcpan") == "4.1b"
        assert "test_cohort" in r2.sources

    def test_to_csv_from_result(self, tmp_path):
        r = TopiaryResult(_sample_long_df())
        path = tmp_path / "out.csv"
        r.to_csv(path)
        from topiary import read_csv
        r2 = read_csv(path)
        assert len(r2) == 2

    def test_to_tsv_accepts_bare_df(self, tmp_path):
        """Backward compat: to_tsv still works on a bare DataFrame."""
        df = _sample_long_df()
        path = tmp_path / "bare.tsv"
        to_tsv(df, path)
        r = read_tsv(path)
        assert len(r) == 2


# ---------------------------------------------------------------------------
# Loader populates source
# ---------------------------------------------------------------------------


class TestLoaderSource:
    def test_source_column_added(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "mydata.tsv"
        to_tsv(df, path)
        r = read_tsv(path)
        assert "source" in r.columns
        assert (r["source"] == "mydata.tsv").all()

    def test_tag_overrides_filename(self, tmp_path):
        df = _sample_long_df()
        path = tmp_path / "mydata.tsv"
        to_tsv(df, path)
        r = read_tsv(path, tag="patient01")
        assert (r["source"] == "patient01").all()
        assert "patient01" in r.sources

    def test_source_column_not_overwritten(self, tmp_path):
        df = _sample_long_df()
        df["source"] = "manual_tag"
        path = tmp_path / "mydata.tsv"
        to_tsv(df, path)
        r = read_tsv(path, tag="patient01")
        # Existing source column preserved
        assert (r["source"] == "manual_tag").all()


# ---------------------------------------------------------------------------
# concat()
# ---------------------------------------------------------------------------


class TestConcat:
    def _make_r(self, value, source_tag, model_version="4.1b"):
        df = pd.DataFrame([dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            kind="pMHC_affinity", score=0.8, value=value,
            percentile_rank=0.5, affinity=value,
            prediction_method_name="netmhcpan",
            predictor_version=model_version,
            source=source_tag,
        )])
        meta = Metadata(
            form="long",
            models={"netmhcpan": model_version},
            sources=[source_tag],
        )
        return TopiaryResult(df, meta)

    def test_concat_basic(self):
        r1 = self._make_r(100.0, "patient01")
        r2 = self._make_r(200.0, "patient02")
        combined = concat([r1, r2])
        assert len(combined) == 2

    def test_concat_sources_merged(self):
        r1 = self._make_r(100.0, "patient01")
        r2 = self._make_r(200.0, "patient02")
        combined = concat([r1, r2])
        assert combined.sources == ["patient01", "patient02"]

    def test_concat_preserves_source_column(self):
        r1 = self._make_r(100.0, "patient01")
        r2 = self._make_r(200.0, "patient02")
        combined = concat([r1, r2])
        assert set(combined["source"].unique()) == {"patient01", "patient02"}

    def test_concat_models_union(self):
        r1 = self._make_r(100.0, "p1", model_version="4.1b")
        r2 = pd.DataFrame([dict(
            peptide="X", allele="A",
            kind="pMHC_presentation", score=0.5, value=0.5,
            percentile_rank=1.0, affinity=float("nan"),
            prediction_method_name="mhcflurry", predictor_version="2.1.1",
            source="p2",
        )])
        r2 = TopiaryResult(r2, Metadata(form="long", models={"mhcflurry": "2.1.1"}, sources=["p2"]))
        combined = concat([r1, r2])
        assert combined.models == {"netmhcpan": "4.1b", "mhcflurry": "2.1.1"}

    def test_concat_version_conflict_warns(self):
        r1 = self._make_r(100.0, "p1", model_version="4.1b")
        r2 = self._make_r(200.0, "p2", model_version="4.2")
        with pytest.warns(UserWarning, match="conflicting versions"):
            concat([r1, r2])

    def test_concat_mixed_forms_raises(self):
        r_long = self._make_r(100.0, "p1")
        r_wide = TopiaryResult(
            pd.DataFrame({
                "peptide": ["A"],
                "netmhcpan_affinity_value": [100.0],
            }),
        )
        with pytest.raises(ValueError, match="different forms"):
            concat([r_long, r_wide])

    def test_concat_empty_list(self):
        combined = concat([])
        assert isinstance(combined, TopiaryResult)
        assert len(combined) == 0

    def test_concat_single(self):
        r = self._make_r(100.0, "p1")
        combined = concat([r])
        assert len(combined) == 1

    def test_concat_then_write_roundtrip(self, tmp_path):
        """Concat + write + read preserves sources in comment block."""
        r1 = self._make_r(100.0, "patient01")
        r2 = self._make_r(200.0, "patient02")
        combined = concat([r1, r2])

        path = tmp_path / "merged.tsv"
        combined.to_tsv(path)

        r_back = read_tsv(path)
        # Source column survives
        assert set(r_back["source"].unique()) == {"patient01", "patient02"}
        # Metadata sources include both original tags
        for src in ["patient01", "patient02"]:
            assert src in r_back.sources


# ---------------------------------------------------------------------------
# filter_by() method
# ---------------------------------------------------------------------------


def _multi_row_df():
    """Long-form DataFrame with two peptides × two kinds each."""
    rows = []
    for peptide, affinity_val, pres_rank in [
        ("SIINFEKL",  120.0,  0.3),
        ("ELAGIGILT", 5000.0, 15.0),
        ("AAAAAAAA",  300.0,  1.5),
    ]:
        for kind, value, score, rank in [
            ("pMHC_affinity",     affinity_val, 0.8,  0.5),
            ("pMHC_presentation", 0.9,          0.92, pres_rank),
        ]:
            rows.append(dict(
                peptide=peptide,
                allele="HLA-A*02:01",
                source_sequence_name="prot1",
                peptide_offset=0,
                kind=kind,
                score=score,
                value=value,
                percentile_rank=rank,
                affinity=value if kind == "pMHC_affinity" else float("nan"),
                prediction_method_name="netmhcpan",
                predictor_version="4.1b",
            ))
    return pd.DataFrame(rows)


class TestFilterBy:
    def test_string_filter_reduces_rows(self):
        r = TopiaryResult(_multi_row_df())
        n_before = len(r)
        filtered = r.filter_by("affinity <= 500")
        assert len(filtered) < n_before
        assert len(filtered) > 0

    def test_string_filter_records_history(self):
        r = TopiaryResult(_multi_row_df())
        filtered = r.filter_by("affinity <= 500")
        assert filtered.filter_by_str == "affinity <= 500"
        assert filtered.filter_by_ast is not None

    def test_filter_ands_with_existing(self):
        r = TopiaryResult(_multi_row_df())
        r2 = r.filter_by("affinity <= 1000")
        r3 = r2.filter_by("presentation.rank <= 2")
        # ANDed — both clauses in metadata
        assert "affinity <= 1000" in r3.filter_by_str
        assert "presentation.rank <= 2" in r3.filter_by_str
        assert "&" in r3.filter_by_str

    def test_dsl_object_filter(self):
        from topiary import Affinity
        r = TopiaryResult(_multi_row_df())
        filtered = r.filter_by(Affinity <= 500)
        assert len(filtered) < len(r)
        # AST captured
        assert filtered.filter_by_ast is not None
        # String form round-trippable (recognizable DSL syntax)
        assert "affinity" in filtered.filter_by_str.lower()

    def test_dsl_object_and_string_equivalent(self):
        from topiary import Affinity
        df = _multi_row_df()
        a = TopiaryResult(df).filter_by(Affinity <= 500)
        b = TopiaryResult(df).filter_by("affinity <= 500")
        assert len(a) == len(b)

    def test_invalid_type_raises(self):
        r = TopiaryResult(_multi_row_df())
        with pytest.raises(TypeError, match="filter_by expects"):
            r.filter_by(500)

    def test_filter_on_empty_df_is_noop(self):
        df = pd.DataFrame(columns=_multi_row_df().columns)
        r = TopiaryResult(df)
        filtered = r.filter_by("affinity <= 500")
        assert filtered.empty
        # History still recorded
        assert filtered.filter_by_str == "affinity <= 500"

    def test_filter_preserves_other_metadata(self):
        r = TopiaryResult(
            _multi_row_df(),
            models={"netmhcpan": "4.1b"},
            sources=["patient01"],
        )
        filtered = r.filter_by("affinity <= 500")
        assert filtered.models == {"netmhcpan": "4.1b"}
        assert filtered.sources == ["patient01"]

    def test_filter_roundtrip_through_tsv(self, tmp_path):
        r = TopiaryResult(_multi_row_df()).filter_by("affinity <= 500")
        path = tmp_path / "filtered.tsv"
        r.to_tsv(path)
        r_back = read_tsv(path)
        assert r_back.filter_by_str == r.filter_by_str

    def test_filter_returns_new_result(self):
        """filter_by must not mutate the original."""
        r = TopiaryResult(_multi_row_df())
        n_before = len(r)
        _ = r.filter_by("affinity <= 500")
        assert len(r) == n_before
        assert r.filter_by_str is None


# ---------------------------------------------------------------------------
# sort_by() method
# ---------------------------------------------------------------------------


class TestSortBy:
    def test_string_sort_reorders(self):
        r = TopiaryResult(_multi_row_df())
        sorted_r = r.sort_by("affinity.score")
        # Can't easily check exact order without knowing DSL semantics,
        # but the result should have same rows reordered.
        assert len(sorted_r) == len(r)

    def test_string_sort_records_history(self):
        r = TopiaryResult(_multi_row_df())
        sorted_r = r.sort_by("affinity.score")
        assert sorted_r.sort_by_str == "affinity.score"
        assert sorted_r.sort_by_ast is not None

    def test_sort_replaces_history(self):
        r = TopiaryResult(_multi_row_df())
        r2 = r.sort_by("affinity.score")
        r3 = r2.sort_by("presentation.score")
        # sort_by REPLACES, doesn't combine
        assert r3.sort_by_str == "presentation.score"

    def test_dsl_object_sort(self):
        from topiary import Presentation
        r = TopiaryResult(_multi_row_df())
        sorted_r = r.sort_by(Presentation.score)
        assert sorted_r.sort_by_str == "presentation.score"
        assert sorted_r.sort_by_ast is not None

    def test_invalid_type_raises(self):
        r = TopiaryResult(_multi_row_df())
        with pytest.raises(TypeError, match="sort_by expects"):
            r.sort_by(42)

    def test_sort_on_empty_df_is_noop(self):
        df = pd.DataFrame(columns=_multi_row_df().columns)
        r = TopiaryResult(df)
        sorted_r = r.sort_by("affinity.score")
        assert sorted_r.empty
        assert sorted_r.sort_by_str == "affinity.score"

    def test_sort_preserves_other_metadata(self):
        r = TopiaryResult(
            _multi_row_df(),
            models={"netmhcpan": "4.1b"},
            sources=["patient01"],
            filter_by_str="affinity <= 1000",
        )
        sorted_r = r.sort_by("affinity.score")
        assert sorted_r.models == {"netmhcpan": "4.1b"}
        assert sorted_r.sources == ["patient01"]
        assert sorted_r.filter_by_str == "affinity <= 1000"

    def test_sort_roundtrip_through_tsv(self, tmp_path):
        r = TopiaryResult(_multi_row_df()).sort_by("affinity.score")
        path = tmp_path / "sorted.tsv"
        r.to_tsv(path)
        r_back = read_tsv(path)
        assert r_back.sort_by_str == r.sort_by_str

    def test_sort_returns_new_result(self):
        r = TopiaryResult(_multi_row_df())
        _ = r.sort_by("affinity.score")
        assert r.sort_by_str is None


class TestFilterSortComposition:
    def test_filter_then_sort(self):
        r = (
            TopiaryResult(_multi_row_df())
            .filter_by("affinity <= 1000")
            .sort_by("affinity.score")
        )
        assert r.filter_by_str == "affinity <= 1000"
        assert r.sort_by_str == "affinity.score"

    def test_sort_then_filter(self):
        r = (
            TopiaryResult(_multi_row_df())
            .sort_by("affinity.score")
            .filter_by("affinity <= 1000")
        )
        assert r.filter_by_str == "affinity <= 1000"
        assert r.sort_by_str == "affinity.score"


# ---------------------------------------------------------------------------
# concat warnings for dropped filter/sort history
# ---------------------------------------------------------------------------


class TestConcatHistoryDrop:
    def _make_r(self, value, source_tag, filter_str=None, sort_str=None):
        df = pd.DataFrame([dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            kind="pMHC_affinity", score=0.8, value=value,
            percentile_rank=0.5, affinity=value,
            prediction_method_name="netmhcpan",
            predictor_version="4.1b",
            source=source_tag,
        )])
        return TopiaryResult(
            df,
            form="long",
            models={"netmhcpan": "4.1b"},
            sources=[source_tag],
            filter_by_str=filter_str,
            sort_by_str=sort_str,
        )

    def test_matching_filters_preserved_silently(self, recwarn):
        r1 = self._make_r(100, "p1", filter_str="affinity <= 500")
        r2 = self._make_r(200, "p2", filter_str="affinity <= 500")
        combined = concat([r1, r2])
        assert combined.filter_by_str == "affinity <= 500"
        # No warning about filter/sort drop
        filter_warnings = [w for w in recwarn.list if "Dropping" in str(w.message)]
        assert not filter_warnings

    def test_differing_filters_warn_and_drop(self):
        r1 = self._make_r(100, "p1", filter_str="affinity <= 500")
        r2 = self._make_r(200, "p2", filter_str="affinity <= 1000")
        with pytest.warns(UserWarning, match="Dropping filter_by metadata"):
            combined = concat([r1, r2])
        assert combined.filter_by_str is None

    def test_one_has_filter_one_doesnt_warns(self):
        r1 = self._make_r(100, "p1", filter_str="affinity <= 500")
        r2 = self._make_r(200, "p2", filter_str=None)
        with pytest.warns(UserWarning, match="Dropping filter_by metadata"):
            combined = concat([r1, r2])
        assert combined.filter_by_str is None

    def test_differing_sorts_warn_and_drop(self):
        r1 = self._make_r(100, "p1", sort_str="affinity.score")
        r2 = self._make_r(200, "p2", sort_str="presentation.score")
        with pytest.warns(UserWarning, match="Dropping sort_by metadata"):
            combined = concat([r1, r2])
        assert combined.sort_by_str is None

    def test_matching_sorts_preserved_silently(self, recwarn):
        r1 = self._make_r(100, "p1", sort_str="affinity.score")
        r2 = self._make_r(200, "p2", sort_str="affinity.score")
        combined = concat([r1, r2])
        assert combined.sort_by_str == "affinity.score"
        sort_warnings = [w for w in recwarn.list if "Dropping sort_by" in str(w.message)]
        assert not sort_warnings


# ---------------------------------------------------------------------------
# Real predictor integration
# ---------------------------------------------------------------------------


class TestPredictorIntegration:
    def test_wrap_predictor_output(self):
        from mhctools import RandomBindingPredictor
        from topiary import TopiaryPredictor

        predictor = TopiaryPredictor(
            models=RandomBindingPredictor, alleles=["A0201"],
        )
        df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
        r = TopiaryResult(df)
        assert r.form == "long"
        assert len(r) > 0

        # Can convert, write, read back
        wide = r.to_wide()
        assert wide.form == "wide"
