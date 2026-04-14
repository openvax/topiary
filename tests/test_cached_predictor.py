"""Tests for topiary.cached.CachedPredictor — core invariants, loader
round-trips, fallback + version enforcement, and integration with
:class:`TopiaryPredictor`."""

import numpy as np
import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import CachedPredictor, TopiaryPredictor
from topiary.cached import mhcflurry_composite_version


ALLELES = ["HLA-A*02:01"]


def _row(
    peptide="SIINFEKLA",
    allele="HLA-A*02:01",
    *,
    score=0.5,
    affinity=150.0,
    percentile_rank=2.0,
    predictor_name="random",
    predictor_version="1.0",
):
    return {
        "peptide": peptide,
        "allele": allele,
        "peptide_length": len(peptide),
        "score": score,
        "affinity": affinity,
        "percentile_rank": percentile_rank,
        "prediction_method_name": predictor_name,
        "predictor_version": predictor_version,
    }


def _df(rows):
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Construction + invariant
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_from_dataframe_minimal(self):
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        assert cache.prediction_method_name == "random"
        assert cache.predictor_version == "1.0"
        assert cache.alleles == ["HLA-A*02:01"]
        assert cache.default_peptide_lengths == [9]

    def test_from_dataframe_backfills_from_args(self):
        df = _df([_row()]).drop(
            columns=["prediction_method_name", "predictor_version"],
        )
        cache = CachedPredictor.from_dataframe(
            df,
            prediction_method_name="mhcflurry",
            predictor_version="2.1.0",
        )
        assert cache.prediction_method_name == "mhcflurry"
        assert cache.predictor_version == "2.1.0"

    def test_from_dataframe_derives_peptide_length(self):
        df = _df([_row()]).drop(columns=["peptide_length"])
        cache = CachedPredictor.from_dataframe(df)
        assert cache.default_peptide_lengths == [9]

    def test_missing_required_columns_raises(self):
        df = _df([_row()]).drop(columns=["prediction_method_name"])
        with pytest.raises(ValueError, match="missing required columns"):
            CachedPredictor(df)

    def test_empty_df_and_no_fallback_raises(self):
        df = pd.DataFrame(columns=list(_row().keys()))
        with pytest.raises(ValueError, match="either .df. .* or .fallback."):
            CachedPredictor(df)


class TestVersionInvariant:
    def test_mixed_versions_reject(self):
        rows = [
            _row(peptide="SIINFEKLA", predictor_version="1.0"),
            _row(peptide="SIINFEKLB", predictor_version="1.1"),
        ]
        with pytest.raises(ValueError, match="multiple"):
            CachedPredictor(_df(rows))

    def test_mixed_predictor_names_reject(self):
        rows = [
            _row(peptide="SIINFEKLA", predictor_name="random"),
            _row(peptide="SIINFEKLB", predictor_name="mhcflurry"),
        ]
        with pytest.raises(ValueError, match="multiple"):
            CachedPredictor(_df(rows))

    def test_version_preserved_in_output_rows(self):
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        out = cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert (out["prediction_method_name"] == "random").all()
        assert (out["predictor_version"] == "1.0").all()


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestPredictPeptides:
    def test_hit_returns_row(self):
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        out = cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert len(out) == 1
        assert out.iloc[0]["peptide"] == "SIINFEKLA"
        assert out.iloc[0]["affinity"] == 150.0

    def test_miss_raises_without_fallback(self):
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        with pytest.raises(KeyError, match="missed"):
            cache.predict_peptides_dataframe(["QQQQQQQQQ"])

    def test_multiple_alleles_cross_product(self):
        rows = [
            _row(peptide="SIINFEKLA", allele="HLA-A*02:01"),
            _row(peptide="SIINFEKLA", allele="HLA-B*07:02"),
            _row(peptide="GILGFVFTL", allele="HLA-A*02:01"),
            _row(peptide="GILGFVFTL", allele="HLA-B*07:02"),
        ]
        cache = CachedPredictor.from_dataframe(_df(rows))
        out = cache.predict_peptides_dataframe(["SIINFEKLA", "GILGFVFTL"])
        assert len(out) == 4
        assert set(out["allele"]) == {"HLA-A*02:01", "HLA-B*07:02"}
        assert set(out["peptide"]) == {"SIINFEKLA", "GILGFVFTL"}


class TestPredictProteins:
    def test_sliding_window_hits(self):
        # 9-mer sliding window over 'MASIINFEKLG' → positions 0..2
        rows = [
            _row(peptide="MASIINFEK"),
            _row(peptide="ASIINFEKL"),
            _row(peptide="SIINFEKLG"),
        ]
        cache = CachedPredictor.from_dataframe(_df(rows))
        out = cache.predict_proteins_dataframe({"prot": "MASIINFEKLG"})
        assert len(out) == 3
        assert sorted(out["offset"].tolist()) == [0, 1, 2]
        assert (out["source_sequence_name"] == "prot").all()

    def test_sliding_window_miss_raises(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(peptide="MASIINFEK")]),
        )
        # Second window 'ASIINFEKL' isn't in the cache
        with pytest.raises(KeyError):
            cache.predict_proteins_dataframe({"prot": "MASIINFEKLG"})


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _matched_fallback(name="random", version=None):
    """A live predictor whose output will match the cache's (name, version)
    once we re-tag its output."""
    return _TaggedRandomPredictor(
        alleles=ALLELES,
        default_peptide_lengths=[9],
        predictor_name=name,
        predictor_version=version,
    )


class _TaggedRandomPredictor(RandomBindingPredictor):
    """RandomBindingPredictor that stamps caller-chosen
    ``prediction_method_name`` / ``predictor_version`` on its output —
    lets us test the version-match invariant deterministically."""

    def __init__(self, *, predictor_name, predictor_version, **kwargs):
        super().__init__(**kwargs)
        self._stamp_name = predictor_name
        self._stamp_version = predictor_version

    def _stamp(self, df):
        df = df.copy()
        df["prediction_method_name"] = self._stamp_name
        df["predictor_version"] = self._stamp_version
        return df

    def predict_peptides_dataframe(self, peptides):
        return self._stamp(super().predict_peptides_dataframe(peptides))

    def predict_proteins_dataframe(self, name_to_seq):
        return self._stamp(super().predict_proteins_dataframe(name_to_seq))


class TestFallback:
    def test_fallback_fills_miss(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row()]),
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        out = cache.predict_peptides_dataframe(["GILGFVFTL"])
        assert len(out) == 1
        assert out.iloc[0]["peptide"] == "GILGFVFTL"

    def test_fallback_populates_cache(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row()]),
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        cache.predict_peptides_dataframe(["GILGFVFTL"])
        # Second call should NOT hit fallback (we can verify by
        # observing the cache's internal df grew and contains it).
        assert ("GILGFVFTL", "HLA-A*02:01", 9) in cache._index
        assert len(cache._df) == 2

    def test_fallback_version_mismatch_rejects(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(predictor_version="1.0")]),
            fallback=_matched_fallback(name="random", version="2.0"),
        )
        with pytest.raises(ValueError, match="version mismatch"):
            cache.predict_peptides_dataframe(["GILGFVFTL"])

    def test_fallback_predictor_name_mismatch_rejects(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(predictor_name="random", predictor_version="1.0")]),
            fallback=_matched_fallback(name="mhcflurry", version="1.0"),
        )
        with pytest.raises(ValueError, match="predictor_name mismatch"):
            cache.predict_peptides_dataframe(["GILGFVFTL"])

    def test_partial_allele_cache_preserves_existing_rows(self):
        """Regression for vaxrank review concern #1: when a cache has
        peptide P for allele A but not B, a fallback call triggered by
        the (P, B) miss returns rows for ALL the fallback's alleles
        including A.  The pre-loaded (P, A) row must not be overwritten
        by the fallback's output."""
        cache = CachedPredictor.from_dataframe(
            _df([_row(
                peptide="SIINFEKLA", allele="HLA-A*02:01",
                affinity=42.0,  # sentinel we can check survives
            )]),
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        # Trigger fallback: SIINFEKLA is a hit for A*02:01 but the
        # fallback's own alleles may include it too, so without the
        # preserve-existing guard, the random-predictor output would
        # overwrite the 42.0 sentinel.
        cache.predict_peptides_dataframe(["SIINFEKLA"])
        preserved = cache._index[("SIINFEKLA", "HLA-A*02:01", 9)]
        assert preserved["affinity"] == 42.0


# ---------------------------------------------------------------------------
# Null / empty version rejection
# ---------------------------------------------------------------------------


class TestNullVersionRejection:
    def test_none_version_rejected(self):
        row = _row()
        row["predictor_version"] = None
        with pytest.raises(ValueError, match="must be a non-empty string"):
            CachedPredictor(_df([row]))

    def test_nan_version_rejected(self):
        row = _row()
        row["predictor_version"] = np.nan
        with pytest.raises(ValueError, match="must be a non-empty string"):
            CachedPredictor(_df([row]))

    def test_empty_string_version_rejected(self):
        row = _row()
        row["predictor_version"] = ""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            CachedPredictor(_df([row]))

    def test_none_predictor_name_rejected(self):
        row = _row()
        row["prediction_method_name"] = None
        with pytest.raises(ValueError, match="must be a non-empty string"):
            CachedPredictor(_df([row]))


# ---------------------------------------------------------------------------
# Empty cache + fallback — lazy identity discovery
# ---------------------------------------------------------------------------


class TestEmptyCacheWithFallback:
    def test_empty_cache_plus_fallback_discovers_identity(self):
        cache = CachedPredictor(
            fallback=_matched_fallback(name="random", version="3.0"),
        )
        # Pre-query: identity is unset
        assert cache.prediction_method_name is None
        assert cache.predictor_version is None
        # First query: identity is adopted from fallback output
        out = cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert len(out) == 1
        assert cache.prediction_method_name == "random"
        assert cache.predictor_version == "3.0"

    def test_empty_cache_plus_fallback_caches_results(self):
        cache = CachedPredictor(
            fallback=_matched_fallback(name="random", version="3.0"),
        )
        cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert ("SIINFEKLA", "HLA-A*02:01", 9) in cache._index

    def test_empty_df_value_and_no_fallback_still_raises(self):
        empty = pd.DataFrame(columns=list(_row().keys()))
        with pytest.raises(ValueError, match="either .df. .* or .fallback."):
            CachedPredictor(empty)

    def test_save_before_any_query_raises(self, tmp_path):
        """Regression for vaxrank review concern #5: saving an
        un-queried empty-cache-plus-fallback would write a schema-only
        file that can't be round-tripped through from_topiary_output.
        Fail fast instead."""
        cache = CachedPredictor(
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        path = tmp_path / "cache.tsv"
        with pytest.raises(ValueError, match="no rows to persist"):
            cache.save(path)
        assert not path.exists()

    def test_save_works_after_fallback_populates(self, tmp_path):
        """Once identity is locked in and rows exist, save() proceeds."""
        cache = CachedPredictor(
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        cache.predict_peptides_dataframe(["SIINFEKLA"])
        path = tmp_path / "cache.tsv"
        cache.save(path)
        assert path.exists()
        reloaded = CachedPredictor.from_topiary_output(path)
        assert reloaded.prediction_method_name == "random"


# ---------------------------------------------------------------------------
# Version equivalence (also_accept_versions)
# ---------------------------------------------------------------------------


class TestAlsoAcceptVersions:
    def test_equivalent_version_accepted(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(predictor_version="2.2.0rc2")]),
            fallback=_matched_fallback(name="random", version="2.2.0"),
            also_accept_versions={"2.2.0"},
        )
        out = cache.predict_peptides_dataframe(["GILGFVFTL"])
        assert len(out) == 1
        # Cache's own identity doesn't change — only acceptance widens.
        assert cache.predictor_version == "2.2.0rc2"

    def test_non_equivalent_version_still_rejects(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(predictor_version="2.2.0rc2")]),
            fallback=_matched_fallback(name="random", version="1.0"),
            also_accept_versions={"2.2.0"},  # different version in set
        )
        with pytest.raises(ValueError, match="version mismatch"):
            cache.predict_peptides_dataframe(["GILGFVFTL"])

    def test_default_empty_set_is_strict(self):
        cache = CachedPredictor.from_dataframe(
            _df([_row(predictor_version="2.2.0rc2")]),
            fallback=_matched_fallback(name="random", version="2.2.0"),
        )
        with pytest.raises(ValueError, match="version mismatch"):
            cache.predict_peptides_dataframe(["GILGFVFTL"])


# ---------------------------------------------------------------------------
# mhcflurry_composite_version helper + auto-default in from_mhcflurry
# ---------------------------------------------------------------------------


class TestMhcflurryCompositeVersion:
    """Tests use stubbed mhcflurry / mhcflurry.downloads modules to
    avoid importing tensorflow (which collides with pandas' libomp on
    macOS and isn't needed to verify the helper's composition logic)."""

    @staticmethod
    def _stub_mhcflurry(monkeypatch, *, version="2.2.1", release="2.2.0"):
        import sys
        import types
        fake_mhcflurry = types.ModuleType("mhcflurry")
        fake_mhcflurry.__version__ = version
        fake_downloads = types.ModuleType("mhcflurry.downloads")
        fake_downloads.get_current_release = lambda: release
        fake_mhcflurry.downloads = fake_downloads
        monkeypatch.setitem(sys.modules, "mhcflurry", fake_mhcflurry)
        monkeypatch.setitem(sys.modules, "mhcflurry.downloads", fake_downloads)

    def test_returns_composite_string(self, monkeypatch):
        self._stub_mhcflurry(monkeypatch, version="2.2.1", release="2.2.0")
        assert mhcflurry_composite_version() == "2.2.1+release-2.2.0"

    def test_raises_when_mhcflurry_missing(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "mhcflurry", None)
        with pytest.raises(RuntimeError, match="not installed"):
            mhcflurry_composite_version()

    def test_raises_when_release_missing(self, monkeypatch):
        self._stub_mhcflurry(monkeypatch, release="")
        with pytest.raises(RuntimeError, match="no active model release"):
            mhcflurry_composite_version()

    def test_from_mhcflurry_autocomposes_version(self, monkeypatch, tmp_path):
        self._stub_mhcflurry(monkeypatch, version="2.2.1", release="2.2.0")
        df = pd.DataFrame([{
            "peptide": "SIINFEKLA",
            "allele": "HLA-A*02:01",
            "mhcflurry_affinity": 100.0,
        }])
        path = tmp_path / "mhcflurry.csv"
        df.to_csv(path, index=False)
        cache = CachedPredictor.from_mhcflurry(path)  # no predictor_version
        assert cache.prediction_method_name == "mhcflurry"
        assert cache.predictor_version == "2.2.1+release-2.2.0"

    def test_from_mhcflurry_explicit_version_overrides_autocompose(
        self, monkeypatch, tmp_path,
    ):
        # Even with mhcflurry stubbed in, an explicit predictor_version wins.
        self._stub_mhcflurry(monkeypatch, version="2.2.1", release="2.2.0")
        df = pd.DataFrame([{
            "peptide": "SIINFEKLA", "allele": "HLA-A*02:01",
            "mhcflurry_affinity": 100.0,
        }])
        path = tmp_path / "mhcflurry.csv"
        df.to_csv(path, index=False)
        cache = CachedPredictor.from_mhcflurry(
            path, predictor_version="custom-label",
        )
        assert cache.predictor_version == "custom-label"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_from_topiary_output_roundtrip_tsv(self, tmp_path):
        path = tmp_path / "cache.tsv"
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        cache.save(path)
        reloaded = CachedPredictor.from_topiary_output(path)
        assert reloaded.prediction_method_name == "random"
        assert reloaded.predictor_version == "1.0"
        out = reloaded.predict_peptides_dataframe(["SIINFEKLA"])
        assert out.iloc[0]["affinity"] == 150.0

    def test_from_topiary_output_roundtrip_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "cache.parquet"
        cache = CachedPredictor.from_dataframe(_df([_row()]))
        cache.save(path)
        reloaded = CachedPredictor.from_topiary_output(path)
        assert reloaded.prediction_method_name == "random"

    def test_from_tsv_with_column_mapping(self, tmp_path):
        # Simulate a third-party TSV with non-canonical column names
        df = pd.DataFrame([{
            "peptide": "SIINFEKLA",
            "allele": "HLA-A*02:01",
            "ic50_nM": 150.0,
            "rank": 2.0,
        }])
        path = tmp_path / "third_party.tsv"
        df.to_csv(path, sep="\t", index=False)
        cache = CachedPredictor.from_tsv(
            path,
            columns={"affinity": "ic50_nM", "percentile_rank": "rank"},
            prediction_method_name="thirdparty",
            predictor_version="0.9",
        )
        out = cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert out.iloc[0]["affinity"] == 150.0
        assert out.iloc[0]["percentile_rank"] == 2.0

    def test_from_mhcflurry(self, tmp_path):
        df = pd.DataFrame([{
            "peptide": "SIINFEKLA",
            "allele": "HLA-A*02:01",
            "mhcflurry_affinity": 100.0,
            "mhcflurry_affinity_percentile": 1.5,
            "mhcflurry_presentation_score": 0.8,
        }])
        path = tmp_path / "mhcflurry.csv"
        df.to_csv(path, index=False)
        cache = CachedPredictor.from_mhcflurry(path, predictor_version="2.0.6")
        assert cache.prediction_method_name == "mhcflurry"
        assert cache.predictor_version == "2.0.6"
        out = cache.predict_peptides_dataframe(["SIINFEKLA"])
        assert out.iloc[0]["affinity"] == 100.0
        assert out.iloc[0]["percentile_rank"] == 1.5
        assert out.iloc[0]["score"] == 0.8


# ---------------------------------------------------------------------------
# TopiaryPredictor integration
# ---------------------------------------------------------------------------


class TestTopiaryPredictorIntegration:
    def test_cache_as_model_predict_from_sequences(self):
        # Pre-populate a cache for every 9-mer of 'MASIINFEKLGGG'
        seq = "MASIINFEKLGGG"
        rows = [
            _row(peptide=seq[i:i + 9])
            for i in range(len(seq) - 8)
        ]
        cache = CachedPredictor.from_dataframe(_df(rows))
        predictor = TopiaryPredictor(models=cache)
        df = predictor.predict_from_named_sequences({"prot": seq})
        assert len(df) == len(seq) - 8
        assert "peptide_offset" in df.columns
        assert "prediction_method_name" in df.columns

    def test_cache_as_model_with_fallback_fills_sequence(self):
        # Cache has only the first 9-mer; fallback covers the rest.
        cache = CachedPredictor.from_dataframe(
            _df([_row(peptide="MASIINFEK")]),
            fallback=_matched_fallback(name="random", version="1.0"),
        )
        predictor = TopiaryPredictor(models=cache)
        df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
        assert len(df) == 5  # 5 sliding 9-mers over the 13-aa sequence
