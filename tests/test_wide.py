"""Tests for topiary.wide — wide/long DataFrame conversion."""

import numpy as np
import pandas as pd
import pytest

from topiary.wide import (
    PREDICTION_COLUMNS,
    _parse_wide_column,
    detect_form,
    from_wide,
    to_wide,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _long_df_single_model():
    """Single model, single kind (affinity)."""
    return pd.DataFrame([
        dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1", n_flank="MAA", c_flank="GGG",
            peptide_offset=3, peptide_length=8,
            kind="pMHC_affinity", score=0.85, value=120.0,
            percentile_rank=0.5, affinity=120.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
        dict(
            peptide="ELAGIGILT", allele="HLA-A*02:01",
            source_sequence_name="prot1", n_flank="", c_flank="",
            peptide_offset=10, peptide_length=9,
            kind="pMHC_affinity", score=0.3, value=5000.0,
            percentile_rank=15.0, affinity=5000.0,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
    ])


def _long_df_multi_kind():
    """Single model, two kinds (affinity + presentation)."""
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
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=3, peptide_length=8,
            kind="pMHC_presentation", score=0.92, value=0.95,
            percentile_rank=0.3, affinity=np.nan,
            prediction_method_name="netmhcpan", predictor_version="4.1b",
        ),
    ])


def _long_df_multi_model():
    """Two models, affinity kind."""
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
            peptide="SIINFEKL", allele="HLA-A*02:01",
            source_sequence_name="prot1",
            peptide_offset=3, peptide_length=8,
            kind="pMHC_affinity", score=0.9, value=85.0,
            percentile_rank=0.3, affinity=85.0,
            prediction_method_name="mhcflurry", predictor_version="2.1.1",
        ),
    ])


# ---------------------------------------------------------------------------
# _parse_wide_column tests
# ---------------------------------------------------------------------------


class TestParseWideColumn:
    def test_simple_affinity(self):
        assert _parse_wide_column("netmhcpan_affinity_value") == (
            "netmhcpan", "affinity", "value"
        )

    def test_affinity_score(self):
        assert _parse_wide_column("netmhcpan_affinity_score") == (
            "netmhcpan", "affinity", "score"
        )

    def test_affinity_rank(self):
        assert _parse_wide_column("mhcflurry_affinity_rank") == (
            "mhcflurry", "affinity", "rank"
        )

    def test_presentation(self):
        assert _parse_wide_column("mhcflurry_presentation_score") == (
            "mhcflurry", "presentation", "score"
        )

    def test_antigen_processing(self):
        assert _parse_wide_column("netmhcpan_antigen_processing_score") == (
            "netmhcpan", "antigen_processing", "score"
        )

    def test_proteasome_cleavage(self):
        assert _parse_wide_column("tool_proteasome_cleavage_rank") == (
            "tool", "proteasome_cleavage", "rank"
        )

    def test_tap_transport(self):
        assert _parse_wide_column("mhcflurry_tap_transport_value") == (
            "mhcflurry", "tap_transport", "value"
        )

    def test_erap_trimming(self):
        assert _parse_wide_column("mhcflurry_erap_trimming_score") == (
            "mhcflurry", "erap_trimming", "score"
        )

    def test_immunogenicity(self):
        assert _parse_wide_column("custom_immunogenicity_value") == (
            "custom", "immunogenicity", "value"
        )

    def test_stability(self):
        assert _parse_wide_column("netmhcstabpan_stability_value") == (
            "netmhcstabpan", "stability", "value"
        )

    def test_model_with_underscores(self):
        assert _parse_wide_column("net_mhc_pan_affinity_value") == (
            "net_mhc_pan", "affinity", "value"
        )

    def test_versioned_model(self):
        assert _parse_wide_column("netmhcpan_4.1b_affinity_value") == (
            "netmhcpan_4.1b", "affinity", "value"
        )

    def test_not_a_prediction_col(self):
        assert _parse_wide_column("gene_expression") is None

    def test_peptide(self):
        assert _parse_wide_column("peptide") is None

    def test_allele(self):
        assert _parse_wide_column("allele") is None

    def test_unknown_kind(self):
        assert _parse_wide_column("tool_bogus_kind_value") is None

    def test_unknown_field(self):
        assert _parse_wide_column("netmhcpan_affinity_median") is None

    def test_empty_string(self):
        assert _parse_wide_column("") is None

    def test_single_word(self):
        assert _parse_wide_column("affinity") is None


# ---------------------------------------------------------------------------
# detect_form tests
# ---------------------------------------------------------------------------


class TestDetectForm:
    def test_long(self):
        df = pd.DataFrame({"peptide": ["A"], "kind": ["pMHC_affinity"]})
        assert detect_form(df) == "long"

    def test_wide(self):
        df = pd.DataFrame({
            "peptide": ["A"],
            "netmhcpan_affinity_value": [100.0],
        })
        assert detect_form(df) == "wide"

    def test_unknown(self):
        df = pd.DataFrame({"peptide": ["A"], "gene_name": ["BRAF"]})
        assert detect_form(df) == "unknown"

    def test_empty_long(self):
        df = pd.DataFrame(columns=["peptide", "kind"])
        assert detect_form(df) == "long"

    def test_empty_wide(self):
        df = pd.DataFrame(columns=["peptide", "mhcflurry_presentation_score"])
        assert detect_form(df) == "wide"


# ---------------------------------------------------------------------------
# to_wide tests
# ---------------------------------------------------------------------------


class TestToWide:
    def test_single_model_single_kind(self):
        df = _long_df_single_model()
        wide = to_wide(df)
        assert "netmhcpan_affinity_value" in wide.columns
        assert "netmhcpan_affinity_score" in wide.columns
        assert "netmhcpan_affinity_rank" in wide.columns
        assert "kind" not in wide.columns
        assert "affinity" not in wide.columns
        assert len(wide) == 2

    def test_single_model_multi_kind(self):
        df = _long_df_multi_kind()
        wide = to_wide(df)
        assert "netmhcpan_affinity_value" in wide.columns
        assert "netmhcpan_presentation_value" in wide.columns
        assert "netmhcpan_presentation_score" in wide.columns
        # One row for the single peptide-allele group
        assert len(wide) == 1

    def test_multi_model(self):
        df = _long_df_multi_model()
        wide = to_wide(df)
        assert "netmhcpan_affinity_value" in wide.columns
        assert "mhcflurry_affinity_value" in wide.columns
        assert len(wide) == 1

    def test_multi_model_values_correct(self):
        df = _long_df_multi_model()
        wide = to_wide(df)
        row = wide.iloc[0]
        assert row["netmhcpan_affinity_value"] == 120.0
        assert row["mhcflurry_affinity_value"] == 85.0

    def test_multi_underscore_kind(self):
        df = pd.DataFrame([dict(
            peptide="SIINFEKL", allele="HLA-A*02:01",
            kind="antigen_processing", score=0.7, value=0.8,
            percentile_rank=1.5,
            prediction_method_name="mhcflurry", predictor_version="2.1.1",
        )])
        wide = to_wide(df)
        assert "mhcflurry_antigen_processing_value" in wide.columns
        assert "mhcflurry_antigen_processing_score" in wide.columns

    def test_version_collision_warns(self):
        df = pd.DataFrame([
            dict(
                peptide="SIINFEKL", allele="HLA-A*02:01",
                kind="pMHC_affinity", score=0.8, value=120.0,
                percentile_rank=0.5,
                prediction_method_name="netmhcpan", predictor_version="4.1b",
            ),
            dict(
                peptide="SIINFEKL", allele="HLA-A*02:01",
                kind="pMHC_affinity", score=0.7, value=200.0,
                percentile_rank=1.0,
                prediction_method_name="netmhcpan", predictor_version="4.2",
            ),
        ])
        with pytest.warns(UserWarning, match="Multiple predictor versions"):
            wide = to_wide(df)
        # Both versioned columns should exist
        assert "netmhcpan_4.1b_affinity_value" in wide.columns
        assert "netmhcpan_4.2_affinity_value" in wide.columns

    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "peptide", "allele", "kind", "score", "value",
            "percentile_rank", "prediction_method_name", "predictor_version",
            "affinity",
        ])
        wide = to_wide(df)
        assert len(wide) == 0
        assert "peptide" in wide.columns

    def test_affinity_column_dropped(self):
        df = _long_df_single_model()
        wide = to_wide(df)
        assert "affinity" not in wide.columns

    def test_extra_columns_preserved(self):
        df = _long_df_single_model()
        df["gene"] = "BRAF"
        df["gene_tpm"] = 42.5
        wide = to_wide(df)
        assert "gene" in wide.columns
        assert "gene_tpm" in wide.columns
        assert wide.iloc[0]["gene"] == "BRAF"

    def test_missing_kind_raises(self):
        df = pd.DataFrame({"peptide": ["A"], "score": [0.5]})
        with pytest.raises(ValueError, match="missing 'kind' column"):
            to_wide(df)

    def test_model_versions_in_attrs(self):
        df = _long_df_multi_model()
        wide = to_wide(df)
        models = wide.attrs.get("topiary_models", {})
        assert models.get("netmhcpan") == "4.1b"
        assert models.get("mhcflurry") == "2.1.1"

    def test_flanks_preserved(self):
        df = _long_df_single_model()
        wide = to_wide(df)
        assert "n_flank" in wide.columns
        assert "c_flank" in wide.columns
        maa_rows = wide[wide["n_flank"] == "MAA"]
        assert len(maa_rows) == 1


# ---------------------------------------------------------------------------
# from_wide tests
# ---------------------------------------------------------------------------


class TestFromWide:
    def test_roundtrip_single_model(self):
        orig = _long_df_single_model()
        wide = to_wide(orig)
        long = from_wide(wide)
        assert "kind" in long.columns
        assert "prediction_method_name" in long.columns
        assert set(long["kind"].unique()) == {"pMHC_affinity"}
        assert len(long) == 2
        # Values round-trip correctly.
        vals = sorted(long["value"].tolist())
        assert vals == sorted(orig["value"].tolist())

    def test_roundtrip_multi_kind(self):
        orig = _long_df_multi_kind()
        wide = to_wide(orig)
        long = from_wide(wide)
        assert set(long["kind"].unique()) == {"pMHC_affinity", "pMHC_presentation"}
        # One peptide × 2 kinds = 2 rows.
        assert len(long) == 2

    def test_roundtrip_multi_model(self):
        orig = _long_df_multi_model()
        wide = to_wide(orig)
        long = from_wide(wide)
        assert set(long["prediction_method_name"].unique()) == {
            "netmhcpan", "mhcflurry"
        }
        assert len(long) == 2

    def test_affinity_reconstructed(self):
        orig = _long_df_multi_kind()
        wide = to_wide(orig)
        long = from_wide(wide)
        aff_rows = long[long["kind"] == "pMHC_affinity"]
        non_aff_rows = long[long["kind"] != "pMHC_affinity"]
        assert not aff_rows["affinity"].isna().any()
        assert non_aff_rows["affinity"].isna().all()

    def test_unknown_columns_preserved(self):
        wide = pd.DataFrame({
            "peptide": ["SIINFEKL"],
            "allele": ["HLA-A*02:01"],
            "gene_name": ["BRAF"],
            "netmhcpan_affinity_value": [120.0],
            "netmhcpan_affinity_score": [0.85],
        })
        long = from_wide(wide)
        assert "gene_name" in long.columns
        assert long.iloc[0]["gene_name"] == "BRAF"

    def test_missing_field_becomes_nan(self):
        wide = pd.DataFrame({
            "peptide": ["SIINFEKL"],
            "allele": ["HLA-A*02:01"],
            "netmhcpan_affinity_value": [120.0],
            "netmhcpan_affinity_score": [0.85],
            # No netmhcpan_affinity_rank column
        })
        long = from_wide(wide)
        assert "percentile_rank" in long.columns
        assert pd.isna(long.iloc[0]["percentile_rank"])

    def test_empty_wide_df(self):
        wide = pd.DataFrame(columns=[
            "peptide", "allele", "netmhcpan_affinity_value",
        ])
        long = from_wide(wide)
        assert "kind" in long.columns
        assert len(long) == 0

    def test_with_metadata_versions(self):
        from topiary.io import Metadata
        wide = pd.DataFrame({
            "peptide": ["SIINFEKL"],
            "allele": ["HLA-A*02:01"],
            "netmhcpan_affinity_value": [120.0],
            "netmhcpan_affinity_score": [0.85],
        })
        meta = Metadata(models={"netmhcpan": "4.1b"})
        long = from_wide(wide, metadata=meta)
        assert long.iloc[0]["predictor_version"] == "4.1b"

    def test_multi_underscore_kind(self):
        wide = pd.DataFrame({
            "peptide": ["SIINFEKL"],
            "allele": ["HLA-A*02:01"],
            "mhcflurry_antigen_processing_score": [0.7],
            "mhcflurry_antigen_processing_value": [0.8],
        })
        long = from_wide(wide)
        assert long.iloc[0]["kind"] == "antigen_processing"
        assert long.iloc[0]["prediction_method_name"] == "mhcflurry"

    def test_no_prediction_columns(self):
        """Wide df with no prediction columns returns input + empty long cols."""
        wide = pd.DataFrame({
            "peptide": ["SIINFEKL"],
            "allele": ["HLA-A*02:01"],
            "gene_name": ["BRAF"],
        })
        long = from_wide(wide)
        assert "kind" in long.columns
        assert len(long) == 1
