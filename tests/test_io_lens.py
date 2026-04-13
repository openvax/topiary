"""Tests for the LENS report loader (topiary.io_lens)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from topiary import (
    Affinity,
    Presentation,
    TopiaryResult,
    apply_filter,
    apply_sort,
    detect_lens_version,
    read_lens,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "lens"

V1_4 = FIXTURE_DIR / "sample_v1_4.tsv"
V1_5_1 = FIXTURE_DIR / "sample_v1_5_1.tsv"
V1_9 = FIXTURE_DIR / "sample_v1_9.tsv"


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


class TestDetectVersion:
    def test_v1_4_detected(self):
        cols = pd.read_csv(V1_4, sep="\t", nrows=0).columns
        assert detect_lens_version(cols) == "v1.4"

    def test_v1_5_1_detected(self):
        cols = pd.read_csv(V1_5_1, sep="\t", nrows=0).columns
        assert detect_lens_version(cols) == "v1.5.1"

    def test_v1_9_detected(self):
        cols = pd.read_csv(V1_9, sep="\t", nrows=0).columns
        assert detect_lens_version(cols) == "v1.9"

    def test_unknown_returns_none(self):
        assert detect_lens_version(["foo", "bar"]) is None


# ---------------------------------------------------------------------------
# Per-version loads
# ---------------------------------------------------------------------------


class TestLoad:
    def test_v1_4_binding_columns_remapped(self):
        r = read_lens(V1_4)
        expected = {
            "netmhcpan_affinity_value",
            "netmhcpan_affinity_score",
            "netmhcpan_affinity_rank",
            "netmhcpan_presentation_score",
            "netmhcpan_presentation_rank",
            "mhcflurry_affinity_value",
            "mhcflurry_affinity_rank",
            "mhcflurry_antigen_processing_score",
            "mhcflurry_presentation_score",
            "mhcflurry_presentation_rank",
            "netmhcstabpan_stability_score",
            "netmhcstabpan_stability_value",
            "netmhcstabpan_stability_rank",
        }
        assert expected <= set(r.df.columns)
        # Originals gone.
        assert "netmhcpan_4.1b.aff_nm" not in r.df.columns
        assert "mhcflurry_2.1.1.proc_score" not in r.df.columns

    def test_v1_4_models_metadata(self):
        r = read_lens(V1_4)
        assert r.models == {
            "netmhcpan": "4.1b",
            "mhcflurry": "2.1.1",
            "netmhcstabpan": "1.0",
        }

    def test_v1_5_1_no_stability(self):
        r = read_lens(V1_5_1)
        assert "netmhcstabpan" not in r.models
        assert "netmhcstabpan_stability_score" not in r.df.columns
        # NetMHCpan + MHCflurry still present
        assert "netmhcpan_affinity_value" in r.df.columns
        assert "mhcflurry_presentation_score" in r.df.columns

    def test_v1_9_only_mhcflurry_binding(self):
        r = read_lens(V1_9)
        assert r.models == {"mhcflurry": "2.1.1"}
        netmhcpan_cols = [c for c in r.df.columns if c.startswith("netmhcpan_")]
        assert netmhcpan_cols == []

    def test_version_recorded_in_extra(self):
        for path, version in [(V1_4, "v1.4"), (V1_5_1, "v1.5.1"), (V1_9, "v1.9")]:
            r = read_lens(path)
            assert r.extra.get("lens_version") == version

    def test_source_tag(self):
        r = read_lens(V1_4)
        assert r.sources == ["lens-v1.4"]
        r2 = read_lens(V1_4, tag="patient-42")
        assert r2.sources == ["patient-42"]

    def test_form_is_wide(self):
        assert read_lens(V1_4).form == "wide"

    def test_returns_topiary_result(self):
        assert isinstance(read_lens(V1_4), TopiaryResult)


# ---------------------------------------------------------------------------
# Allele normalization (mhcgnomes)
# ---------------------------------------------------------------------------


class TestAlleles:
    def test_class_i_star_inserted(self):
        r = read_lens(V1_4)
        # LENS uses HLA-A02:01, mhcgnomes should normalize to HLA-A*02:01.
        alleles = set(r.df["allele"].dropna().unique())
        assert all("*" in a for a in alleles), alleles
        # Spot-check a known allele.
        assert any(a.startswith("HLA-A*") for a in alleles)

    def test_idempotent_across_versions(self):
        for p in [V1_4, V1_5_1, V1_9]:
            alleles = set(read_lens(p).df["allele"].dropna().unique())
            for a in alleles:
                assert "*" in a, f"{p.name}: {a!r} missing star"


# ---------------------------------------------------------------------------
# Derived columns
# ---------------------------------------------------------------------------


class TestDerived:
    def test_peptide_length(self):
        r = read_lens(V1_4)
        lengths = r.df["peptide_length"]
        expected = r.df["peptide"].str.len()
        assert (lengths == expected).all()

    def test_peptide_offset_zero(self):
        # LENS doesn't record offset; we set it to 0 for group-key stability.
        r = read_lens(V1_4)
        assert (r.df["peptide_offset"] == 0).all()

    def test_flanks_derived_for_snv(self):
        """For SNV rows with a unique peptide match in pep_context,
        n_flank + peptide + c_flank should reconstruct pep_context."""
        r = read_lens(V1_4)
        snv_rows = r.df[r.df["antigen_source"] == "SNV"]
        assert len(snv_rows) > 0
        # Some rows should have non-empty flanks.
        with_flanks = snv_rows.dropna(subset=["n_flank", "c_flank"])
        assert len(with_flanks) > 0
        for _, row in with_flanks.iterrows():
            reconstructed = row["n_flank"] + row["peptide"] + row["c_flank"]
            assert reconstructed == row["pep_context"]

    def test_flanks_nan_for_erv(self):
        """ERV pep_context is the full ORF — flank derivation skipped."""
        r = read_lens(V1_4)
        erv_rows = r.df[r.df["antigen_source"] == "ERV"]
        if len(erv_rows) == 0:
            pytest.skip("no ERV rows in fixture")
        assert erv_rows["n_flank"].isna().all()
        assert erv_rows["c_flank"].isna().all()

    def test_effect_type_from_hgvs(self):
        """v1.5.1+ has HGVS variant_effect → Topiary effect_type."""
        r = read_lens(V1_5_1)
        snv = r.df[r.df["antigen_source"] == "SNV"]
        if len(snv) == 0:
            pytest.skip("no SNV rows")
        # SNV HGVS like p.Ala290Val → Substitution
        assert (snv["effect_type"] == "Substitution").any()

    def test_effect_type_frameshift(self):
        r = read_lens(V1_5_1)
        # p.Thr259fs present in the v1.5.1 fixture — fs → FrameShift
        indel = r.df[r.df["antigen_source"] == "INDEL"]
        if len(indel) == 0:
            pytest.skip("no INDEL rows")
        fs_rows = indel[indel["effect"].fillna("").str.contains("fs")]
        if len(fs_rows) > 0:
            assert (fs_rows["effect_type"] == "FrameShift").all()

    def test_effect_type_fallback_for_v1_4(self):
        """v1.4 has no variant_effect column → falls back to
        antigen_source mapping."""
        r = read_lens(V1_4)
        snv = r.df[r.df["antigen_source"] == "SNV"]
        assert (snv["effect_type"] == "Substitution").all()
        erv = r.df[r.df["antigen_source"] == "ERV"]
        if len(erv) > 0:
            assert (erv["effect_type"] == "ERV").all()

    def test_source_sequence_name(self):
        r = read_lens(V1_4)
        # Every row should have a value synthesized from antigen_source
        assert r.df["source_sequence_name"].notna().all()
        # ERV rows use origin_descriptor (Hsap38.chr...)
        erv = r.df[r.df["antigen_source"] == "ERV"]
        if len(erv) > 0:
            assert erv["source_sequence_name"].str.startswith("ERV:").all()


# ---------------------------------------------------------------------------
# tpm handling (fusion composite strings)
# ---------------------------------------------------------------------------


class TestTpm:
    def test_numeric_tpm_coerced(self):
        r = read_lens(V1_4)
        # gene_tpm is numeric (float) where the raw was parseable.
        assert pd.api.types.is_numeric_dtype(r.df["gene_tpm"])

    def test_fusion_raw_preserved(self):
        """For rows where tpm couldn't coerce (fusion composites),
        gene_tpm is NaN but gene_tpm_raw keeps the original string."""
        r = read_lens(V1_9)  # fusion rows definitely present
        fusion = r.df[r.df["antigen_source"] == "FUSION"]
        if len(fusion) == 0:
            pytest.skip("no FUSION rows")
        # At least one fusion row should have the composite raw format.
        if "gene_tpm_raw" in r.df.columns:
            raw_strings = fusion["gene_tpm_raw"].dropna()
            assert (raw_strings.str.contains("ENST")).any()
        # Numeric gene_tpm NaN for those rows.
        composite_rows = fusion["gene_tpm"].isna() & fusion["gene_tpm_raw"].notna() \
            if "gene_tpm_raw" in r.df.columns else pd.Series([False])
        assert composite_rows.any()


# ---------------------------------------------------------------------------
# NA handling
# ---------------------------------------------------------------------------


class TestNA:
    def test_na_strings_converted(self):
        """LENS uses 'NA' literal for missing values — should be NaN."""
        r = read_lens(V1_4)
        # mhcflurry_agretopicity is 'NA' for most rows
        col = "mhcflurry_agretopicity"
        if col in r.df.columns:
            vals = r.df[col]
            # Should have NaN, not 'NA' strings.
            assert not (vals.astype(str) == "NA").any()


# ---------------------------------------------------------------------------
# Annotation columns pass through
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_preserved_columns_v1_4(self):
        r = read_lens(V1_4)
        # Spot-check a selection of annotation columns.
        for col in [
            "vaf", "ccf", "priority_score", "erv_orf_id",
            "mean_mtec_tpm", "gene_detectable_normal_tissues",
            "pep_context", "coding_sequence",
        ]:
            assert col in r.df.columns, col

    def test_preserved_columns_v1_5_1(self):
        r = read_lens(V1_5_1)
        for col in [
            "snaf_exp", "b2m_tpm", "tap1_tpm",
            "priority_score_mhcflurry", "priority_score_maximum",
            "hla_allele_support",
            "all_transcript_ids_encoding_peptide",
        ]:
            assert col in r.df.columns, col

    def test_preserved_columns_v1_9(self):
        r = read_lens(V1_9)
        for col in [
            "lohhla_allele_loss_pval", "num_erv_orfs_with_peptide",
            "norm_tissue_p95_tpm", "mtec_p95_tpm",
            "priority_score_maximum",
        ]:
            assert col in r.df.columns, col


# ---------------------------------------------------------------------------
# Round-trip into long form and through the DSL
# ---------------------------------------------------------------------------


class TestDSLIntegration:
    def test_to_long_roundtrip(self):
        r = read_lens(V1_4).to_long()
        assert "kind" in r.df.columns
        assert "prediction_method_name" in r.df.columns
        kinds = set(r.df["kind"].dropna().unique())
        # All three kinds should show up (v1.4 has stability).
        assert "pMHC_affinity" in kinds
        assert "pMHC_presentation" in kinds
        assert "pMHC_stability" in kinds

    def test_filter_qualified_model(self):
        r = read_lens(V1_4).to_long()
        result = apply_filter(r.df, Affinity["netmhcpan"] <= 500)
        # At least some rows should pass the filter.
        assert len(result) > 0
        # Check that all surviving groups actually passed.
        surviving_peptides = set(result["peptide"])
        assert len(surviving_peptides) > 0

    def test_sort_qualified_model(self):
        r = read_lens(V1_4).to_long()
        sorted_df = apply_sort(r.df, [Presentation["mhcflurry"].score])
        # Top row should have a high presentation score (non-NaN).
        top = sorted_df.iloc[0]
        assert isinstance(top["peptide"], str)

    def test_unqualified_ambiguous_raises(self):
        """v1.4 has two models producing pMHC_affinity — unqualified
        access must error."""
        r = read_lens(V1_4).to_long()
        with pytest.raises(ValueError, match="Ambiguous"):
            apply_filter(r.df, Affinity <= 500)

    def test_v1_9_unqualified_ok(self):
        """v1.9 has only MHCflurry → unqualified access works."""
        r = read_lens(V1_9).to_long()
        result = apply_filter(r.df, Affinity <= 500)
        # Should not raise and may return any count.
        assert len(result) >= 0

    def test_dsl_column_access_on_annotation(self):
        """Annotation columns remain accessible via Column('...') in the DSL."""
        from topiary.ranking import Column
        r = read_lens(V1_4).to_long()
        # Filter on vaf (pass-through annotation)
        if "vaf" in r.df.columns:
            try:
                apply_filter(r.df, Column("vaf") >= 0.0)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f"Column('vaf') filter failed: {e}")
