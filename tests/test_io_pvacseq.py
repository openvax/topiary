"""Tests for the pVACseq report loader (topiary.io_pvacseq)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from topiary import (
    Affinity,
    Column,
    Presentation,
    SelfProteome,
    TopiaryResult,
    apply_filter,
    apply_sort,
    detect_pvacseq_format,
    melt_pvacseq_algorithms,
    read_pvacseq,
    read_tsv,
    stack_results,
    to_tsv,
    wt,
)
from topiary.io_pvacseq import _classify_effect, _reconstruct_wt_peptide

FIXTURE_DIR = Path(__file__).parent / "data" / "pvacseq"

MHC_I_AGG = FIXTURE_DIR / "mhc_i_aggregated.tsv"
MHC_II_AGG = FIXTURE_DIR / "mhc_ii_aggregated.tsv"
MHC_I_ALL = FIXTURE_DIR / "mhc_i_all_epitopes.tsv"
MHC_II_ALL = FIXTURE_DIR / "mhc_ii_all_epitopes.tsv"
MHC_I_AGG_PRESENTATION = FIXTURE_DIR / "mhc_i_aggregated_presentation.tsv"
MHC_I_ALL_PRESENTATION = FIXTURE_DIR / "mhc_i_all_epitopes_presentation.tsv"


def _data_row_count(path):
    return sum(1 for _ in open(path)) - 1


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_aggregated_detected(self):
        cols = pd.read_csv(MHC_I_AGG, sep="\t", nrows=0).columns
        assert detect_pvacseq_format(cols) == "aggregated"

    def test_all_epitopes_detected(self):
        cols = pd.read_csv(MHC_I_ALL, sep="\t", nrows=0).columns
        assert detect_pvacseq_format(cols) == "all_epitopes"

    def test_unrelated_columns_return_none(self):
        assert detect_pvacseq_format(["foo", "bar", "Peptide"]) is None


# ---------------------------------------------------------------------------
# Aggregated flavor
# ---------------------------------------------------------------------------


class TestLoadAggregated:
    def test_returns_topiary_result_in_long_form(self):
        r = read_pvacseq(MHC_I_AGG)
        assert isinstance(r, TopiaryResult)
        assert r.form == "long"
        assert r.df is r.long_df
        assert "pvacseq_affinity_value" in r.wide_df.columns
        assert len(r.wide_df) == len(r.df)

    def test_one_row_per_variant(self):
        r = read_pvacseq(MHC_I_AGG)
        assert len(r) == _data_row_count(MHC_I_AGG)

    def test_legacy_affinity_only_file_gains_no_rows_or_kinds(self):
        r = read_pvacseq(MHC_I_AGG)
        assert len(r) == _data_row_count(MHC_I_AGG)
        assert set(r.df["kind"]) == {"pMHC_affinity"}

    def test_presentation_percentile_becomes_a_separate_aggregate_row(self):
        r = read_pvacseq(MHC_I_AGG_PRESENTATION)
        assert len(r) == 2

        row = r.df[r.df["kind"] == "pMHC_presentation"].iloc[0]
        assert row["prediction_method_name"] == "pvacseq"
        assert row["percentile_rank"] == pytest.approx(0.46)
        assert row["wt_percentile_rank"] == pytest.approx(7.2)
        assert pd.isna(row["score"])
        assert pd.isna(row["value"])
        assert pd.isna(row["affinity"])

    def test_metadata_records_flavor(self):
        r = read_pvacseq(MHC_I_AGG)
        assert r.extra["pvacseq_format"] == "aggregated"
        assert any("pvacseq-aggregated" in s for s in r.sources)

    def test_long_form_schema_present(self):
        r = read_pvacseq(MHC_I_AGG)
        required = {
            "peptide", "allele", "kind", "score", "value", "affinity",
            "percentile_rank", "prediction_method_name", "predictor_version",
            "peptide_offset", "peptide_length", "source_sequence_name",
            "wt_peptide", "wt_value", "wt_percentile_rank", "wt_affinity",
            "wt_score",
        }
        assert required <= set(r.df.columns)

    def test_alleles_normalized(self):
        r = read_pvacseq(MHC_I_AGG)
        # MHC-I sample carries HLA-A/B/C only; mhcgnomes leaves the prefix.
        assert (r.df["allele"].dropna().str.startswith("HLA-")).all()

    def test_mhc_ii_alleles_get_hla_prefix(self):
        r = read_pvacseq(MHC_II_AGG)
        # Raw "DRB1*04:05" / "DPA1*02:01-DPB1*01:01" gain the HLA- prefix
        # from mhcgnomes; assertion is loose enough to survive future
        # re-slicing of the fixture to include other class II loci.
        assert (r.df["allele"].dropna().str.startswith("HLA-D")).all()

    def test_ic50_mt_populates_value_affinity_score(self):
        r = read_pvacseq(MHC_I_AGG)
        row = r.df.iloc[0]
        assert math.isclose(row["value"], 76.11, abs_tol=0.01)
        assert row["value"] == row["affinity"] == row["score"]

    def test_ic50_wt_populates_wt_columns(self):
        r = read_pvacseq(MHC_I_AGG)
        row = r.df.iloc[0]
        assert math.isclose(row["wt_value"], 61.796, abs_tol=0.01)
        assert row["wt_value"] == row["wt_affinity"] == row["wt_score"]

    def test_missense_wt_peptide_reconstructed(self):
        r = read_pvacseq(MHC_I_AGG)
        # First HCC1395 aggregated row: AA Change=E806V, Pos=8,
        # Best Peptide=AERMGFTVV → WT peptide AERMGFTEV.
        first = r.df.iloc[0]
        assert first["aa_change"] == "E806V"
        assert first["peptide"] == "AERMGFTVV"
        assert first["wt_peptide"] == "AERMGFTEV"
        assert first["wt_peptide_length"] == 9

    def test_effect_type_substitution_for_missense(self):
        r = read_pvacseq(MHC_I_AGG)
        missense = r.df[r.df["aa_change"].str.match(r"^[A-Z]\d+[A-Z]$", na=False)]
        assert (missense["effect_type"] == "Substitution").all()

    def test_dsl_filter_round_trip(self):
        r = read_pvacseq(MHC_I_AGG)
        strong = apply_filter(r.df, Affinity.value <= 500)
        assert 0 < len(strong) <= len(r)
        assert (strong["value"] <= 500).all()

    def test_dsl_sort_round_trip(self):
        r = read_pvacseq(MHC_I_AGG)
        sorted_df = apply_sort(r.df, [Affinity.value])
        values = sorted_df["value"].dropna().tolist()
        assert values == sorted(values)

    def test_wt_aware_sort(self):
        # The wt-scoped accessor reads pvacseq's wt_* columns end-to-end.
        r = read_pvacseq(MHC_I_AGG)
        # Sort by MT-vs-WT IC50 delta — exercises wt.Affinity.value lookup.
        sorted_df = apply_sort(
            r.df, [Affinity.value - wt.Affinity.value], sort_direction="asc"
        )
        diffs = (sorted_df["value"] - sorted_df["wt_value"]).dropna().tolist()
        assert diffs == sorted(diffs)

    def test_tag_overrides_source_label(self):
        r = read_pvacseq(MHC_I_AGG, tag="patient-42")
        assert r.sources == ["patient-42"]

    def test_round_trip_through_topiary_tsv(self, tmp_path):
        r = read_pvacseq(MHC_I_AGG)
        out_path = tmp_path / "roundtrip.tsv"
        to_tsv(r, out_path)
        reloaded = read_tsv(out_path)
        assert reloaded.form == "long"
        assert len(reloaded) == len(r)
        # Column set survives exactly (read_pvacseq itself stamps a
        # `source` column, so no diff on that axis either).
        assert set(reloaded.df.columns) == set(r.df.columns)
        # NaN-aware equality for every column we control.
        for col in ("peptide", "allele", "wt_peptide", "value",
                    "percentile_rank", "wt_value", "wt_percentile_rank"):
            assert reloaded.df[col].equals(r.df[col]), (
                f"column {col} did not round-trip"
            )

    def test_predictor_version_is_na(self):
        # Pinned at pd.NA so stacking files doesn't trigger the
        # "conflicting versions" warning in topiary.stack_results.
        r = read_pvacseq(MHC_I_AGG)
        assert r.df["predictor_version"].isna().all()
        assert r.df["wt_predictor_version"].isna().all()

    def test_mhc_class_derived_from_alleles(self):
        r = read_pvacseq(MHC_I_AGG)
        assert (r.df["mhc_class"] == "I").all()
        r2 = read_pvacseq(MHC_II_AGG)
        assert (r2.df["mhc_class"] == "II").all()

    def test_derive_mhc_class_public_utility(self):
        # Available at the package root for stamping mhc_class onto a
        # DataFrame produced outside the pvacseq loader (e.g. a fresh
        # TopiaryPredictor result).
        from topiary import derive_mhc_class
        alleles = pd.Series([
            "HLA-A*02:01", "HLA-B*07:02", "HLA-C*06:02",
            "HLA-DRB1*04:05", "HLA-DPA1*02:01/DPB1*01:01",
            "UNKNOWN", None,
        ])
        classes = derive_mhc_class(alleles)
        assert classes.tolist()[:3] == ["I", "I", "I"]
        assert classes.tolist()[3:5] == ["II", "II"]
        assert pd.isna(classes.iloc[5]) and pd.isna(classes.iloc[6])

    def test_source_column_per_row(self):
        # Per-row provenance matches read_tsv convention.
        r = read_pvacseq(MHC_I_AGG)
        assert r.df["source"].nunique() == 1
        assert r.df["source"].iloc[0] == r.sources[0]

    def test_tag_propagates_into_source_column(self):
        r = read_pvacseq(MHC_I_AGG, tag="patient-42")
        assert (r.df["source"] == "patient-42").all()

    def test_mutation_interval_for_missense(self):
        r = read_pvacseq(MHC_I_AGG)
        # First fixture row: Best Peptide=AERMGFTVV, Pos=8 → 0-based [7, 8).
        first = r.df.iloc[0]
        assert bool(first["contains_mutant_residues"]) is True
        assert int(first["mutation_start_in_peptide"]) == 7
        assert int(first["mutation_end_in_peptide"]) == 8

    def test_mutation_interval_dtype_is_int64(self):
        r = read_pvacseq(MHC_I_AGG)
        assert str(r.df["mutation_start_in_peptide"].dtype) == "Int64"
        assert str(r.df["mutation_end_in_peptide"].dtype) == "Int64"
        assert str(r.df["contains_mutant_residues"].dtype) == "boolean"


# ---------------------------------------------------------------------------
# all_epitopes flavor
# ---------------------------------------------------------------------------


class TestLoadAllEpitopes:
    def test_format_recorded(self):
        r = read_pvacseq(MHC_I_ALL)
        assert r.extra["pvacseq_format"] == "all_epitopes"

    def test_median_mt_ic50_becomes_value(self):
        r = read_pvacseq(MHC_I_ALL)
        row = r.df.iloc[0]
        assert math.isclose(row["value"], 76.11, abs_tol=0.01)

    def test_presentation_algorithms_become_native_rows(self):
        r = read_pvacseq(MHC_I_ALL_PRESENTATION)
        presentation = r.df[r.df["kind"] == "pMHC_presentation"]

        assert set(presentation["prediction_method_name"]) == {
            "pvacseq", "bigmhc_el", "mhcflurry", "netmhcpan",
        }
        mhcflurry = presentation[
            presentation["prediction_method_name"] == "mhcflurry"
        ].iloc[0]
        assert mhcflurry["score"] == pytest.approx(0.91)
        assert mhcflurry["value"] == pytest.approx(0.91)
        assert mhcflurry["percentile_rank"] == pytest.approx(0.12)
        assert mhcflurry["wt_score"] == pytest.approx(0.11)
        assert mhcflurry["wt_value"] == pytest.approx(0.11)
        assert mhcflurry["wt_percentile_rank"] == pytest.approx(5.0)
        assert pd.isna(mhcflurry["affinity"])
        assert pd.isna(mhcflurry["wt_affinity"])

        netmhcpan = presentation[
            presentation["prediction_method_name"] == "netmhcpan"
        ].iloc[0]
        assert netmhcpan["score"] == pytest.approx(0.64)
        assert netmhcpan["percentile_rank"] == pytest.approx(0.8)

        bigmhc = presentation[
            presentation["prediction_method_name"] == "bigmhc_el"
        ].iloc[0]
        assert bigmhc["score"] == pytest.approx(0.72)
        assert bigmhc["percentile_rank"] == pytest.approx(0.6)

    def test_presentation_rows_are_reachable_from_dsl(self):
        r = read_pvacseq(MHC_I_ALL_PRESENTATION)
        scores = Presentation["mhcflurry"].score.evaluate(r.df)
        assert scores == pytest.approx(0.91)

    def test_presentation_kind_support_records_each_source(self):
        r = read_pvacseq(MHC_I_ALL_PRESENTATION)
        support = r.extra["kind_support"]
        assert "pMHC_affinity" in support["pvacseq"]
        for method in ("pvacseq", "bigmhc_el", "mhcflurry", "netmhcpan"):
            assert support[method]["pMHC_presentation"] == {
                "mhc_dependence": "single_allele",
                "mhc_class": "I",
            }

    def test_netmhciipan_el_name_is_preserved(self, tmp_path):
        df = pd.read_csv(MHC_I_ALL_PRESENTATION, sep="\t")
        mhcflurry_columns = [
            column for column in df.columns
            if column.startswith("MHCflurryEL Presentation")
        ]
        df = df.drop(columns=mhcflurry_columns)
        df = df.rename(columns={
            column: column.replace("NetMHCpanEL", "NetMHCIIpanEL")
            for column in df.columns
            if column.startswith("NetMHCpanEL")
        })
        path = tmp_path / "NetMHCIIpanEL.tsv"
        df.to_csv(path, sep="\t", index=False)

        rows = read_pvacseq(path).df
        row = rows[
            (rows["kind"] == "pMHC_presentation")
            & (rows["prediction_method_name"] == "netmhciipan")
        ].iloc[0]
        assert row["score"] == pytest.approx(0.64)
        assert row["percentile_rank"] == pytest.approx(0.8)

    def test_processing_and_immunogenicity_are_native_rows(self):
        r = read_pvacseq(MHC_I_ALL_PRESENTATION)

        processing = r.df[r.df["kind"] == "antigen_processing"]
        assert processing["prediction_method_name"].tolist() == ["mhcflurry"]
        assert processing["score"].iloc[0] == pytest.approx(0.75)
        assert processing["percentile_rank"].iloc[0] == pytest.approx(0.9)
        assert r.extra["kind_support"]["mhcflurry"]["antigen_processing"] == {
            "mhc_dependence": "none",
            "mhc_class": "none",
        }

        immunogenicity = r.df[r.df["kind"] == "immunogenicity"]
        assert set(immunogenicity["prediction_method_name"]) == {
            "pvacseq", "bigmhc_im", "prime",
        }
        bigmhc = immunogenicity[
            immunogenicity["prediction_method_name"] == "bigmhc_im"
        ].iloc[0]
        assert bigmhc["score"] == pytest.approx(0.83)
        assert bigmhc["percentile_rank"] == pytest.approx(0.4)

    def test_processing_summary_percentile_uses_shared_vocabulary(
        self, tmp_path,
    ):
        df = pd.read_csv(MHC_I_ALL_PRESENTATION, sep="\t")
        df["Median MT Processing Percentile"] = 1.2
        df["Median WT Processing Percentile"] = 3.4
        path = tmp_path / "processing-summary.tsv"
        df.to_csv(path, sep="\t", index=False)

        rows = read_pvacseq(path).df
        summary = rows[
            (rows["kind"] == "antigen_processing")
            & (rows["prediction_method_name"] == "pvacseq")
        ].iloc[0]
        assert summary["percentile_rank"] == pytest.approx(1.2)
        assert summary["wt_percentile_rank"] == pytest.approx(3.4)

    def test_best_presentation_summary_is_used_when_median_is_absent(
        self, tmp_path,
    ):
        df = pd.read_csv(MHC_I_ALL_PRESENTATION, sep="\t").rename(columns={
            "Median MT Presentation Percentile": (
                "Best MT Presentation Percentile"
            ),
            "Median WT Presentation Percentile": (
                "Corresponding WT Presentation Percentile"
            ),
        })
        path = tmp_path / "best-presentation.tsv"
        df.to_csv(path, sep="\t", index=False)

        rows = read_pvacseq(path).df
        aggregate = rows[
            (rows["kind"] == "pMHC_presentation")
            & (rows["prediction_method_name"] == "pvacseq")
        ].iloc[0]
        assert aggregate["percentile_rank"] == pytest.approx(0.46)
        assert aggregate["wt_percentile_rank"] == pytest.approx(7.2)

    def test_wt_peptide_from_wt_epitope_seq(self):
        r = read_pvacseq(MHC_I_ALL)
        # All fixture rows ship a WT epitope.
        assert r.df["wt_peptide"].notna().all()
        assert (r.df["wt_peptide_length"] == r.df["wt_peptide"].str.len()).all()

    def test_per_algorithm_columns_pass_through(self):
        r = read_pvacseq(MHC_I_ALL)
        expected = {
            "pvacseq_netmhcpan_ic50_mt",
            "pvacseq_netmhcpan_ic50_wt",
            "pvacseq_netmhcpan_pct_mt",
            "pvacseq_netmhcpan_pct_wt",
            "pvacseq_mhcflurry_ic50_mt",
            "pvacseq_mhcflurry_ic50_wt",
            "pvacseq_mhcflurry_pct_mt",
            "pvacseq_mhcflurry_pct_wt",
        }
        assert expected <= set(r.df.columns)

    def test_unknown_affinity_percentile_inherits_companion_kind(
        self, tmp_path,
    ):
        df = pd.read_csv(MHC_I_ALL, sep="\t")
        df["BrandNew MT IC50 Score"] = 123.0
        df["BrandNew WT IC50 Score"] = 456.0
        df["BrandNew MT Percentile"] = 1.2
        df["BrandNew WT Percentile"] = 3.4
        path = tmp_path / "unknown-affinity.tsv"
        df.to_csv(path, sep="\t", index=False)

        loaded = read_pvacseq(path)
        assert loaded.df["pvacseq_brand_new_pct_mt"].eq(1.2).all()
        assert loaded.df["pvacseq_brand_new_pct_wt"].eq(3.4).all()

        row = melt_pvacseq_algorithms(loaded).df
        row = row[row["prediction_method_name"] == "brand_new"].iloc[0]
        assert row["value"] == pytest.approx(123.0)
        assert row["percentile_rank"] == pytest.approx(1.2)
        assert row["wt_value"] == pytest.approx(456.0)
        assert row["wt_percentile_rank"] == pytest.approx(3.4)

    @pytest.mark.parametrize("column", [
        "BrandNew MT Score",
        "BrandNew Score WT",
    ])
    def test_ambiguous_prediction_column_is_preserved_and_reported(
        self, tmp_path, column,
    ):
        df = pd.read_csv(MHC_I_ALL, sep="\t")
        df[column] = 0.42
        path = tmp_path / "ambiguous-score.tsv"
        df.to_csv(path, sep="\t", index=False)

        with pytest.warns(UserWarning, match=column):
            loaded = read_pvacseq(path)

        assert loaded.df[column].eq(0.42).all()

    def test_na_per_algorithm_value_passes_as_nan(self):
        # Fixture row 1 has "NA" in NetMHCpan columns.
        r = read_pvacseq(MHC_I_ALL)
        assert pd.isna(r.df["pvacseq_netmhcpan_ic50_mt"].iloc[1])

    def test_presentation_fixture_stays_one_wide_row(self):
        wide = read_pvacseq(MHC_I_ALL_PRESENTATION).wide_df

        assert len(wide) == 1
        assert wide.loc[0, "mhcflurry_presentation_wt_score"] == pytest.approx(
            0.11,
        )
        assert wide.loc[0, "bigmhc_im_immunogenicity_wt_rank"] == pytest.approx(
            12.0,
        )

    def test_calis_support_is_allele_independent(self, tmp_path):
        df = pd.read_csv(MHC_I_ALL, sep="\t")
        df["Calis MT Score"] = 0.6
        df["Calis WT Score"] = -0.1
        path = tmp_path / "calis.tsv"
        df.to_csv(path, sep="\t", index=False)

        result = read_pvacseq(path)

        assert result.extra["kind_support"]["calis"]["immunogenicity"] == {
            "mhc_dependence": "none",
            "mhc_class": "I",
        }

    def test_annotation_columns_renamed(self):
        r = read_pvacseq(MHC_I_ALL)
        assert {"gene_expression", "pvacseq_tumor_dna_vaf", "variant_type"} <= set(r.df.columns)

    def test_effect_type_substitution_for_missense(self):
        r = read_pvacseq(MHC_I_ALL)
        assert (r.df["effect_type"] == "Substitution").all()

    def test_mhc_ii_all_epitopes_loads(self):
        # Class-II all_epitopes uses heterodimer alleles and a different
        # algorithm column ("NetMHCIIpan ..." instead of "NetMHCpan ...").
        r = read_pvacseq(MHC_II_ALL)
        assert r.extra["pvacseq_format"] == "all_epitopes"
        alleles = set(r.df["allele"].dropna())
        assert all(a.startswith("HLA-D") for a in alleles)
        # Heterodimer separator gets normalized by mhcgnomes.
        assert any("/" in a for a in alleles)
        # Per-algorithm passthrough handles NetMHCIIpan the same way.
        assert {
            "pvacseq_netmhciipan_ic50_mt",
            "pvacseq_netmhciipan_ic50_wt",
            "pvacseq_netmhciipan_pct_mt",
            "pvacseq_netmhciipan_pct_wt",
        } <= set(r.df.columns)

    def test_wt_peptide_length_is_nullable_int(self):
        r = read_pvacseq(MHC_I_ALL)
        assert str(r.df["wt_peptide_length"].dtype) == "Int64"

    def test_chr_coord_variant_fallback_when_index_missing(self, tmp_path):
        # Strip the "Index" column from the fixture; _parse_all_epitopes
        # should fall back to building variant from chr/start/ref/alt.
        df = pd.read_csv(MHC_I_ALL, sep="\t").drop(columns=["Index"])
        path = tmp_path / "no_index.tsv"
        df.to_csv(path, sep="\t", index=False)
        r = read_pvacseq(path)
        first = r.df.iloc[0]
        assert first["variant"] == "chr1-154590262-T-A"


# ---------------------------------------------------------------------------
# Stacking files via topiary.stack_results
# ---------------------------------------------------------------------------


class TestStackMultipleFiles:
    def test_mhc_i_plus_mhc_ii_via_stack_results(self):
        combined = stack_results([
            read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG),
        ])
        assert len(combined) == _data_row_count(MHC_I_AGG) + _data_row_count(MHC_II_AGG)
        alleles = set(combined.df["allele"].dropna())
        assert any(a.startswith(("HLA-A", "HLA-B", "HLA-C")) for a in alleles)
        assert any("D" in a for a in alleles)

    def test_stack_mixed_flavors(self):
        combined = stack_results([
            read_pvacseq(MHC_I_ALL), read_pvacseq(MHC_II_AGG),
        ])
        assert len(combined) == _data_row_count(MHC_I_ALL) + _data_row_count(MHC_II_AGG)

    def test_stack_preserves_per_row_source_and_mhc_class(self):
        # Vaxrank wants to combine MHC-I + MHC-II in one ranking run and
        # split or filter by class afterward.
        combined = stack_results([
            read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG),
        ])
        # Two distinct provenance labels.
        assert combined.df["source"].nunique() == 2
        # mhc_class lets downstream filter by class without parsing alleles.
        assert set(combined.df["mhc_class"]) == {"I", "II"}
        class_i_rows = (combined.df["mhc_class"] == "I").sum()
        class_ii_rows = (combined.df["mhc_class"] == "II").sum()
        assert class_i_rows == _data_row_count(MHC_I_AGG)
        assert class_ii_rows == _data_row_count(MHC_II_AGG)


# ---------------------------------------------------------------------------
# WT peptide reconstruction edge cases
# ---------------------------------------------------------------------------


class TestReconstructWtPeptide:
    def test_missense_substitutes_at_pos(self):
        assert _reconstruct_wt_peptide("AERMGFTVV", 8, "E806V") == "AERMGFTEV"

    def test_position_outside_peptide_returns_none(self):
        # Flanking peptide that doesn't span the mutation: Pos points past
        # the peptide's length.  We refuse to guess.
        assert _reconstruct_wt_peptide("AERMGFTVV", 99, "E806V") is None

    def test_pos_aa_change_disagreement_returns_none(self):
        # AA Change says mutant residue is V, but position 1 of "AERMGFTVV"
        # is A — disagreement, so refuse rather than corrupt the peptide.
        assert _reconstruct_wt_peptide("AERMGFTVV", 1, "E806V") is None

    def test_non_missense_aa_change_returns_none(self):
        # Frameshift / indel / multi-residue formats don't match the
        # single-AA regex.
        assert _reconstruct_wt_peptide("AERMGFTVV", 5, "FS342") is None
        assert _reconstruct_wt_peptide("AERMGFTVV", 5, "EE764-765EK") is None

    def test_non_string_inputs_return_none(self):
        assert _reconstruct_wt_peptide(None, 5, "E806V") is None
        assert _reconstruct_wt_peptide("AERMGFTVV", 5, None) is None
        assert _reconstruct_wt_peptide("AERMGFTVV", "not-a-number", "E806V") is None


# ---------------------------------------------------------------------------
# Effect-type classifier
# ---------------------------------------------------------------------------


class TestClassifyEffect:
    def test_missense(self):
        assert _classify_effect("E806V") == "Substitution"

    def test_frameshift_pvacseq_format(self):
        # pVACseq frameshifts: "FS<pos>" or "FS<start>-<end>".
        assert _classify_effect("FS342") == "FrameShift"
        assert _classify_effect("FS113-120") == "FrameShift"
        assert _classify_effect("fs778") == "FrameShift"

    def test_missense_with_fs_substring_is_not_frameshift(self):
        # Strict regex prevents a missense whose residues contain F/S
        # from being misclassified.  "F206S" is a Substitution.
        assert _classify_effect("F206S") == "Substitution"

    def test_multi_residue_substitution(self):
        assert _classify_effect("EE764-765EK") == "Substitution"

    def test_variant_type_overrides_aa_change(self):
        # When Variant Type is provided, it wins over AA-Change-shape parsing.
        assert _classify_effect("FS342", variant_type="missense") == "Substitution"

    def test_unknown_returns_none(self):
        assert _classify_effect(None) is None
        assert _classify_effect("nonsense-garbage") is None


# ---------------------------------------------------------------------------
# Mutation interval derivation on flanking-only peptides
# ---------------------------------------------------------------------------


class TestFlankingPeptideHandling:
    def test_pos_outside_peptide_yields_no_mutation_interval(self, tmp_path):
        # Construct a 1-row fixture where Pos points past peptide_length —
        # mirrors the 3 HCC1395 rows where pVACseq's Best Peptide is a
        # flank-only candidate (mutation outside the 9-mer).
        agg_path = tmp_path / "flanking.tsv"
        cols = pd.read_csv(MHC_I_AGG, sep="\t", nrows=0).columns.tolist()
        df = pd.read_csv(MHC_I_AGG, sep="\t").head(1).copy()
        df.loc[df.index[0], "Pos"] = 99  # past the peptide length
        df.to_csv(agg_path, sep="\t", index=False, columns=cols)
        r = read_pvacseq(agg_path)
        row = r.df.iloc[0]
        assert bool(row["contains_mutant_residues"]) is False
        assert pd.isna(row["mutation_start_in_peptide"])
        assert pd.isna(row["mutation_end_in_peptide"])
        # And the unrelated WT-peptide reconstruction also bails.
        assert pd.isna(row["wt_peptide"])


# ---------------------------------------------------------------------------
# kind_support metadata
# ---------------------------------------------------------------------------


class TestKindSupport:
    def test_mhc_i_file_records_single_allele_class_i(self):
        r = read_pvacseq(MHC_I_AGG)
        ks = r.extra["kind_support"]
        assert ks == {
            "pvacseq": {
                "pMHC_affinity": {
                    "mhc_dependence": "single_allele",
                    "mhc_class": "I",
                },
            },
        }

    def test_mhc_ii_file_records_single_allele_class_ii(self):
        r = read_pvacseq(MHC_II_AGG)
        assert r.extra["kind_support"]["pvacseq"]["pMHC_affinity"]["mhc_class"] == "II"

    def test_stack_summary_can_be_recomputed_post_stack(self):
        # stack_results doesn't merge kind_support — callers can recompute
        # from the combined allele column if needed.
        from topiary.io_pvacseq import _summarize_mhc_class
        combined = stack_results([
            read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG),
        ])
        assert _summarize_mhc_class(combined.df["allele"]) == "both"


# ---------------------------------------------------------------------------
# melt_pvacseq_algorithms
# ---------------------------------------------------------------------------


class TestMeltAlgorithms:
    def test_aggregated_is_noop(self):
        # Aggregated TSVs only carry Median scores; melt produces the same frame.
        r = read_pvacseq(MHC_I_AGG)
        melted = melt_pvacseq_algorithms(r)
        assert len(melted) == len(r)
        assert melted.df.columns.equals(r.df.columns)

    def test_all_epitopes_expands_one_row_per_algorithm(self):
        r = read_pvacseq(MHC_I_ALL)
        melted = melt_pvacseq_algorithms(r)
        # The MHC-I all_epitopes fixture has NetMHCpan + MHCflurry per-algo
        # columns, so each (peptide, allele) row gets 2 new sibling rows.
        n_rows_original = len(r)
        n_rows_melted = len(melted)
        assert n_rows_melted == n_rows_original * 3  # median + 2 algos

    def test_melt_adds_affinity_rows_without_cloning_presentation_rows(self):
        r = read_pvacseq(MHC_I_ALL_PRESENTATION)
        melted = melt_pvacseq_algorithms(r)

        assert len(r) == 9
        assert len(melted) == 11  # two algorithm-specific affinity rows added
        presentation = melted.df[
            melted.df["kind"] == "pMHC_presentation"
        ]
        affinity = melted.df[melted.df["kind"] == "pMHC_affinity"]
        assert len(presentation) == 4
        assert set(affinity["prediction_method_name"]) == {
            "pvacseq", "mhcflurry", "netmhcpan",
        }

    def test_melted_rows_have_distinct_method_names(self):
        r = read_pvacseq(MHC_I_ALL)
        melted = melt_pvacseq_algorithms(r)
        methods = set(melted.df["prediction_method_name"].dropna())
        assert {"pvacseq", "mhcflurry", "netmhcpan"} <= methods

    def test_per_algorithm_value_lifts_from_passthrough_column(self):
        r = read_pvacseq(MHC_I_ALL)
        melted = melt_pvacseq_algorithms(r)
        # First fixture row's NetMHCpan MT IC50 was 20.16; after melt that
        # value should appear on the netmhcpan row for that peptide.
        netmhcpan_rows = melted.df[melted.df["prediction_method_name"] == "netmhcpan"]
        first = netmhcpan_rows.iloc[0]
        assert math.isclose(first["value"], 20.16, abs_tol=0.01)

    def test_score_only_affinity_does_not_inherit_aggregate_value(
        self, tmp_path,
    ):
        df = pd.read_csv(MHC_I_ALL, sep="\t")
        algorithm_columns = [
            column for column in df.columns
            if column.startswith(("NetMHCpan ", "MHCflurry "))
        ]
        df = df.drop(columns=algorithm_columns)
        df["Random MT Score"] = 0.8
        df["Random WT Score"] = 0.2
        path = tmp_path / "score-only-affinity.tsv"
        df.to_csv(path, sep="\t", index=False)

        row = melt_pvacseq_algorithms(read_pvacseq(path)).df
        row = row[row["prediction_method_name"] == "random"].iloc[0]

        assert row["score"] == pytest.approx(0.8)
        assert row["wt_score"] == pytest.approx(0.2)
        assert pd.isna(row["value"])
        assert pd.isna(row["affinity"])
        assert pd.isna(row["wt_value"])
        assert pd.isna(row["wt_affinity"])

    def test_kind_support_extends_with_algorithm_models(self):
        r = read_pvacseq(MHC_I_ALL)
        melted = melt_pvacseq_algorithms(r)
        ks = melted.extra["kind_support"]
        assert {"pvacseq", "netmhcpan", "mhcflurry"} <= set(ks.keys())
        for name in ("netmhcpan", "mhcflurry"):
            assert ks[name]["pMHC_affinity"]["mhc_class"] == "I"

    def test_dsl_per_algorithm_selector_after_melt(self):
        # The whole point of melting: Affinity['mhcflurry'] reaches per-algo rows.
        r = read_pvacseq(MHC_I_ALL)
        melted = melt_pvacseq_algorithms(r)
        passing_groups = apply_filter(
            melted.df, Affinity["mhcflurry"].value <= 100
        )
        # The filter keeps every row of any (peptide, allele) group whose
        # mhcflurry row passes; check the mhcflurry rows themselves.
        mhcflurry_rows = passing_groups[
            passing_groups["prediction_method_name"] == "mhcflurry"
        ]
        assert len(mhcflurry_rows) >= 1
        assert (mhcflurry_rows["value"] <= 100).all()


# ---------------------------------------------------------------------------
# Integration: vaxrank-shaped filters + exclude_by composition
# ---------------------------------------------------------------------------


class TestVaxrankComposition:
    def test_vaxrank_shape_filter_expression(self):
        # Mirrors the shape of filter vaxrank applies after swapping its
        # epitope_io.py loader for topiary.read_pvacseq.  Fully native
        # DSL: numeric + categorical clauses in one expression via the
        # IsIn nodes introduced for class-I/II filtering.
        combined = stack_results([
            read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG),
        ])
        strong = apply_filter(
            combined.df,
            (Affinity.value <= 500)
            & Column("mhc_class").eq("I")
            & Column("contains_mutant_residues").eq(True),
        )
        assert (strong["value"] <= 500).all()
        assert (strong["mhc_class"] == "I").all()
        assert strong["contains_mutant_residues"].all()
        assert len(strong) > 0

    def test_vaxrank_shape_filter_via_parsed_string(self):
        # Same filter shape via the string DSL — what a CLI flag would feed.
        from topiary import parse
        combined = stack_results([
            read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG),
        ])
        node = parse('affinity.value <= 500 & mhc_class == "I"')
        strong = apply_filter(combined.df, node)
        assert (strong["value"] <= 500).all()
        assert (strong["mhc_class"] == "I").all()
        assert len(strong) > 0

    def test_exclude_by_filters_loaded_pvacseq_rows(self):
        # exclude_by removes predicted peptides found in a reference proteome.
        # Use the first fixture peptide itself as the "reference" — verify
        # the row gets dropped and the rest survive.
        r = read_pvacseq(MHC_I_AGG)
        sentinel_peptide = r.df["peptide"].iloc[0]
        from topiary.inputs import exclude_by
        kept = exclude_by(r.df, {"sentinel": sentinel_peptide}, mode="exact")
        assert sentinel_peptide not in set(kept["peptide"])
        assert len(kept) == len(r) - 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unrecognized_file_raises(self, tmp_path):
        bad = tmp_path / "bad.tsv"
        bad.write_text("foo\tbar\n1\t2\n")
        with pytest.raises(ValueError, match="Could not detect pVACseq format"):
            read_pvacseq(bad)
