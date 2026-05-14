"""Tests for the pVACseq report loader (topiary.io_pvacseq)."""

import math
from pathlib import Path

import pandas as pd
import pytest

from topiary import (
    Affinity,
    TopiaryResult,
    apply_filter,
    apply_sort,
    concat,
    detect_pvacseq_format,
    read_pvacseq,
    read_tsv,
    to_tsv,
    wt,
)
from topiary.io_pvacseq import _classify_effect, _reconstruct_wt_peptide

FIXTURE_DIR = Path(__file__).parent / "data" / "pvacseq"

MHC_I_AGG = FIXTURE_DIR / "mhc_i_aggregated.tsv"
MHC_II_AGG = FIXTURE_DIR / "mhc_ii_aggregated.tsv"
MHC_I_ALL = FIXTURE_DIR / "mhc_i_all_epitopes.tsv"
MHC_II_ALL = FIXTURE_DIR / "mhc_ii_all_epitopes.tsv"


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

    def test_one_row_per_variant(self):
        r = read_pvacseq(MHC_I_AGG)
        assert len(r) == _data_row_count(MHC_I_AGG)

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
        # Raw "DRB1*04:05" / "DRB4*01:03" gain the HLA- prefix from mhcgnomes.
        assert (r.df["allele"].dropna().str.startswith("HLA-DR")).all()

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
        # read_tsv adds a "source" provenance column on read; otherwise the
        # column set should round-trip exactly.
        assert set(reloaded.df.columns) - {"source"} == set(r.df.columns)
        for col in ("peptide", "allele", "wt_peptide"):
            assert (
                reloaded.df[col].fillna("") == r.df[col].fillna("")
            ).all(), f"column {col} did not round-trip"
        for col in ("value", "percentile_rank", "wt_value", "wt_percentile_rank"):
            assert (
                reloaded.df[col].fillna(-1) == r.df[col].fillna(-1)
            ).all(), f"column {col} did not round-trip"


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

    def test_na_per_algorithm_value_passes_as_nan(self):
        # Fixture row 1 has "NA" in NetMHCpan columns.
        r = read_pvacseq(MHC_I_ALL)
        assert pd.isna(r.df["pvacseq_netmhcpan_ic50_mt"].iloc[1])

    def test_annotation_columns_renamed(self):
        r = read_pvacseq(MHC_I_ALL)
        assert {"gene_expression", "tumor_dna_vaf", "variant_type"} <= set(r.df.columns)

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
# Combining files via topiary.concat
# ---------------------------------------------------------------------------


class TestConcatMultipleFiles:
    def test_mhc_i_plus_mhc_ii_via_concat(self):
        combined = concat([read_pvacseq(MHC_I_AGG), read_pvacseq(MHC_II_AGG)])
        assert len(combined) == _data_row_count(MHC_I_AGG) + _data_row_count(MHC_II_AGG)
        alleles = set(combined.df["allele"].dropna())
        assert any(a.startswith(("HLA-A", "HLA-B", "HLA-C")) for a in alleles)
        assert any("D" in a for a in alleles)

    def test_concat_mixed_flavors(self):
        combined = concat([read_pvacseq(MHC_I_ALL), read_pvacseq(MHC_II_AGG)])
        assert len(combined) == _data_row_count(MHC_I_ALL) + _data_row_count(MHC_II_AGG)


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
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unrecognized_file_raises(self, tmp_path):
        bad = tmp_path / "bad.tsv"
        bad.write_text("foo\tbar\n1\t2\n")
        with pytest.raises(ValueError, match="Could not detect pVACseq format"):
            read_pvacseq(bad)
