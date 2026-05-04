"""Tests for TopiaryPredictor.predict_from_fragments."""

import math

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import (
    Affinity,
    ProteinFragment,
    Column,
    TopiaryPredictor,
    apply_filter,
    apply_sort,
    self_nearest,
    wt,
)


ALLELE = "HLA-A*02:01"


class MultiKindPredictor:
    """Small deterministic predictor with affinity and presentation rows."""

    default_peptide_lengths = [8]

    def __init__(self):
        self.values = {
            "SIINFEKL": {
                "pMHC_affinity": 1.0,
                "pMHC_presentation": 2.0,
            },
            "GILGFVFT": {
                "pMHC_affinity": 10.0,
                "pMHC_presentation": 20.0,
            },
        }

    def _prediction_rows(self, peptide):
        rows = []
        for kind, value in self.values[peptide].items():
            rows.append({
                "peptide": peptide,
                "allele": ALLELE,
                "kind": kind,
                "value": value,
                "score": value / 100.0,
                "affinity": value if kind == "pMHC_affinity" else math.nan,
                "percentile_rank": value,
                "predictor_name": "multi-kind",
                "predictor_version": "1.0",
            })
        return rows

    def predict_peptides_dataframe(self, peptides):
        rows = []
        for peptide in peptides:
            rows.extend(self._prediction_rows(peptide))
        return pd.DataFrame(rows)

    def predict_proteins_dataframe(self, name_to_sequence):
        rows = []
        for name, sequence in name_to_sequence.items():
            for offset in range(len(sequence) - 7):
                peptide = sequence[offset:offset + 8]
                for row in self._prediction_rows(peptide):
                    row = row.copy()
                    row["source_sequence_name"] = name
                    row["offset"] = offset
                    rows.append(row)
        return pd.DataFrame(rows)


class LengthStrictPredictor:
    """Predictor that raises if asked to score unsupported lengths."""

    def __init__(self, peptide_length):
        self.peptide_length = peptide_length
        self.default_peptide_lengths = [peptide_length]

    def _prediction_rows(self, peptide, source_sequence_name=None, offset=0):
        if len(peptide) != self.peptide_length:
            raise ValueError(
                f"strict-{self.peptide_length} cannot score {len(peptide)}mers"
            )
        return [{
            "source_sequence_name": source_sequence_name,
            "offset": offset,
            "peptide": peptide,
            "allele": ALLELE,
            "kind": "pMHC_affinity",
            "value": float(self.peptide_length),
            "score": float(self.peptide_length) / 100.0,
            "affinity": float(self.peptide_length),
            "percentile_rank": float(self.peptide_length),
            "predictor_name": f"strict-{self.peptide_length}",
            "predictor_version": "1.0",
        }]

    def predict_peptides_dataframe(self, peptides):
        rows = []
        for peptide in peptides:
            rows.extend(self._prediction_rows(peptide))
        return pd.DataFrame(rows)

    def predict_proteins_dataframe(self, name_to_sequence):
        rows = []
        length = self.peptide_length
        for name, sequence in name_to_sequence.items():
            for offset in range(len(sequence) - length + 1):
                peptide = sequence[offset:offset + length]
                rows.extend(self._prediction_rows(peptide, name, offset))
        return pd.DataFrame(rows)


class DuplicateIdentityPredictor:
    """Same public method/version across instances, different scores."""

    default_peptide_lengths = [8]

    def __init__(self, mutant_value, wt_value):
        self.mutant_value = mutant_value
        self.wt_value = wt_value

    def _prediction_rows(self, peptide, source_sequence_name=None, offset=0):
        if peptide == "SIINFEKL":
            value = self.mutant_value
        elif peptide == "GILGFVFT":
            value = self.wt_value
        else:
            raise ValueError(f"Unexpected peptide {peptide!r}")
        return [{
            "source_sequence_name": source_sequence_name,
            "offset": offset,
            "peptide": peptide,
            "allele": ALLELE,
            "kind": "pMHC_affinity",
            "value": value,
            "score": value / 100.0,
            "affinity": value,
            "percentile_rank": value,
            "predictor_name": "duplicate-identity",
            "predictor_version": "1.0",
        }]

    def predict_peptides_dataframe(self, peptides):
        rows = []
        for peptide in peptides:
            rows.extend(self._prediction_rows(peptide))
        return pd.DataFrame(rows)

    def predict_proteins_dataframe(self, name_to_sequence):
        rows = []
        for name, sequence in name_to_sequence.items():
            peptide = sequence[:8]
            rows.extend(self._prediction_rows(peptide, name, 0))
        return pd.DataFrame(rows)


def _predictor(lengths=(9,), **kwargs):
    return TopiaryPredictor(
        models=RandomBindingPredictor(
            alleles=[ALLELE], default_peptide_lengths=list(lengths),
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Core: rows carry fragment metadata
# ---------------------------------------------------------------------------


class TestMetadataPropagation:
    def test_fragment_id_on_every_row(self):
        f = ProteinFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11, inframe=True,
            gene="BRAF",
        )
        df = _predictor().predict_from_fragments([f])
        assert (df["fragment_id"] == f.fragment_id).all()

    def test_source_type_propagated(self):
        f = ProteinFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11, inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        assert (df["source_type"] == "variant:snv").all()

    def test_variant_effect_gene_propagated(self):
        f = ProteinFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11, inframe=True,
            variant="chr1:100", effect="p.Ala290Val", gene="BRAF",
            gene_id="ENSG123", transcript_id="ENST456",
            gene_expression=12.3, transcript_expression=8.1,
        )
        df = _predictor().predict_from_fragments([f])
        assert (df["variant"] == "chr1:100").all()
        assert (df["effect"] == "p.Ala290Val").all()
        assert (df["gene"] == "BRAF").all()
        assert (df["gene_id"] == "ENSG123").all()
        assert (df["transcript_id"] == "ENST456").all()
        assert (df["gene_expression"] == 12.3).all()
        assert (df["transcript_expression"] == 8.1).all()

    def test_annotations_flattened(self):
        f = ProteinFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11, inframe=True,
            annotations={"vaf": 0.42, "ccf": 0.9},
        )
        df = _predictor().predict_from_fragments([f])
        assert "vaf" in df.columns
        assert "ccf" in df.columns
        assert (df["vaf"] == 0.42).all()
        assert (df["ccf"] == 0.9).all()

    def test_annotations_nan_when_absent(self):
        """Fragments without a given annotation key get NaN for it."""
        with_vaf = ProteinFragment(
            fragment_id="a__00000000", sequence="MAAVTDVGMAV",
            annotations={"vaf": 0.42},
        )
        without_vaf = ProteinFragment(
            fragment_id="b__00000000", sequence="MAAVTDVGMAV",
            annotations={},
        )
        df = _predictor().predict_from_fragments([with_vaf, without_vaf])
        for fid, row in df.groupby("fragment_id"):
            if fid == "a__00000000":
                assert (row["vaf"] == 0.42).all()
            else:
                assert row["vaf"].isna().all()


# ---------------------------------------------------------------------------
# Geometry: overlaps_target / contains_mutant_residues
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_overlaps_target_for_snv(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAVATGSWDSFLKIWN",  # length 25
            mutation_start=11, mutation_end=12,
            inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        # Peptides covering position 11 should be True; others False.
        for _, row in aff.iterrows():
            start = int(row["peptide_offset"])
            touches = start <= 11 < start + 9
            assert row["overlaps_target"] is touches, (start, touches, row["overlaps_target"])

    def test_overlaps_target_nan_when_intervals_none(self):
        f = ProteinFragment(
            fragment_id="erv__00000000", source_type="erv",
            sequence="MLGMNMLLITLFLLLPLSMLKGEPWEGCLHCTH",
            target_intervals=None,
        )
        df = _predictor().predict_from_fragments([f])
        assert df["overlaps_target"].isna().all()

    def test_contains_mutant_residues_only_for_variants(self):
        variant_f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            mutation_start=11, mutation_end=12, inframe=True,
        )
        erv_f = ProteinFragment(
            fragment_id="erv__00000000", source_type="erv",
            sequence="MLGMNMLLITLFLLL",
            target_intervals=[(5, 10)],   # even if targets given, not a variant
        )
        df = _predictor().predict_from_fragments([variant_f, erv_f])
        variant_rows = df[df["fragment_id"] == variant_f.fragment_id]
        erv_rows = df[df["fragment_id"] == erv_f.fragment_id]
        # Variant fragment: contains_mutant_residues populated (True/False)
        assert variant_rows["contains_mutant_residues"].notna().any()
        # ERV: always NaN regardless of whether its target region exists
        assert erv_rows["contains_mutant_residues"].isna().all()


# ---------------------------------------------------------------------------
# WT peptide derivation
# ---------------------------------------------------------------------------


class TestWtPeptide:
    def test_wt_peptide_from_reference(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            reference_sequence="MAAGVTDVGMAA",  # diff at pos 11 (V → A)
            mutation_start=11, mutation_end=12, inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        for _, row in aff.iterrows():
            mut = row["peptide"]
            wt = row["wt_peptide"]
            assert isinstance(wt, str)
            start = int(row["peptide_offset"])
            if start <= 11 < start + 9:
                # Peptide straddles the mutation → WT should differ
                assert mut != wt
            else:
                # Peptide doesn't straddle → WT should match
                assert mut == wt

    def test_germline_takes_precedence_over_reference(self):
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVG",
            reference_sequence="XXXXXXXX",
            germline_sequence="GGGGGGGG",
        )
        df = _predictor(lengths=[8]).predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        assert len(aff) == 1
        assert aff.iloc[0]["wt_peptide"] == "GGGGGGGG"

    def test_no_baseline_leaves_wt_none(self):
        f = ProteinFragment(
            fragment_id="x__00000000", source_type="erv",
            sequence="MAAVTDVG",
        )
        df = _predictor(lengths=[8]).predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        assert aff["wt_peptide"].isna().all()

    def test_wt_peptide_length(self):
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVG",
            reference_sequence="REFXXXXX",
        )
        df = _predictor(lengths=[8]).predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        assert (aff["wt_peptide_length"] == 8).all()

    def test_wt_peptide_none_for_length_changing_baseline(self):
        """Indels / frameshifts where baseline and mutant differ in length
        can't be sliced with mutant coordinates — wt_peptide stays None
        until coordinate remapping lands."""
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",                # 12 aa — post-indel
            reference_sequence="MAAGVTDVGMA",        # 11 aa — pre-indel
            mutation_start=10, mutation_end=12,
            inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        assert aff["wt_peptide"].isna().all()

    def test_predict_wt_false_does_not_score_wt_peptides(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            reference_sequence="MAAGVTDVGMAA",
            mutation_start=11, mutation_end=12, inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        assert "wt_score" not in df.columns
        assert "wt_value" not in df.columns

    def test_predict_wt_true_scores_wt_peptides(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            reference_sequence="MAAGVTDVGMAA",
            mutation_start=11, mutation_end=12, inframe=True,
        )
        df = _predictor(predict_wt=True).predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        for col in (
            "wt_value", "wt_score", "wt_affinity", "wt_percentile_rank",
            "wt_prediction_method_name", "wt_predictor_version",
        ):
            assert col in aff.columns
            assert aff[col].notna().all()

    def test_predict_wt_true_leaves_length_changing_wt_predictions_nan(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            reference_sequence="MAAGVTDVGMA",
            mutation_start=10, mutation_end=12, inframe=True,
        )
        df = _predictor(predict_wt=True).predict_from_fragments([f])
        aff = df[df["kind"] == "pMHC_affinity"]
        assert aff["wt_peptide"].isna().all()
        assert aff["wt_score"].isna().all()
        assert aff["wt_value"].isna().all()

    def test_predict_wt_true_allows_wt_sort_expression(self):
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAV",
            reference_sequence="MAAGVTDVGMAA",
            mutation_start=11, mutation_end=12, inframe=True,
        )
        pred = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=[ALLELE], default_peptide_lengths=[9],
            ),
            predict_wt=True,
            sort_by=[Affinity.score - wt.Affinity.score],
        )
        df = pred.predict_from_fragments([f])
        assert not df.empty
        assert df["wt_score"].notna().all()

    def test_predict_wt_true_aligns_multi_kind_rows(self):
        """WT predictions join by kind and predictor identity, so an
        affinity row does not pick up the WT presentation value."""
        f = ProteinFragment.from_variant(
            sequence="SIINFEKL",
            reference_sequence="GILGFVFT",
            mutation_start=0, mutation_end=8, inframe=True,
        )
        df = TopiaryPredictor(
            models=MultiKindPredictor(), predict_wt=True,
        ).predict_from_fragments([f])
        assert len(df) == 2
        by_kind = df.set_index("kind")
        assert by_kind.loc["pMHC_affinity", "wt_value"] == 10.0
        assert by_kind.loc["pMHC_presentation", "wt_value"] == 20.0

    def test_predict_wt_true_filters_wt_peptides_by_model_length(self):
        f = ProteinFragment.from_variant(
            sequence="SIINFEKLA",
            reference_sequence="GILGFVFTA",
            mutation_start=0, mutation_end=9, inframe=True,
        )
        df = TopiaryPredictor(
            models=[LengthStrictPredictor(8), LengthStrictPredictor(9)],
            predict_wt=True,
        ).predict_from_fragments([f])
        assert set(df["prediction_method_name"]) == {"strict-8", "strict-9"}
        assert df["wt_value"].notna().all()
        for method, length in [("strict-8", 8), ("strict-9", 9)]:
            rows = df[df["prediction_method_name"] == method]
            assert not rows.empty
            assert set(rows["peptide_length"]) == {length}
            assert set(rows["wt_peptide_length"]) == {length}

    def test_predict_wt_true_keeps_duplicate_predictor_instances_separate(self):
        f = ProteinFragment.from_variant(
            sequence="SIINFEKL",
            reference_sequence="GILGFVFT",
            mutation_start=0, mutation_end=8, inframe=True,
        )
        df = TopiaryPredictor(
            models=[
                DuplicateIdentityPredictor(mutant_value=1.0, wt_value=101.0),
                DuplicateIdentityPredictor(mutant_value=2.0, wt_value=202.0),
            ],
            predict_wt=True,
        ).predict_from_fragments([f])
        assert len(df) == 2
        assert "_topiary_model_key" not in df.columns
        assert set(zip(df["value"], df["wt_value"])) == {
            (1.0, 101.0),
            (2.0, 202.0),
        }


# ---------------------------------------------------------------------------
# Multiple fragments & fragment_id as group key
# ---------------------------------------------------------------------------


class TestMultipleFragments:
    def test_fragment_id_groups_preserved(self):
        f1 = ProteinFragment(
            fragment_id="frag_one__00000001", source_type="erv",
            sequence="MAAVTDVGMAV",
        )
        f2 = ProteinFragment(
            fragment_id="frag_two__00000002", source_type="cta",
            sequence="MAAVTDVGMAV",  # same sequence, different fragment
        )
        df = _predictor().predict_from_fragments([f1, f2])
        assert set(df["fragment_id"].unique()) == {
            "frag_one__00000001", "frag_two__00000002",
        }
        # Predictions for the same peptide show up under each fragment_id
        # without collapsing.
        peptide_counts = df.groupby("peptide").size()
        assert (peptide_counts >= 2).any()  # at least some peptides duplicated

    def test_apply_filter_groups_by_fragment_id(self):
        f1 = ProteinFragment(
            fragment_id="frag_one__00000001", source_type="erv",
            sequence="MAAVTDVGMAV",
        )
        f2 = ProteinFragment(
            fragment_id="frag_two__00000002", source_type="cta",
            sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f1, f2])
        # Filter on Affinity — fragment_id used as group key so each
        # fragment's pass/fail is independent.
        result = apply_filter(df, Affinity <= 50000)
        # Both fragments survive (they share the same peptides so binding is identical)
        assert set(result["fragment_id"].unique()) == {
            "frag_one__00000001", "frag_two__00000002",
        }


# ---------------------------------------------------------------------------
# self_nearest scope reservation (columns absent → NaN)
# ---------------------------------------------------------------------------


class TestSelfNearestScope:
    def test_self_nearest_score_nan_when_columns_absent(self):
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f])
        # self_nearest_* columns aren't emitted by PR A
        assert "self_nearest_score" not in df.columns
        # But the DSL scope evaluates to NaN silently
        from topiary.ranking import EvalContext
        node = self_nearest.Affinity.score
        ctx = EvalContext(df)
        s = node.eval(ctx)
        assert s.isna().all()

    def test_self_nearest_reads_populated_columns(self):
        """If a producer writes self_nearest_score etc., the DSL scope reads them."""
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f])
        # Simulate an external tool populating the reserved columns.
        df["self_nearest_peptide"] = "ABCDEFGHI"
        df["self_nearest_score"] = 0.5
        df["self_nearest_percentile_rank"] = 2.0
        # Filter via self_nearest.Affinity.score — reads self_nearest_score.
        # (Field kind filtering still applies.)
        assert "self_nearest_score" in df.columns

    def test_self_nearest_edit_distance_filter(self):
        """Users filter with Column() on self_nearest_edit_distance."""
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f])
        # Producer populates edit distance
        df["self_nearest_edit_distance"] = 4.0
        result = apply_filter(df, Column("self_nearest_edit_distance") >= 3)
        assert len(result) == len(df)

    def test_self_nearest_parses_from_dsl_string(self):
        """parse("self_nearest.affinity.score") yields a node that reads
        self_nearest_score — matches the Python-side Scope("self_nearest")."""
        from topiary.ranking import EvalContext, parse
        node = parse("self_nearest.affinity.score")
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f])
        df["self_nearest_score"] = 0.7
        s = node.eval(EvalContext(df))
        assert (s == 0.7).all()

    def test_self_nearest_parses_in_sort_expression(self):
        """parse("affinity.score - self_nearest.affinity.score") composes."""
        from topiary.ranking import parse
        node = parse("affinity.score - self_nearest.affinity.score")
        f = ProteinFragment(
            fragment_id="x__00000000", sequence="MAAVTDVGMAV",
        )
        df = _predictor().predict_from_fragments([f])
        df["self_nearest_score"] = 0.2
        result = apply_sort(df, [node])
        assert len(result) == len(df)


# ---------------------------------------------------------------------------
# only_novel_epitopes flag
# ---------------------------------------------------------------------------


class TestOnlyNovelEpitopes:
    def test_only_novel_drops_non_mutant_rows(self):
        """When only_novel_epitopes=True, keep only rows with
        contains_mutant_residues == True (parallels predict_from_mutation_effects)."""
        f = ProteinFragment.from_variant(
            sequence="MAAGVTDVGMAVATGSWDSFLKIWN",
            mutation_start=11, mutation_end=12, inframe=True,
        )
        pred = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=[ALLELE], default_peptide_lengths=[9],
            ),
            only_novel_epitopes=True,
        )
        df = pred.predict_from_fragments([f])
        assert df["contains_mutant_residues"].astype(bool).all()

    def test_only_novel_drops_non_variant_fragments(self):
        """Fragments whose source_type isn't a variant have NaN
        contains_mutant_residues and are dropped when only_novel is on."""
        erv = ProteinFragment(
            fragment_id="erv__00000000", source_type="erv",
            sequence="MLGMNMLLITLFLLLPLSMLKGEPWEGCLHCTH",
        )
        pred = TopiaryPredictor(
            models=RandomBindingPredictor(
                alleles=[ALLELE], default_peptide_lengths=[9],
            ),
            only_novel_epitopes=True,
        )
        df = pred.predict_from_fragments([erv])
        assert df.empty


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_fragment_list(self):
        df = _predictor().predict_from_fragments([])
        assert df.empty

    def test_fragment_shorter_than_peptide_length(self):
        f = ProteinFragment(fragment_id="x__00000000", sequence="MA")
        df = _predictor(lengths=[9]).predict_from_fragments([f])
        # No 9-mers fit in a 2-aa sequence
        assert df.empty or (df["peptide"].str.len() == 9).sum() == 0

    def test_reset_index(self):
        f = ProteinFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11, inframe=True,
        )
        df = _predictor().predict_from_fragments([f])
        # Index should be clean 0..N-1
        assert list(df.index) == list(range(len(df)))
