"""Regression tests for the PR B refactor of predict_from_variants /
predict_from_mutation_effects onto AntigenFragment internally.

These tests assert:
  1. Legacy output columns are still present with their original
     semantics (contains_mutant_residues, mutation_start_in_peptide,
     mutation_end_in_peptide, absolute peptide_offset, transcript_name).
  2. New fragment-derived columns (fragment_id, source_type,
     overlaps_target, wt_peptide) are now populated on the variant path.
  3. _fragment_from_effect: given a varcode Effect with known offsets,
     produces a fragment with the correct target_intervals,
     reference_sequence, and source_type classification.
"""

from unittest.mock import MagicMock

from mhctools import RandomBindingPredictor
from topiary import TopiaryPredictor

# Intentional import of private symbols — these regression tests pin the
# implementation-detail plumbing (annotation keys, effect→fragment
# adapter) that the refactor relies on.  Moves break tests deliberately.
from topiary.predictor import (
    _MUTATION_END_KEY,
    _MUTATION_START_KEY,
    _SUBSEQ_OFFSET_KEY,
    _fragment_from_effect,
)

from .data import cancer_test_variants


ALLELES = ["HLA-A*02:01"]


def _predictor(**kwargs):
    return TopiaryPredictor(
        models=RandomBindingPredictor(
            alleles=ALLELES, default_peptide_lengths=[9],
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Legacy column contract
# ---------------------------------------------------------------------------


class TestLegacyColumnContract:
    def test_required_columns_present(self):
        df = _predictor().predict_from_variants(cancer_test_variants)
        required = {
            "peptide", "peptide_offset", "peptide_length", "allele",
            "kind", "score", "value", "affinity", "percentile_rank",
            "prediction_method_name", "predictor_version",
            "variant", "effect", "effect_type",
            "gene", "gene_id", "transcript_id", "transcript_name",
            "contains_mutant_residues",
            "mutation_start_in_peptide", "mutation_end_in_peptide",
        }
        missing = required - set(df.columns)
        assert not missing, f"Missing legacy columns: {missing}"

    def test_peptide_offset_is_absolute(self):
        """Offset should be relative to the full mutant protein, not
        to the sliding-window subsequence."""
        df = _predictor().predict_from_variants(cancer_test_variants)
        # BRAF V600E: mutation at residue 599 — peptides around it should
        # have offsets in the ~591–599 range, not 0–8.
        braf = df[df["gene"] == "BRAF"]
        assert not braf.empty
        assert braf["peptide_offset"].min() > 50, (
            "peptide_offset appears to be fragment-local, not absolute protein"
        )

    def test_mutation_interval_populated_when_overlapping(self):
        df = _predictor().predict_from_variants(cancer_test_variants)
        aff = df[df["kind"] == "pMHC_affinity"]
        mutant = aff[aff["contains_mutant_residues"].eq(True)]
        nonmutant = aff[aff["contains_mutant_residues"].eq(False)]
        # Mutant rows have both interval endpoints set
        assert mutant["mutation_start_in_peptide"].notna().all()
        assert mutant["mutation_end_in_peptide"].notna().all()
        # Non-mutant rows get None for both
        assert nonmutant["mutation_start_in_peptide"].isna().all()
        assert nonmutant["mutation_end_in_peptide"].isna().all()

    def test_internal_annotation_keys_not_leaked(self):
        """Internal bookkeeping keys stashed on fragment.annotations
        must not appear as columns on the returned DataFrame."""
        df = _predictor().predict_from_variants(cancer_test_variants)
        for leaked in (
            "_subsequence_offset",
            "_mutation_start_in_protein",
            "_mutation_end_in_protein",
        ):
            assert leaked not in df.columns, f"Internal key leaked: {leaked}"

    def test_only_novel_epitopes_drops_non_mutant_rows(self):
        # padding=16 gives a 33-residue subseq for a single-aa mutation,
        # so the outer 9-mers don't overlap the mutation and we can
        # verify that only_novel_epitopes=True drops them.
        df_all = _predictor(
            padding_around_mutation=16,
        ).predict_from_variants(cancer_test_variants)
        df_novel = _predictor(
            padding_around_mutation=16, only_novel_epitopes=True,
        ).predict_from_variants(cancer_test_variants)
        assert len(df_novel) < len(df_all)
        assert df_novel["contains_mutant_residues"].eq(True).all()


# ---------------------------------------------------------------------------
# New fragment-derived columns now flow through the variant path
# ---------------------------------------------------------------------------


class TestFragmentColumnsOnVariantPath:
    def test_fragment_id_and_source_type_populated(self):
        df = _predictor().predict_from_variants(cancer_test_variants)
        assert "fragment_id" in df.columns
        assert "source_type" in df.columns
        assert df["fragment_id"].notna().all()
        # Both BRAF V600E and TP53 R248W are single-residue substitutions
        assert (df["source_type"] == "variant:snv").all()

    def test_overlaps_target_matches_contains_mutant(self):
        """For variant fragments, overlaps_target and
        contains_mutant_residues derive from the same interval — they
        should agree row-by-row."""
        df = _predictor().predict_from_variants(cancer_test_variants)
        aff = df[df["kind"] == "pMHC_affinity"]
        assert (aff["overlaps_target"] == aff["contains_mutant_residues"]).all()

    def test_wt_peptide_populated_for_substitutions(self):
        """Substitution effects have matched-length original/mutant
        proteins, so wt_peptide is derived from the reference slice."""
        df = _predictor().predict_from_variants(cancer_test_variants)
        aff = df[df["kind"] == "pMHC_affinity"]
        # Every affinity row should get a WT peptide for these SNVs.
        assert aff["wt_peptide"].notna().all()
        # And mutant-containing peptides should differ from their WT.
        mutant = aff[aff["contains_mutant_residues"].eq(True)]
        assert (mutant["peptide"] != mutant["wt_peptide"]).all()


# ---------------------------------------------------------------------------
# _fragment_from_effect unit tests (varcode-free via mock)
# ---------------------------------------------------------------------------


def _make_mock_effect(
    mutant_protein="MAAGVTDVGMAVATGSWDSFLK",
    original_protein="MAAGVTDVGMAVATGSWDSFLK",
    mutation_start=11,
    mutation_end=12,
    cls_name="Substitution",
    variant_desc="chr1:100 A>T",
    effect_desc="p.A12V",
    gene_name="FAKE",
    gene_id="ENSG000",
    transcript_id="ENST000",
    transcript_name="FAKE-001",
):
    # Build a real class so type(effect).__name__ returns cls_name —
    # which is what _fragment_from_effect uses to pick source_type.
    effect_cls = type(cls_name, (), {})
    e = effect_cls()
    e.mutant_protein_sequence = mutant_protein
    e.original_protein_sequence = original_protein
    e.aa_mutation_start_offset = mutation_start
    e.aa_mutation_end_offset = mutation_end
    variant_obj = MagicMock()
    variant_obj.short_description = variant_desc
    e.variant = variant_obj
    e.short_description = effect_desc
    e.gene_name = gene_name
    e.gene_id = gene_id
    e.transcript_id = transcript_id
    e.transcript_name = transcript_name
    return e


class TestFragmentFromEffect:
    def test_substitution_populates_reference(self):
        effect = _make_mock_effect(
            mutant_protein="MAAGVTDVGMAVATGSWDSFLK",  # V at pos 11
            original_protein="MAAGVTDVGMAAATGSWDSFLK",  # A at pos 11
            mutation_start=11, mutation_end=12,
            cls_name="Substitution",
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag is not None
        assert frag.source_type == "variant:snv"
        assert frag.reference_sequence is not None
        # Length-match preserved (substitution)
        assert len(frag.reference_sequence) == len(frag.sequence)
        # target_intervals is relative to the subsequence
        ti = frag.target_intervals
        assert len(ti) == 1
        rel_start, rel_end = ti[0]
        # Subsequence starts at max(0, 11-8)=3, so relative pos = 11-3 = 8
        assert rel_start == 8
        assert rel_end == 9

    def test_frameshift_gets_no_reference_sequence(self):
        """Frameshift effects have divergent mutant/original protein
        lengths — reference_sequence stays None so that the wt_peptide
        derivation is not misled into slicing mismatched coords."""
        effect = _make_mock_effect(
            mutant_protein="MAAGVTDVGMAVXXXXX",   # 17 aa (frameshifted)
            original_protein="MAAGVTDVGMAAATGSWDSFLK",  # 22 aa
            mutation_start=11, mutation_end=12,
            cls_name="FrameShift",
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag is not None
        assert frag.source_type == "variant:frameshift"
        assert frag.reference_sequence is None

    def test_subsequence_offset_stashed_in_annotations(self):
        effect = _make_mock_effect(mutation_start=50, mutation_end=51)
        frag = _fragment_from_effect(
            effect,
            padding_around_mutation=8,
            gene_expression=12.3,
        )
        # Mutation at 50, padding 8 → expected seq_start = 42.
        # But the mock protein is only 22 aa, so everything clips.
        # Validate that annotations were populated at all.
        assert _SUBSEQ_OFFSET_KEY in frag.annotations
        assert _MUTATION_START_KEY in frag.annotations
        assert _MUTATION_END_KEY in frag.annotations
        assert frag.annotations[_MUTATION_START_KEY] == 50
        assert frag.annotations[_MUTATION_END_KEY] == 51
        assert frag.gene_expression == 12.3

    def test_returns_none_for_effect_without_protein(self):
        effect = _make_mock_effect(mutant_protein=None)
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag is None

    def test_stop_codon_trims_subsequence(self):
        """Stop codon in the mutant protein should truncate the subseq
        at the stop — preserves legacy behavior of
        protein_subsequences_around_mutations."""
        effect = _make_mock_effect(
            mutant_protein="MAAGVTDVGM*AVATGSWDSFLK",  # stop at pos 10
            original_protein="MAAGVTDVGMAVATGSWDSFLK",
            mutation_start=8, mutation_end=9,
            cls_name="Substitution",
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag is not None
        # Subsequence should not contain the '*' character
        assert "*" not in frag.sequence

    def test_premature_stop_source_type(self):
        """PrematureStop effects map to the documented ``variant:stop_gain``."""
        effect = _make_mock_effect(cls_name="PrematureStop")
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag.source_type == "variant:stop_gain"

    def test_multi_residue_substitution_becomes_indel(self):
        """Block substitutions (span > 1) collapse to the documented
        ``variant:indel`` rather than introducing a new vocabulary term."""
        effect = _make_mock_effect(
            mutation_start=10, mutation_end=13,   # 3-residue block
            cls_name="Substitution",
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag.source_type == "variant:indel"

    def test_unknown_effect_class_falls_back_to_lowercase(self):
        """Unlisted effect classes yield ``variant:<classname_lower>``
        so new varcode effect types remain representable without a
        Topiary change."""
        effect = _make_mock_effect(cls_name="SomeNovelEffect")
        frag = _fragment_from_effect(effect, padding_around_mutation=8)
        assert frag.source_type == "variant:somenoveleffect"


# ---------------------------------------------------------------------------
# Legacy expression-dict plumbing
# ---------------------------------------------------------------------------


class TestExpressionDictPlumbing:
    """Regression: legacy gene_expression_dict / transcript_expression_dict
    still propagate into the column output through the refactored path."""

    def test_transcript_expression_dict_populates_column(self):
        expr_dict = {
            transcript_id: 42.0
            for v in cancer_test_variants
            for transcript_id in v.transcript_ids
        }
        df = _predictor().predict_from_variants(
            cancer_test_variants, transcript_expression_dict=expr_dict,
        )
        assert "transcript_expression" in df.columns
        assert (df["transcript_expression"] == 42.0).all()

    def test_gene_expression_dict_populates_column(self):
        expr_dict = {
            gene_id: 7.5
            for v in cancer_test_variants
            for gene_id in v.gene_ids
        }
        df = _predictor().predict_from_variants(
            cancer_test_variants, gene_expression_dict=expr_dict,
        )
        assert "gene_expression" in df.columns
        assert (df["gene_expression"] == 7.5).all()
