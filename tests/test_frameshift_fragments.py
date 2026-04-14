"""Regression tests pinning _fragment_from_effect behavior on varcode
FrameShift / FrameShiftTruncation effects.

Codex review of PR B flagged a suspected bug: the adapter passes
``inframe=True`` when forwarding a FrameShift to
``ProteinFragment.from_variant``, which the reviewer claimed would
limit ``target_intervals`` to just the reported mutation span and drop
downstream novel 9-mers under ``only_novel_epitopes=True``.

These tests verify the opposite: because varcode's FrameShift sets
``aa_mutation_end_offset == len(mutant_protein_sequence)`` (i.e. the
reported mutation interval already extends to the end of the novel
tail), the ``inframe=True`` path produces the same target_intervals
as ``inframe=False`` would — every 9-mer overlapping the shift is
correctly flagged ``contains_mutant_residues=True`` and survives
``only_novel_epitopes`` filtering.

We cover:
  * the varcode invariant itself (via a real ``varcode.effects.FrameShift``);
  * ``_fragment_from_effect`` target_intervals across a grid of shift
    positions, tail lengths, and paddings;
  * per-peptide overlaps_target for every sliding window inside the
    subsequence;
  * ``inframe=True`` vs ``inframe=False`` equivalence for frameshifts;
  * FrameShiftTruncation (empty interval, no novel residues);
  * end-to-end ``predict_from_mutation_effects`` + ``only_novel_epitopes``
    behavior on a mocked FrameShift.
"""

from unittest.mock import MagicMock

import pytest
from mhctools import RandomBindingPredictor

from topiary import TopiaryPredictor
from topiary.predictor import _fragment_from_effect
from topiary.protein_fragment import ProteinFragment


ALLELES = ["HLA-A*02:01"]
REFERENCE_PROTEIN = "MAAGVTDVGMAVATGSWDSFLKQWERTYPASDFGHJKLZXCVBNM"  # 45 aa


def _make_mock_frameshift(
    *,
    original_protein=REFERENCE_PROTEIN,
    shift_at=11,
    shifted_tail="XYZQRSTUV",
    gene_name="FAKE",
    gene_id="ENSG0FAKE",
    transcript_id="ENST0FAKE",
    transcript_name="FAKE-001",
    variant_desc="chr1:100 A>AT",
    effect_desc="p.A12fs",
):
    """Build a mock FrameShift that mirrors varcode's real invariants.

    Specifically: ``mutant_protein_sequence = original[:shift_at] + shifted_tail``
    and ``aa_mutation_end_offset = shift_at + len(shifted_tail)`` — both
    assertions from varcode's ``FrameShift.__init__`` path.
    """
    prefix = original_protein[:shift_at]
    mutant = prefix + shifted_tail
    effect = type("FrameShift", (), {})()
    effect.mutant_protein_sequence = mutant
    effect.original_protein_sequence = original_protein
    effect.aa_mutation_start_offset = shift_at
    effect.aa_mutation_end_offset = shift_at + len(shifted_tail)
    variant_obj = MagicMock()
    variant_obj.short_description = variant_desc
    effect.variant = variant_obj
    effect.short_description = effect_desc
    effect.gene_name = gene_name
    effect.gene_id = gene_id
    effect.transcript_id = transcript_id
    effect.transcript_name = transcript_name
    return effect


def _make_mock_frameshift_truncation(
    *,
    original_protein=REFERENCE_PROTEIN,
    truncate_at=11,
    **kwargs,
):
    """Mock for varcode FrameShiftTruncation: the frameshift hits an
    immediate stop so no novel residues are added.  Invariants:
    ``aa_mutation_end_offset == aa_mutation_start_offset == truncate_at``
    and ``mutant_protein_sequence == original[:truncate_at]``.
    """
    mutant = original_protein[:truncate_at]
    effect = type("FrameShiftTruncation", (), {})()
    effect.mutant_protein_sequence = mutant
    effect.original_protein_sequence = original_protein
    effect.aa_mutation_start_offset = truncate_at
    effect.aa_mutation_end_offset = truncate_at
    variant_obj = MagicMock()
    variant_obj.short_description = kwargs.get("variant_desc", "chr1:100 A>AT")
    effect.variant = variant_obj
    effect.short_description = kwargs.get("effect_desc", "p.A12fs*")
    effect.gene_name = kwargs.get("gene_name", "FAKE")
    effect.gene_id = kwargs.get("gene_id", "ENSG0FAKE")
    effect.transcript_id = kwargs.get("transcript_id", "ENST0FAKE")
    effect.transcript_name = kwargs.get("transcript_name", "FAKE-001")
    return effect


def _predictor(**kwargs):
    return TopiaryPredictor(
        models=RandomBindingPredictor(
            alleles=ALLELES, default_peptide_lengths=[9],
        ),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Varcode invariant — sanity check against the actual varcode implementation
# ---------------------------------------------------------------------------


class TestVarcodeFrameShiftInvariant:
    """Pin the varcode-side contract that the rest of this file relies on:
    for ``varcode.effects.FrameShift``, ``aa_mutation_end_offset`` equals
    ``len(mutant_protein_sequence)``.  If a future varcode release breaks
    this invariant, these tests fail fast and the adapter needs a rethink.
    """

    def test_real_varcode_frameshift_end_equals_protein_length(self):
        from varcode.effects import FrameShift

        transcript = MagicMock()
        transcript.protein_sequence = REFERENCE_PROTEIN
        variant = MagicMock()
        shifted = "XYZQRSTUV"
        fs = FrameShift(
            variant=variant,
            transcript=transcript,
            aa_mutation_start_offset=11,
            shifted_sequence=shifted,
        )
        assert fs.aa_mutation_end_offset == len(fs.mutant_protein_sequence)
        assert fs.mutant_protein_sequence == REFERENCE_PROTEIN[:11] + shifted
        # And our mock helper mirrors that
        mock = _make_mock_frameshift(shift_at=11, shifted_tail=shifted)
        assert mock.aa_mutation_end_offset == fs.aa_mutation_end_offset
        assert mock.mutant_protein_sequence == fs.mutant_protein_sequence

    def test_real_varcode_frameshift_truncation_invariant(self):
        from varcode.effects import FrameShiftTruncation

        transcript = MagicMock()
        transcript.protein_sequence = REFERENCE_PROTEIN
        variant = MagicMock()
        fs = FrameShiftTruncation(
            variant=variant,
            transcript=transcript,
            stop_codon_offset=11,
        )
        assert fs.aa_mutation_start_offset == fs.aa_mutation_end_offset == 11
        assert fs.mutant_protein_sequence == REFERENCE_PROTEIN[:11]


# ---------------------------------------------------------------------------
# _fragment_from_effect: target_intervals across a grid of FrameShift shapes
# ---------------------------------------------------------------------------


# (shift_at, shifted_tail, padding_around_mutation)
_FRAMESHIFT_GRID = [
    (11, "XYZQRSTUV",         8),   # short tail, mid-protein
    (11, "XYZQRSTUV",         16),  # larger padding (no effect on FS)
    (5,  "XYZQRSTUV",         8),   # shift near N-terminus
    (5,  "XYZQRSTUV",         16),
    (40, "KLMNOP",            8),   # shift near C-terminus
    (20, "ABCDEFGHIJKLMN",    8),   # longer tail
    (20, "ABCDEFGHIJKLMNOP",  20),
    (11, "X",                 8),   # minimum 1-residue tail
    (30, "NOVELRESIDUES",     0),   # zero padding on the mutation side
]


class TestFrameShiftTargetInterval:
    """_fragment_from_effect on varcode-style FrameShift: verify
    target_intervals covers the full downstream novel tail in the
    resulting subsequence, under a grid of shift positions / tail
    lengths / paddings.
    """

    @pytest.mark.parametrize("shift_at,shifted_tail,padding", _FRAMESHIFT_GRID)
    def test_target_interval_spans_full_downstream_tail(
        self, shift_at, shifted_tail, padding,
    ):
        effect = _make_mock_frameshift(
            shift_at=shift_at, shifted_tail=shifted_tail,
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)
        assert frag is not None
        assert frag.source_type == "variant:frameshift"
        assert len(frag.target_intervals) == 1
        rel_start, rel_end = frag.target_intervals[0]

        mutant_len = shift_at + len(shifted_tail)
        expected_seq_start = max(0, shift_at - padding)
        # varcode invariant -> mut_end == mutant_len, no '*' in tail, so
        # seq_end = min(mutant_len, mut_end + padding) = mutant_len
        expected_seq_end = mutant_len

        assert len(frag.sequence) == expected_seq_end - expected_seq_start
        # Target interval in subsequence coordinates
        assert rel_start == shift_at - expected_seq_start
        # Critically: interval extends to the END of the subsequence
        # (the full novel tail is within the target region)
        assert rel_end == len(frag.sequence), (
            f"target interval rel_end={rel_end} does not reach end of "
            f"subsequence (len={len(frag.sequence)}). Downstream peptides "
            f"would be misclassified as reference."
        )

    @pytest.mark.parametrize("shift_at,shifted_tail,padding", _FRAMESHIFT_GRID)
    def test_reference_sequence_is_none_for_frameshift(
        self, shift_at, shifted_tail, padding,
    ):
        """Frameshift mutant protein and original protein have divergent
        lengths / residue mappings — reference_sequence must stay None
        so wt_peptide derivation doesn't emit mis-aligned slices."""
        effect = _make_mock_frameshift(
            shift_at=shift_at, shifted_tail=shifted_tail,
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)
        assert frag.reference_sequence is None


# ---------------------------------------------------------------------------
# Per-peptide overlaps_target: every 9-mer inside the shifted tail should
# be marked as novel; 9-mers entirely upstream of the shift should not.
# ---------------------------------------------------------------------------


class TestFrameShiftPerPeptideOverlap:
    @pytest.mark.parametrize("shift_at,shifted_tail,padding", _FRAMESHIFT_GRID)
    @pytest.mark.parametrize("peptide_length", [8, 9, 10, 11])
    def test_every_peptide_touching_shift_is_novel(
        self, shift_at, shifted_tail, padding, peptide_length,
    ):
        effect = _make_mock_frameshift(
            shift_at=shift_at, shifted_tail=shifted_tail,
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)
        seq_start = max(0, shift_at - padding)
        rel_shift = shift_at - seq_start

        if len(frag.sequence) < peptide_length:
            pytest.skip(
                f"subsequence ({len(frag.sequence)} aa) smaller than "
                f"peptide_length ({peptide_length}); no windows to enumerate"
            )

        seen_novel, seen_reference = False, False
        for offset in range(0, len(frag.sequence) - peptide_length + 1):
            overlaps = frag.peptide_overlaps_target(offset, peptide_length)
            peptide_end = offset + peptide_length
            # Ground truth: peptide is novel iff it overlaps the shift
            # in subsequence coords, i.e. peptide_end > rel_shift.
            expected = peptide_end > rel_shift
            assert overlaps is expected, (
                f"shift_at={shift_at} tail={shifted_tail!r} padding={padding} "
                f"peptide_length={peptide_length} offset={offset}: "
                f"expected overlaps_target={expected} but got {overlaps}"
            )
            if overlaps:
                seen_novel = True
            else:
                seen_reference = True

        # We expect at least one novel peptide (the shift is in-subsequence)
        assert seen_novel, "no novel peptides enumerated — check fixture"
        # When padding + peptide_length allows, we should also see some
        # reference-only peptides.  Skip the assertion when the shift is
        # near position 0 (all peptides unavoidably straddle the shift).
        if rel_shift >= peptide_length:
            assert seen_reference, "expected some reference-only peptides"


# ---------------------------------------------------------------------------
# inframe=True / inframe=False equivalence for frameshifts — justifies the
# cleanup Codex suggested (pass inframe=False for self-documentation) and
# confirms that the current inframe=True behavior is not buggy.
# ---------------------------------------------------------------------------


class TestInframeFlagEquivalenceOnFrameShift:
    """For varcode-style FrameShift effects where mut_end == len(protein),
    from_variant(inframe=True) and from_variant(inframe=False) produce
    identical target_intervals.  This equivalence is what makes the
    current adapter correct despite using ``inframe=True``.
    """

    @pytest.mark.parametrize("shift_at,shifted_tail,padding", _FRAMESHIFT_GRID)
    def test_inframe_true_and_false_agree(
        self, shift_at, shifted_tail, padding,
    ):
        effect = _make_mock_frameshift(
            shift_at=shift_at, shifted_tail=shifted_tail,
        )
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)

        # Re-build the same subsequence manually and invoke from_variant
        # both ways to compare.
        mutant_protein = effect.mutant_protein_sequence
        mut_start = effect.aa_mutation_start_offset
        mut_end = effect.aa_mutation_end_offset
        seq_start = max(0, mut_start - padding)
        first_stop = mutant_protein.find("*")
        if first_stop < 0:
            first_stop = len(mutant_protein)
        seq_end = min(first_stop, mut_end + padding)
        subsequence = mutant_protein[seq_start:seq_end]

        frag_inframe_true = ProteinFragment.from_variant(
            sequence=subsequence,
            mutation_start=mut_start - seq_start,
            mutation_end=mut_end - seq_start,
            inframe=True,
            source_type="variant:frameshift",
            variant=effect.variant.short_description,
        )
        frag_inframe_false = ProteinFragment.from_variant(
            sequence=subsequence,
            mutation_start=mut_start - seq_start,
            mutation_end=mut_end - seq_start,
            inframe=False,
            source_type="variant:frameshift",
            variant=effect.variant.short_description,
        )
        # Exact equality of the target intervals is the key property
        assert frag_inframe_true.target_intervals == frag_inframe_false.target_intervals
        assert frag_inframe_true.target_intervals == frag.target_intervals


# ---------------------------------------------------------------------------
# FrameShiftTruncation: empty interval, no novel residues
# ---------------------------------------------------------------------------


class TestFrameShiftTruncation:
    @pytest.mark.parametrize("truncate_at,padding", [
        (11, 8), (11, 16), (5, 8), (30, 8), (40, 20),
    ])
    def test_empty_target_interval(self, truncate_at, padding):
        """FrameShiftTruncation adds no novel residues; mut_start==mut_end.
        The target interval is empty (a point)."""
        effect = _make_mock_frameshift_truncation(truncate_at=truncate_at)
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)
        assert frag is not None
        assert frag.source_type == "variant:frameshift"
        assert len(frag.target_intervals) == 1
        rel_start, rel_end = frag.target_intervals[0]
        assert rel_start == rel_end, (
            "FrameShiftTruncation should yield an empty target interval"
        )

    @pytest.mark.parametrize("truncate_at,padding,peptide_length", [
        (11, 8, 9), (20, 12, 9), (30, 16, 10),
    ])
    def test_no_peptides_are_novel(self, truncate_at, padding, peptide_length):
        effect = _make_mock_frameshift_truncation(truncate_at=truncate_at)
        frag = _fragment_from_effect(effect, padding_around_mutation=padding)
        for offset in range(0, len(frag.sequence) - peptide_length + 1):
            assert not frag.peptide_overlaps_target(offset, peptide_length), (
                f"peptide at offset={offset} erroneously marked novel for "
                "FrameShiftTruncation (no novel residues exist)"
            )


# ---------------------------------------------------------------------------
# End-to-end: predict_from_mutation_effects on a mocked FrameShift must
# keep the downstream novel peptides under only_novel_epitopes=True.
# ---------------------------------------------------------------------------


def _patch_groupby_variant_for_single_effect(effect):
    """varcode EffectCollection has a ``groupby_variant`` method used by
    the predictor.  We wrap a single mock effect in a minimal shim that
    returns itself grouped by its (mock) variant."""
    # Use a list-of-effects wrapper that behaves enough like an
    # EffectCollection for predict_from_mutation_effects' usage.
    from varcode import EffectCollection

    # Mock enough of the effect to satisfy the silent/noncoding filter:
    # filter_silent_and_noncoding_effects checks isinstance(NonsilentCodingMutation).
    # We bypass that by constructing a real EffectCollection of real effects.
    return EffectCollection([effect])


class TestFrameShiftEndToEnd:
    """Integration: a mocked FrameShift goes all the way through
    predict_from_mutation_effects and survives only_novel_epitopes."""

    def _make_real_frameshift_effect(self, shift_at=11, shifted_tail="XYZQRSTUV"):
        """Construct a real varcode FrameShift attached to a variant /
        transcript mock, which is enough for the predictor path to run."""
        from varcode.effects import FrameShift

        transcript = MagicMock()
        transcript.protein_sequence = REFERENCE_PROTEIN
        transcript.gene_id = "ENSG0FAKE"
        transcript.gene_name = "FAKE"
        transcript.id = "ENST0FAKE"
        transcript.name = "FAKE-001"
        variant = MagicMock()
        variant.short_description = "chr1:100 A>AT"
        fs = FrameShift(
            variant=variant,
            transcript=transcript,
            aa_mutation_start_offset=shift_at,
            shifted_sequence=shifted_tail,
        )
        # The predictor uses effect.gene_name / gene_id / transcript_id /
        # transcript_name; varcode's FrameShift derives these from
        # transcript, so our MagicMock attributes are picked up.
        return fs

    @pytest.mark.parametrize("shift_at,shifted_tail,padding", [
        (11, "XYZQRSTUVWXYZ", 8),
        (15, "ABCDEFGHIJKL", 12),
        (5,  "NOVELRESIDUES", 8),
    ])
    def test_all_downstream_9mers_survive_only_novel_epitopes(
        self, shift_at, shifted_tail, padding,
    ):
        effect = self._make_real_frameshift_effect(
            shift_at=shift_at, shifted_tail=shifted_tail,
        )
        frag = _fragment_from_effect(
            effect, padding_around_mutation=padding,
        )
        assert frag is not None
        mutant_protein = effect.mutant_protein_sequence

        # Build an EffectCollection containing just this effect and run
        # it through predict_from_mutation_effects with only_novel_epitopes.
        from varcode import EffectCollection

        predictor_novel = _predictor(
            padding_around_mutation=padding,
            only_novel_epitopes=True,
        )
        predictor_all = _predictor(padding_around_mutation=padding)

        effects = EffectCollection([effect])
        df_all = predictor_all.predict_from_mutation_effects(effects)
        df_novel = predictor_novel.predict_from_mutation_effects(effects)

        aff_all = df_all[df_all["kind"] == "pMHC_affinity"]
        aff_novel = df_novel[df_novel["kind"] == "pMHC_affinity"]
        assert not aff_all.empty
        assert not aff_novel.empty

        # Every surviving row is marked novel, obviously.
        assert aff_novel["contains_mutant_residues"].eq(True).all()

        # Every 9-mer in aff_all whose peptide_offset + peptide_length
        # crosses shift_at must be flagged novel (and thus appear in
        # aff_novel).
        crosses_shift = (
            aff_all["peptide_offset"].astype(int)
            + aff_all["peptide_length"].astype(int)
            > shift_at
        )
        expected_novel_peptides = set(aff_all.loc[crosses_shift, "peptide"])
        actual_novel_peptides = set(aff_novel["peptide"])
        missing = expected_novel_peptides - actual_novel_peptides
        assert not missing, (
            f"{len(missing)} novel peptides that cross the frameshift at "
            f"position {shift_at} were incorrectly dropped by "
            f"only_novel_epitopes: {sorted(missing)[:3]}..."
        )

        # And the peptides at or past the C-terminus of the novel tail
        # must be present, pinning the "Codex concern" directly: a peptide
        # whose offset is shift_at+1 (fully downstream) must appear.
        deep_downstream = aff_all[
            aff_all["peptide_offset"].astype(int) > shift_at
        ]
        if not deep_downstream.empty:
            # Each should be present in aff_novel
            for peptide in deep_downstream["peptide"]:
                assert peptide in actual_novel_peptides, (
                    f"deep-downstream peptide {peptide!r} (offset past "
                    f"shift position {shift_at}) was dropped — this is the "
                    "exact regression Codex worried about"
                )

    def test_overlaps_target_matches_contains_mutant_on_frameshift(self):
        """On the variant path, overlaps_target and contains_mutant_residues
        derive from the same target_intervals — they must agree row-by-row
        even for frameshifts."""
        effect = self._make_real_frameshift_effect()
        from varcode import EffectCollection

        df = _predictor().predict_from_mutation_effects(EffectCollection([effect]))
        aff = df[df["kind"] == "pMHC_affinity"]
        assert not aff.empty
        assert (aff["overlaps_target"] == aff["contains_mutant_residues"]).all()
