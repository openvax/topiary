"""Running isovar for real, interchangeably with reference translation.

Give `fragments_from_variants` an alignment file and the protein context
around each mutation is assembled from RNA reads; leave it out and the same
variants are translated from the reference. Both arms return
`ProteinFragment`s with the same core.

**Nothing here imports isovar.** The RNA arm is exercised through a fake, so
these run in an environment that has never installed it — which is the claim
the module makes and CI is such an environment.
"""

import pytest
from varcode import NonsilentCodingMutation

from topiary import (
    DEFAULT_PROTEIN_SEQUENCE_LENGTH,
    ProteinFragment,
    fragments_from_effects,
    fragments_from_variants,
)


class _ProteinSequence:
    amino_acids = "MKTVRQERLKSIVRILEDAAWQ"
    mutation_start_idx = 10
    mutation_end_idx = 12
    gene_name = "BRAF"
    transcript_ids = ["ENST1"]
    transcript_names = ["BRAF-204"]
    num_supporting_fragments = 27
    num_supporting_reads = 52


class _Result:
    def __init__(self, supported=True, passes=True, variant="v"):
        self.top_protein_sequence = _ProteinSequence() if supported else None
        self.variant = variant
        self.passes_all_filters = passes
        self.num_total_fragments = 61
        self.num_total_reads = 118
        self.num_alt_fragments = 30
        self.num_alt_reads = 58
        self.num_ref_fragments = 31
        self.num_ref_reads = 60


class _FakeCreator:
    def __init__(self, protein_sequence_length=21,
                 variant_sequence_assembly=False):
        self.protein_sequence_length = protein_sequence_length
        self.variant_sequence_assembly = variant_sequence_assembly


class _FakeIsovar:
    """Stands in for the isovar module, recording how it was called."""

    def __init__(self, results):
        self._results = results
        self.calls = []

    def run_isovar(self, **kwargs):
        self.calls.append(kwargs)
        return self._results


def _fake(monkeypatch, results):
    module = _FakeIsovar(results)
    monkeypatch.setattr("topiary.io_isovar._check_isovar", lambda: module)
    monkeypatch.setattr(
        "topiary.io_isovar.ProteinSequenceCreator", _FakeCreator,
        raising=False,
    )
    import sys
    import types
    stub = types.ModuleType("isovar.protein_sequence_creator")
    stub.ProteinSequenceCreator = _FakeCreator
    monkeypatch.setitem(sys.modules, "isovar", types.ModuleType("isovar"))
    monkeypatch.setitem(sys.modules, "isovar.protein_sequence_creator", stub)
    return module


@pytest.fixture
def isovar(monkeypatch):
    return _fake(monkeypatch, [_Result()])


# ---------------------------------------------------------------------------
# The RNA arm
# ---------------------------------------------------------------------------


def test_an_alignment_file_assembles_from_rna(isovar):
    fragments = fragments_from_variants(["v"], alignment_file=object())

    assert len(fragments) == 1
    assert fragments[0].annotations["sequence_source"] == "isovar_assembly"


def test_assembly_is_turned_on():
    """isovar defaults it off, and with it off a single read must span the
    whole window — so a longer context yields fewer variants rather than
    longer sequences, and "carrying the phasing the reads support" would
    not be true of the result."""
    import types

    import topiary.io_isovar as module

    captured = {}

    class _Creator(_FakeCreator):
        def __init__(self, **kwargs):
            captured.update(kwargs)
            super().__init__(**kwargs)

    stub = types.ModuleType("isovar.protein_sequence_creator")
    stub.ProteinSequenceCreator = _Creator
    fake = _FakeIsovar([_Result()])

    import sys
    sys.modules["isovar"] = types.ModuleType("isovar")
    sys.modules["isovar.protein_sequence_creator"] = stub
    original = module._check_isovar
    module._check_isovar = lambda: fake
    try:
        fragments_from_variants(["v"], alignment_file=object())
    finally:
        module._check_isovar = original
        del sys.modules["isovar"], sys.modules["isovar.protein_sequence_creator"]

    assert captured["variant_sequence_assembly"] is True


def test_the_default_window_matches_isovar_rather_than_exceeding_it():
    """Asking for more context than the reads support returns fewer
    variants, not longer sequences."""
    assert DEFAULT_PROTEIN_SEQUENCE_LENGTH == 21


def test_isovar_knobs_are_passed_through(isovar):
    fragments_from_variants(
        ["v"], alignment_file=object(),
        transcript_id_whitelist={"ENST1"},
        filter_thresholds={"min_num_alt_reads": 3},
        min_shared_fragments_for_phasing=4,
    )

    call = isovar.calls[0]
    assert call["transcript_id_whitelist"] == {"ENST1"}
    assert call["filter_thresholds"] == {"min_num_alt_reads": 3}
    assert call["min_shared_fragments_for_phasing"] == 4


def test_the_read_counts_come_through_as_measured(isovar):
    fragment = fragments_from_variants(["v"], alignment_file=object())[0]

    assert fragment.n_rna_alt == 30            # fragments preferred
    assert fragment.n_alt_reads == 58          # reads also carried
    assert not fragment.is_approximate("n_alt_reads")


# ---------------------------------------------------------------------------
# Filters have to filter
# ---------------------------------------------------------------------------


def test_a_result_failing_its_filters_is_dropped(monkeypatch):
    """isovar records filter outcomes and never drops anything, so without
    this a caller's thresholds — and isovar's own defaults — annotate
    results that then flow on as RNA-backed evidence."""
    _fake(monkeypatch, [_Result(passes=False)])

    assert fragments_from_variants(["v"], alignment_file=object()) == []


def test_filtering_can_be_turned_off(monkeypatch):
    _fake(monkeypatch, [_Result(passes=False)])

    fragments = fragments_from_variants(
        ["v"], alignment_file=object(), require_passing_filters=False,
    )

    assert len(fragments) == 1


def test_a_filtered_out_result_can_fall_back_to_reference(monkeypatch):
    _fake(monkeypatch, [_Result(passes=False, variant="v")])
    monkeypatch.setattr(
        "topiary.io_isovar.fragments_from_effects",
        lambda effects, padding, **kw: [
            ProteinFragment(fragment_id="ref", sequence="MKTV"),
        ],
    )
    monkeypatch.setattr("topiary.io_isovar._effects_for", lambda variants: [])

    fragments = fragments_from_variants(
        ["v"], alignment_file=object(), allow_reference_fallback=True,
    )

    assert [f.fragment_id for f in fragments] == ["ref"]


# ---------------------------------------------------------------------------
# Conflicting requests are refused rather than silently resolved
# ---------------------------------------------------------------------------


def test_a_creator_and_a_length_together_are_refused(isovar):
    """The creator's length would have won silently, breaking the
    guarantee protein_sequence_length exists to make."""
    with pytest.raises(ValueError, match="pass one"):
        fragments_from_variants(
            ["v"], alignment_file=object(),
            protein_sequence_length=31,
            protein_sequence_creator=_FakeCreator(protein_sequence_length=15),
        )


def test_a_caller_supplied_creator_is_used(isovar):
    mine = _FakeCreator(protein_sequence_length=31)

    fragments_from_variants(
        ["v"], alignment_file=object(), protein_sequence_creator=mine,
    )

    assert isovar.calls[0]["protein_sequence_creator"] is mine


def test_isovar_only_kwargs_are_refused_without_an_alignment_file():
    """They were silently swallowed, so a caller who forgot the BAM got a
    full result set built with none of their configuration."""
    with pytest.raises(TypeError, match="only apply when an alignment_file"):
        fragments_from_variants(["v"], transcript_id_whitelist={"E"})


def test_a_nonsense_protein_sequence_length_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        fragments_from_variants(["v"], protein_sequence_length=0)


def test_a_padding_too_small_for_an_epitope_is_refused():
    """check_padding_around_mutation exists for exactly this; the earlier
    ad-hoc derivation bypassed it and produced fragments no 9-mer fits."""
    with pytest.raises(ValueError):
        fragments_from_variants(
            ["v"], padding_around_mutation=2, epitope_lengths=(9,),
        )


# ---------------------------------------------------------------------------
# The reference arm, exercised rather than stubbed
# ---------------------------------------------------------------------------


class _Effect(NonsilentCodingMutation):
    """A real NonsilentCodingMutation, because the arm filters on the type.

    The earlier version of this test used a duck-typed stand-in, which
    `filter_silent_and_noncoding_effects` correctly discarded — so the
    test exercised nothing. A fake that the code under test rejects is
    not a fake, it is a hole.
    """

    # Several of these are read-only properties on the real class.
    mutant_protein_sequence = "MKTVRQERLKSIVRILEDAAWQ"
    original_protein_sequence = "MKTVRQERLKAIVRILEDAAWQ"
    aa_mutation_start_offset = 10
    aa_mutation_end_offset = 11
    gene_name = "BRAF"
    gene_id = "ENSG1"
    transcript_id = "ENST1"
    transcript_name = "BRAF-204"
    short_description = "p.A11R"
    modifies_protein_sequence = True

    def __init__(self, variant="v", gene="BRAF", transcript="ENST1"):
        self.gene_name = gene
        self.transcript_id = transcript
        self.variant = type("V", (), {"short_description": variant})()

    def __hash__(self):
        return hash((self.transcript_id, self.short_description))

    def __eq__(self, other):
        return self is other


def test_the_reference_arm_translates_an_effect():
    """Real translation, not a stub — the arm the docs sell as half the
    feature had no coverage at all before."""
    fragments = fragments_from_effects(
        [_Effect()], padding_around_mutation=8,
    )

    assert len(fragments) == 1
    assert fragments[0].sequence
    assert fragments[0].gene == "BRAF"


def test_a_reference_fragment_says_where_its_sequence_came_from():
    fragment = fragments_from_effects([_Effect()], padding_around_mutation=8)[0]

    assert fragment.annotations["sequence_source"] == "varcode_translation"


def test_a_reference_fragment_has_no_read_counts():
    """No RNA was consulted, and it says so rather than saying zero."""
    fragment = fragments_from_effects([_Effect()], padding_around_mutation=8)[0]

    assert not fragment.is_known("n_alt_reads")
    assert not fragment.is_usable_as_biology("n_alt_reads")


def test_expression_is_attached_when_given():
    fragment = fragments_from_effects(
        [_Effect()], padding_around_mutation=8,
        gene_expression={"ENSG1": 12.5},
    )[0]

    assert fragment.gene_expression == 12.5


def test_no_effects_yields_no_fragments():
    assert fragments_from_effects([], padding_around_mutation=8) == []


def test_the_two_arms_are_read_the_same_way(isovar):
    """Interchangeable means the caller's next line does not change."""
    def support(fragment):
        """What a consumer should write: ask for the evidence, not a unit."""
        return fragment.n_rna_alt

    rna = fragments_from_variants(["v"], alignment_file=object())[0]
    reference = fragments_from_effects([_Effect()], padding_around_mutation=8)[0]

    assert isinstance(rna, ProteinFragment)
    assert isinstance(reference, ProteinFragment)
    assert support(rna) == 30                      # fragments, preferred
    assert rna.rna_evidence_subject() == "fragments"
    assert support(reference) is None
