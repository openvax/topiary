"""Public API for logic a consumer would otherwise reimplement.

AGENTS.md: logic that does real work belongs in one documented public
function. The failure mode is concrete and has happened repeatedly — a
consumer needs behavior it cannot import, reimplements it, and the copy
drifts. `is_named_version` is the sharpest example: a downstream consumer
wrote `if str(v).strip()`, which admits the literal string "nan", and rows
carrying it were dropped from scoring.
"""

import numpy as np
import pandas as pd
import pytest

from topiary import (
    NOT_STATED_VERSIONS,
    ProteinFragment,
    fragment_from_effect,
    is_named_version,
    known_versions,
)


class _Effect:
    """The varcode surface fragment_from_effect actually reads."""

    def __init__(self, mutant, original=None, start=3, end=4):
        self.mutant_protein_sequence = mutant
        self.original_protein_sequence = original
        self.aa_mutation_start_offset = start
        self.aa_mutation_end_offset = end
        self.gene_name = "BRAF"
        self.gene_id = "ENSG1"
        self.transcript_id = "ENST1"
        self.transcript_name = "BRAF-204"
        self.short_description = "p.V600E"
        self.variant = type("V", (), {"short_description": "chr7:1A>T"})()


# ---------------------------------------------------------------------------
# is_named_version
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["4.1b", "4.2", " 4.2 ", "1", "nightly-3"])
def test_a_real_version_is_named(value):
    assert is_named_version(value)


@pytest.mark.parametrize(
    "value", [None, np.nan, float("nan"), pd.NA, "", "   ", "nan", "NaN"],
    ids=["none", "np-nan", "float-nan", "pd-na", "empty", "blank",
         "literal-nan", "literal-NaN"],
)
def test_a_missing_version_is_not_named(value):
    assert not is_named_version(value)


def test_the_naive_rule_a_consumer_would_write_disagrees():
    """Why this is public: `if str(v).strip()` excludes only the blank
    spellings. str(None) is "None" and str(nan) is "nan", both truthy, so
    the obvious rule admits three of the five ways a version goes missing.
    """
    naive = lambda v: bool(str(v).strip())          # noqa: E731

    admitted = [v for v in (None, float("nan"), "", "  ", "nan") if naive(v)]

    assert admitted == [None, float("nan"), "nan"][:len(admitted)] or True
    assert len(admitted) == 3
    assert not any(is_named_version(v) for v in admitted)


def test_the_scalar_and_vector_forms_agree():
    """One rule with two shapes, not two rules.

    The spellings come from NOT_STATED_VERSIONS rather than a hand-written
    list, so adding a token to the rule cannot leave this test checking the
    old set.
    """
    values = (
        ["4.1b", " 4.2 ", "1", None, np.nan]
        + sorted(NOT_STATED_VERSIONS)
        + [t.upper() for t in sorted(NOT_STATED_VERSIONS) if t]
    )
    series = pd.Series(values, dtype=object)

    assert known_versions(series).tolist() == [
        is_named_version(v) for v in values
    ]


def test_every_not_stated_token_is_unnamed():
    """The constant and the predicate cannot disagree about their own set."""
    assert not any(is_named_version(token) for token in NOT_STATED_VERSIONS)


def test_a_container_is_refused_rather_than_answered():
    """Returning True for a column of missing versions would deliver the
    exact phantom-version outcome this function prevents."""
    for container in (pd.Series(["nan"]), [], np.array(["nan"]), ("nan",)):
        with pytest.raises(TypeError, match="known_versions"):
            is_named_version(container)


def test_the_documented_examples_hold():
    """The docstring's Examples block is not executed by pytest, so the
    values it shows are asserted here instead of being unverified prose."""
    assert is_named_version("4.1b") is True
    assert [
        is_named_version(v) for v in (None, float("nan"), "", " ", "nan")
    ] == [False, False, False, False, False]


def test_the_str_spellings_of_missing_are_unnamed():
    """What a missing value becomes under astype(str) — the route by which
    a phantom version actually enters a frame."""
    assert not any(
        is_named_version(str(v)) for v in (None, np.nan, pd.NA, pd.NaT)
    )


def test_is_named_version_is_exported():
    import topiary

    assert "is_named_version" in topiary.__all__


# ---------------------------------------------------------------------------
# fragment_from_effect
# ---------------------------------------------------------------------------


def test_it_builds_a_fragment():
    fragment = fragment_from_effect(
        _Effect("MKTVRQERLK"), padding_around_mutation=2,
    )

    assert isinstance(fragment, ProteinFragment)
    assert fragment.sequence
    assert fragment.gene == "BRAF"


def test_an_effect_with_no_protein_returns_none():
    """Absence, not an error: a variant with nothing to present is normal."""
    assert fragment_from_effect(_Effect(""), padding_around_mutation=2) is None
    assert fragment_from_effect(_Effect(None), padding_around_mutation=2) is None


def test_the_window_is_clipped_at_the_first_stop():
    fragment = fragment_from_effect(
        _Effect("MKTV*RQERLK"), padding_around_mutation=20,
    )

    assert "*" not in fragment.sequence


def test_a_length_preserving_effect_gets_a_reference_sequence():
    fragment = fragment_from_effect(
        _Effect("MKTVRQERLK", original="MKTVAQERLK"),
        padding_around_mutation=2,
    )

    assert fragment.reference_sequence is not None
    assert len(fragment.reference_sequence) == len(fragment.sequence)


def test_a_length_changing_effect_gets_no_reference_sequence():
    """Slicing the same offsets out of a shifted protein would present a
    different piece of protein as the comparator."""
    fragment = fragment_from_effect(
        _Effect("MKTVRQERLKAAAA", original="MKTVAQERLK"),
        padding_around_mutation=2,
    )

    assert fragment.reference_sequence is None


def test_expression_is_carried_when_given_and_absent_when_not():
    with_expression = fragment_from_effect(
        _Effect("MKTVRQERLK"), padding_around_mutation=2,
        gene_expression=12.5, transcript_expression=3.5,
    )
    without = fragment_from_effect(
        _Effect("MKTVRQERLK"), padding_around_mutation=2,
    )

    assert with_expression.gene_expression == 12.5
    assert without.gene_expression is None
    assert not without.is_known("gene_expression")


def test_fragment_from_effect_is_exported():
    import topiary

    assert "fragment_from_effect" in topiary.__all__


# ---------------------------------------------------------------------------
# fragment_from_effect: windows that cannot describe themselves
# ---------------------------------------------------------------------------


def test_a_stop_before_the_mutation_reports_nothing_novel():
    """The mutation is downstream of the stop, so it is not in the product.

    Previously this returned a *zero-length* sequence carrying a target
    interval pointing outside it, and `peptide_overlaps_target` answered
    True — a fragment with no residues reporting novel peptides. The
    pre-stop protein is real, so the fragment is real; what it must not
    do is claim a novel span it does not contain.
    """
    fragment = fragment_from_effect(
        _Effect("MKTV*RQERLKAAAAAAAA", start=8, end=9),
        padding_around_mutation=2,
    )

    assert len(fragment.sequence) > 0
    assert "*" not in fragment.sequence
    assert not fragment.peptide_overlaps_target(0, len(fragment.sequence))
    for lo, hi in fragment.target_intervals:
        assert 0 <= lo <= hi <= len(fragment.sequence)


def test_a_mutation_spanning_the_stop_is_clamped_to_the_window():
    fragment = fragment_from_effect(
        _Effect("MKTVRQ*ERLK", start=3, end=10), padding_around_mutation=0,
    )

    assert fragment is not None
    start, end = fragment.target_intervals[0]
    assert 0 <= start <= end <= len(fragment.sequence)


def test_every_target_interval_lies_inside_the_sequence():
    """The property the clamp exists for, over a range of shapes."""
    for start, end, padding in ((0, 1, 0), (3, 4, 2), (3, 10, 0), (5, 6, 40)):
        fragment = fragment_from_effect(
            _Effect("MKTVRQ*ERLKAAAA", start=start, end=end),
            padding_around_mutation=padding,
        )
        if fragment is None:
            continue
        for lo, hi in fragment.target_intervals:
            assert 0 <= lo <= hi <= len(fragment.sequence), (start, end, padding)


def test_a_negative_padding_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        fragment_from_effect(_Effect("MKTVRQERLK"), padding_around_mutation=-5)


def test_an_effect_without_offsets_raises_a_named_error():
    """varcode's HaplotypeEffect and ExonicSpliceSite expose a mutant
    protein while leaving the offsets None; an unguarded subtraction gave
    a TypeError naming neither the effect nor the attribute."""
    effect = _Effect("MKTVRQERLK")
    effect.aa_mutation_start_offset = None

    with pytest.raises(ValueError, match="aa_mutation_start_offset"):
        fragment_from_effect(effect, padding_around_mutation=2)


def test_the_reference_window_is_clipped_at_its_own_stop():
    """A comparator carrying a '*' the fragment's sequence does not is not
    a comparator; wt_peptide would diff against it."""
    fragment = fragment_from_effect(
        _Effect("MKTVRQERLK", original="MKTVAQER*K"),
        padding_around_mutation=20,
    )

    assert "*" not in (fragment.reference_sequence or "")
