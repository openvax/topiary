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
    ProteinFragment,
    fragment_from_effect,
    is_named_version,
)
from topiary.ranking.nodes import _known_versions


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
        self.variant = type("V", (), {"short_description": "chr7:1A>T"})()

    def short_description(self):
        return "p.V600E"


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
    """Documents why this is public: `if str(v).strip()` gets "nan" wrong."""
    naive = lambda v: bool(str(v).strip())          # noqa: E731

    assert naive("nan") and not is_named_version("nan")


def test_the_scalar_and_vector_forms_agree():
    """One rule with two shapes, not two rules — the defect this repo keeps
    producing is exactly two paths answering one question differently."""
    values = ["4.1b", "", "  ", "nan", "NaN", " 4.2 ", None, np.nan]
    series = pd.Series(values, dtype=object)

    vectorized = _known_versions(series).tolist()
    scalar = [is_named_version(v) for v in values]

    assert vectorized == scalar


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
