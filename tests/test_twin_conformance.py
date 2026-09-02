"""Functions that answer the same question must answer it the same way.

The defect class this exists for, three instances of it in one release
cycle:

- ``attach_rna_evidence`` let pandas align a misaligned Series, producing
  an all-null column; ``attach_dna_evidence`` assigned positionally,
  producing a misaligned one. Both silent, in opposite directions.
- pVACseq's aggregated and all-epitopes branches attached read evidence
  under different vocabularies.
- ``CachedPredictor.concat`` raises on any duplicate key while the
  constructor accepts contradictory rows silently (topiary#231).

None of these were caught by tests, because every test exercised one
door. A test that exercises one door can never see a divergence; only a
test that drives both through the same battery can.

**Adding a twin here is the point.** When you write a function that
answers a question an existing function already answers -- an assay
variant, a format branch, a second constructor -- add the pair to
``TWINS`` rather than trusting review to notice the next divergence.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import pandas as pd
import pytest

from topiary.evidence import attach_dna_evidence, attach_rna_evidence


@dataclass(frozen=True)
class Twin:
    """Two callables that must behave identically on the args they share.

    Parameters
    ----------
    name : str
        Shown in test ids.
    left, right : callable
        Both take a DataFrame first and the shared arguments by keyword.
    shared : dict
        Left's argument name -> right's name for the same quantity. Names
        differ legitimately (``overlapping`` vs ``depth``); what must not
        differ is what they do with the value.
    """

    name: str
    left: Callable
    right: Callable
    shared: Dict[str, str] = field(default_factory=dict)

    def calls(self, arg, value):
        """``((fn, kwargs), (fn, kwargs))`` passing *value* as *arg*."""
        return (
            (self.left, {arg: value}),
            (self.right, {self.shared[arg]: value}),
        )


TWINS = (
    Twin(
        name="rna/dna evidence",
        left=attach_rna_evidence,
        right=attach_dna_evidence,
        # RNA calls the coverage argument `overlapping`, DNA calls it
        # `depth`; they are the same quantity per assay.
        shared={"overlapping": "depth", "vaf": "vaf"},
    ),
)

FRAME = pd.DataFrame({"x": [1, 2]}, index=[10, 11])


def _ids(twin):
    return [f"{twin.name}:{arg}" for arg in twin.shared]


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_a_misaligned_series_is_refused_by_both_or_neither(twin):
    """The instance that motivated this file.

    One door aligning while the other assigns positionally loses data
    either way, and the caller is told nothing.
    """
    misaligned = pd.Series([100, 200])  # RangeIndex, frame is [10, 11]

    for arg in twin.shared:
        outcomes = []
        for fn, kwargs in twin.calls(arg, misaligned):
            try:
                fn(FRAME, **kwargs)
                outcomes.append("accepted")
            except Exception as exc:
                outcomes.append(type(exc).__name__)
        assert outcomes[0] == outcomes[1], (
            f"{twin.name}: {arg!r} -> {outcomes[0]}, "
            f"{twin.shared[arg]!r} -> {outcomes[1]}"
        )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_a_wrong_length_sequence_is_refused_by_both_or_neither(twin):
    for arg in twin.shared:
        outcomes = []
        for fn, kwargs in twin.calls(arg, [1, 2, 3]):
            try:
                fn(FRAME, **kwargs)
                outcomes.append("accepted")
            except Exception as exc:
                outcomes.append(type(exc).__name__)
        assert outcomes[0] == outcomes[1], (
            f"{twin.name}: wrong-length {arg!r} -> {outcomes[0]}, "
            f"{twin.shared[arg]!r} -> {outcomes[1]}"
        )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_an_aligned_series_and_a_bare_sequence_agree(twin):
    """A bare sequence has no index to honour, so it is positional.

    Both doors must read it that way, and must agree with the aligned
    Series carrying the same numbers -- otherwise the convenience of
    passing a list quietly means something different per door.
    """
    values = [100, 200]
    aligned = pd.Series(values, index=FRAME.index)

    for twin_arg, other_arg in ((a, twin.shared[a]) for a in twin.shared):
        for fn, arg in ((twin.left, twin_arg), (twin.right, other_arg)):
            from_series = fn(FRAME, **{arg: aligned})
            from_list = fn(FRAME, **{arg: values})
            added = [c for c in from_series.columns if c != "x"]
            for column in added:
                pd.testing.assert_series_equal(
                    from_series[column], from_list[column],
                    check_names=False,
                )


@pytest.mark.parametrize("twin", TWINS, ids=lambda t: t.name)
def test_absent_input_writes_no_column_on_either_side(twin):
    """Omit-not-null, checked as a property of the pair.

    This rule was applied to the DNA side first and reached the RNA side
    two releases later; a pair-level assertion would have failed the day
    they diverged.
    """
    bare_left = twin.left(FRAME)
    bare_right = twin.right(FRAME)
    for out, label in ((bare_left, "left"), (bare_right, "right")):
        nulled = [
            c for c in out.columns
            if c != "x" and not out[c].notna().any()
        ]
        assert not nulled, (
            f"{twin.name} ({label}): wrote all-null columns for absent "
            f"inputs: {nulled}"
        )
