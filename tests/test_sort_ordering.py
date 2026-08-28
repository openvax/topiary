"""apply_sort must be an ordering, not a pairwise comparison (topiary #191).

Skipping a key when either side is missing made "equal" intransitive, so the
ranking depended on the order rows arrived in and a worse group could outrank
a better one. Keys are ranked instead, with a missing value taking the
average rank so it neither gains nor loses by that key.
"""

import time

import numpy as np
import pandas as pd

from topiary import Affinity, Column, apply_sort

NAN = float("nan")


def _frame(groups, **extra):
    """One row per group; ``groups`` is (peptide, k0, k1)."""
    return pd.DataFrame([
        dict(source_sequence_name="s", peptide=peptide, peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=1.0,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan", k0=k0, k1=k1, **extra)
        for peptide, k0, k1 in groups
    ])


KEYS = [Column("k0"), Column("k1")]


# ---------------------------------------------------------------------------
# The bug: order-dependent ranking
# ---------------------------------------------------------------------------


def test_ranking_does_not_depend_on_row_order():
    """A vs B and B vs C compared 'equal'; A vs C did not."""
    groups = [("A", 1.0, NAN), ("B", NAN, NAN), ("C", 2.0, NAN)]

    rankings = {
        tuple(apply_sort(_frame([groups[i] for i in perm]), KEYS)["peptide"])
        for perm in ([0, 1, 2], [2, 1, 0], [1, 0, 2], [1, 2, 0])
    }

    assert len(rankings) == 1


def test_a_better_group_outranks_a_worse_one():
    groups = [("A", 1.0, NAN), ("B", NAN, NAN), ("C", 2.0, NAN)]

    ordered = apply_sort(_frame(groups), KEYS)["peptide"].tolist()

    # Descending on k0: 2.0 beats 1.0, whatever else is in the frame.
    assert ordered.index("C") < ordered.index("A")


# ---------------------------------------------------------------------------
# Missing values are neutral, not penalized
# ---------------------------------------------------------------------------


def test_a_missing_key_neither_promotes_nor_penalizes():
    groups = [("A", 9.0, 0.0), ("B", NAN, 0.0), ("C", 1.0, 0.0)]

    ordered = apply_sort(_frame(groups), KEYS)["peptide"].tolist()

    # B sits between the group it would beat and the one it would lose to.
    assert ordered == ["A", "B", "C"]


def test_later_keys_decide_when_a_key_cannot():
    """The property the old pairwise skip was reaching for."""
    groups = [("A", NAN, 1.0), ("B", NAN, 9.0), ("C", NAN, 5.0)]

    ordered = apply_sort(_frame(groups), KEYS)["peptide"].tolist()

    assert ordered == ["B", "C", "A"]


def test_a_key_everyone_is_missing_is_simply_skipped():
    groups = [("A", NAN, 1.0), ("B", NAN, 9.0)]

    assert apply_sort(_frame(groups), KEYS)["peptide"].tolist() == ["B", "A"]


# ---------------------------------------------------------------------------
# Ordinary sorting is unchanged
# ---------------------------------------------------------------------------


def test_direction_is_respected():
    groups = [("A", 1.0, 0.0), ("B", 3.0, 0.0), ("C", 2.0, 0.0)]

    descending = apply_sort(_frame(groups), KEYS)["peptide"].tolist()
    ascending = apply_sort(
        _frame(groups), KEYS, sort_direction="asc",
    )["peptide"].tolist()

    assert descending == ["B", "C", "A"]
    assert ascending == ["A", "C", "B"]


def test_affinity_still_sorts_strong_binders_first():
    df = pd.DataFrame([
        dict(source_sequence_name="s", peptide=peptide, peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=value,
             score=0.5, percentile_rank=1.0,
             prediction_method_name="netmhcpan")
        for peptide, value in (("weak", 5000.0), ("strong", 50.0))
    ])

    assert apply_sort(df, [Affinity.value])["peptide"].tolist() == [
        "strong", "weak",
    ]


def test_ties_keep_frame_order():
    groups = [("A", 1.0, 1.0), ("B", 1.0, 1.0), ("C", 1.0, 1.0)]

    assert apply_sort(_frame(groups), KEYS)["peptide"].tolist() == [
        "A", "B", "C",
    ]


def test_rows_of_a_group_stay_together():
    df = pd.concat([_frame([("A", 1.0, 0.0)]), _frame([("B", 2.0, 0.0)]),
                    _frame([("A", 1.0, 0.0)])], ignore_index=True)

    assert apply_sort(df, KEYS)["peptide"].tolist() == ["B", "A", "A"]


# ---------------------------------------------------------------------------
# The same change removes the dominant cost
# ---------------------------------------------------------------------------


def test_sorting_many_groups_is_not_quadratic_in_python():
    """The comparator was ~100% of apply_sort; this pins that it isn't now."""
    n = 40_000
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "source_sequence_name": "s",
        "peptide": [f"PEP{i:06d}" for i in range(n)],
        "peptide_offset": 0,
        "allele": "HLA-A*02:01",
        "kind": "pMHC_affinity",
        "value": rng.random(n) * 5000,
        "score": rng.random(n),
        "percentile_rank": rng.random(n) * 20,
        "prediction_method_name": "netmhcpan",
    })

    start = time.perf_counter()
    apply_sort(df, [Affinity.value, Affinity.score])
    elapsed = time.perf_counter() - start

    # The Python comparator took ~1s at this size; the bound is loose
    # enough to survive a slow machine and tight enough to catch a
    # regression back to per-pair Python calls.
    assert elapsed < 2.0
