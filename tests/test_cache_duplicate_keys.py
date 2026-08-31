"""A cache key with two answers has no defined result (topiary).

`CachedPredictor.concat` already guarded overlapping keys and raised by
default. The plain constructor did not — so the same question had two
answers depending on which door the rows came in, and a lookup on a
duplicated key returned one value with nothing said.

Reachable directly: after blank alleles were collapsed onto one spelling in
5.35.1, a frame carrying two spellings of "no allele" for one peptide became
exactly this case. Found while checking whether that fix could merge two
rows into one key.
"""

import pandas as pd
import pytest

from topiary import CachedPredictor


def _rows(pairs, peptide="SIINFEKLA"):
    return pd.DataFrame([
        dict(peptide=peptide, peptide_length=9, allele=allele,
             kind="proteasome_cleavage", value=value, score=value,
             percentile_rank=1.0, prediction_method_name="netchop",
             predictor_version="1.0")
        for allele, value in pairs
    ])


# ---------------------------------------------------------------------------
# A key with two answers is refused
# ---------------------------------------------------------------------------


def test_two_predictions_for_one_key_are_refused():
    with pytest.raises(ValueError, match="more than once with different"):
        CachedPredictor(_rows([("HLA-A*02:01", 0.1), ("HLA-A*02:01", 0.9)]))


def test_two_spellings_of_no_allele_are_refused_when_they_disagree():
    """The case 5.35.1's collapse made reachable: "None" and "nan" are one
    key now, so two different scores under them are one key with two
    answers."""
    with pytest.raises(ValueError, match="more than once with different"):
        CachedPredictor(_rows([("None", 0.1), ("nan", 0.9)]))


def test_the_error_names_the_offending_key():
    with pytest.raises(ValueError) as excinfo:
        CachedPredictor(_rows([("HLA-A*02:01", 0.1), ("HLA-A*02:01", 0.9)]))

    message = str(excinfo.value)
    assert "SIINFEKLA" in message
    assert "on_overlap" in message      # points at the resolution concat offers


# ---------------------------------------------------------------------------
# What is not a conflict
# ---------------------------------------------------------------------------


def test_identical_duplicates_are_dropped_not_refused():
    """A file listing the same prediction twice is not a contradiction."""
    cache = CachedPredictor(
        _rows([("HLA-A*02:01", 0.5), ("HLA-A*02:01", 0.5)])
    )

    assert len(cache._df) == 1


def test_two_spellings_of_no_allele_agreeing_collapse_quietly():
    cache = CachedPredictor(_rows([("None", 0.5), ("nan", 0.5)]))

    assert len(cache._df) == 1
    assert set(cache._df["allele"]) == {""}


def test_distinct_alleles_are_untouched():
    cache = CachedPredictor(
        _rows([("HLA-A*02:01", 0.1), ("HLA-B*07:02", 0.9)])
    )

    assert len(cache._df) == 2


def test_distinct_peptides_are_untouched():
    cache = pd.concat([
        _rows([("HLA-A*02:01", 0.1)], peptide="SIINFEKLA"),
        _rows([("HLA-A*02:01", 0.9)], peptide="KLQAAMAVL"),
    ], ignore_index=True)

    assert len(CachedPredictor(cache)._df) == 2


def test_a_single_row_cache_is_fine():
    assert len(CachedPredictor(_rows([("HLA-A*02:01", 0.5)]))._df) == 1


# ---------------------------------------------------------------------------
# The two doors now agree
# ---------------------------------------------------------------------------


def test_the_constructor_and_concat_agree_about_a_conflicting_key():
    """concat raised and the constructor silently picked; both raise now."""
    left = _rows([("HLA-A*02:01", 0.1)])
    right = _rows([("HLA-A*02:01", 0.9)])

    with pytest.raises(ValueError):
        CachedPredictor.concat([CachedPredictor(left), CachedPredictor(right)])
    with pytest.raises(ValueError):
        CachedPredictor(pd.concat([left, right], ignore_index=True))


def test_concat_can_still_be_told_which_wins():
    left = _rows([("HLA-A*02:01", 0.1)])
    right = _rows([("HLA-A*02:01", 0.9)])

    merged = CachedPredictor.concat(
        [CachedPredictor(left), CachedPredictor(right)], on_overlap="last",
    )

    assert merged._df["score"].tolist() == [0.9]


# ---------------------------------------------------------------------------
# Each guard owns its own case
# ---------------------------------------------------------------------------


def test_a_multi_model_cache_gets_the_multi_model_error():
    """A cache spanning two models also has two rows per key.

    `_unique_version_pair` already owns that case and names it precisely.
    Running the duplicate check first would answer "duplicate key" to a
    frame whose actual problem is that it holds two models — a worse
    message for a case that has a good one.
    """
    df = pd.DataFrame([
        dict(peptide="SIINFEKLA", peptide_length=9, allele="HLA-A*02:01",
             kind="pMHC_affinity", value=value, score=0.5,
             percentile_rank=1.0, prediction_method_name=method,
             predictor_version="1.0")
        for method, value in (("netmhcpan", 75.0), ("mhcflurry", 120.0))
    ])

    with pytest.raises(ValueError, match="span multiple"):
        CachedPredictor(df)


def test_a_single_model_cache_still_reaches_the_duplicate_check():
    """The ordering must not shadow the check it was reordered around."""
    with pytest.raises(ValueError, match="more than once with different"):
        CachedPredictor(_rows([("HLA-A*02:01", 0.1), ("HLA-A*02:01", 0.9)]))
