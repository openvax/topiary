"""A spelling of "nothing" must not become an allele (topiary, cache axis).

Found by following a report from a downstream consumer, who hit the same
shape in their own store: one predictor's unversioned evidence split across
four buckets because NaN, "nan", None and "" were four different keys.

`CachedPredictor._normalize` rejected null identity columns for
prediction_method_name, predictor_version and kind — with a comment saying
why — and then applied `astype(str)` to `allele` and `peptide` without the
same guard. The reasoning was written down and not applied one field over.
"""

import numpy as np
import pandas as pd
import pytest

from topiary import CachedPredictor


def _row(allele, peptide="SIINFEKLA", kind="proteasome_cleavage"):
    return dict(
        peptide=peptide, peptide_length=9, allele=allele, kind=kind,
        value=0.9, score=0.9, percentile_rank=1.0,
        prediction_method_name="netchop", predictor_version="1.0",
    )


def _cache(rows):
    return CachedPredictor(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Every spelling of "no allele" is one bucket
# ---------------------------------------------------------------------------


def test_the_spellings_of_a_missing_allele_collapse():
    """None, NaN and "" are one key, not three.

    The cache keys on (peptide, allele, peptide_length, kind), so three
    spellings meant one predictor's allele-free evidence sat in three
    buckets while still looking present in the store.
    """
    cache = _cache([
        _row(None, "AAAAAAAAA"), _row(np.nan, "CCCCCCCCC"), _row("", "DDDDDDDDD"),
    ])

    assert set(cache._df["allele"]) == {""}


def test_an_allele_free_row_is_still_accepted():
    """Rejecting it would break caching for proteasome_cleavage and friends."""
    cache = _cache([_row(None)])

    assert len(cache._df) == 1


def test_a_real_allele_is_untouched():
    cache = _cache([_row("HLA-A*02:01")])

    assert set(cache._df["allele"]) == {"HLA-A*02:01"}


# ---------------------------------------------------------------------------
# .alleles is what the cache can answer for
# ---------------------------------------------------------------------------


def test_a_missing_allele_is_not_reported_as_an_allele():
    """It surfaced as the literal string "None" through the mhctools
    protocol, offering callers an allele they cannot predict for."""
    cache = _cache([_row("HLA-A*02:01"), _row(None, "KLQAAMAVL")])

    assert cache.alleles == ["HLA-A*02:01"]


@pytest.mark.parametrize("missing", [None, np.nan, "", "  "],
                         ids=["none", "nan", "empty", "blank"])
def test_no_spelling_of_missing_reaches_the_allele_list(missing):
    cache = _cache([_row("HLA-A*02:01"), _row(missing, "KLQAAMAVL")])

    assert cache.alleles == ["HLA-A*02:01"]


def test_an_all_allele_free_cache_reports_no_alleles():
    cache = _cache([_row(None), _row(np.nan, "KLQAAMAVL")])

    assert cache.alleles == []


# ---------------------------------------------------------------------------
# peptide is never legitimately absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, np.nan, ""],
                         ids=["none", "nan", "empty"])
def test_a_missing_peptide_is_refused(missing):
    """Unlike allele, there is no such thing as a peptide-free prediction;
    astype(str) would have made it the peptide named "None"."""
    with pytest.raises(ValueError, match="'peptide' must be a non-empty"):
        _cache([_row("HLA-A*02:01", peptide=missing)])


def test_the_existing_identity_guards_still_hold():
    for column in ("prediction_method_name", "predictor_version", "kind"):
        rows = [_row("HLA-A*02:01")]
        rows[0][column] = None
        with pytest.raises(ValueError, match=column):
            _cache(rows)
