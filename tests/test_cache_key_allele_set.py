"""The genotype belongs in the cache key (topiary #229).

`_CACHE_COLUMNS` already said why:

    # The genotype a haplotype-mode prediction was scored against; blank
    # for per-allele rows. Without it a cached presentation row reads as
    # a prediction for its deconvolved best allele.

and then `allele_set` was not in `_KEY_COLS`. The reasoning was written down
and not applied one field over — the third instance of that shape this week.

It is exactly the argument the key already makes for `n_flank`/`c_flank`: the
same peptide in two different contexts produces different scores, so the
context is part of the identity.
"""

import numpy as np
import pandas as pd

from topiary import CachedPredictor

GENOTYPE_A = "HLA-A*02:01,HLA-B*07:02"
GENOTYPE_B = "HLA-A*02:01,HLA-B*44:02"


def _row(**kwargs):
    row = dict(
        peptide="SIINFEKL", peptide_length=8, allele="HLA-A*02:01",
        kind="pMHC_presentation", value=None, score=0.5,
        percentile_rank=None, prediction_method_name="mhcflurry",
        predictor_version="2.1",
    )
    row.update(kwargs)
    return row


def _cache(rows):
    return CachedPredictor(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Two genotypes are two predictions
# ---------------------------------------------------------------------------


def test_two_genotypes_sharing_a_best_allele_are_two_entries():
    """MHCflurry haplotype mode reports the deconvolved best allele, so two
    genotypes collide on every other key column. They are two questions."""
    cache = _cache([
        _row(score=0.9, allele_set=GENOTYPE_A),
        _row(score=0.3, allele_set=GENOTYPE_B),
    ])

    assert len(cache._df) == 2
    assert len(cache._index) == 2
    assert sorted(cache._df["score"]) == [0.3, 0.9]


def test_each_genotype_is_addressable():
    cache = _cache([
        _row(score=0.9, allele_set=GENOTYPE_A),
        _row(score=0.3, allele_set=GENOTYPE_B),
    ])

    keys = set(cache._index)
    assert CachedPredictor._row_key_from_values(
        "SIINFEKL", "HLA-A*02:01", 8, "pMHC_presentation",
        None, None, GENOTYPE_A,
    ) in keys
    assert CachedPredictor._row_key_from_values(
        "SIINFEKL", "HLA-A*02:01", 8, "pMHC_presentation",
        None, None, GENOTYPE_B,
    ) in keys


def test_a_per_allele_row_coexists_with_a_genotype_row():
    """Blank genotype is its own entry, the way absent flanks are."""
    cache = _cache([
        _row(score=0.5, allele_set=""),
        _row(score=0.9, allele_set=GENOTYPE_A),
    ])

    assert len(cache._index) == 2


# ---------------------------------------------------------------------------
# One spelling of "no genotype"
# ---------------------------------------------------------------------------
#
# Adding a dimension to a key is a chance to add the absent-vs-empty bug on a
# new axis, which is what #229 warned about. Every spelling of "no genotype"
# has to be the same key.


def test_every_spelling_of_no_genotype_is_one_key():
    keys = {
        CachedPredictor._row_key_from_values(
            "SIINFEKL", "HLA-A*02:01", 8, "k", None, None, spelling,
        )
        for spelling in (None, np.nan, "", "   ", "nan", "None")
    }

    assert len(keys) == 1


def test_a_frame_with_no_genotype_column_still_keys():
    """allele_set is optional; the key shape must be stable without it."""
    cache = _cache([_row(score=0.5)])

    assert len(cache._index) == 1


def test_two_spellings_of_no_genotype_do_not_split_the_cache():
    cache = _cache([
        _row(score=0.5, peptide="SIINFEKL", allele_set=None),
        _row(score=0.7, peptide="KLQAAMAV", allele_set="nan"),
    ])

    assert set(cache._df["allele_set"]) == {""}
    assert len(cache._index) == 2


# ---------------------------------------------------------------------------
# What stays out of the key, and why
# ---------------------------------------------------------------------------
#
# Provenance says where a peptide was found, not what was predicted about it.
# Keying on it would defeat the cache: the same peptide would be re-predicted
# once per source protein.


def test_provenance_does_not_split_a_prediction():
    cache = _cache([
        _row(score=0.5, source_sequence_name="BRAF", peptide_offset=1),
        _row(score=0.5, source_sequence_name="KRAS", peptide_offset=9),
    ])

    assert len(cache._index) == 1


def test_sample_name_does_not_split_a_prediction():
    """Same peptide, same allele, same model — the same prediction."""
    cache = _cache([
        _row(score=0.5, sample_name="S1"),
        _row(score=0.5, sample_name="S2"),
    ])

    assert len(cache._index) == 1


def test_flanks_still_split_a_prediction():
    """The precedent allele_set follows — unchanged."""
    cache = _cache([
        _row(score=0.5, n_flank="AAA", c_flank="CCC"),
        _row(score=0.9, n_flank="GGG", c_flank="TTT"),
    ])

    assert len(cache._index) == 2
