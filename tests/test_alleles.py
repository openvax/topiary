import pandas as pd
import pytest
from mhcgnomes.errors import ParseError

from topiary.alleles import canonical_allele_name, canonical_allele_names_or_raw


def test_canonical_allele_name_normalizes_spellings():
    assert canonical_allele_name("HLA-A0201") == "HLA-A*02:01"
    assert canonical_allele_name("HLA-A*02:01") == "HLA-A*02:01"


def test_canonical_allele_name_raises_on_unparseable():
    with pytest.raises(ParseError):
        canonical_allele_name("not an allele")


def test_names_or_raw_passes_through_unparseable_and_non_strings():
    values = ["HLA-A0201", "not an allele", "", None, 7, "HLA-A0201"]
    result = canonical_allele_names_or_raw(values)
    assert result == ["HLA-A*02:01", "not an allele", "", None, 7, "HLA-A*02:01"]


def test_names_or_raw_accepts_a_series_with_missing_values():
    values = pd.Series(["HLA-A0201", None], dtype=object)
    result = canonical_allele_names_or_raw(values)
    assert result[0] == "HLA-A*02:01"
    assert pd.isna(result[1])


def test_both_are_exported_from_the_package():
    import topiary

    assert topiary.canonical_allele_name is canonical_allele_name
    assert topiary.canonical_allele_names_or_raw is canonical_allele_names_or_raw
    assert {"canonical_allele_name", "canonical_allele_names_or_raw"} <= set(topiary.__all__)


def test_lenient_and_strict_agree_on_parseable_alleles():
    # Two doors, one answer (AGENTS.md): the lenient reader path must give
    # the same name as the strict path wherever the strict path succeeds.
    alleles = ["HLA-A0201", "HLA-B*07:02", "H-2-Kb", "HLA-DRB1*01:01"]
    assert canonical_allele_names_or_raw(alleles) == [canonical_allele_name(a) for a in alleles]
