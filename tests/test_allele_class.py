"""derive_mhc_class must parse alleles, not match name prefixes.

AGENTS.md: "Use mhcgnomes for MHC allele parsing. Never startswith("HLA-") or
other string hacks — alleles aren't always human." The implementation did
exactly that, reading HLA-A/B/C as class I and HLA-D* as class II, so every
other real allele came back NA.

NA is not a harmless answer: it drops a row from the `class_i` *and* `class_ii`
filters alike, so the peptide isn't in either view.
"""

import pandas as pd
import pytest

from topiary import Column, apply_filter, derive_mhc_class


def _class_of(allele):
    return derive_mhc_class(pd.Series([allele])).iloc[0]


@pytest.mark.parametrize("allele", [
    "HLA-A*02:01", "HLA-B*07:02", "HLA-C*06:02",
])
def test_classical_human_class_i(allele):
    assert _class_of(allele) == "I"


@pytest.mark.parametrize("allele", ["HLA-E*01:01", "HLA-F*01:01", "HLA-G*01:01"])
def test_non_classical_human_class_i(allele):
    """Ib genes are class I; the prefix test returned NA for all of them."""
    assert _class_of(allele) == "I"


@pytest.mark.parametrize("allele", [
    "HLA-DRB1*01:01", "HLA-DQA1*01:01", "HLA-DPB1*01:01",
])
def test_human_class_ii(allele):
    assert _class_of(allele) == "II"


@pytest.mark.parametrize("allele, mhc_class", [
    ("H2-Kb", "I"), ("H2-Db", "I"), ("H2-IAb", "II"),
    ("BoLA-N*01301", "I"), ("Mamu-A1*001:01", "I"), ("SLA-1*01:01", "I"),
])
def test_non_human_mhc(allele, mhc_class):
    """Alleles aren't always human — the rule this violated."""
    assert _class_of(allele) == mhc_class


@pytest.mark.parametrize("value", ["", "   ", "not-an-allele", None, 42])
def test_what_cannot_be_parsed_is_na(value):
    assert pd.isna(_class_of(value))


def test_the_series_keeps_its_index():
    alleles = pd.Series(["HLA-A*02:01", "H2-Kb"], index=[7, 9])

    assert derive_mhc_class(alleles).index.tolist() == [7, 9]


def test_an_empty_series_is_handled():
    assert derive_mhc_class(pd.Series([], dtype=object)).empty


def test_repeated_alleles_are_parsed_once():
    """Parsing per row would be the obvious slow implementation."""
    import time

    alleles = pd.Series(["HLA-A*02:01", "HLA-B*07:02"] * 20_000)

    start = time.perf_counter()
    derive_mhc_class(alleles)

    assert time.perf_counter() - start < 1.0


def test_the_class_filters_now_see_these_alleles():
    """The consequence: NA dropped the row from both class views."""
    df = pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele=allele, kind="pMHC_affinity", value=50.0, score=0.5,
             percentile_rank=1.0, prediction_method_name="netmhcpan")
        for allele in ("HLA-A*02:01", "HLA-E*01:01", "HLA-DRB1*01:01")
    ])
    df["mhc_class"] = derive_mhc_class(df["allele"])

    class_i = apply_filter(df, Column("mhc_class").eq("I"))
    class_ii = apply_filter(df, Column("mhc_class").eq("II"))

    assert sorted(class_i["allele"]) == ["HLA-A*02:01", "HLA-E*01:01"]
    assert class_ii["allele"].tolist() == ["HLA-DRB1*01:01"]


def test_a_species_without_a_gene_has_no_class():
    """`HLA` parses, but names no gene — so it has no class of its own."""
    assert pd.isna(_class_of("HLA"))


def test_the_class_comes_from_mhcgnomes_predicates():
    """Not from string-matching its Ia / Ib / IIa / IIb labels.

    Collapsing those by prefix would be the same guess this function
    exists to stop making, one layer up — so the predicates are what's
    consulted, and this pins that they agree for every case above.
    """
    import mhcgnomes

    for allele in ("HLA-A*02:01", "HLA-E*01:01", "H2-Kb", "BoLA-N*01301",
                   "HLA-DRB1*01:01", "H2-IAb"):
        parsed = mhcgnomes.parse(allele)
        expected = "I" if parsed.is_class1 else "II" if parsed.is_class2 else pd.NA

        assert _class_of(allele) == expected, allele
