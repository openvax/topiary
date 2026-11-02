"""Which mhcgnomes results count as one allele, and what each refusal says."""

import logging
import pickle

import mhcgnomes
import numpy as np
import pandas as pd
import pytest

import topiary
from topiary.alleles import (
    AlleleError,
    allele_expression_caveats,
    allele_rejection_reason,
    canonical_allele_name,
    canonical_allele_names_or_raw,
    log_allele_caveat_once,
    parse_allele,
)


@pytest.fixture(autouse=True)
def fresh_caveat_log():
    """Each test sees the once-per-process caveat log from the start."""
    log_allele_caveat_once.cache_clear()
    yield
    log_allele_caveat_once.cache_clear()


# Accepted spellings and their canonical names. Every one of these was
# accepted before this module checked result types, with the same name.
ACCEPTED = [
    ("HLA-A0201", "HLA-A*02:01"),
    ("A0201", "HLA-A*02:01"),
    ("HLA-A*02:01", "HLA-A*02:01"),
    (" HLA-A*02:01 ", "HLA-A*02:01"),
    ("HLA-A*02:01:01:01", "HLA-A*02:01:01:01"),
    ("HLA-A*99:99", "HLA-A*99:99"),  # mhcgnomes does not check existence
    ("HLA-A*02:01:01G", "HLA-A*02:01:01G"),
    ("HLA-A*02:01P", "HLA-A*02:01P"),
    ("HLA-A*24:02:01:02L", "HLA-A*24:02:01:02L"),
    ("HLA-A*32:11Q", "HLA-A*32:11Q"),
    ("HLA-A*02:01 K66A mutant", "HLA-A*02:01 K66A mutant"),
    ("DRB1*01:01", "HLA-DRB1*01:01"),
    ("DQB1*06:02", "HLA-DQB1*06:02"),  # a beta chain alone; no pairing inferred
    ("HLA-DQA1*01:01-DQB1*06:02", "HLA-DQA1*01:01/DQB1*06:02"),
    ("HLA-DPA1*01:03/DPB1*04:01", "HLA-DPA1*01:03/DPB1*04:01"),
    ("H-2-Kb", "H2-K*b"),  # one letter field names the mouse allele
    ("H2-Db", "H2-D*b"),
    ("H2-IAb", "H2-AA*b/AB*b"),
    ("RT1-Aa", "RT1-A*a"),
    ("SLA-1*CDY", "SLA-1*CDY"),
    ("88*001:01", "DLA-88*001:01"),  # only dogs have an 88 gene
    ("Mamu-A1*001:01", "Mamu-A1*001:01"),
    ("BoLA-T2C", "BoLA-T2c"),  # AlleleWithoutGene: a named molecule
]


@pytest.mark.parametrize("value,expected", ACCEPTED)
def test_accepted_spellings_canonicalize(value, expected):
    assert canonical_allele_name(value) == expected


@pytest.mark.parametrize("value,expected", ACCEPTED)
def test_canonical_names_are_fixed_points(value, expected):
    assert canonical_allele_name(expected) == expected


@pytest.mark.parametrize("value,expected", ACCEPTED)
def test_parse_allele_returns_the_mhcgnomes_result(value, expected):
    parsed = parse_allele(value)
    assert isinstance(parsed, (mhcgnomes.Allele, mhcgnomes.Pair, mhcgnomes.AlleleWithoutGene))
    assert parsed.to_string() == expected
    assert allele_rejection_reason(parsed) is None


# Each rejection: the input, the mhcgnomes type it parsed as (None when it
# did not parse) and a pattern its message must match.
REJECTED = [
    (None, None, r"^missing allele: got None$"),
    (float("nan"), None, r"^missing allele: got nan$"),
    (pd.NA, None, r"^missing allele: got <NA>$"),
    ("", None, r"^missing allele: got ''$"),
    ("   ", None, r"^missing allele: got '   '$"),
    ("nan", None, r"^missing allele: got 'nan'$"),
    (7, None, r"^an allele must be a string, not int: 7$"),
    (["HLA-A*02:01"], None, r"^an allele must be a string, not list"),
    ("not an allele", None, r"could not parse 'not an allele' as one MHC allele"),
    ("A*02:01,B*07:02", None, r"could not parse 'A\*02:01,B\*07:02' as one MHC allele.*one allele per value"),
    ("Kb", None, r"could not parse 'Kb'.*read as human.*with its prefix.*species="),
    ("A1*001:01", None, r"could not parse 'A1\*001:01'.*read as human"),
    ("A2", mhcgnomes.Serotype,
     r"^'A2' is a serotype \(HLA-A2\), not an allele; give one of its \d+ alleles, such as HLA-A\*02:01"),
    ("Cw7", mhcgnomes.Serotype, r"^'Cw7' is a serotype \(HLA-Cw7\), not an allele; .*HLA-C\*07"),
    ("A02", mhcgnomes.Supertype, r"^'A02' is a supertype \(HLA A02 supertype\), not an allele"),
    ("H2-b", mhcgnomes.Haplotype, r"^'H2-b' is a haplotype \(H2-b\), not an allele; give one of its"),
    ("SahaI*49/82", mhcgnomes.AmbiguousAlleles, r"is an ambiguous set of alleles"),
    ("HLA-A", mhcgnomes.Gene, r"^'HLA-A' is a gene \(HLA-A\), not an allele; give one specific allele, such as HLA-A\*02:01$"),
    ("HLA-DR", mhcgnomes.Class2Locus, r"^'HLA-DR' is a class II locus \(HLA-DR\), not an allele"),
    ("class I", mhcgnomes.MhcClass, r"^'class I' is an MHC class \(human class I\), not an allele"),
    ("HLA", mhcgnomes.Species, r"^'HLA' is a species \(HLA\), not an allele"),
    ("A*02", mhcgnomes.Allele, r"^'A\*02' is the allele group HLA-A\*02 \(one field\) rather than a specific allele"),
    ("HLA-DRB1*04", mhcgnomes.Allele, r"is the allele group HLA-DRB1\*04 \(one field\)"),
    ("Mamu-A01", mhcgnomes.Allele, r"is the allele group Mamu-A\*01 \(one field\)"),
    ("HLA-A*02:01N", mhcgnomes.Allele, r"^'HLA-A\*02:01N' is the null allele HLA-A\*02:01N, which is not expressed"),
    ("HLA-DQA1*01:01/DQB1*06:02N", mhcgnomes.Pair, r"contains the null allele HLA-DQB1\*06:02N"),
    ("HLA-A*02:01/HLA-A*02:02", mhcgnomes.Pair,
     r"pairs HLA-A\*02:01 with HLA-A\*02:02, which is not a class II alpha/beta heterodimer"),
    ("HLA-DRB1*01:01/DRA*01:01", mhcgnomes.Pair, r"pairs HLA-DRB1\*01:01 with HLA-DRA\*01:01"),
    ("HLA-DRA*01/DRB1*04:01", mhcgnomes.Pair, r"contains the allele group HLA-DRA\*01 \(one field\)"),
]


@pytest.mark.parametrize("value,parsed_type,message", REJECTED, ids=[repr(case[0]) for case in REJECTED])
def test_each_rejection_says_what_and_why(value, parsed_type, message):
    with pytest.raises(AlleleError, match=message) as caught:
        parse_allele(value)
    error = caught.value
    assert isinstance(error, ValueError)
    assert error.value is value
    if parsed_type is None:
        assert error.parsed is None
    else:
        assert type(error.parsed) is parsed_type
        assert allele_rejection_reason(error.parsed) in str(error)


@pytest.mark.parametrize("value,parsed_type,message", REJECTED, ids=[repr(case[0]) for case in REJECTED])
def test_canonical_allele_name_refuses_the_same_inputs(value, parsed_type, message):
    with pytest.raises(AlleleError, match=message):
        canonical_allele_name(value)


def test_an_unparseable_value_chains_the_mhcgnomes_error():
    with pytest.raises(AlleleError) as caught:
        parse_allele("not an allele")
    assert isinstance(caught.value.__cause__, mhcgnomes.ParseError)


def test_allele_error_pickles_without_the_unpicklable_mhcgnomes_result():
    error = AlleleError("'A2' is a serotype", "A2", mhcgnomes.parse("A2"))
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is AlleleError
    assert str(copy) == str(error)
    assert copy.value == "A2"
    assert copy.parsed is None


def test_rejection_reason_examples_come_from_the_serotype_members():
    serotype = mhcgnomes.parse("A2")
    reason = allele_rejection_reason(serotype)
    assert f"one of its {len(serotype.alleles)} alleles" in reason
    assert serotype.alleles[0].to_string() in reason


# Species handling.

@pytest.mark.parametrize("value,species,expected", [
    ("Kb", "H2", "H2-K*b"),
    ("Kb", "mouse", "H2-K*b"),
    ("Kb", "Mus musculus", "H2-K*b"),
    ("H2-Kb", "H2", "H2-K*b"),
    ("A1*001:01", "Mamu", "Mamu-A1*001:01"),
    ("A*02:01", "HLA", "HLA-A*02:01"),
])
def test_species_reads_unprefixed_names(value, species, expected):
    assert canonical_allele_name(value, species=species) == expected


def test_species_rejects_a_name_whose_prefix_says_otherwise():
    with pytest.raises(AlleleError, match=r"of species 'H2'.*does not match expected species 'Mus musculus'"):
        parse_allele("HLA-A*02:01", species="H2")


def test_species_still_rejects_non_alleles():
    with pytest.raises(AlleleError, match=r"is a serotype"):
        parse_allele("A2", species="HLA")


def test_unknown_species_is_a_call_error_not_an_allele_error():
    with pytest.raises(ValueError, match=r"species 'nonsense' is not one species mhcgnomes knows") as caught:
        parse_allele("Kb", species="nonsense")
    assert not isinstance(caught.value, AlleleError)
    with pytest.raises(ValueError, match="nonsense"):
        canonical_allele_names_or_raw(["Kb"], species="nonsense")


def test_no_species_hint_once_a_species_is_given():
    with pytest.raises(AlleleError) as caught:
        parse_allele("not an allele", species="H2")
    assert "read as human" not in str(caught.value)


# Expression annotations.

@pytest.mark.parametrize("value,code", [
    ("HLA-A*24:02:01:02L", "L"),
    ("HLA-A*32:11Q", "Q"),
    ("HLA-A*02:01S", "S"),
    ("HLA-A*02:01C", "C"),
    ("HLA-A*02:01A", "A"),
    ("HLA-A*02:01Ps", "Ps"),
])
def test_expression_caveats_name_the_allele_and_annotation(value, code):
    (caveat,) = allele_expression_caveats(mhcgnomes.parse(value))
    assert caveat.startswith(f"{mhcgnomes.parse(value).to_string()}, annotated {code} (")


def test_group_annotations_and_plain_alleles_have_no_caveats():
    for value in ("HLA-A*02:01", "HLA-A*02:01:01G", "HLA-A*02:01P", "H2-Kb", "BoLA-T2C", "A2"):
        assert allele_expression_caveats(mhcgnomes.parse(value)) == ()


def test_pair_caveats_cover_both_chains():
    caveats = allele_expression_caveats(mhcgnomes.parse("HLA-DQA1*01:01L/DQB1*06:02Q"))
    assert [caveat.split(",")[0] for caveat in caveats] == ["HLA-DQA1*01:01L", "HLA-DQB1*06:02Q"]


def test_an_annotated_allele_is_accepted_with_one_warning_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger="topiary.alleles"):
        for _ in range(3):
            assert canonical_allele_name("HLA-A*24:02:01:02L") == "HLA-A*24:02:01:02L"
    assert len(caplog.records) == 1
    assert "HLA-A*24:02:01:02L, annotated L (low cell-surface expression)" in caplog.text


def test_log_allele_caveat_once_logs_each_caveat_once(caplog):
    with caplog.at_level(logging.WARNING, logger="topiary.alleles"):
        log_allele_caveat_once("X, annotated L (low cell-surface expression)")
        log_allele_caveat_once("X, annotated L (low cell-surface expression)")
        log_allele_caveat_once("Y, annotated Q (questionable expression)")
    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        "Accepting allele X, annotated L (low cell-surface expression)",
        "Accepting allele Y, annotated Q (questionable expression)",
    ]


# The lenient door, for table readers.

def test_names_or_raw_keeps_rejected_values_and_warns_once_per_distinct_value(caplog):
    values = ["HLA-A0201", "A2", "A2", "HLA-A*02:01N", "not an allele", "HLA-A0201"]
    with caplog.at_level(logging.WARNING, logger="topiary.alleles"):
        result = canonical_allele_names_or_raw(values)
    assert result == ["HLA-A*02:01", "A2", "A2", "HLA-A*02:01N", "not an allele", "HLA-A*02:01"]
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 3
    assert all(message.startswith("Allele value kept as given: ") for message in messages)
    assert "'A2' is a serotype (HLA-A2)" in messages[0]
    assert "'HLA-A*02:01N' is the null allele" in messages[1]
    assert "could not parse 'not an allele'" in messages[2]


def test_names_or_raw_passes_missing_and_non_string_values_silently(caplog):
    values = ["", "   ", "nan", None, np.nan, pd.NA, 7]
    with caplog.at_level(logging.WARNING, logger="topiary.alleles"):
        result = canonical_allele_names_or_raw(values)
    assert result[:4] == ["", "   ", "nan", None]
    assert pd.isna(result[4]) and result[5] is pd.NA and result[6] == 7
    assert caplog.records == []


def test_names_or_raw_accepts_a_series_with_missing_values():
    values = pd.Series(["HLA-A0201", None], dtype=object)
    result = canonical_allele_names_or_raw(values)
    assert result[0] == "HLA-A*02:01"
    assert pd.isna(result[1])


def test_names_or_raw_passes_species_through():
    assert canonical_allele_names_or_raw(["Kb", "H2-Db"], species="H2") == ["H2-K*b", "H2-D*b"]


def test_names_or_raw_does_not_absorb_other_errors(monkeypatch):
    def broken(value, species=None):
        raise RuntimeError("not an allele problem")
    monkeypatch.setattr("topiary.alleles.canonical_allele_name", broken)
    with pytest.raises(RuntimeError):
        canonical_allele_names_or_raw(["HLA-A*02:01"])


# Two doors, one answer (AGENTS.md): the lenient path gives the strict
# path's name wherever the strict path accepts, and keeps the input
# wherever it refuses.

@pytest.mark.parametrize("value", [case[0] for case in ACCEPTED] + [
    case[0] for case in REJECTED if isinstance(case[0], str) and case[0].strip() and case[0] != "nan"])
def test_lenient_and_strict_agree(value):
    try:
        expected = canonical_allele_name(value)
    except AlleleError:
        expected = value
    assert canonical_allele_names_or_raw([value]) == [expected]


def test_the_api_is_exported_from_the_package():
    names = {
        "AlleleError", "allele_expression_caveats", "allele_rejection_reason",
        "canonical_allele_name", "canonical_allele_names_or_raw", "parse_allele",
    }
    assert names <= set(topiary.__all__)
    for name in names:
        assert getattr(topiary, name) is getattr(topiary.alleles, name)
