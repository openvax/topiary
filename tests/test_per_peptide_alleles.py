"""Per-peptide declared allele sets (topiary #219, #220).

`alleles=` closed #182 for one genotype, but it declared **one** set for the
whole frame. LENS emits one row per (peptide, allele) that passed its own
threshold, so peptides in a single file arrive reported against different
subsets of a patient's genotype — 2 to 8 distinct sets per file across every
LENS fixture vaxrank has. Passing the union broadcasts each peptide's
allele-free evidence onto alleles it was never reported against.

How this hides is the point: vaxrank swapped in the union and every fixture
came out byte-identical, because their default expression contains an affinity
term, so the invented groups read NaN. An expression reading only peptide-level
evidence gives each invented group a real number. Fixtures being unchanged was
not evidence.
"""

import warnings

import pandas as pd
import pytest

from topiary import (
    EvalContext,
    describe_default_versions,
    peptide_view,
    resolve_default_versions,
)
from topiary.ranking import parse

UNION = ["HLA-A*02:01", "HLA-B*07:02"]
PER_PEPTIDE = {
    "SIINFEKLA": ["HLA-A*02:01"],
    "KLQAAMAVL": ["HLA-B*07:02"],
}
NEVER_PREDICTED = {("SIINFEKLA", "HLA-B*07:02"), ("KLQAAMAVL", "HLA-A*02:01")}


def _frame():
    """Two peptides, each scored against a different allele, plus
    allele-free cleavage evidence for both."""
    rows = []
    for peptide, allele in (
        ("SIINFEKLA", "HLA-A*02:01"), ("KLQAAMAVL", "HLA-B*07:02"),
    ):
        rows.append(dict(
            source_sequence_name="s", peptide=peptide, peptide_offset=0,
            allele=allele, kind="pMHC_affinity", value=100.0, score=0.5,
            percentile_rank=1.0, prediction_method_name="netmhcpan",
            predictor_version="1",
        ))
        rows.append(dict(
            source_sequence_name="s", peptide=peptide, peptide_offset=0,
            allele=None, kind="proteasome_cleavage", value=0.9, score=0.9,
            percentile_rank=1.0, prediction_method_name="netchop",
            predictor_version="1",
        ))
    return pd.DataFrame(rows)


def _scored_pairs(alleles):
    """(peptide, allele) pairs that get a real number from cleavage alone."""
    node = peptide_view(parse("proteasome_cleavage.score"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        values = node.eval(EvalContext(_frame(), alleles=alleles))
    return {
        (key[1], key[3]) for key, value in values.items()
        if pd.notna(value) and pd.notna(key[3])
    }


# ---------------------------------------------------------------------------
# The failure a union hides
# ---------------------------------------------------------------------------


def test_a_union_scores_pairings_that_were_never_predicted():
    """The bug, stated as a test: this is what per-peptide sets prevent."""
    assert _scored_pairs(UNION) & NEVER_PREDICTED == NEVER_PREDICTED


def test_per_peptide_sets_score_only_real_pairings():
    assert not _scored_pairs(PER_PEPTIDE) & NEVER_PREDICTED


def test_per_peptide_sets_still_reach_each_peptide_s_own_alleles():
    """Narrowing must not throw away what the declaration was for."""
    assert _scored_pairs(PER_PEPTIDE) == {
        ("SIINFEKLA", "HLA-A*02:01"), ("KLQAAMAVL", "HLA-B*07:02"),
    }


def test_the_group_index_gains_only_declared_pairs():
    per = EvalContext(_frame(), alleles=PER_PEPTIDE).group_index
    union = EvalContext(_frame(), alleles=UNION).group_index

    assert len(per) == 4
    assert len(union) == 6


# ---------------------------------------------------------------------------
# The three forms
# ---------------------------------------------------------------------------


def test_a_flat_sequence_behaves_as_before():
    """The existing contract is untouched."""
    index = EvalContext(_frame(), alleles=UNION).group_index

    assert ("s", "SIINFEKLA", 0, "HLA-B*07:02") in list(index)


def test_a_mapping_keyed_by_peptide():
    index = list(EvalContext(_frame(), alleles=PER_PEPTIDE).group_index)

    assert ("s", "SIINFEKLA", 0, "HLA-A*02:01") in index
    assert ("s", "SIINFEKLA", 0, "HLA-B*07:02") not in index


def test_a_mapping_keyed_by_the_group_key_tuple():
    mapping = {
        ("s", "SIINFEKLA", 0): ["HLA-A*02:01"],
        ("s", "KLQAAMAVL", 0): ["HLA-B*07:02"],
    }

    index = list(EvalContext(_frame(), alleles=mapping).group_index)

    assert ("s", "SIINFEKLA", 0, "HLA-A*02:01") in index
    assert ("s", "SIINFEKLA", 0, "HLA-B*07:02") not in index


def test_a_callable_receives_the_peptide_keys():
    seen = []

    def alleles_for(keys):
        seen.append(keys)
        return ["HLA-A*02:01"] if keys["peptide"] == "SIINFEKLA" else []

    index = list(EvalContext(_frame(), alleles=alleles_for).group_index)

    assert {k["peptide"] for k in seen} == {"SIINFEKLA", "KLQAAMAVL"}
    assert ("s", "SIINFEKLA", 0, "HLA-A*02:01") in index
    assert ("s", "KLQAAMAVL", 0, "HLA-A*02:01") not in index


def test_an_undeclared_peptide_keeps_only_its_observed_groups():
    """It must not inherit another peptide's genotype."""
    index = list(
        EvalContext(_frame(), alleles={"SIINFEKLA": ["HLA-A*02:01"]}).group_index
    )

    assert ("s", "KLQAAMAVL", 0, "HLA-B*07:02") in index   # observed
    assert ("s", "KLQAAMAVL", 0, "HLA-A*02:01") not in index


# ---------------------------------------------------------------------------
# A declaration that cannot take effect is an error
# ---------------------------------------------------------------------------


def test_a_mapping_key_that_matches_nothing_is_refused():
    """Otherwise a typo is indistinguishable from a deliberate omission."""
    with pytest.raises(ValueError, match="not in the frame"):
        EvalContext(
            _frame(), alleles={"SIINFEKLZ": ["HLA-A*02:01"]},
        ).group_index


def test_a_bare_string_for_one_peptide_is_refused():
    with pytest.raises(ValueError, match="not the string"):
        EvalContext(
            _frame(), alleles={"SIINFEKLA": "HLA-A*02:01"},
        ).group_index


def test_a_peptide_keyed_mapping_needs_peptide_in_the_group_key():
    with pytest.raises(ValueError, match="not one of the group keys"):
        EvalContext(
            _frame(), group_keys=["source_sequence_name", "allele"],
            alleles={"SIINFEKLA": ["HLA-A*02:01"]},
        ).group_index


# ---------------------------------------------------------------------------
# #220: the candidates behind the resolved version
# ---------------------------------------------------------------------------


def _versioned(versions):
    return pd.DataFrame([
        dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
             allele="HLA-A*02:01", kind="pMHC_affinity", value=1.0, score=0.5,
             percentile_rank=1.0, prediction_method_name="netmhcpan",
             predictor_version=v)
        for v in versions
    ])


def test_describe_returns_the_candidates():
    df = _versioned(["4.1b", "4.2"])

    assert describe_default_versions(df) == {
        ("pMHC_affinity", "netmhcpan"): ["4.1b", "4.2"],
    }


def test_describe_and_resolve_agree_on_which_pairs_are_ambiguous():
    """They zip, so a caller can report winner against candidates."""
    df = _versioned(["4.1b", "4.2"])

    assert set(describe_default_versions(df)) == set(resolve_default_versions(df))


def test_the_winner_is_the_last_candidate():
    df = _versioned(["4.1b", "4.2"])
    key = ("pMHC_affinity", "netmhcpan")

    assert describe_default_versions(df)[key][-1] == (
        resolve_default_versions(df)[key]
    )
    assert describe_default_versions(df)[key][0] == (
        resolve_default_versions(df, prefer="oldest")[key]
    )


def test_describe_orders_candidates_by_pep_440():
    assert describe_default_versions(_versioned(["4.10", "4.9"])) == {
        ("pMHC_affinity", "netmhcpan"): ["4.9", "4.10"],
    }


@pytest.mark.parametrize("versions", [
    ["4.2", "nan"], ["4.2", ""], ["4.2"],
], ids=["literal-nan", "blank", "single"])
def test_describe_applies_the_same_is_it_a_version_rule(versions):
    """The rule a consumer would otherwise re-implement, and get wrong."""
    assert describe_default_versions(_versioned(versions)) == {}


def test_describe_handles_a_frame_without_versions():
    df = _versioned(["4.1b", "4.2"]).drop(columns=["predictor_version"])

    assert describe_default_versions(df) == {}
