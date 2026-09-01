"""A peptide-level row that names an allele is not broadcast (topiary #232).

Peptide-level evidence — cleavage, TAP, ERAP, integrated processing — usually
carries no allele, and then there is exactly one thing a reference to it can
mean: this peptide's value, for every allele. So topiary projects it.

But a producer that writes such a row *onto one allele* is saying something
narrower, and projecting it anyway credits a score to alleles the row
explicitly did not name. That is a value attributed to something that did not
state it — the same family as the stringified-null group keys, one level up.

It also blocks the mechanism a consumer needs for allele attribution
(openvax/vaxrank#349): writing the row onto an allele is the natural way to say
"credit this here only", and it was being discarded, so every attribution
policy produced identical per-allele scores.
"""

import warnings

import pandas as pd
import pytest

from topiary import EvalContext, evaluate_scores, is_stated, peptide_view
from topiary.ranking import parse

ALLELES = ("HLA-A*02:01", "HLA-B*07:02")


def _row(kind, allele, score, method="mhcflurry", allele_set=""):
    return dict(
        source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
        allele=allele, allele_set=allele_set, kind=kind, value=score,
        score=score, percentile_rank=1.0, prediction_method_name=method,
        predictor_version="1",
    )


def _affinity_rows():
    return [
        _row("pMHC_affinity", allele, value, method="netmhcpan")
        for allele, value in zip(ALLELES, (50.0, 900.0))
    ]


def _by_allele(series):
    """Scores keyed by allele, for groups that name one.

    A blank-allele row forms its own group (deliberate, since 5.35.0 — a
    blank cell is stated-but-empty, not a stringified null), so
    "by allele" has to mean groups that actually name an allele. Using
    `pd.notna` here would let that group in and compare it against real
    alleles.
    """
    return {
        key[3]: (None if pd.isna(value) else round(float(value), 3))
        for key, value in series.items()
        if is_stated(key[3])
    }


def _eval(rows, expression="antigen_processing['mhcflurry'].score", **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return parse(expression).eval(
            EvalContext(pd.DataFrame(_affinity_rows() + rows), **kwargs)
        )


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------


def test_a_named_allele_is_not_projected_onto_the_others():
    scores = _by_allele(_eval([
        _row("antigen_processing", "HLA-A*02:01", 0.8),
    ]))

    assert scores["HLA-A*02:01"] == 0.8
    assert scores["HLA-B*07:02"] is None


def test_an_allele_free_row_still_broadcasts():
    """The case projection exists for, unchanged."""
    scores = _by_allele(_eval([_row("antigen_processing", None, 0.8)]))

    assert scores == {"HLA-A*02:01": 0.8, "HLA-B*07:02": 0.8}


def test_a_named_row_and_a_free_row_compose():
    """The named row claims its allele; the free row fills the rest."""
    scores = _by_allele(_eval([
        _row("antigen_processing", "HLA-A*02:01", 0.8),
        _row("antigen_processing", None, 0.3),
    ]))

    assert scores == {"HLA-A*02:01": 0.8, "HLA-B*07:02": 0.3}


def test_two_named_rows_each_keep_their_own_allele():
    """Two answers to two questions, not one peptide contradicting itself."""
    scores = _by_allele(_eval([
        _row("antigen_processing", "HLA-A*02:01", 0.9),
        _row("antigen_processing", "HLA-B*07:02", 0.1),
    ]))

    assert scores == {"HLA-A*02:01": 0.9, "HLA-B*07:02": 0.1}


@pytest.mark.parametrize("blank", [None, "", "  ", "nan"],
                         ids=["none", "empty", "blank", "literal-nan"])
def test_every_spelling_of_no_allele_broadcasts(blank):
    """"Names an allele" uses the same is_stated rule as everything else,
    so a frame through astype(str) does not stop broadcasting."""
    scores = _by_allele(_eval([_row("antigen_processing", blank, 0.8)]))

    assert scores == {"HLA-A*02:01": 0.8, "HLA-B*07:02": 0.8}


# ---------------------------------------------------------------------------
# peptide_view says the same thing
# ---------------------------------------------------------------------------


def test_peptide_view_respects_a_named_allele():
    df = pd.DataFrame(_affinity_rows() + [
        _row("antigen_processing", "HLA-A*02:01", 0.8),
    ])

    scores = evaluate_scores(df, peptide_view(parse("antigen_processing.score")))

    assert scores.notna().sum() < len(scores)


def test_peptide_view_still_broadcasts_an_allele_free_row():
    df = pd.DataFrame(_affinity_rows() + [
        _row("antigen_processing", None, 0.8),
    ])

    scores = evaluate_scores(df, peptide_view(parse("antigen_processing.score")))

    assert (scores.dropna() == 0.8).all()


# ---------------------------------------------------------------------------
# haplotype is exempt, deliberately
# ---------------------------------------------------------------------------


def test_a_haplotype_row_still_projects_across_the_genotype():
    """mhctools stamps a genotype-level score with the allele it
    deconvolved as the best presenter. That allele is an artifact of
    reporting, not a restriction — treating it as one would strand a
    joint score on a single allele, which is the failure projection
    exists to prevent."""
    kind_support = {
        "mhcflurry": {
            "pMHC_presentation": {
                "mhc_dependence": "haplotype", "mhc_class": "I",
            },
            "pMHC_affinity": {
                "mhc_dependence": "single_allele", "mhc_class": "I",
            },
        },
    }
    scores = _by_allele(_eval(
        [_row("pMHC_presentation", "HLA-A*02:01", 0.9,
              allele_set="HLA-A*02:01,HLA-B*07:02")],
        expression="presentation.score",
        kind_support=kind_support,
    ))

    assert scores["HLA-A*02:01"] == 0.9
    assert scores["HLA-B*07:02"] == 0.9


# ---------------------------------------------------------------------------
# The conflict guard still guards
# ---------------------------------------------------------------------------


def test_two_allele_free_rows_that_disagree_still_raise():
    """That is a peptide contradicting itself, and still an error."""
    df = pd.DataFrame(_affinity_rows() + [
        _row("antigen_processing", None, 0.9),
        _row("antigen_processing", None, 0.1),
    ])

    with pytest.raises(ValueError, match="several different"):
        evaluate_scores(df, peptide_view(parse("antigen_processing.score")))


def test_two_named_rows_that_disagree_do_not_raise():
    """They disagree about different alleles, which is not a disagreement."""
    df = pd.DataFrame(_affinity_rows() + [
        _row("antigen_processing", "HLA-A*02:01", 0.9),
        _row("antigen_processing", "HLA-B*07:02", 0.1),
    ])

    scores = evaluate_scores(df, peptide_view(parse("antigen_processing.score")))

    assert scores.notna().any()


# ---------------------------------------------------------------------------
# What this unblocks
# ---------------------------------------------------------------------------


def test_an_attribution_policy_can_now_change_the_scores():
    """The mechanism vaxrank#349 needs: writing the row onto chosen
    alleles must change per-allele scores. Every policy previously
    produced identical output, which is why a narrowing knob could
    compute an answer and then fail to apply it."""
    whole_genotype = _by_allele(_eval([
        _row("antigen_processing", None, 0.8),
    ]))
    best_allele_only = _by_allele(_eval([
        _row("antigen_processing", "HLA-A*02:01", 0.8),
    ]))

    assert whole_genotype != best_allele_only
