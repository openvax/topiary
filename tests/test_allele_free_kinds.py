"""Allele-free predictions in an allele-keyed frame (topiary #182, #183).

An antigen-processing prediction carries no allele, so it lands in a group
of its own rather than in any of the peptide's per-allele groups. Two
consequences the DSL has to handle explicitly:

* a filter on an allele-scoped kind has nothing to say about that group,
  and dropping it silently removes evidence a later ``peptide_view()``
  would read (#183);
* a peptide whose *only* evidence is allele-free has no per-allele group
  at all, so a consumer keyed by patient allele can't read it (#182).
"""

import warnings

import pandas as pd
import pytest

from topiary.ranking import (
    EvalContext,
    apply_filter,
    apply_sort,
    evaluate_scores,
    parse,
)

GROUP_KEYS = ["prediction_id", "peptide", "peptide_offset", "allele"]
PATIENT_ALLELES = ["HLA-A*02:01", "HLA-B*07:02"]

PROCESSING = parse("peptide_view(processing[mhcflurry].score)")


def _row(allele, kind, value=None, score=0.0, peptide="SIINFEKL",
         prediction_id="p1"):
    return {
        "prediction_id": prediction_id, "source_sequence_name": "ctx",
        "peptide": peptide, "peptide_offset": 2, "peptide_length": 8,
        "allele": allele, "n_flank": "", "c_flank": "",
        "prediction_method_name": "mhcflurry", "predictor_version": "2.1.1",
        "kind": kind, "value": value, "affinity": value,
        "percentile_rank": None, "score": score,
    }


def _affinity_and_processing(processing_score=0.77):
    return pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0),
        _row("HLA-B*07:02", "pMHC_affinity", 60.0),
        _row("", "antigen_processing", None, processing_score),
    ])


def _scores(df, node=PROCESSING, **kwargs):
    ctx = EvalContext(df, group_keys=GROUP_KEYS, **kwargs)
    return node.eval(ctx).reindex(ctx.group_index).tolist()


# ---------------------------------------------------------------------------
# #183 — a filter on an allele-scoped kind must not drop allele-free evidence
# ---------------------------------------------------------------------------


def test_allele_scoped_filter_keeps_allele_free_evidence():
    df = _affinity_and_processing()

    kept = apply_filter(
        df, parse("affinity.value <= 500"), group_keys=GROUP_KEYS,
    )

    assert sorted(kept["kind"].unique()) == [
        "antigen_processing", "pMHC_affinity",
    ]
    # The whole point: the value is still there to broadcast afterwards.
    assert _scores(kept) == [0.77, 0.77, 0.77]


def test_filter_that_reads_the_allele_free_kind_still_decides_it():
    """Riding along is for predicates that say nothing, not for exemption."""
    df = _affinity_and_processing(processing_score=0.4)

    rejected = apply_filter(
        df, parse("peptide_view(processing[mhcflurry].score) >= 0.9"),
        group_keys=GROUP_KEYS,
    )
    accepted = apply_filter(
        df, parse("peptide_view(processing[mhcflurry].score) >= 0.3"),
        group_keys=GROUP_KEYS,
    )

    assert len(rejected) == 0
    assert "antigen_processing" in set(accepted["kind"])


def test_excluded_peptide_takes_its_allele_free_evidence_with_it():
    df = _affinity_and_processing()

    kept = apply_filter(
        df, parse("affinity.value <= 10"), group_keys=GROUP_KEYS,
    )

    assert len(kept) == 0


def test_evidence_rides_along_only_with_its_own_peptide():
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0, peptide="AAA",
             prediction_id="p1"),
        _row("", "antigen_processing", None, 0.9, peptide="AAA",
             prediction_id="p1"),
        _row("HLA-A*02:01", "pMHC_affinity", 9000.0, peptide="BBB",
             prediction_id="p2"),
        _row("", "antigen_processing", None, 0.2, peptide="BBB",
             prediction_id="p2"),
    ])

    kept = apply_filter(
        df, parse("affinity.value <= 500"), group_keys=GROUP_KEYS,
    )

    assert set(kept["peptide"]) == {"AAA"}
    assert sorted(kept["kind"].unique()) == [
        "antigen_processing", "pMHC_affinity",
    ]


def test_filter_then_score_matches_scoring_the_unfiltered_frame():
    """The pipeline shape that made this silent: filter, then score."""
    df = _affinity_and_processing()

    before = _scores(df)
    after = _scores(
        apply_filter(df, parse("affinity.value <= 500"), group_keys=GROUP_KEYS)
    )

    assert before == after == [0.77, 0.77, 0.77]


def test_allele_free_rows_are_not_kept_when_the_filter_is_boolean_false():
    """A False answer about the peptide is still a False answer."""
    df = _affinity_and_processing()

    kept = apply_filter(
        df, parse("affinity.value <= 500 & affinity.value >= 1000"),
        group_keys=GROUP_KEYS,
    )

    assert len(kept) == 0


# ---------------------------------------------------------------------------
# #182 — declaring the alleles a peptide should be evaluated against
# ---------------------------------------------------------------------------


def test_processing_only_peptide_reaches_the_patient_alleles():
    """No allele-scoped rows at all: the genotype has to come from outside."""
    df = pd.DataFrame([_row("", "antigen_processing", None, 0.77)])

    without = _scores(df)
    with_genotype = EvalContext(
        df, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
    )
    scores = PROCESSING.eval(with_genotype).reindex(with_genotype.group_index)

    # Without the genotype there is one group, and it names no allele.
    assert without == [0.77]
    per_allele = {
        key[-1]: value for key, value in scores.items() if key[-1]
    }
    assert per_allele == {
        "HLA-A*02:01": 0.77, "HLA-B*07:02": 0.77,
    }


def test_declared_alleles_union_with_observed_ones():
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0),
        _row("", "antigen_processing", None, 0.9),
    ])

    ctx = EvalContext(df, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES)

    assert [key[-1] for key in ctx.group_index] == [
        "HLA-A*02:01", "", "HLA-B*07:02",
    ]


def test_declared_groups_have_no_rows_so_allele_scoped_fields_are_nan():
    """An allele with no prediction of its own reads NaN, which is true."""
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0),
        _row("", "antigen_processing", None, 0.9),
    ])

    ctx = EvalContext(df, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES)
    affinity = parse("affinity.value").eval(ctx).reindex(ctx.group_index)

    assert affinity.tolist()[0] == 50.0
    assert pd.isna(affinity.tolist()[2])
    assert PROCESSING.eval(ctx).reindex(ctx.group_index).tolist() == [
        0.9, 0.9, 0.9,
    ]


def test_declared_alleles_do_not_invent_rows():
    """Group keys grow; the frame does not."""
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0),
        _row("", "antigen_processing", None, 0.9),
    ])

    scores = evaluate_scores(
        df, PROCESSING, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
    )
    kept = apply_filter(
        df, parse("affinity.value <= 500"), group_keys=GROUP_KEYS,
        alleles=PATIENT_ALLELES,
    )
    ordered = apply_sort(
        df, [PROCESSING], group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
    )

    assert len(scores) == len(df)
    assert len(kept) == len(df)
    assert len(ordered) == len(df)


def test_alleles_forwarded_by_every_entry_point():
    df = pd.DataFrame([_row("", "antigen_processing", None, 0.77)])

    for call in (
        lambda: apply_filter(
            df, parse("peptide_view(processing[mhcflurry].score) >= 0.5"),
            group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
        ),
        lambda: apply_sort(
            df, [PROCESSING], group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
        ),
        lambda: evaluate_scores(
            df, PROCESSING, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES,
        ),
    ):
        assert len(call()) == 1


# ---------------------------------------------------------------------------
# Rejected declarations
# ---------------------------------------------------------------------------


def test_bare_string_allele_set_is_rejected():
    df = _affinity_and_processing()

    with pytest.raises(ValueError, match="not the string 'HLA-A"):
        EvalContext(df, group_keys=GROUP_KEYS, alleles="HLA-A*02:01")


def test_empty_allele_set_is_rejected():
    df = _affinity_and_processing()

    with pytest.raises(ValueError, match="non-empty sequence"):
        EvalContext(df, group_keys=GROUP_KEYS, alleles=[])


def test_blank_allele_entry_is_rejected():
    """Allele-free is a property of a row, not an allele you can declare."""
    df = _affinity_and_processing()

    with pytest.raises(ValueError, match="must all name an allele"):
        EvalContext(df, group_keys=GROUP_KEYS, alleles=["HLA-A*02:01", ""])


def test_alleles_are_ignored_without_an_allele_group_key():
    df = _affinity_and_processing()

    ctx = EvalContext(
        df, group_keys=["prediction_id", "peptide", "peptide_offset"],
        alleles=PATIENT_ALLELES,
    )

    assert len(ctx.group_index) == 1


# ---------------------------------------------------------------------------
# #186 — an unwrapped allele-free kind means the same thing, and says so
# ---------------------------------------------------------------------------


def test_unwrapped_allele_free_kind_is_projected():
    """The config strings users have already written keep working."""
    df = _affinity_and_processing()
    ctx = EvalContext(df, group_keys=GROUP_KEYS)

    with pytest.warns(UserWarning, match="carries no allele"):
        scores = parse("processing[mhcflurry].score").eval(ctx)

    assert scores.reindex(ctx.group_index).tolist() == [0.77, 0.77, 0.77]


def test_unwrapped_and_wrapped_agree():
    df = _affinity_and_processing()

    with pytest.warns(UserWarning):
        bare = _scores(df, parse("processing[mhcflurry].score"))

    assert bare == _scores(df, PROCESSING)


def test_the_warning_names_the_explicit_form():
    df = _affinity_and_processing()
    ctx = EvalContext(df, group_keys=GROUP_KEYS)

    with pytest.warns(UserWarning) as caught:
        parse("processing[mhcflurry].score").eval(ctx)

    message = str(caught[0].message)
    assert "peptide_view(" in message


def test_explicit_peptide_view_does_not_warn():
    df = _affinity_and_processing()
    ctx = EvalContext(df, group_keys=GROUP_KEYS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert PROCESSING.eval(ctx).reindex(ctx.group_index).tolist() == [
            0.77, 0.77, 0.77,
        ]


def test_per_allele_kinds_are_left_alone():
    """Choosing which allele's row to read stays the caller's decision."""
    df = _affinity_and_processing()
    ctx = EvalContext(df, group_keys=GROUP_KEYS)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        affinity = parse("affinity.value").eval(ctx).reindex(ctx.group_index)

    assert affinity.tolist()[:2] == [50.0, 60.0]
    assert pd.isna(affinity.tolist()[2])


def test_no_projection_without_an_allele_group_key():
    """Nothing to project onto; the plain read is already right."""
    df = _affinity_and_processing()
    keys = ["prediction_id", "peptide", "peptide_offset"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ctx = EvalContext(df, group_keys=keys)
        scores = parse("processing[mhcflurry].score").eval(ctx)

    assert scores.reindex(ctx.group_index).tolist() == [0.77]


def test_unwrapped_reference_works_in_a_filter():
    df = _affinity_and_processing()

    with pytest.warns(UserWarning, match="carries no allele"):
        kept = apply_filter(
            df, parse("processing[mhcflurry].score >= 0.5"),
            group_keys=GROUP_KEYS,
        )

    assert len(kept) == 3


def test_unwrapped_reference_reaches_declared_alleles():
    """Auto-projection composes with a declared genotype."""
    df = pd.DataFrame([_row("", "antigen_processing", None, 0.77)])
    ctx = EvalContext(df, group_keys=GROUP_KEYS, alleles=PATIENT_ALLELES)

    with pytest.warns(UserWarning, match="carries no allele"):
        scores = parse("processing[mhcflurry].score").eval(ctx)

    per_allele = {
        key[-1]: value
        for key, value in scores.reindex(ctx.group_index).items()
        if key[-1]
    }
    assert per_allele == {"HLA-A*02:01": 0.77, "HLA-B*07:02": 0.77}


def test_inconsistent_values_still_raise_when_unwrapped():
    """Auto-projection doesn't soften the one-value-per-peptide rule."""
    df = pd.DataFrame([
        _row("HLA-A*02:01", "pMHC_affinity", 50.0),
        _row("", "antigen_processing", None, 0.30),
        _row("", "antigen_processing", None, 0.90),
    ])
    ctx = EvalContext(df, group_keys=GROUP_KEYS)

    with pytest.warns(UserWarning), pytest.raises(
        ValueError, match="carry several different score",
    ):
        parse("processing[mhcflurry].score").eval(ctx)
