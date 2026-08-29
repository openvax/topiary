"""KIND_MHC_DEPENDENCE: what a prediction kind is about, before any rows.

Downstream has to answer "does this kind describe a peptide-MHC pair, or the
peptide alone?" with no predictor and no rows in hand — on external-input runs
there is no ``kind_support`` at all (topiary #195).

Row inspection cannot answer it. A peptide-level record and an allele-scoped
record that arrived with a blank allele both scan as "allele-free", and
treating the second as the first invents per-allele evidence for alleles no
model scored.
"""

import warnings

import pandas as pd
import pytest

from topiary import KIND_ALIASES, KIND_MHC_DEPENDENCE
from topiary.ranking import EvalContext, evaluate_scores, parse
from topiary.ranking.nodes import (
    _MHC_DEPENDENCE_VALUES,
    _filter_kind_method_version,
    _kind_value,
    _resolve_mhc_dependence,
)


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------


def test_every_known_kind_is_classified():
    """Completeness drift is what a downstream copy of this table can't catch."""
    known = {_kind_value(kind) for kind in KIND_ALIASES.values()}

    assert known == set(KIND_MHC_DEPENDENCE)


def test_values_come_from_the_mhctools_vocabulary():
    assert set(KIND_MHC_DEPENDENCE.values()) <= _MHC_DEPENDENCE_VALUES


def test_pmhc_kinds_are_per_allele():
    """The prefix names a peptide-MHC pair, so it can't be peptide-level."""
    for kind, dependence in KIND_MHC_DEPENDENCE.items():
        if kind.startswith("pMHC_"):
            assert dependence == "single_allele", kind


def test_processing_pathway_kinds_are_peptide_level():
    """Cleavage, transport and trimming happen before or apart from loading."""
    for kind in (
        "antigen_processing", "proteasome_cleavage", "endolysosomal_cleavage",
        "erap_trimming", "tap_transport",
    ):
        assert KIND_MHC_DEPENDENCE[kind] == "none"


def test_the_mapping_is_read_only():
    with pytest.raises(TypeError):
        KIND_MHC_DEPENDENCE["pMHC_affinity"] = "none"


# ---------------------------------------------------------------------------
# Resolution order: kind_support > allele_set > the kind's default
# ---------------------------------------------------------------------------


def _row(kind, allele="HLA-A*02:01", score=0.5, **extra):
    row = dict(source_sequence_name="s", peptide="SIINFEKLA", peptide_offset=0,
               allele=allele, kind=kind, value=None, score=score,
               percentile_rank=1.0, prediction_method_name="mhcflurry")
    row.update(extra)
    return row


def _dependence(df, kind_name, **ctx_kwargs):
    from mhctools import Kind

    kind = getattr(Kind, kind_name)
    ctx = EvalContext(df, **ctx_kwargs)
    sub = _filter_kind_method_version(ctx, kind, None, None)
    return _resolve_mhc_dependence(ctx, kind, sub)


def test_the_kind_decides_when_nothing_else_does():
    df = pd.DataFrame([_row("pMHC_affinity"), _row("antigen_processing", allele="")])

    assert _dependence(df, "pMHC_affinity") == "single_allele"
    assert _dependence(df, "antigen_processing") == "none"


def test_kind_support_still_overrides():
    df = pd.DataFrame([_row("pMHC_presentation")])
    support = {"mhcflurry": {"pMHC_presentation": {"mhc_dependence": "haplotype"}}}

    assert _dependence(df, "pMHC_presentation", kind_support=support) == "haplotype"


def test_an_allele_set_still_overrides():
    df = pd.DataFrame([
        _row("pMHC_presentation", allele_set="HLA-A*02:01,HLA-B*07:02"),
    ])

    assert _dependence(df, "pMHC_presentation") == "haplotype"


# ---------------------------------------------------------------------------
# The hazard: a malformed allele-scoped row must not become peptide-level
# ---------------------------------------------------------------------------


def test_an_allele_scoped_kind_with_no_allele_stays_allele_scoped():
    df = pd.DataFrame([_row("pMHC_affinity", allele="")])

    with pytest.warns(UserWarning, match="carry no allele"):
        assert _dependence(df, "pMHC_affinity") == "single_allele"


def test_an_all_blank_allele_kind_is_not_projected():
    """The shape that actually reproduced the bug.

    Every row of the kind must be blank-allele: with even one real
    allele present, the old row scan saw a counter-example and already
    answered ``single_allele``. Only when the whole kind scanned as
    allele-free did it project, spreading one prediction across alleles
    the model never scored (found in the openvax/vaxrank#348 review, on
    a frame this shape).
    """
    df = pd.DataFrame([
        # The malformed prediction: affinity, no allele, and the only
        # affinity row in the frame.
        _row("pMHC_affinity", allele="", score=0.5, value=7.0),
        _row("pMHC_stability", allele="HLA-A*02:01", score=0.9, value=100.0),
        _row("pMHC_stability", allele="HLA-B*07:02", score=0.8, value=200.0),
    ])

    with pytest.warns(UserWarning, match="carry no allele"):
        scores = evaluate_scores(df, parse("affinity.value"))

    # Only the group the row is actually in sees it; the two real
    # alleles read NaN, which is the truth — no affinity was predicted
    # for either.
    assert scores.tolist()[0] == 7.0
    assert pd.isna(scores.tolist()[1]) and pd.isna(scores.tolist()[2])


def test_a_blank_allele_among_real_ones_is_also_kept_per_allele():
    """The adjacent path: a mixed frame, which the row scan already got right."""
    df = pd.DataFrame([
        _row("pMHC_affinity", allele="HLA-A*02:01", score=0.9),
        _row("pMHC_affinity", allele="HLA-B*07:02", score=0.1),
        _row("pMHC_affinity", allele="", score=0.5),
    ])

    with pytest.warns(UserWarning, match="carry no allele"):
        scores = evaluate_scores(df, parse("affinity.score"))

    assert scores.tolist()[:2] == [0.9, 0.1]


def test_a_genuinely_peptide_level_kind_still_projects():
    df = pd.DataFrame([
        _row("pMHC_affinity", allele="HLA-A*02:01", score=0.9),
        _row("pMHC_affinity", allele="HLA-B*07:02", score=0.1),
        _row("antigen_processing", allele="", score=0.77),
    ])

    with pytest.warns(UserWarning, match="carries no allele"):
        scores = evaluate_scores(df, parse("processing.score"))

    assert scores.tolist() == [0.77, 0.77, 0.77]


def test_no_warning_when_the_alleles_are_there():
    df = pd.DataFrame([_row("pMHC_affinity")])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _dependence(df, "pMHC_affinity") == "single_allele"


def test_an_unknown_kind_still_falls_back_to_the_rows():
    """A kind from a newer mhctools has no entry; read what's there."""
    df = pd.DataFrame([_row("some_future_kind", allele="")])
    ctx = EvalContext(df)
    sub = ctx.df[ctx.df["kind"] == "some_future_kind"]

    assert _resolve_mhc_dependence(ctx, "some_future_kind", sub) == "none"


# ---------------------------------------------------------------------------
# The public resolver — usable with no predictor and no context
# ---------------------------------------------------------------------------


def test_the_kind_alone_answers():
    """External-input runs have no predictor, so this must work bare."""
    from topiary import mhc_dependence

    assert mhc_dependence("antigen_processing") == "none"
    assert mhc_dependence("pMHC_affinity") == "single_allele"


def test_kind_objects_and_strings_both_work():
    from mhctools import Kind

    from topiary import mhc_dependence

    assert mhc_dependence(Kind.pMHC_affinity) == mhc_dependence("pMHC_affinity")


def test_precedence_kind_support_beats_everything():
    from topiary import mhc_dependence

    rows = pd.DataFrame([_row("pMHC_presentation")])
    support = {"mhcflurry": {"pMHC_presentation": {"mhc_dependence": "haplotype"}}}

    assert mhc_dependence("pMHC_presentation", kind_support=support, rows=rows) == (
        "haplotype"
    )


def test_precedence_allele_set_beats_the_table():
    from topiary import mhc_dependence

    rows = pd.DataFrame([
        _row("pMHC_presentation", allele_set="HLA-A*02:01,HLA-B*07:02"),
    ])

    # The table's default for this kind is single_allele; the data says
    # otherwise and the data is more specific.
    assert mhc_dependence("pMHC_presentation", rows=rows) == "haplotype"


def test_precedence_the_table_beats_the_rows():
    """The case row inspection gets wrong."""
    from topiary import mhc_dependence

    rows = pd.DataFrame([_row("pMHC_affinity", allele="")])

    with pytest.warns(UserWarning, match="carry no allele"):
        assert mhc_dependence("pMHC_affinity", rows=rows) == "single_allele"


def test_rows_may_be_a_whole_frame():
    from topiary import mhc_dependence

    rows = pd.DataFrame([
        _row("pMHC_affinity"), _row("antigen_processing", allele=""),
    ])

    assert mhc_dependence("antigen_processing", rows=rows) == "none"
    assert mhc_dependence("pMHC_affinity", rows=rows) == "single_allele"


def test_vocabulary_comes_from_mhctools():
    """A restated copy is exactly what drifts."""
    from mhctools import MHC_DEPENDENCE_VALUES as upstream

    from topiary import MHC_DEPENDENCE_VALUES

    assert MHC_DEPENDENCE_VALUES is upstream


def test_unknown_key_raises_rather_than_defaulting():
    with pytest.raises(KeyError):
        KIND_MHC_DEPENDENCE["not_a_kind"]
