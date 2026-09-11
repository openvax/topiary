"""``serum_half_life`` / ``blood_half_life`` reach the DSL (topiary #300).

Both kinds are declared allele-free in :data:`KIND_MHC_DEPENDENCE` --
degradation of a free peptide has no MHC context -- so a prediction for
them lands in a group of its own, the same shape :mod:`topiary.ranking`
already gives ``antigen_processing`` (see ``tests/test_allele_free_kinds.py``).
Reaching them at all first requires the parser to recognize the names,
which depends on :data:`KIND_ALIASES` carrying them; #300 found the two
disagreeing on the reporter's environment. With the mhctools floor this
project already declares (``>=3.39.0``, which is where mhctools' own
``Kind`` gained these two constants), the two registries agree and every
form below parses -- these tests exist so a future regression here is a
CI failure, not a field report.

Also pins the parser's now-specific diagnostic for a name that still
doesn't resolve: a bracket after an unrecognized identifier used to read
as "Cannot use ['...'] on Column", naming the bracket instead of the
kind that was never registered.
"""

import warnings

import pandas as pd
import pytest

from topiary import evaluate_scores
from topiary.ranking import EvalContext, parse

GROUP_KEYS = ["prediction_id", "peptide", "peptide_offset", "allele"]
PATIENT_ALLELES = ["HLA-A*02:01", "HLA-B*07:02"]


def _row(allele, kind, value=None, score=0.0, peptide="SIINFEKL"):
    return {
        "prediction_id": "p1", "source_sequence_name": "ctx",
        "peptide": peptide, "peptide_offset": 2, "peptide_length": 8,
        "allele": allele, "n_flank": "", "c_flank": "",
        "prediction_method_name": "plifepred2", "predictor_version": "1.0",
        "kind": kind, "value": value, "affinity": None,
        "percentile_rank": None, "score": score,
    }


def _affinity_and_half_life(kind, half_life_score=4.2):
    """One peptide: two per-allele affinity rows, one allele-free half-life row.

    The shape ``tests/test_allele_free_kinds.py`` already established for
    ``antigen_processing`` -- reused here for the two kinds #300 is about,
    so the same projection machinery is what's under test, not a new one.
    """
    return pd.DataFrame([
        _row(PATIENT_ALLELES[0], "pMHC_affinity", 50.0),
        _row(PATIENT_ALLELES[1], "pMHC_affinity", 60.0),
        _row("", kind, half_life_score, score=half_life_score),
    ])


@pytest.mark.parametrize("kind", ["serum_half_life", "blood_half_life"])
class TestHalfLifeReachesTheDSL:
    def test_bracketed_method_form_parses(self, kind):
        parse(f"{kind}[plifepred2].score")

    def test_dotted_value_form_parses(self, kind):
        parse(f"{kind}.value")

    def test_bare_rank_field_parses(self, kind):
        parse(f"{kind}.rank")

    def test_peptide_view_wrapped_form_reaches_every_allele(self, kind):
        """The explicit form: no warning, one score per group."""
        df = _affinity_and_half_life(kind)
        ctx = EvalContext(df, group_keys=GROUP_KEYS)
        node = parse(f"peptide_view({kind}[plifepred2].score)")

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scores = node.eval(ctx).reindex(ctx.group_index)

        assert scores.tolist() == [4.2, 4.2, 4.2]

    def test_unwrapped_form_still_projects_with_a_warning(self, kind):
        """A config string written before ``peptide_view`` existed still
        works, the same backward-compatible path #186 gives every other
        allele-free kind."""
        df = _affinity_and_half_life(kind)
        ctx = EvalContext(df, group_keys=GROUP_KEYS)

        with pytest.warns(UserWarning, match="carries no allele"):
            scores = parse(f"{kind}[plifepred2].score").eval(ctx)

        assert scores.reindex(ctx.group_index).tolist() == [4.2, 4.2, 4.2]

    def test_evaluate_scores_reaches_every_patient_allele(self, kind):
        """The exact claim the two blocked vaxrank tests make (#300):
        every allele-free kind reaches every patient allele."""
        df = _affinity_and_half_life(kind)
        node = parse(f"peptide_view({kind}[plifepred2].score)")

        scores = evaluate_scores(df, node, group_keys=GROUP_KEYS)

        assert scores.tolist() == [4.2, 4.2, 4.2]

    def test_kind_reachable_without_a_bracket_too(self, kind):
        """Not only the bracket form the original report used."""
        df = _affinity_and_half_life(kind)
        node = parse(f"peptide_view({kind}.score)")

        scores = evaluate_scores(df, node, group_keys=GROUP_KEYS)

        assert scores.tolist() == [4.2, 4.2, 4.2]


# ---------------------------------------------------------------------------
# A name that still doesn't resolve names itself, not the bracket
# ---------------------------------------------------------------------------


def test_unregistered_kind_before_a_bracket_names_the_kind():
    with pytest.raises(ValueError, match="Unknown prediction kind 'not_a_real_kind'"):
        parse("not_a_real_kind[plifepred2].score")


def test_unregistered_kind_error_lists_available_kinds():
    """The registered-kind list this environment actually has, not a
    fixed two names -- an older mhctools genuinely lacks the half-life
    entries (that gap is what #300 traced back to), so this checks the
    diagnostic mechanism, not this environment's exact kind set."""
    from topiary import KIND_ALIASES

    with pytest.raises(ValueError) as excinfo:
        parse("not_a_real_kind[plifepred2]")

    message = str(excinfo.value)
    assert "affinity" in message
    for spelling in sorted(KIND_ALIASES):
        assert spelling in message


def test_explicit_column_bracket_error_is_unchanged():
    """The one legitimate case that still reads as a Column subscript --
    an explicit ``column(...)`` call is not an identifier that could have
    been a kind, so it keeps the original generic message."""
    with pytest.raises(ValueError, match=r"Cannot use \['\.\.\.'\] on Column"):
        parse("column(peptide)['x']")
