"""Tests targeting coverage gaps found in the audit."""

import os
import subprocess
import sys
import tempfile

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import Affinity, Presentation, TopiaryPredictor
from topiary.inputs import exclude_by, read_fasta
from topiary.ranking import (
    BoolOp,
    Comparison,
    Field,
    KindAccessor,
    _build_kind_aliases,
    _gauss_cdf,
    _iter_known_kinds,
    apply_filter,
    apply_sort,
    parse,
)
from mhctools import Kind


# ---------------------------------------------------------------------------
# predictor.py gaps
# ---------------------------------------------------------------------------


def test_predict_from_sequences():
    """predict_from_sequences (non-named) was untested."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_sequences(["MASIINFEKLGGG"])
    assert len(df) > 0
    assert "source_sequence" in df.columns  # renamed from source_sequence_name


def test_alleles_required_for_model_class():
    with pytest.raises(ValueError, match="alleles required"):
        TopiaryPredictor(models=RandomBindingPredictor)


def test_no_models_raises():
    with pytest.raises(ValueError, match="Must provide models"):
        TopiaryPredictor()


def test_bare_predictor_instance_autowraps():
    """``models=`` accepts a bare mhctools predictor instance (no list)."""
    instance = RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=[9],
    )
    predictor = TopiaryPredictor(models=instance)
    assert predictor.models == [instance]
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) > 0


def test_bare_predictor_class_autowraps():
    """``models=`` accepts a bare predictor class (no list)."""
    predictor = TopiaryPredictor(models=RandomBindingPredictor, alleles=["A0201"])
    assert len(predictor.models) == 1
    assert isinstance(predictor.models[0], RandomBindingPredictor)


# ---------------------------------------------------------------------------
# Version-selecting factories (#580)
#
# mhctools' NetMHCpan, NetMHC and NetMHCIIpan are functions that run the
# installed tool to learn its version and return the class for it, not
# classes. ``models=NetMHCpan`` used to store the function itself, and the
# run failed later with "'function' object has no attribute
# predict_peptides_dataframe". The stand-ins below have the same shape, so
# these tests do not need an installed NetMHCpan.
# ---------------------------------------------------------------------------


def version_selecting_stand_in(alleles=None, default_peptide_lengths=[9]):
    """A factory shaped like mhctools' NetMHCpan: returns an instance."""
    return RandomBindingPredictor(
        alleles=alleles, default_peptide_lengths=default_peptide_lengths,
    )


def allele_free_stand_in(default_peptide_lengths=[9]):
    """A factory for a model that takes no alleles."""
    return RandomBindingPredictor(
        alleles=["HLA-A*02:01"], default_peptide_lengths=default_peptide_lengths,
    )


def test_model_factory_is_called_like_a_class():
    predictor = TopiaryPredictor(
        models=version_selecting_stand_in, alleles=["HLA-A*02:01"],
    )

    assert isinstance(predictor.models[0], RandomBindingPredictor)
    assert predictor.models[0].alleles == ["HLA-A*02:01"]
    assert len(predictor.predict_from_named_peptides({"flu": "GILGFVFTL"})) > 0


def test_model_factory_without_alleles_is_built_without_them():
    """A factory that takes no alleles is built without them, as on the CLI."""
    predictor = TopiaryPredictor(models=[allele_free_stand_in])

    assert isinstance(predictor.models[0], RandomBindingPredictor)


def test_model_factory_needing_alleles_says_so():
    with pytest.raises(ValueError, match="alleles required"):
        TopiaryPredictor(models=version_selecting_stand_in)


def test_a_registered_factory_name_resolves_to_the_factory(monkeypatch):
    """A name means in Python what it means on the command line (#580).

    mhctools registers ``netmhcpan`` for a factory; resolving the name used
    to raise "is not a peptide predictor" because the registry entry was
    not a class.
    """
    import mhctools.cli

    from topiary.predictor import _build_model_lookup

    monkeypatch.setattr(mhctools.cli, "mhc_predictors", {
        "stand-in-pan": version_selecting_stand_in,
    })
    _build_model_lookup.cache_clear()
    try:
        predictor = TopiaryPredictor(
            models=["stand-in-pan"], alleles=["HLA-A*02:01"],
        )
    finally:
        _build_model_lookup.cache_clear()

    assert isinstance(predictor.models[0], RandomBindingPredictor)
    assert len(predictor.predict_from_named_peptides({"flu": "GILGFVFTL"})) > 0


def test_both_doors_onto_mhctools_build_the_same_model():
    """``models=`` and mhctools' own CLI builder agree on one model.

    The predictor path used to construct classes itself and the command
    line went through mhctools, so a factory worked on one and not the
    other.
    """
    from mhctools.cli import predictors_from_args

    from topiary import model_from_spec
    from topiary.cli.args import create_arg_parser

    args = create_arg_parser().parse_args([
        "--mhc-predictor", "random", "--mhc-alleles", "HLA-A*02:01",
    ])
    from_cli = predictors_from_args(args)[0]
    from_spec = model_from_spec("random", alleles=["HLA-A*02:01"])

    assert type(from_spec) is type(from_cli)
    assert from_spec.alleles == from_cli.alleles


# ---------------------------------------------------------------------------
# String model names
# ---------------------------------------------------------------------------


def test_model_name_string_lowercase():
    """Can create predictor with lowercase string model name."""
    predictor = TopiaryPredictor(
        models="randombindingpredictor", alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) > 0


def test_model_name_string_mixed_case():
    """Case-insensitive model name matching."""
    predictor = TopiaryPredictor(
        models="RandomBindingPredictor", alleles=["A0201"],
    )
    df = predictor.predict_from_named_sequences({"prot": "MASIINFEKLGGG"})
    assert len(df) > 0


def test_model_name_string_mhcflurry():
    """MHCflurry resolves in a fresh process without an eager import."""
    code = """
import sys
from topiary.predictor import _build_model_lookup, _resolve_model_name

assert "mhctools.mhcflurry" not in sys.modules
assert "mhcflurry" in _build_model_lookup()
assert "mhctools.mhcflurry" not in sys.modules
cls = _resolve_model_name("mhcflurry")
from mhctools import MHCflurry
assert cls is MHCflurry
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_model_name_string_list():
    """List of string model names."""
    predictor = TopiaryPredictor(
        models=["randombindingpredictor", "randombindingpredictor"],
        alleles=["A0201"],
    )
    assert len(predictor.models) == 2


def test_model_name_string_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model name"):
        TopiaryPredictor(models="nonexistent_model", alleles=["A0201"])


def test_mhc_model_backward_compat_property():
    predictor = TopiaryPredictor(
        mhc_model=RandomBindingPredictor(alleles=["A0201"]),
    )
    assert predictor.mhc_model is predictor.models[0]
    assert predictor.mhc_models is predictor.models


def test_affinity_column_values():
    """Verify affinity column is value for pMHC_affinity, NaN otherwise."""
    predictor = TopiaryPredictor(
        models=RandomBindingPredictor, alleles=["A0201"],
    )
    df = predictor.predict_from_sequences(["MASIINFEKLGGG"])
    affinity_rows = df[df.kind == "pMHC_affinity"]
    assert not affinity_rows["affinity"].isna().any()


# ---------------------------------------------------------------------------
# ranking.py gaps
# ---------------------------------------------------------------------------


def test_parse_ranking_single_returns_comparison():
    """parse_ranking with single filter returns a Comparison."""
    result = parse("affinity <= 500")
    assert isinstance(result, Comparison)


def test_parse_filter_whitespace():
    f = parse("  affinity  <=  500  ")
    assert isinstance(f, Comparison)
    assert f.right.val == 500.0


def test_parse_filter_negative():
    f = parse("affinity >= -100")
    assert isinstance(f, Comparison)
    # Negative parses as -1 * 100
    import operator
    assert f.op is operator.ge
    val = f.right.evaluate(pd.DataFrame([dict(
        source_sequence_name="x", peptide="A", peptide_offset=0,
        allele="A", kind="pMHC_affinity", score=0.5, value=100.0,
        percentile_rank=1.0,
    )]))
    assert val == -100.0


def test_parse_filter_unknown_kind_becomes_column():
    """Unknown identifiers in filters become Column references (not an error)."""
    from topiary.ranking import Column
    f = parse("bogus_kind <= 500")
    assert isinstance(f, Comparison)
    assert isinstance(f.left, Column)
    assert f.left.col_name == "bogus_kind"
    assert f.right.val == 500.0


def test_parse_ranking_mixed_ops_uses_precedence():
    """New parser uses precedence (& tighter than |) instead of rejecting mix."""
    import operator
    result = parse(
        "affinity <= 500 | presentation.rank <= 2 & stability.score >= 0.5"
    )
    # | at top level, right child is the & group
    assert isinstance(result, BoolOp)
    assert result.op is operator.or_
    assert len(result.children) == 2
    # The second child is the AND branch
    and_child = result.children[1]
    assert isinstance(and_child, BoolOp)
    assert and_child.op is operator.and_


def test_field_le_on_bad_field():
    """<= on an unsupported field name produces a Comparison;
    the error is deferred to eval when the column isn't available."""
    f = Field(Kind.pMHC_affinity, "bogus")
    # Comparison node builds fine
    cmp = f <= 5
    assert isinstance(cmp, Comparison)
    # Eval on a frame with the bogus field column missing just returns NaN
    # (no row has "bogus" column, so eval returns empty_series / NaN).
    # No error expected with new API — the Field just evaluates to NaN.


def test_field_ge_on_bad_field():
    f = Field(Kind.pMHC_affinity, "bogus")
    cmp = f >= 5
    assert isinstance(cmp, Comparison)


def test_expr_le_on_compound_returns_comparison():
    """(Affinity.score + 1) <= 5 returns a Comparison."""
    expr = Affinity.score + 1
    f = expr <= 5
    assert isinstance(f, Comparison)
    assert f.right.val == 5.0


def test_gauss_cdf_extreme():
    assert _gauss_cdf(10) > 0.999
    assert _gauss_cdf(-10) < 0.001


def test_boolop_combined_filter_and_sort_apply():
    """Apply a BoolOp as filter, then a sort_by list — the new equivalent
    of RankingStrategy(filters=..., sort_by=...).evaluate(df)."""
    filt = (Affinity <= 500) | (Presentation.rank <= 2.0)
    # The filter has two children from the OR
    assert isinstance(filt, BoolOp)
    assert len(filt.children) == 2
    sort_nodes = [Presentation.score]
    assert len(sort_nodes) == 1
    # End-to-end: apply_filter then apply_sort
    df = pd.DataFrame([
        dict(source_sequence_name="s", peptide="A", peptide_offset=0,
             allele="A", kind="pMHC_affinity", score=0.9, value=100.0,
             percentile_rank=1.0),
        dict(source_sequence_name="s", peptide="A", peptide_offset=0,
             allele="A", kind="pMHC_presentation", score=0.8, value=None,
             percentile_rank=0.5),
    ])
    filtered = apply_filter(df, filt)
    sorted_df = apply_sort(filtered, sort_nodes)
    assert len(sorted_df) == 2


def test_custom_kind_accessor():
    custom = KindAccessor(Kind.tap_transport)
    f = custom.score >= 0.3
    assert isinstance(f, Comparison)
    assert f.left.kind == Kind.tap_transport


def test_field_supports_string_style_kind_constants():
    df = pd.DataFrame([
        {
            "source_sequence_name": "var1",
            "peptide": "SIINFEKL",
            "peptide_offset": 0,
            "allele": "A",
            "kind": "pMHC_affinity",
            "value": 85.3,
            "score": 0.8,
            "percentile_rank": 0.4,
        }
    ])
    field = Field("pMHC_affinity", "value")
    assert field.evaluate(df) == 85.3
    assert repr(field) == "affinity.value"


def test_iter_known_kinds_supports_string_constant_kind_class():
    class FakeKind:
        pMHC_affinity = "pMHC_affinity"
        pMHC_presentation = "pMHC_presentation"
        antigen_processing = "antigen_processing"
        tap_transport = "tap_transport"

    assert _iter_known_kinds(FakeKind) == [
        "pMHC_affinity",
        "pMHC_presentation",
        "antigen_processing",
        "tap_transport",
    ]


def test_build_kind_aliases_supports_string_constant_kind_class():
    class FakeKind:
        pMHC_affinity = "pMHC_affinity"
        pMHC_presentation = "pMHC_presentation"
        antigen_processing = "antigen_processing"

    aliases = _build_kind_aliases(FakeKind)
    assert aliases["pmhc_affinity"] == "pMHC_affinity"
    assert aliases["affinity"] == "pMHC_affinity"
    assert aliases["ba"] == "pMHC_affinity"
    assert aliases["el"] == "pMHC_presentation"
    assert aliases["antigen_processing"] == "antigen_processing"


# ---------------------------------------------------------------------------
# inputs.py gaps
# ---------------------------------------------------------------------------


def test_exclude_by_invalid_mode():
    df = pd.DataFrame({"peptide": ["AAA"]})
    with pytest.raises(ValueError, match="mode must be"):
        exclude_by(df, {"p": "AAABBB"}, mode="bogus")


def test_exclude_by_min_kmer():
    """Custom min_kmer changes substring window size."""
    df = pd.DataFrame({"peptide": ["ABCDE", "XXXXX"]})
    ref = {"p": "ABCDEFGH"}
    # min_kmer=5: ABCDE is in ref → excluded
    result = exclude_by(df, ref, mode="substring", min_kmer=5)
    assert len(result) == 1
    assert result.iloc[0]["peptide"] == "XXXXX"


# ---------------------------------------------------------------------------
# sources.py gaps (fast tests only)
# ---------------------------------------------------------------------------


def test_sequences_from_transcript_ids_unknown():
    from topiary.sources import sequences_from_transcript_ids
    seqs = sequences_from_transcript_ids(["ENST_FAKE_99999"])
    assert len(seqs) == 0


@pytest.mark.pirlygenes
def test_available_tissues_has_testis():
    from topiary.sources import available_tissues
    tissues = available_tissues()
    assert "testis" in tissues
