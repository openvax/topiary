"""Tests targeting coverage gaps found in the audit."""

import os
import tempfile

import pandas as pd
import pytest
from mhctools import RandomBindingPredictor

from topiary import Affinity, Presentation, TopiaryPredictor
from topiary.inputs import exclude_by, read_fasta
from topiary.ranking import (
    EpitopeFilter,
    Field,
    KindAccessor,
    RankingStrategy,
    _build_kind_aliases,
    _gauss_cdf,
    _iter_known_kinds,
    apply_ranking_strategy,
    parse_filter,
    parse_ranking,
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
    """MHCflurry can be referenced by string name."""
    predictor = TopiaryPredictor(
        models="mhcflurry", alleles=["A0201"],
    )
    assert len(predictor.models) == 1


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


def test_parse_ranking_single_returns_strategy():
    """parse_ranking with single filter can be used directly."""
    result = parse_ranking("affinity <= 500")
    assert isinstance(result, EpitopeFilter)


def test_parse_filter_whitespace():
    f = parse_filter("  affinity  <=  500  ")
    assert f.max_value == 500.0


def test_parse_filter_negative():
    f = parse_filter("affinity >= -100")
    assert f.min_value == -100.0


def test_parse_filter_unknown_kind_becomes_column():
    """Unknown identifiers in filters become ColumnFilter (not an error)."""
    from topiary.ranking import ColumnFilter
    f = parse_filter("bogus_kind <= 500")
    assert isinstance(f, ColumnFilter)
    assert f.col_name == "bogus_kind"
    assert f.max_value == 500.0


def test_parse_ranking_mixed_ops_error():
    with pytest.raises(ValueError, match="Cannot mix"):
        parse_ranking("affinity <= 500 | presentation.rank <= 2 & stability.score >= 0.5")


def test_field_le_on_bad_field():
    """<= on an unsupported field name should raise."""
    f = Field(Kind.pMHC_affinity, "bogus")
    with pytest.raises(ValueError):
        f <= 5


def test_field_ge_on_bad_field():
    f = Field(Kind.pMHC_affinity, "bogus")
    with pytest.raises(ValueError):
        f >= 5


def test_expr_le_on_compound_returns_expr_filter():
    """(Affinity.score + 1) <= 5 returns an ExprFilter."""
    from topiary.ranking import ExprFilter
    expr = Affinity.score + 1
    f = expr <= 5
    assert isinstance(f, ExprFilter)
    assert f.max_value == 5.0


def test_gauss_cdf_extreme():
    assert _gauss_cdf(10) > 0.999
    assert _gauss_cdf(-10) < 0.001


def test_ranking_strategy_sort_by_preserves_filters():
    strategy = (Affinity <= 500) | (Presentation.rank <= 2.0)
    sorted_strategy = strategy.sort_by(Presentation.score)
    assert len(sorted_strategy.filters) == 2
    assert len(sorted_strategy.sort_by) == 1


def test_custom_kind_accessor():
    custom = KindAccessor(Kind.tap_transport)
    f = custom.score >= 0.3
    assert f.kind == Kind.tap_transport


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


def test_available_tissues_has_testis():
    pytest.importorskip("pirlygenes")
    from topiary.sources import available_tissues
    tissues = available_tissues()
    assert "testis" in tissues
