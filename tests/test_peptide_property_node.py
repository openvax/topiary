"""Tests for the PeptideProperty DSL node and its singletons (#95)."""

import math

import numpy as np
import pandas as pd
import pytest

from topiary import (
    Affinity,
    Aromaticity,
    Charge,
    Hydrophobicity,
    MolecularWeight,
    PeptideProperty,
    apply_filter,
    apply_sort,
    parse,
    wt,
)
from topiary.ranking import EvalContext


def _make_df(rows):
    return pd.DataFrame(rows)


def _basic_rows(peptides, alleles=("HLA-A*02:01",)):
    out = []
    for i, p in enumerate(peptides):
        for a in alleles:
            out.append(dict(
                source_sequence_name="seq1",
                peptide=p,
                peptide_offset=i,
                allele=a,
            ))
    return out


# ---------------------------------------------------------------------------
# Compute correctness via .evaluate
# ---------------------------------------------------------------------------


class TestPeptidePropertyCompute:
    def test_charge_basic(self):
        df = _make_df(_basic_rows(["KKRR"]))
        assert Charge.evaluate(df) == pytest.approx(4.0)

    def test_charge_negative(self):
        df = _make_df(_basic_rows(["DDEE"]))
        assert Charge.evaluate(df) == pytest.approx(-4.0)

    def test_aromaticity(self):
        df = _make_df(_basic_rows(["AFWY"]))
        assert Aromaticity.evaluate(df) == 3

    def test_hydrophobicity_mean(self):
        df = _make_df(_basic_rows(["AAAA"]))
        # A = 1.8 on Kyte-Doolittle; mean of four A's is 1.8.
        assert Hydrophobicity.evaluate(df) == pytest.approx(1.8)

    def test_molecular_weight(self):
        df = _make_df(_basic_rows(["A"]))
        # Single residue: just A's residue mass (no peptide bonds).
        assert MolecularWeight.evaluate(df) == pytest.approx(89.09)


# ---------------------------------------------------------------------------
# Composition with arithmetic / abs / clip / comparison
# ---------------------------------------------------------------------------


class TestPeptidePropertyComposition:
    def test_abs_charge(self):
        df = _make_df(_basic_rows(["DDEE"]))
        assert abs(Charge).evaluate(df) == pytest.approx(4.0)

    def test_neg_charge(self):
        df = _make_df(_basic_rows(["KKRR"]))
        assert (-Charge).evaluate(df) == pytest.approx(-4.0)

    def test_clip(self):
        df = _make_df(_basic_rows(["FWYFW"]))  # aromaticity = 5
        assert Aromaticity.clip(lo=0, hi=3).evaluate(df) == pytest.approx(3.0)

    def test_arithmetic_mix(self):
        df = _make_df(_basic_rows(["KKRR"]))
        expr = 0.5 * Aromaticity + 0.1 * abs(Charge)
        # aromaticity = 0, |charge| = 4 → 0 + 0.4 = 0.4
        assert expr.evaluate(df) == pytest.approx(0.4)

    def test_comparison(self):
        df = _make_df(_basic_rows(["KKRR"]))
        cmp_node = Charge >= 0
        result = cmp_node.eval(EvalContext(df))
        assert bool(result.iloc[0]) is True

    def test_filter_application(self):
        df = _make_df(_basic_rows(["KKRR", "DDEE", "AAAA"]))
        kept = apply_filter(df, Charge >= 0)
        assert sorted(kept["peptide"].tolist()) == ["AAAA", "KKRR"]

    def test_sort_by_property(self):
        df = _make_df(_basic_rows(["AFWY", "AAAA", "FFFF"]))
        ordered = apply_sort(df, [Aromaticity])
        # desc by default: FFFF (4), AFWY (3), AAAA (0)
        assert ordered["peptide"].tolist() == ["FFFF", "AFWY", "AAAA"]


# ---------------------------------------------------------------------------
# Parser support
# ---------------------------------------------------------------------------


class TestPeptidePropertyParser:
    def test_parse_atom(self):
        node = parse("charge")
        assert isinstance(node, PeptideProperty)
        assert node.name == "charge"

    def test_parse_comparison(self):
        df = _make_df(_basic_rows(["KKRR", "DDEE"]))
        node = parse("charge >= 0")
        kept = apply_filter(df, node)
        assert kept["peptide"].tolist() == ["KKRR"]

    def test_parse_clip(self):
        df = _make_df(_basic_rows(["FWYFW"]))
        node = parse("aromaticity.clip(0, 3)")
        assert node.evaluate(df) == pytest.approx(3.0)

    def test_parse_composite(self):
        df = _make_df(_basic_rows(["KKRR"]))
        node = parse("0.5 * aromaticity + 0.1 * abs(charge)")
        assert node.evaluate(df) == pytest.approx(0.4)

    def test_parse_all_four_singletons(self):
        for name in ("charge", "aromaticity", "hydrophobicity", "molecular_weight"):
            node = parse(name)
            assert isinstance(node, PeptideProperty)
            assert node.name == name


# ---------------------------------------------------------------------------
# Scoped access (wt., shuffled., self.)
# ---------------------------------------------------------------------------


class TestPeptidePropertyScope:
    def test_wt_charge_python_api(self):
        df = _make_df([
            dict(
                source_sequence_name="seq1", peptide="DDEE", peptide_offset=0,
                allele="HLA-A*02:01", wt_peptide="KKRR",
            ),
        ])
        assert wt.charge.evaluate(df) == pytest.approx(4.0)

    def test_wt_charge_via_parser(self):
        df = _make_df([
            dict(
                source_sequence_name="seq1", peptide="DDEE", peptide_offset=0,
                allele="HLA-A*02:01", wt_peptide="KKRR",
            ),
        ])
        node = parse("wt.charge")
        assert node.evaluate(df) == pytest.approx(4.0)

    def test_wt_missing_column_is_nan(self):
        df = _make_df(_basic_rows(["DDEE"]))  # no wt_peptide column
        result = wt.charge.evaluate(df)
        assert math.isnan(result)

    def test_unknown_scope_attr_raises(self):
        with pytest.raises(AttributeError, match="Unknown kind"):
            wt.not_a_property


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPeptidePropertyEdgeCases:
    def test_empty_df(self):
        df = pd.DataFrame(columns=[
            "source_sequence_name", "peptide", "peptide_offset", "allele",
        ])
        # .evaluate scalar wrapper returns NaN on empty input.
        assert math.isnan(Charge.evaluate(df))

    def test_missing_peptide_column_returns_empty(self):
        df = _make_df([dict(
            source_sequence_name="seq1", peptide_offset=0, allele="HLA-A*02:01",
        )])
        # No peptide column → eval returns NaN-filled empty_series.
        # Need a group_keys override since auto-detection requires `peptide`.
        ctx = EvalContext(df, group_keys=["source_sequence_name", "peptide_offset"])
        result = Charge.eval(ctx)
        assert result.isna().all()

    def test_repr_unscoped(self):
        assert repr(Charge) == "charge"

    def test_repr_scoped(self):
        assert repr(wt.charge) == "wt.charge"

    def test_to_expr_string_roundtrip(self):
        node = parse("0.3 * aromaticity + 0.7 * abs(charge)")
        roundtripped = parse(node.to_expr_string())
        df = _make_df(_basic_rows(["KKRR"]))
        assert node.evaluate(df) == pytest.approx(roundtripped.evaluate(df))

    def test_multiple_peptides_indexed_correctly(self):
        df = _make_df(_basic_rows(["KKRR", "DDEE", "AAAA"]))
        ctx = EvalContext(df)
        result = Charge.eval(ctx)
        # Order is the unique-group-tuple order from MultiIndex.
        peptide_to_value = {
            tup[ctx.group_keys.index("peptide")]: val
            for tup, val in result.items()
        }
        assert peptide_to_value["KKRR"] == pytest.approx(4.0)
        assert peptide_to_value["DDEE"] == pytest.approx(-4.0)
        assert peptide_to_value["AAAA"] == pytest.approx(0.0)

    def test_composes_with_kind_field(self):
        # Make a frame with an affinity row so Affinity.value works,
        # alongside the peptide-intrinsic Charge node.
        df = _make_df([
            dict(
                source_sequence_name="seq1", peptide="KKRR", peptide_offset=0,
                allele="HLA-A*02:01", kind="pMHC_affinity",
                score=0.8, value=120.0, percentile_rank=0.5,
                prediction_method_name="netmhcpan",
            ),
        ])
        expr = 0.5 * Aromaticity + 0.1 * abs(Charge)
        # aromaticity = 0, |charge| = 4 → 0.4
        assert expr.evaluate(df) == pytest.approx(0.4)

        # And a real composite mixing Affinity with property nodes.
        mixed = Affinity.value + 100.0 * Charge
        assert mixed.evaluate(df) == pytest.approx(120.0 + 400.0)
