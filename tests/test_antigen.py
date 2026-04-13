"""Tests for topiary.antigen (AntigenFragment + helpers)."""

import json
import tempfile
from pathlib import Path

import pytest

from topiary import AntigenFragment, make_fragment_id
from topiary.antigen import _sanitize_prefix, _default_prefix, collect_annotations
from topiary.io_antigen import read_antigens, write_antigens, iter_antigens


# ---------------------------------------------------------------------------
# fragment_id construction
# ---------------------------------------------------------------------------


class TestMakeFragmentId:
    def test_shape(self):
        fid = make_fragment_id("BRAF_p.V600E", "MAVS")
        assert "__" in fid
        prefix, rest = fid.split("__")
        assert prefix == "BRAF_p.V600E"
        assert len(rest) == 8  # default hash_length
        assert all(c in "0123456789abcdef" for c in rest)

    def test_deterministic(self):
        a = make_fragment_id("gene_X", "MAVS", variant="v1")
        b = make_fragment_id("gene_X", "MAVS", variant="v1")
        assert a == b

    def test_variant_changes_hash(self):
        a = make_fragment_id("gene_X", "MAVS", variant="v1")
        b = make_fragment_id("gene_X", "MAVS", variant="v2")
        assert a != b

    def test_sequence_changes_hash(self):
        a = make_fragment_id("X", "MAVS")
        b = make_fragment_id("X", "MAVT")
        assert a != b

    def test_prefix_sanitization(self):
        # Spaces, slashes, etc. collapse to single underscore
        fid = make_fragment_id("gene name / with   weird", "X")
        prefix, _ = fid.split("__")
        assert prefix == "gene_name_with_weird"

    def test_preserves_safe_chars(self):
        fid = make_fragment_id("BRAF_p.V600E:mut-1", "X")
        prefix, _ = fid.split("__")
        assert prefix == "BRAF_p.V600E:mut-1"

    def test_empty_prefix(self):
        fid = make_fragment_id("", "X")
        assert fid.startswith("__")
        assert len(fid) == 10  # "" + "__" + 8 hex chars


# ---------------------------------------------------------------------------
# Dataclass identity / hashing
# ---------------------------------------------------------------------------


class TestIdentity:
    def _sample(self, **overrides):
        kwargs = dict(
            fragment_id="test__abcdef01",
            source_type="variant:snv",
            sequence="MAAV",
            target_intervals=[(2, 3)],
        )
        kwargs.update(overrides)
        return AntigenFragment(**kwargs)

    def test_equality_by_id(self):
        a = self._sample()
        b = self._sample(sequence="DIFFERENT", target_intervals=None)
        # Same fragment_id → equal even though content differs
        assert a == b

    def test_hashable_in_set(self):
        a = self._sample()
        b = self._sample()
        assert {a, b} == {a}

    def test_inequality_by_id(self):
        a = self._sample(fragment_id="test__aaaaaaaa")
        b = self._sample(fragment_id="test__bbbbbbbb")
        assert a != b

    def test_hash_ignores_unhashable_fields(self):
        # target_intervals is a list, annotations a dict — both unhashable.
        # Auto-generated __hash__ would blow up; ours is fine because it keys
        # on fragment_id.
        f = self._sample(annotations={"vaf": 0.5})
        hash(f)  # does not raise


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestSerialization:
    def _rich(self):
        return AntigenFragment(
            fragment_id="BRAF_p.V600E__abcdef01",
            source_type="variant:snv",
            sequence="MAAVTDVGMAV",
            reference_sequence="MAAVTDVGMAA",
            germline_sequence=None,
            target_intervals=[(10, 11)],
            variant="chr7:140753336",
            effect="p.Val600Glu",
            effect_type="Substitution",
            gene="BRAF",
            gene_id="ENSG00000157764",
            transcript_id="ENST00000288602",
            gene_expression=12.3,
            transcript_expression=8.1,
            annotations={"vaf": 0.42, "ccf": 0.9, "caller": "mutect2"},
        )

    def test_to_dict_shape(self):
        d = self._rich().to_dict()
        assert d["fragment_id"] == "BRAF_p.V600E__abcdef01"
        # target_intervals serialized as lists, not tuples
        assert d["target_intervals"] == [[10, 11]]

    def test_dict_roundtrip(self):
        f = self._rich()
        f2 = AntigenFragment.from_dict(f.to_dict())
        assert f == f2
        assert f2.target_intervals == [(10, 11)]  # restored as tuples
        assert f2.annotations == f.annotations

    def test_json_roundtrip(self):
        f = self._rich()
        f2 = AntigenFragment.from_json(f.to_json())
        assert f == f2
        assert f2.sequence == f.sequence
        assert f2.target_intervals == [(10, 11)]

    def test_json_pretty(self):
        f = self._rich()
        s = f.to_json(indent=2)
        assert "\n" in s
        assert AntigenFragment.from_json(s) == f

    def test_from_dict_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown AntigenFragment field"):
            AntigenFragment.from_dict({
                "fragment_id": "x__00000000", "sequence": "M",
                "bogus_field": 1,
            })

    def test_from_dict_missing_optional_defaults(self):
        f = AntigenFragment.from_dict({
            "fragment_id": "x__00000000", "sequence": "M",
        })
        assert f.source_type is None
        assert f.target_intervals is None
        assert f.annotations == {}

    def test_none_target_intervals_serializes(self):
        f = AntigenFragment(fragment_id="x__00000000", sequence="M")
        d = f.to_dict()
        assert d["target_intervals"] is None
        f2 = AntigenFragment.from_dict(d)
        assert f2.target_intervals is None

    def test_empty_target_intervals(self):
        f = AntigenFragment(fragment_id="x__00000000", sequence="M", target_intervals=[])
        f2 = AntigenFragment.from_json(f.to_json())
        assert f2.target_intervals == []


# ---------------------------------------------------------------------------
# Stringification
# ---------------------------------------------------------------------------


class TestStringification:
    def test_str_is_compact(self):
        f = AntigenFragment(
            fragment_id="BRAF_p.V600E__abcdef01",
            source_type="variant:snv",
            sequence="MAAVTDVGMAV",
            target_intervals=[(10, 11)],
            gene="BRAF",
        )
        s = str(f)
        assert "BRAF_p.V600E__abcdef01" in s
        assert "11 aa" in s
        assert "variant:snv" in s
        assert "1 target interval" in s
        assert "gene=BRAF" in s

    def test_repr_unambiguous(self):
        f = AntigenFragment(fragment_id="x__00000000", sequence="M")
        r = repr(f)
        # dataclass-generated repr contains class name and fragment_id
        assert "AntigenFragment" in r
        assert "fragment_id='x__00000000'" in r


# ---------------------------------------------------------------------------
# Geometry: peptide_overlaps_target
# ---------------------------------------------------------------------------


class TestOverlaps:
    def _f(self, intervals):
        return AntigenFragment(
            fragment_id="x__00000000",
            sequence="A" * 20,
            target_intervals=intervals,
        )

    def test_none_intervals_never_overlap(self):
        f = self._f(None)
        assert f.peptide_overlaps_target(0, 9) is False
        assert f.has_target is False

    def test_empty_intervals_never_overlap(self):
        f = self._f([])
        assert f.peptide_overlaps_target(0, 9) is False
        assert f.has_target is False

    def test_single_interval_exact(self):
        # Target at [5, 6) — a single residue at position 5
        f = self._f([(5, 6)])
        assert f.has_target is True
        # Peptide covering [5, 5+9) — overlaps
        assert f.peptide_overlaps_target(5, 9) is True
        # Peptide ending at 5 — no overlap (half-open)
        assert f.peptide_overlaps_target(0, 5) is False
        # Peptide starting at 6 — no overlap
        assert f.peptide_overlaps_target(6, 9) is False

    def test_multiple_intervals(self):
        # Target at two disjoint positions
        f = self._f([(3, 4), (15, 16)])
        assert f.peptide_overlaps_target(0, 9) is True    # covers pos 3
        assert f.peptide_overlaps_target(5, 9) is False   # [5, 14): misses both
        assert f.peptide_overlaps_target(10, 9) is True   # covers 10-19, includes 15

    def test_interval_inside_peptide(self):
        f = self._f([(5, 8)])
        assert f.peptide_overlaps_target(3, 9) is True  # [3, 12) fully contains [5, 8)


# ---------------------------------------------------------------------------
# from_variant / from_junction
# ---------------------------------------------------------------------------


class TestClassmethods:
    def test_from_variant_inframe_snv(self):
        f = AntigenFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=10, mutation_end=11,
            inframe=True,
            gene="BRAF", effect="p.Ala290Val",
        )
        assert f.source_type == "variant:snv"
        assert f.target_intervals == [(10, 11)]
        assert f.gene == "BRAF"
        assert f.fragment_id.startswith("BRAF_p.Ala290Val")

    def test_from_variant_inframe_indel(self):
        f = AntigenFragment.from_variant(
            sequence="MAAVTDVGMAV",
            mutation_start=8, mutation_end=11,
            inframe=True,
        )
        assert f.source_type == "variant:indel"
        assert f.target_intervals == [(8, 11)]

    def test_from_variant_frameshift(self):
        f = AntigenFragment.from_variant(
            sequence="MAAVTDVGMAV",  # length 11
            mutation_start=5, mutation_end=5,
            inframe=False,
        )
        assert f.source_type == "variant:frameshift"
        # Targets everything from the shift to the end
        assert f.target_intervals == [(5, 11)]

    def test_from_junction_crossing_only(self):
        f = AntigenFragment.from_junction(
            sequence="AAAAAABBBBBB",
            junction_position=6,
            novel_downstream=False,
            gene="EWSR1", variant="EWSR1-FLI1",
        )
        assert f.source_type == "sv:fusion"
        assert f.target_intervals == [(5, 7)]  # junction ± 1

    def test_from_junction_novel_downstream(self):
        f = AntigenFragment.from_junction(
            sequence="AAAAAABBBBBB",  # length 12
            junction_position=6,
            novel_downstream=True,
        )
        assert f.target_intervals == [(6, 12)]

    def test_from_variant_fragment_id_prefix(self):
        # Provide all three fields; prefix should concatenate
        f = AntigenFragment.from_variant(
            sequence="ABC", mutation_start=1, mutation_end=2, inframe=True,
            gene="GENE", effect="p.Ala1Val", variant="chr1:100",
        )
        prefix, _ = f.fragment_id.split("__")
        assert "GENE" in prefix
        assert "p.Ala1Val" in prefix
        assert "chr1:100" in prefix


# ---------------------------------------------------------------------------
# effective_baseline property
# ---------------------------------------------------------------------------


class TestEffectiveBaseline:
    def test_germline_takes_precedence(self):
        f = AntigenFragment(
            fragment_id="x__00000000", sequence="M",
            reference_sequence="REF", germline_sequence="GERM",
        )
        assert f.effective_baseline == "GERM"

    def test_fallback_to_reference(self):
        f = AntigenFragment(
            fragment_id="x__00000000", sequence="M",
            reference_sequence="REF", germline_sequence=None,
        )
        assert f.effective_baseline == "REF"

    def test_both_none(self):
        f = AntigenFragment(fragment_id="x__00000000", sequence="M")
        assert f.effective_baseline is None


# ---------------------------------------------------------------------------
# TSV IO
# ---------------------------------------------------------------------------


class TestTsvIo:
    def _fragments(self):
        return [
            AntigenFragment.from_variant(
                sequence="MAAVTDVGMAV",
                reference_sequence="MAAVTDVGMAA",
                mutation_start=10, mutation_end=11,
                inframe=True,
                gene="BRAF", effect="p.Val600Glu",
                variant="chr7:140753336",
                annotations={"vaf": 0.42, "ccf": 0.9},
            ),
            AntigenFragment(
                fragment_id="erv_Hsap38.chr7__abcdef01",
                source_type="erv",
                sequence="MLGMNMLL",
                target_intervals=None,
                annotations={"erv_orf_id": "X", "erv_norm_cpm": 1.2},
            ),
        ]

    def test_roundtrip(self, tmp_path):
        frags = self._fragments()
        p = tmp_path / "antigens.tsv"
        write_antigens(frags, p)
        loaded = read_antigens(p)
        for a, b in zip(frags, loaded):
            assert a == b
            assert a.sequence == b.sequence
            assert a.target_intervals == b.target_intervals
            assert a.annotations == b.annotations

    def test_file_is_tsv(self, tmp_path):
        frags = self._fragments()
        p = tmp_path / "antigens.tsv"
        write_antigens(frags, p)
        first_line = p.read_text().splitlines()[0]
        assert "\t" in first_line
        assert first_line.startswith("fragment_id\t")

    def test_unknown_column_raises(self, tmp_path):
        p = tmp_path / "bad.tsv"
        p.write_text(
            "fragment_id\tsequence\tfoobar\n"
            "x__00000000\tM\tvalue\n"
        )
        with pytest.raises(ValueError, match="Unknown antigen-TSV column"):
            read_antigens(p)

    def test_missing_optional_columns(self, tmp_path):
        p = tmp_path / "minimal.tsv"
        p.write_text("fragment_id\tsequence\nx__00000000\tMAVS\n")
        loaded = read_antigens(p)
        assert len(loaded) == 1
        assert loaded[0].sequence == "MAVS"
        assert loaded[0].annotations == {}
        assert loaded[0].target_intervals is None

    def test_iter_antigens(self, tmp_path):
        frags = self._fragments()
        p = tmp_path / "antigens.tsv"
        write_antigens(frags, p)
        streamed = list(iter_antigens(p))
        assert len(streamed) == len(frags)
        for a, b in zip(frags, streamed):
            assert a == b


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilities:
    def test_sanitize_prefix(self):
        assert _sanitize_prefix("  hello  world/foo ") == "hello_world_foo"
        assert _sanitize_prefix("already.clean-text:ok") == "already.clean-text:ok"
        assert _sanitize_prefix("") == ""

    def test_default_prefix_skips_none(self):
        assert _default_prefix("BRAF", None, "chr1:100") == "BRAF_chr1:100"
        assert _default_prefix(None, None, None) == ""

    def test_collect_annotations(self):
        frags = [
            AntigenFragment(fragment_id="a__00000000", sequence="M",
                            annotations={"x": 1, "y": 2}),
            AntigenFragment(fragment_id="b__00000000", sequence="M",
                            annotations={"y": 3, "z": 4}),
        ]
        assert collect_annotations(frags) == {"x", "y", "z"}
