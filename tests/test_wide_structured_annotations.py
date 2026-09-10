"""Wide conversion has to survive the annotations topiary itself produces.

An RNA-derived :class:`ProteinFragment` carries
``supporting_reference_transcripts`` — every transcript consistent with
the assembled sequence, as a list — and prediction preserves it.  The
wide conversion grouped rows by merging on every non-prediction column,
which required each of them to be hashable, so ordinary RNA results
reached ``TypeError: unhashable type: 'list'`` (issue #287).

Grouping is a question about which rows belong together, not about what
the cells contain, so these tests pin both halves: rows group by the
*value* of a structured annotation, and the annotation that comes out is
the object that went in.
"""

import math

import numpy as np
import pandas as pd
import pytest

from topiary import (
    ProteinFragment,
    TopiaryPredictor,
    detect_form,
    from_wide,
    to_wide,
)

ALLELE = "HLA-A*02:01"
TRANSCRIPTS = ["ENST00000288602", "ENST00000496384"]


class TwoKindPredictor:
    """Two kinds per peptide, so siblings must share one wide row."""

    default_peptide_lengths = [8]

    def _rows(self, peptide):
        return [
            {
                "peptide": peptide, "allele": ALLELE, "kind": kind,
                "value": value, "score": value / 100.0,
                "affinity": value if kind == "pMHC_affinity" else math.nan,
                "percentile_rank": value,
                "predictor_name": "two-kind", "predictor_version": "1.0",
            }
            for kind, value in (("pMHC_affinity", 40.0),
                                ("pMHC_presentation", 0.8))
        ]

    def predict_peptides_dataframe(self, peptides):
        return pd.DataFrame([r for p in peptides for r in self._rows(p)])

    def predict_proteins_dataframe(self, name_to_sequence):
        rows = []
        for name, sequence in name_to_sequence.items():
            for offset in range(len(sequence) - 7):
                for row in self._rows(sequence[offset:offset + 8]):
                    rows.append({**row, "source_sequence_name": name,
                                 "offset": offset})
        return pd.DataFrame(rows)


def _long_row(**extra):
    row = {
        "fragment_id": "f", "peptide": "SIINFEKL", "allele": ALLELE,
        "kind": "pMHC_affinity", "prediction_method_name": "test",
        "value": 50.0,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------


def test_list_valued_annotation_converts_to_wide():
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=["ENST1"]),
    ])

    wide = to_wide(frame)

    assert len(wide) == 1
    assert wide["supporting_reference_transcripts"].iloc[0] == ["ENST1"]


def test_the_annotation_comes_back_as_a_list_not_a_stand_in():
    # Deduplicating through a hashable surrogate would hand the caller
    # the surrogate; the value has to be the object that went in.
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=TRANSCRIPTS),
    ])

    wide = to_wide(frame)
    cell = wide["supporting_reference_transcripts"].iloc[0]

    assert isinstance(cell, list)
    assert cell == TRANSCRIPTS


# ---------------------------------------------------------------------------
# Grouping is by value, and siblings stay together
# ---------------------------------------------------------------------------


def test_prediction_siblings_are_not_split_by_a_list_annotation():
    # Two kinds for one peptide are two long rows and one wide row.  The
    # lists are equal but distinct objects, so grouping cannot lean on
    # identity.
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=list(TRANSCRIPTS)),
        _long_row(kind="pMHC_presentation", value=0.8,
                  supporting_reference_transcripts=list(TRANSCRIPTS)),
    ])

    wide = to_wide(frame)

    assert len(wide) == 1
    assert wide["supporting_reference_transcripts"].iloc[0] == TRANSCRIPTS
    assert "test_affinity_value" in wide.columns
    assert "test_presentation_value" in wide.columns


def test_different_transcript_lists_stay_different_groups():
    frame = pd.DataFrame([
        _long_row(fragment_id="f", supporting_reference_transcripts=["E1"]),
        _long_row(fragment_id="g", supporting_reference_transcripts=["E2"]),
    ])

    wide = to_wide(frame)

    assert len(wide) == 2
    assert sorted(
        tuple(v) for v in wide["supporting_reference_transcripts"]
    ) == [("E1",), ("E2",)]


def test_transcript_order_is_part_of_the_annotation():
    # Two orders of the same transcripts are two different annotations,
    # so they are not silently merged into one row.
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=["E1", "E2"]),
        _long_row(kind="pMHC_presentation", value=0.8,
                  supporting_reference_transcripts=["E2", "E1"]),
    ])

    assert len(to_wide(frame)) == 2


def test_rows_missing_the_same_annotation_group_together():
    # Comparing the raw cells would split these, because nan != nan.
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=None),
        _long_row(kind="pMHC_presentation", value=0.8,
                  supporting_reference_transcripts=None),
    ])

    assert len(to_wide(frame)) == 1


def test_a_list_and_a_missing_value_are_different_groups():
    frame = pd.DataFrame([
        _long_row(supporting_reference_transcripts=["E1"]),
        _long_row(kind="pMHC_presentation", value=0.8,
                  supporting_reference_transcripts=None),
    ])

    assert len(to_wide(frame)) == 2


# ---------------------------------------------------------------------------
# Other structured annotation values topiary can carry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected_type", [
    (["E1", "E2"], list),
    (("E1", "E2"), tuple),
    ({"assembly": "GRCh38", "caller": "isovar"}, dict),
    (np.array([1, 2, 3]), np.ndarray),
    ({"E1", "E2"}, set),
])
def test_structured_annotations_survive_with_their_type(value, expected_type):
    frame = pd.DataFrame([_long_row(structured=value)])

    wide = to_wide(frame)

    assert len(wide) == 1
    assert isinstance(wide["structured"].iloc[0], expected_type)


def test_a_mapping_groups_by_content_not_key_order():
    frame = pd.DataFrame([
        _long_row(structured={"a": 1, "b": 2}),
        _long_row(kind="pMHC_presentation", value=0.8,
                  structured={"b": 2, "a": 1}),
    ])

    assert len(to_wide(frame)) == 1


def test_nested_containers_group_by_content():
    frame = pd.DataFrame([
        _long_row(structured={"ids": ["E1", "E2"]}),
        _long_row(kind="pMHC_presentation", value=0.8,
                  structured={"ids": ["E1", "E2"]}),
    ])

    assert len(to_wide(frame)) == 1


# ---------------------------------------------------------------------------
# Composed: fragment -> prediction -> long/wide -> long
# ---------------------------------------------------------------------------


def _rna_fragment():
    return ProteinFragment(
        fragment_id="rna__abc123",
        source_type="variant:snv",
        sequence="MASIINFEKLGG",
        gene="OVA",
        transcript_id=TRANSCRIPTS[0],
        annotations={
            "sequence_source": "isovar",
            "supporting_reference_transcripts": TRANSCRIPTS,
        },
    )


def test_rna_fragment_predictions_reach_wide_form():
    df = TopiaryPredictor(models=[TwoKindPredictor()]).predict_from_fragments(
        [_rna_fragment()]
    )

    assert "supporting_reference_transcripts" in df.columns
    assert detect_form(df) == "long"

    wide = to_wide(df)

    assert detect_form(wide) == "wide"
    assert not wide.empty
    # Every peptide of one fragment carries the same transcript list, and
    # the two kinds per peptide share a row rather than splitting.
    assert len(wide) == len(df) // 2
    for cell in wide["supporting_reference_transcripts"]:
        assert cell == TRANSCRIPTS


def test_rna_fragment_predictions_round_trip_through_wide():
    predicted = TopiaryPredictor(
        models=[TwoKindPredictor()]
    ).predict_from_fragments([_rna_fragment()])

    restored = from_wide(to_wide(predicted))

    assert len(restored) == len(predicted)
    for cell in restored["supporting_reference_transcripts"]:
        assert cell == TRANSCRIPTS
    assert set(restored["kind"]) == set(predicted["kind"])
    assert sorted(restored["peptide"]) == sorted(predicted["peptide"])


def test_two_rna_fragments_keep_their_own_transcript_lists():
    other = ProteinFragment(
        fragment_id="rna__def456",
        source_type="variant:snv",
        sequence="MAGILGFVFTL",
        gene="FLU",
        transcript_id="ENST00000999999",
        annotations={
            "sequence_source": "isovar",
            "supporting_reference_transcripts": ["ENST00000999999"],
        },
    )
    df = TopiaryPredictor(models=[TwoKindPredictor()]).predict_from_fragments(
        [_rna_fragment(), other]
    )

    wide = to_wide(df)

    by_fragment = {
        fragment_id: {tuple(v) for v in rows["supporting_reference_transcripts"]}
        for fragment_id, rows in wide.groupby("fragment_id")
    }
    assert by_fragment["rna__abc123"] == {tuple(TRANSCRIPTS)}
    assert by_fragment["rna__def456"] == {("ENST00000999999",)}
