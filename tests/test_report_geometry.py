"""Coordinate scope, missingness and ambiguity at the report/fragment boundary."""

import pandas as pd
import numpy as np
import pytest

from topiary import (
    fragments_from_dataframe, map_peptide_intervals,
    mutation_intervals_from_positions,
)


@pytest.mark.parametrize("positions,expected", [
    (2, [(1, 2)]), (2.0, [(1, 2)]), ("2,7-9", [(1, 2), (6, 9)]),
    ("3,2,3-4", [(1, 4)]), (99, []), (0, []),
    (None, None), (pd.NA, None), ("NA", None), ("?", None),
    ("2,?", None), ("4-2", None), (2.5, None), (True, None),
    (np.bool_(True), None), (np.int64(2), [(1, 2)]),
])
def test_report_positions_preserve_disjoint_and_unknown_geometry(positions, expected):
    assert mutation_intervals_from_positions(positions, 12) == expected


@pytest.mark.parametrize("context,peptide,intervals,expected", [
    ("AASIINFEKLCC", "SIINFEKL", [(2, 3)], [(4, 5)]),
    ("AASIINFEKLCC", "SIINFEKL", [(3, 3)], [(5, 5)]),
    ("AASIINFEKLCC", "SIINFEKL", [], None),
    ("SIINFEKL", "SIINFEKL", [], []),
    ("SIINFEKLSIINFEKL", "SIINFEKL", [(2, 3)], None),
    ("AAAA", "AAA", [(1, 2)], None),  # overlapping matches also ambiguous
    ("AASIINFEKLCC", "NOTHERE", [(1, 2)], None),
    ("SIINFEKL", None, [(1, 2)], None),
])
def test_peptide_coordinates_require_a_unique_occurrence(context, peptide, intervals, expected):
    assert map_peptide_intervals(context, peptide, intervals) == expected


@pytest.mark.parametrize("interval", [(1.5, 2), (True, 2), (np.bool_(True), 2), (-1, 2), (4, 3), (1, 99)])
def test_bad_report_coordinates_cannot_select_arbitrary_windows(interval):
    with pytest.raises(ValueError, match="interval"):
        map_peptide_intervals("SIINFEKL", "SIINFEKL", [interval])


def test_context_conversion_rebases_targets_without_fabricating_wt_flanks():
    row = dict(peptide="SIINFEKL", pep_context="AASIINFEKLCC", wt_peptide="SIINAEKL",
               source_type="variant:snv", mutation_start_in_peptide=4,
               mutation_end_in_peptide=5, source="report.tsv")
    fragment = fragments_from_dataframe(pd.DataFrame([row]))[0]
    assert fragment.target_intervals == [(6, 7)]
    assert fragment.reference_sequence is None
    assert fragment.annotations["reported_wt_peptide"] == "SIINAEKL"
    assert fragment.annotations["source"] == "report.tsv"
    peptide = fragments_from_dataframe(pd.DataFrame([row]), sequence_column="peptide")[0]
    assert peptide.target_intervals == [(4, 5)]
    assert peptide.reference_sequence == "SIINAEKL"


def test_a_negative_reported_peptide_does_not_label_an_entire_context_negative():
    row = dict(peptide="SIINFEKL", pep_context="AASIINFEKLCC", contains_mutant_residues=False)
    assert fragments_from_dataframe(pd.DataFrame([row]))[0].target_intervals is None
    assert fragments_from_dataframe(pd.DataFrame([row]), sequence_column="peptide")[0].target_intervals == []


def test_a_positive_flag_without_coordinates_does_not_mark_every_residue():
    row = dict(peptide="SIINFEKL", contains_mutant_residues=True)
    fragment = fragments_from_dataframe(pd.DataFrame([row]))[0]
    assert fragment.target_intervals is None
    assert fragment.annotations["reported_contains_mutant_residues"] is True


def test_interval_lists_are_supported_and_conflicts_are_not_row_order_dependent():
    rows = [dict(peptide="SIINFEKL", mutation_intervals_in_peptide=[[1, 2], [6, 7]])]
    assert fragments_from_dataframe(pd.DataFrame(rows))[0].target_intervals == [(1, 2), (6, 7)]
    rows.append(dict(peptide="SIINFEKL", mutation_intervals_in_peptide=[[2, 3]]))
    for records in (rows, rows[::-1]):
        with pytest.raises(ValueError, match="target_intervals"):
            fragments_from_dataframe(pd.DataFrame(records))


def test_explicit_reference_and_germline_sequences_keep_their_scope():
    row = dict(peptide="SIINFEKL", reference_sequence="SIINAEKL", germline_sequence="SIINVEKL")
    fragment = fragments_from_dataframe(pd.DataFrame([row]))[0]
    assert fragment.reference_sequence == "SIINAEKL"
    assert fragment.germline_sequence == fragment.effective_baseline == "SIINVEKL"


def test_pvacseq_missing_peptide_retains_unknown_geometry(tmp_path):
    from topiary import read_pvacseq
    from tests.report_geometry_helpers import write_pvacseq_geometry_report
    from tests.test_twin_conformance import PVACSEQ_MUTATION_GEOMETRY_TWINS

    for flavor in PVACSEQ_MUTATION_GEOMETRY_TWINS:
        path = write_pvacseq_geometry_report(tmp_path / f"{flavor}.tsv", flavor)
        source = pd.read_csv(path, sep="\t")
        source["Best Peptide" if flavor == "aggregated" else "MT Epitope Seq"] = None
        source.to_csv(path, sep="\t", index=False)
        frame = read_pvacseq(path).long_df
        assert frame.contains_mutant_residues.isna().all()
        assert fragments_from_dataframe(frame) == []
