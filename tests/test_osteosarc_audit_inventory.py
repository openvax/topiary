"""Source gaps and conflicts must not turn into biological negative results."""

import pytest

from scripts.osteosarc_variant_audit import fetch_snapshot, variant_inventory


INDEX = '''<table><tr data-vaccines="1">
<td><a href="/variant/example/">GENE</a></td><td>chr1:10</td><td>p.A1V</td>
<td></td><td></td><td></td><td></td><td></td><td></td></tr></table>'''
HEADER = "variant_id\tchrom\tpos\tref\talt\n"


@pytest.mark.parametrize("rows,status", [
    ("", "missing_literal_allele"),
    ("example\tchr1\t10\tA\tdup\n", "non_literal_allele"),
    ("example\tchr1\t11\tA\tC\n", "conflicting_coordinates"),
    ("example\tchr1\t10\tA\tC\nexample\tchr1\t10\tA\tG\n", "ambiguous_literal_allele"),
    ("example\tchr1\t10\tA\tC\nexample\tchr1\t10\tA\tC\n", "ready"),
])
def test_inventory_preserves_one_outcome_without_guessing(rows, status):
    record, = variant_inventory(INDEX, HEADER + rows)
    assert record["variant_id"] == "example"
    assert record["input_status"] == status
    assert ("allele_key" in record) == (status == "ready")


@pytest.mark.parametrize("html", ["", INDEX + INDEX])
def test_missing_or_duplicate_source_ids_are_errors(html):
    with pytest.raises(ValueError, match="Missing or duplicate"):
        variant_inventory(html, HEADER)


def test_incomplete_cached_acquisition_is_not_silently_reused(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    with pytest.raises(FileNotFoundError):
        fetch_snapshot("https://example.org/input", path)
