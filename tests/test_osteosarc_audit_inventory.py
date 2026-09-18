"""Source gaps and conflicts must not turn into biological negative results."""

import pytest

from scripts.osteosarc_variant_audit import fetch_snapshot, variant_inventory


INDEX = '''<table><tr data-vaccines="1">
<td><a href="/variant/example/">GENE</a></td><td>chr1:10</td><td>p.A1V</td>
<td></td><td></td><td></td><td></td><td></td><td></td></tr></table>'''
OTHER = '''<table><tr data-vaccines="0">
<td><a href="/variant/other/">OTHER</a></td><td>chr2:20</td><td>p.G2V</td>
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


@pytest.mark.parametrize("pos", ["", " ", "NA", "nan", "10.0", "1e1", "010", "0", "-10", "ten"])
def test_an_unusable_position_is_one_entrys_status_not_an_abort(pos):
    """Parsing reads every cell as text; one bad cell cannot stop the rest."""
    index = INDEX.replace("</table>", "") + OTHER.replace("<table>", "")
    rows = f"example\tchr1\t{pos}\tA\tC\nother\tchr2\t20\tG\tT\n"

    example, other = variant_inventory(index, HEADER + rows)

    assert example["input_status"] == "non_literal_allele"
    assert example["candidate_alleles"] == [["chr1", pos.strip(), "A", "C"]]
    assert "allele_key" not in example
    assert other["input_status"] == "ready"
    assert other["allele_key"] == "GRCh38:chr2:20:G>T"


def test_a_missing_allele_column_is_a_schema_error():
    with pytest.raises(ValueError, match=r"schema changed.*\['pos'\]"):
        variant_inventory(INDEX, "variant_id\tchrom\tref\talt\nexample\tchr1\tA\tC\n")


@pytest.mark.parametrize("vaccines", ["", "NA", "1.5"])
def test_an_unreadable_vaccine_count_is_a_schema_error(vaccines):
    with pytest.raises(ValueError, match="schema changed: data-vaccines"):
        variant_inventory(INDEX.replace('data-vaccines="1"', f'data-vaccines="{vaccines}"'), HEADER)


@pytest.mark.parametrize("html", ["", INDEX + INDEX])
def test_missing_or_duplicate_source_ids_are_errors(html):
    with pytest.raises(ValueError, match="Missing or duplicate"):
        variant_inventory(html, HEADER)


def test_incomplete_cached_acquisition_is_not_silently_reused(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"original")
    with pytest.raises(FileNotFoundError):
        fetch_snapshot("https://example.org/input", path)
