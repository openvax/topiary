"""Source gaps and conflicts must not turn into biological negative results."""

import json
from pathlib import Path
import runpy

import pytest

from scripts.osteosarc_rna_overlay import digest, write_json
from scripts.osteosarc_variant_audit import fetch_snapshot, variant_inventory


INDEX = '''<table><tr data-vaccines="1">
<td><a href="/variant/example/">GENE</a></td><td>chr1:10</td><td>p.A1V</td>
<td></td><td></td><td></td><td></td><td></td><td></td></tr></table>'''
OTHER = '''<table><tr data-vaccines="0">
<td><a href="/variant/other/">OTHER</a></td><td>chr2:20</td><td>p.G2V</td>
<td></td><td></td><td></td><td></td><td></td><td></td></tr></table>'''
HEADER = "variant_id\tgene\tchrom\tpos\tref\talt\n"


@pytest.mark.parametrize("rows,status", [
    ("", "missing_literal_allele"),
    ("example\tGENE\tchr1\t10\tA\tdup\n", "non_literal_allele"),
    ("example\tGENE\tchr1\t11\tA\tC\n", "conflicting_coordinates"),
    ("example\tGENE\tchr1\t10\tA\tC\nexample\tGENE\tchr1\t10\tA\tG\n", "ambiguous_literal_allele"),
    ("example\tGENE\tchr1\t10\tA\tC\nexample\tGENE\tchr1\t10\tA\tC\n", "ready"),
])
def test_inventory_preserves_one_outcome_without_guessing(rows, status):
    record, = variant_inventory(INDEX, HEADER + rows)
    assert record["variant_id"] == "example"
    assert record["input_status"] == status
    assert ("allele_key" in record) == (status == "ready")


@pytest.mark.parametrize("pos", ["", " ", "NA", "nan", "10.0", "1e1", "0", "-10", "ten"])
def test_an_unusable_position_is_one_entrys_status_not_an_abort(pos):
    """Parsing reads every cell as text; one bad cell cannot stop the rest."""
    index = INDEX.replace("</table>", "") + OTHER.replace("<table>", "")
    rows = f"example\tGENE\tchr1\t{pos}\tA\tC\nother\tOTHER\tchr2\t20\tG\tT\n"

    example, other = variant_inventory(index, HEADER + rows)

    assert example["input_status"] == "malformed_source_row"
    assert example["candidate_alleles"] == []
    assert example["parse_errors"][0]["values"]["pos"] == pos
    assert "allele_key" not in example
    assert other["input_status"] == "ready"
    assert other["allele_key"] == "GRCh38:chr2:20:G>T"


@pytest.mark.parametrize("malformed", ["first", "later"])
@pytest.mark.parametrize("fields", ["\textra", ""], ids=["extra-field", "short-row"])
def test_a_malformed_row_is_its_entrys_status_and_shifts_nothing(malformed, fields):
    """A row with the wrong number of fields cannot be read column by column."""
    index = INDEX.replace("</table>", "") + OTHER.replace("<table>", "")
    bad = "example\tGENE\tchr1\t10\tA\tC" + fields if fields else "example\tGENE\tchr1\t10"
    good = "other\tOTHER\tchr2\t20\tG\tT"
    rows = [bad, good] if malformed == "first" else [good, bad]

    records = {r["variant_id"]: r for r in variant_inventory(index, HEADER + "\n".join(rows) + "\n")}

    assert records["example"]["input_status"] == "malformed_source_row"
    assert records["other"]["input_status"] == "ready"
    assert records["other"]["allele_key"] == "GRCh38:chr2:20:G>T"


def test_resumed_audits_refuse_legacy_fragment_files(tmp_path):
    from scripts.osteosarc_variant_audit import fragment_files

    (tmp_path / "B.tsv").write_text("")
    (tmp_path / "A.tsv").write_text("")
    assert [path.name for path in fragment_files(tmp_path)] == ["A.tsv", "B.tsv"]
    assert fragment_files(tmp_path / "absent") == []
    (tmp_path / "C.json").write_text("")
    with pytest.raises(ValueError, match="pre-5.63.*C.json"):
        fragment_files(tmp_path)


def test_a_missing_allele_column_is_a_schema_error():
    with pytest.raises(ValueError, match="Missing or duplicate.*columns"):
        variant_inventory(INDEX, "variant_id\tgene\tchrom\tref\talt\nexample\tGENE\tchr1\tA\tC\n")


@pytest.mark.parametrize("vaccines", ["", "NA", "1.5"])
def test_an_unreadable_vaccine_count_is_a_schema_error(vaccines):
    with pytest.raises(ValueError, match="invalid literal"):
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


@pytest.mark.parametrize("changed", [None, "bam_sha256", "inventory_sha256", "reference_sha256"])
def test_fixture_export_requires_the_audited_inputs(tmp_path, changed):
    """Re-pinning cannot pair valid new sources with outcomes from an old run."""
    regenerate = runpy.run_path(str(
        Path(__file__).parent / "data/osteosarc_all_variants/regenerate.py"))["regenerate"]
    source, destination = tmp_path / "audit", tmp_path / "export"
    (source / "source").mkdir(parents=True)
    (source / "reference").mkdir()
    bam = source / "source/t2-all-variant-regions.bam"
    bam.write_bytes(b"verified alignment bytes")
    index = bam.with_suffix(".bam.bai")
    index.write_bytes(b"verified index bytes")
    write_json(source / "source/bam.receipt.json",
               dict(sha256=digest(bam), index_sha256=digest(index)))
    write_json(source / "inventory.json", dict(sources={}, variants=[]))
    write_json(source / "checked-inventory.json", dict(variants=[]))
    write_json(source / "reference/manifest.json", dict(files={}))
    run = dict(bam_sha256=digest(bam),
               inventory_sha256=digest(source / "checked-inventory.json"),
               reference_sha256=digest(source / "reference/manifest.json"))
    if changed is not None:
        run[changed] = "0" * 64
    write_json(source / "run.json", run)

    if changed is not None:
        with pytest.raises(ValueError, match=f"Audit run input mismatch: {changed}"):
            regenerate(source, destination)
        assert not destination.exists()
    else:
        regenerate(source, destination)
        assert json.loads((destination / "run.json").read_text()) == run
        assert (destination / "source/t2-all-variant-regions.bam").read_bytes() == bam.read_bytes()
