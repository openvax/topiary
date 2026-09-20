"""Bundling must preserve required reads and exclude reads outside test loci."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pysam
import pytest
from osteosarc import Cache, IntegrityError, digest

from scripts.generate_sid_fixtures import generate_sid_fixtures
from topiary import osteosarc_fixture_paths
from .sid_data import ROOT, sid_fixture_paths


pytestmark = pytest.mark.osteosarc
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def fail_network(*args, **kwargs):
    raise AssertionError("Bundled fixture workflow attempted network access")


@pytest.mark.parametrize("name", list(MANIFEST["read_selections"]))
def test_only_test_locus_records_are_bundled(name, tmp_path):
    selection = MANIFEST["read_selections"][name]
    path = sid_fixture_paths()[name]
    if name.endswith(".sam.gz"):
        expanded = tmp_path / "reads.sam"
        expanded.write_bytes(gzip.decompress(path.read_bytes()))
        path = expanded
    records = Counter()
    with pysam.AlignmentFile(path) as handle:
        for read in handle:
            assert not read.is_unmapped
            assert any(read.reference_name == contig and read.reference_start < end
                       and start < read.reference_end for contig, start, end in selection["regions"])
            records[read.to_string()] += 1
    assert sum(records.values()) == selection["selected_records"]
    assert hashlib.sha256(json.dumps(sorted(records.items()), separators=(",", ":")).encode()).hexdigest() == (
        selection["sam_records_sha256"])


def test_generate_complete_bundle_offline_removes_irrelevant_read(tmp_path, monkeypatch):
    """Real extraction, manifest rewriting, nested export and offline verification."""
    import requests

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    source = tmp_path / "inputs"
    source.mkdir()
    recipe = json.loads((ROOT / "sid-fixtures.json").read_text())
    for asset in recipe["sources"]:
        target = source / asset["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / asset["filename"], target)
    selection = next(r for r in recipe["reads"] if "NTF3" in r["filename"])
    bam = source / selection["filename"]
    with pysam.AlignmentFile(bam) as original:
        header = original.header
        records = list(original)
        expected = Counter(r.to_string() for r in records)
    # Deliberate test noise, never exported as scientific data: one extra
    # alignment of the first read a megabase away from the exercised locus.
    noise = pysam.AlignedSegment.fromstring(records[0].to_string(), header)
    noise.reference_start += 1_000_000
    with pysam.AlignmentFile(bam, "wb", header=header) as handle:
        for read in [*records, noise]:
            handle.write(read)
    pysam.index(str(bam))
    selection.pop("preserve_all_records")
    for asset in recipe["sources"]:
        path = source / asset["filename"]
        asset.update(sha256=digest(path), size_bytes=path.stat().st_size)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    output = tmp_path / "export"
    manifest = generate_sid_fixtures(recipe_path, output, source_directory=source,
                                     cache=Cache(tmp_path / "cache", offline=True))
    paths = osteosarc_fixture_paths(manifest, directory=output)
    shared = output / "osteosarc_shared/vaccine-rna-v1"
    osteosarc_fixture_paths(json.loads((shared / "manifest.json").read_text()), directory=shared)
    with pysam.AlignmentFile(paths[selection["filename"]]) as generated:
        assert generated.has_index()
        assert Counter(r.to_string() for r in generated) == expected
    counts = manifest["read_selections"][selection["filename"]]
    assert counts["original_records"] == counts["selected_records"] + 1
    assert counts["selected_records"] == 15
    assert len(list((output / "osteosarc_shared/vaccine-rna-v1").glob("*.bam"))) == 1
    assert (output / "osteosarc_all_variants/expected.json").read_bytes() == (
        ROOT / "osteosarc_all_variants/expected.json").read_bytes()
    with pytest.raises(FileExistsError):
        generate_sid_fixtures(recipe_path, output, source_directory=source)
    victim = paths[selection["filename"]]
    victim.write_bytes(victim.read_bytes()[:-1])
    with pytest.raises(IntegrityError, match="changed"):
        osteosarc_fixture_paths(manifest, directory=output)


def test_whole_bundle_cli_exports_nested_paths_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(Cache, "fetch", fail_network)
    cache = Cache(tmp_path / "cache", offline=True)
    for asset in MANIFEST["assets"]:
        cache.import_file(ROOT / asset["filename"], asset["url"],
                          sha256=asset["sha256"], size=asset["size_bytes"])
    output = tmp_path / "export"
    subprocess.run([sys.executable, "-m", "scripts.osteosarc_test_data", "--offline",
                    "--cache-root", str(cache.root), "--output", str(output)], check=True)
    assert len(osteosarc_fixture_paths(MANIFEST, directory=output)) == 104


@pytest.mark.parametrize("name", ["../escape.bam", "/absolute.bam", "group/../escape.bam", "group//a.bam",
                                  "group/./a.bam", "group\\a.bam", "group/a:b.bam"])
def test_nested_export_rejects_unsafe_names(name, tmp_path):
    manifest = dict(MANIFEST, assets=[dict(MANIFEST["assets"][0], filename=name)])
    with pytest.raises(ValueError, match="Unsafe"):
        osteosarc_fixture_paths(manifest, directory=tmp_path)


def test_nested_export_rejects_symlink_directory(tmp_path):
    asset = next(a for a in MANIFEST["assets"] if a["filename"].startswith("osteosarc/"))
    manifest = dict(MANIFEST, assets=[asset])
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "osteosarc").symlink_to(ROOT / "osteosarc", target_is_directory=True)
    with pytest.raises(IntegrityError, match="changed"):
        osteosarc_fixture_paths(manifest, directory=tmp_path)
