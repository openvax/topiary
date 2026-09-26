"""Sid reads keep their selected records; checked-in fixtures stay pinned."""

from collections import Counter
import hashlib
import json
import subprocess
import sys

import pysam
import pytest
from osteosarc import Cache, IntegrityError

from topiary import osteosarc_fixture_paths
from .sid_data import ROOT, sid_read


pytestmark = pytest.mark.osteosarc
MANIFEST = json.loads((ROOT / "manifest.json").read_text())


def fail_network(*args, **kwargs):
    raise AssertionError("Bundled fixture workflow attempted network access")


@pytest.mark.parametrize("name", list(MANIFEST["read_selections"]))
def test_only_test_locus_records_are_bundled(name):
    """Each openvax-v1 export holds exactly the originally selected records."""
    selection = MANIFEST["read_selections"][name]
    records = Counter()
    with pysam.AlignmentFile(sid_read(name)) as handle:
        for read in handle:
            assert not read.is_unmapped
            assert any(read.reference_name == contig and read.reference_start < end
                       and start < read.reference_end for contig, start, end in selection["regions"])
            records[read.to_string()] += 1
    assert sum(records.values()) == selection["selected_records"]
    assert hashlib.sha256(json.dumps(sorted(records.items()), separators=(",", ":")).encode()).hexdigest() == (
        selection["sam_records_sha256"])


def test_whole_bundle_cli_exports_nested_paths_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(Cache, "fetch", fail_network)
    cache = Cache(tmp_path / "cache", offline=True)
    for asset in MANIFEST["assets"]:
        cache.import_file(ROOT / asset["filename"], asset["url"],
                          sha256=asset["sha256"], size=asset["size_bytes"])
    output = tmp_path / "export"
    subprocess.run([sys.executable, "-m", "scripts.osteosarc_test_data", "--offline",
                    "--cache-root", str(cache.root), "--output", str(output)], check=True)
    assert len(osteosarc_fixture_paths(MANIFEST, directory=output)) == len(MANIFEST["assets"])


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
