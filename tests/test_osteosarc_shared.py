"""Osteosarc's public cache/extractor, driven by unchanged original-read assets."""

from collections import Counter
import json
from pathlib import Path
import shutil

import pytest

from topiary import osteosarc_fixture_paths


ROOT = Path(__file__).parent / "data/osteosarc_shared"
SOURCE = ROOT / "vaccine-rna-v1"
MANIFEST = json.loads((SOURCE / "manifest.json").read_text())
pytestmark = pytest.mark.osteosarc


def no_download(*args, **kwargs):
    raise AssertionError("Offline regression attempted acquisition")


def test_all_shared_inputs_keep_identity_and_records():
    import pysam
    from osteosarc import assembly_from_header

    paths = osteosarc_fixture_paths(MANIFEST, directory=SOURCE)
    assert len(paths) == 98
    assert sum(p.stat().st_size for p in paths.values()) == 4126730
    assert len(MANIFEST["cases"]) == 49
    assert len({c["variant"]["variant_id"] for c in MANIFEST["cases"]}) == 44
    for case in MANIFEST["cases"]:
        with pysam.AlignmentFile(paths[case["bam"]]) as bam:
            assert bam.has_index()
            assert assembly_from_header(bam.header.to_dict()) == case["variant"]["assembly"]
            assert sum(1 for _ in bam) == case["selected_record_count"]
    # Acquisition never silently substitutes the corrected MAP2 complex allele.
    map2 = next(c["variant"] for c in MANIFEST["cases"] if c["variant"]["gene"] == "MAP2")
    assert len(map2["ref"]) > 1 and len(map2["alt"]) == 1


@pytest.mark.parametrize("door", ["export", "shared_cache"])
def test_offline_paths_reuse_other_consumers_objects_and_reject_tampering(tmp_path, monkeypatch, door):
    from datacache import Cache as DownloadCache
    from osteosarc import Cache, IntegrityError

    manifest = dict(MANIFEST, assets=MANIFEST["assets"][:2], cases=[])
    root = tmp_path / "openvax"
    shared = DownloadCache("openvax", cache_root=root / "objects/sha256")
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps(manifest))
    # Materialize precisely the datacache/Vaxrank object convention, with no
    # Osteosarc URL receipts. Public import_file must adopt these offline.
    for asset in manifest["assets"]:
        name = asset["filename"]
        path = Path(shared.local_path(filename=asset["sha256"] + "".join(Path(name).suffixes)))
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, path)
        shutil.copyfile(SOURCE / name, export / name)
    monkeypatch.setattr(Cache, "fetch", no_download)
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(root))
    kwargs = dict(directory=export) if door == "export" else dict(cache=Cache(offline=True))
    before = {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in (root / "objects/sha256").iterdir()}
    paths = osteosarc_fixture_paths(manifest, **kwargs)
    assert {p: (p.stat().st_ino, p.stat().st_mtime_ns) for p in before} == before
    assert [p.read_bytes() for p in paths.values()] == [
        (SOURCE / a["filename"]).read_bytes() for a in manifest["assets"]]
    victim = next(iter(paths.values()))
    original = victim.read_bytes()
    victim.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(IntegrityError, match="changed"):
        osteosarc_fixture_paths(manifest, **kwargs)


def test_offline_export_requires_its_unchanged_manifest(tmp_path):
    from osteosarc import IntegrityError

    with pytest.raises(IntegrityError, match="manifest is absent"):
        osteosarc_fixture_paths(MANIFEST, directory=tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(MANIFEST, data_version="different")))
    with pytest.raises(IntegrityError, match="manifest is absent or changed"):
        osteosarc_fixture_paths(MANIFEST, directory=tmp_path)


def test_missing_offline_objects_do_not_download(tmp_path, monkeypatch):
    from osteosarc import Cache, OfflineError
    import osteosarc.cache

    monkeypatch.setattr(osteosarc.cache.subprocess, "run", no_download)
    with pytest.raises(OfflineError):
        osteosarc_fixture_paths(MANIFEST, cache=Cache(tmp_path, offline=True))


def test_extracted_ntf3_reads_preserve_every_original_record(tmp_path):
    import pysam
    from osteosarc import Cache, Region, extract_reads

    case = next(c for c in MANIFEST["cases"] if c["variant"]["gene"] == "NTF3")
    paths = osteosarc_fixture_paths(MANIFEST, directory=SOURCE)
    allele = case["variant"]
    region = Region(allele["chrom"], allele["pos"] - 101,
                    allele["pos"] + len(allele["ref"]) + 100, allele["assembly"])
    cache = Cache(tmp_path / "cache", offline=True)
    subset = extract_reads(paths[case["bam"]], [region], cache=cache)
    with pysam.AlignmentFile(paths[case["bam"]]) as before, subset.open() as after:
        # to_string retains names, flags, all coordinates, CIGAR, sequence,
        # qualities, mate coordinates and typed tags; Counter keeps duplicates.
        assert Counter(r.to_string() for r in before.fetch(region.contig, region.start, region.end)) == Counter(
            r.to_string() for r in after)
        assert before.header.to_dict()["SQ"] == after.header.to_dict()["SQ"]
    assert subset.receipt["request"]["source_sha256"] == next(
        a["sha256"] for a in MANIFEST["assets"] if a["filename"] == case["bam"])
    assert extract_reads(paths[case["bam"]], [region], cache=cache).receipt == subset.receipt


def test_audit_acquisition_uses_osteosarc_and_retains_historical_alleles(tmp_path, monkeypatch):
    import pysam
    from osteosarc import Cache, digest
    from scripts.osteosarc_variant_audit import acquire, fetch_snapshot

    case = next(c for c in MANIFEST["cases"] if c["variant"]["gene"] == "NTF3")
    record = dict(case["variant"], input_status="ready")
    source = tmp_path / "source"
    source.mkdir()
    (tmp_path / "inventory.json").write_text(json.dumps(dict(variants=[record])))
    url = (f"https://api.genome.ucsc.edu/getData/sequence?genome=hg38;chrom={record['chrom']}"
           f";start={record['pos'] - 1};end={record['pos'] - 1 + len(record['ref'])}")
    reference = tmp_path / "reference-response.json"
    # Use the original independently verified genomic window, not a fabricated
    # matching REF response. No corrected allele is substituted on acquisition.
    checked = record["genomic_validation"]
    offset = checked["allele_offset"]
    reference.write_text(json.dumps({"dna": checked["sequence"][offset:offset + len(record["ref"])]}))
    cache = Cache(tmp_path / "cache", offline=True)
    cache.import_file(reference, url)
    import osteosarc.cache
    # Read extraction may invoke local samtools; only downloading is forbidden.
    original_run = osteosarc.cache.subprocess.run

    def local_only(command, **kwargs):
        assert command[0] != "curl", "Unexpected network download"
        return original_run(command, **kwargs)

    monkeypatch.setattr(osteosarc.cache.subprocess, "run", local_only)
    acquire(tmp_path, SOURCE / (case["bam"] + ".bai"), cache=cache, alignment_source=SOURCE / case["bam"])
    stored = json.loads((tmp_path / "checked-inventory.json").read_text())["variants"][0]
    assert {k: stored[k] for k in record} == record
    receipt = json.loads((source / "bam.receipt.json").read_text())
    assert receipt["osteosarc_extraction"]["records"] > 0
    exported = source / "t2-all-variant-regions.bam"
    assert receipt["sha256"] == digest(exported)
    with pysam.AlignmentFile(SOURCE / case["bam"]) as before, pysam.AlignmentFile(exported) as after:
        assert Counter(r.to_string() for r in before.fetch(record["chrom"], record["pos"] - 101,
                                                          record["pos"] + len(record["ref"]) + 100)) == Counter(
            r.to_string() for r in after)
    pinned_source = source / "reference" / (record["variant_id"] + ".json")
    assert pinned_source.read_bytes() == reference.read_bytes()
    pinned_source.write_text("tampered")
    with pytest.raises(ValueError, match="Cached input changed"):
        fetch_snapshot(url, pinned_source, cache=cache)
