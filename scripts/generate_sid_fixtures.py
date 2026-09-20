"""Generate the compact Sid test corpus with Osteosarc, entirely offline if cached.

The recipe pins original archived inputs and the exact loci exercised by tests.
Only selected regional reads and small metadata/reference excerpts are exported.
Use --live-alignments to verify the same selection against original remote BAMs
with indexed range requests. Neither mode downloads a complete remote BAM.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pysam
from osteosarc import Cache, Region, digest, extract_reads
from osteosarc import __version__ as osteosarc_version


DEFAULT = Path(__file__).resolve().parents[1] / "tests/data/sid-fixtures.json"


def write_json(path, value):
    """Keep generated provenance portable and deterministic."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def generate_sid_fixtures(recipe_path, destination, *, cache=None, source_directory=None,
                          live_alignments=False):
    """Export only the original reads needed by Topiary's Sid regression tests.

    Parameters
    ----------
    recipe_path : path-like
        Versioned input checksums, source URLs and zero-based half-open regions.
    destination : path-like
        New output directory. Existing destinations are refused.
    cache : osteosarc.Cache, optional
        Shared verified acquisition cache; use offline=True to forbid downloads.
    source_directory : path-like, optional
        Local historical export to import instead of fetching the pinned inputs.
    live_alignments : bool
        Compare extraction from the original indexed remote alignments with
        the pinned regional inputs. Record selection and expectations must agree.

    Returns
    -------
    dict
        Export manifest with file sizes, hashes and per-alignment read counts.
        Empty alignment subsets are retained when they test zero coverage.
    """
    recipe_path, destination = Path(recipe_path), Path(destination)
    recipe = json.loads(recipe_path.read_text())
    for asset in recipe["sources"]:
        name = asset["filename"]
        if (not isinstance(name, str) or any(c in name for c in "\\:")
                or any(part in ("", ".", "..") for part in name.split("/"))):
            raise ValueError(f"Unsafe source filename: {name!r}")
    cache = cache if cache is not None else Cache()
    destination.mkdir(parents=True, exist_ok=False)
    sources = {}
    for asset in recipe["sources"]:
        name = asset["filename"]
        if source_directory is not None:
            receipt = cache.import_file(Path(source_directory) / name, asset["url"],
                                        sha256=asset["sha256"], size=asset["size_bytes"])
        else:
            receipt = cache.fetch(asset["url"], sha256=asset["sha256"], size=asset["size_bytes"])
        sources[name] = cache.path(receipt)
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[name], target)

    selections = {}
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        for i, selection in enumerate(recipe["reads"]):
            name = selection["filename"]
            regions = [Region(*region, "GRCh38") for region in selection["regions"]]
            source = sources[name]
            if name.endswith(".sam.gz"):
                sam = work / f"{i}.sam"
                sam.write_bytes(gzip.decompress(source.read_bytes()))
                source = work / f"{i}.bam"
                pysam.sort("--no-PG", "-o", str(source), str(sam))
            index = work / f"{i}.bai"
            pysam.index("-o", str(index), str(source))
            subset = extract_reads(source, regions, index=index, cache=cache)
            with pysam.AlignmentFile(source) as before, subset.open() as after:
                original = Counter(read.to_string() for read in before)
                selected = Counter(read.to_string() for read in after)
            if not selected <= original:
                raise ValueError(f"Extraction changed original SAM records: {name}")
            if selection.get("preserve_all_records") and selected != original:
                raise ValueError(f"A pinned test observation lies outside its selected loci: {name}")
            if live_alignments:
                live = extract_reads(selection["original_alignment_url"], regions, cache=cache)
                with live.open() as bam:
                    observed = Counter(read.to_string() for read in bam)
                if selection.get("preserve_all_records"):
                    # These fixtures have an explicit historical template selection.
                    observed = Counter({record: observed[record] for record in selected})
                if observed != selected:
                    raise ValueError(f"Live source differs from pinned observations: {name}")
            target = destination / name
            if selected != original:
                if name.endswith(".sam.gz"):
                    with subset.open() as bam:
                        contents = (str(bam.header) + "".join(r.to_string() + "\n" for r in bam)).encode()
                    target.write_bytes(gzip.compress(contents, mtime=0))
                else:
                    shutil.copyfile(subset.path, target)
                    shutil.copyfile(subset.index_path, str(target) + ".bai")
            if name.endswith(".bam") and not Path(str(target) + ".bai").exists():
                shutil.copyfile(subset.index_path, str(target) + ".bai")
            selections[name] = dict(
                regions=selection["regions"], source_sha256=digest(sources[name]),
                original_alignment_url=selection["original_alignment_url"],
                original_records=sum(original.values()), selected_records=sum(selected.values()),
                sam_records_sha256=hashlib.sha256(
                    json.dumps(sorted(selected.items()), separators=(",", ":")).encode()).hexdigest(),
            )

    indels = destination / "osteosarc_indels"
    path = indels / "manifest.json"
    manifest = json.loads(path.read_text())
    for dataset in manifest["datasets"].values():
        target = indels / dataset["file"]
        selected = selections["osteosarc_indels/" + dataset["file"]]
        dataset.update(sha256=digest(target),
                       sam_sha256=hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest(),
                       records=selected["selected_records"], region=selected["regions"][0],
                       fixture_selection=dict(selected, osteosarc_version=osteosarc_version))
    write_json(path, manifest)

    # The shared workflow exercises NTF3. Preserve its original source objects
    # and identity; the other 48 read sets are not necessary for Topiary tests.
    shared = destination / "osteosarc_shared"
    source = shared / "vaccine-rna-v1"
    path = source / "manifest.json"
    upstream_digest = digest(path)
    manifest = json.loads(path.read_text())
    manifest["cases"] = [c for c in manifest["cases"] if c["case_id"] in recipe["shared_cases"]]
    needed = {name for case in manifest["cases"] for name in (case["bam"], case["bam"] + ".bai")}
    manifest["assets"] = [a for a in manifest["assets"] if a["filename"] in needed]
    for asset in manifest["assets"]:
        path = source / asset["filename"]
        asset.update(sha256=digest(path), size_bytes=path.stat().st_size)
    manifest["scope"] = "NTF3 reads required for Topiary's shared-input reconstruction and ranking workflow"
    manifest["upstream_manifest_sha256"] = upstream_digest
    write_json(source / "manifest.json", manifest)
    # Only the manifest's membership changed; the NTF3 read records above were
    # required to stay identical, so derived scientific expectations are intact.
    translation_path = shared / "translation-v1.json"
    translation = json.loads(translation_path.read_text())
    translation["parent_dataset_sha256"] = digest(source / "manifest.json")
    write_json(translation_path, translation)
    prediction_path = shared / "prediction-contract-v1.json"
    prediction = json.loads(prediction_path.read_text())
    prediction["parent_translation_sha256"] = digest(translation_path)
    write_json(prediction_path, prediction)

    for group, filename, receipt_name in (
        ("osteosarc_all_variants", "t2-all-variant-regions.bam", "source/bam.receipt.json"),
        ("osteosarc_rna_overlay", "t2-pvac-regions.bam", "acquisition.json"),
    ):
        root = destination / group
        name = f"{group}/source/{filename}"
        bam = destination / name
        path = root / receipt_name
        metadata = json.loads(path.read_text())
        receipt = metadata if group == "osteosarc_all_variants" else metadata["alignment"]
        receipt.update(
            sha256=digest(bam), index_sha256=digest(str(bam) + ".bai"), bytes=bam.stat().st_size,
            regions=[f"{c}:{start + 1}-{end}" for c, start, end in selections[name]["regions"]],
            fixture_selection=dict(selections[name], osteosarc_version=osteosarc_version),
        )
        # The acquisition command remains provenance of the original regional
        # source. The smaller fixture's selection is recorded separately above.
        if group == "osteosarc_rna_overlay":
            metadata["fixture_note"] = "Only test-locus reads retained through Osteosarc; regional index bundled."
        write_json(path, metadata)
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        files = manifest["files"] if "files" in manifest else manifest
        files["source/" + filename + ".bai"] = digest(str(bam) + ".bai")
        for name in files:
            files[name] = digest(root / name)
        write_json(path, manifest)

    shutil.copyfile(recipe_path, destination / "sid-fixtures.json")
    assets = [dict(filename=p.relative_to(destination).as_posix(), sha256=digest(p),
                   size_bytes=p.stat().st_size,
                   url=recipe["export_url"] + p.relative_to(destination).as_posix())
              for p in sorted(destination.rglob("*")) if p.is_file()]
    manifest = dict(schema_version=1, dataset="osteosarc", data_version="topiary-sid-v1",
                    groups=recipe["groups"], assets=assets, read_selections=selections,
                    recipe_sha256=digest(recipe_path), total_size_bytes=sum(a["size_bytes"] for a in assets))
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--live-alignments", action="store_true")
    args = parser.parse_args()
    manifest = generate_sid_fixtures(
        args.recipe, args.output, cache=Cache(args.cache_root, offline=args.offline),
        source_directory=args.source_directory, live_alignments=args.live_alignments)
    print(f"Generated {len(manifest['assets'])} assets, {manifest['total_size_bytes']:,} bytes")
    for name, counts in manifest["read_selections"].items():
        print(f"{name}: {counts['original_records']} -> {counts['selected_records']} records")
