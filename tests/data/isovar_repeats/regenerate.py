"""Extract the original RNA records reproducing Topiary #397 with Osteosarc."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import osteosarc
from osteosarc import Cache, Region, digest, extract_reads
from osteosarc.reads import ReadFilter
import pysam


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path, help="New output directory")
parser.add_argument("--cache-root", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
recipe = json.loads((root / "recipe.json").read_text())
source = root.parent / recipe["source"]
if digest(source) != recipe["source_sha256"]:
    raise ValueError("Source RNA checksum differs from the reviewed corpus")
args.output.mkdir(parents=True, exist_ok=False)
subset = extract_reads(source, [Region(*recipe["region"])], cache=Cache(args.cache_root),
                       filters=ReadFilter(query_names=tuple(recipe["query_names"])))
with pysam.AlignmentFile(source) as original, subset.open() as selected:
    expected = Counter(r.to_string() for r in original if r.query_name in recipe["query_names"])
    actual = Counter(r.to_string() for r in selected)
if actual != expected:
    raise ValueError("Extraction changed the selected original RNA records")
for source_path, name in ((subset.path, "reads.bam"), (subset.index_path, "reads.bam.bai")):
    shutil.copyfile(source_path, args.output / name)
manifest = dict(
    osteosarc_version=osteosarc.__version__, recipe_sha256=digest(root / "recipe.json"),
    source_sha256=digest(source), records=sum(actual.values()), templates=len(recipe["query_names"]),
    files={name: digest(args.output / name) for name in ("reads.bam", "reads.bam.bai")},
)
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
