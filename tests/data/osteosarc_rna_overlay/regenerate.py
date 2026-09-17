"""Pin unchanged regional BAM and original expression rows from a verified cache.

Usage: python tests/data/osteosarc_rna_overlay/regenerate.py CACHE_DIRECTORY
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(cache):
    root = Path(__file__).resolve().parent
    source = root / "source"
    source.mkdir(exist_ok=True)
    original = json.loads((cache / "acquisition.json").read_text())
    acquisition = deepcopy(original)
    acquisition["sources"] = {}
    panel = pd.read_csv(root.parent / "pvacseq/osteosarc/2025.04.27.variant-input.tsv", sep="\t")
    for filename, key, identifiers in (
        ("t2.genes.results", "gene_id", set(panel.ensembl_gene_id)),
        ("t2.isoforms.results", "transcript_id", set(panel.transcript_name)),
    ):
        path = cache / "source" / filename
        receipt = original["sources"][filename]
        assert digest(path) == receipt["sha256"]
        lines = path.read_bytes().splitlines(keepends=True)
        column = lines[0].decode().rstrip().split("\t").index(key)
        selected = []
        for i, line in enumerate(lines[1:], 1):
            identity = line.decode().split("\t")[column]
            if identity.rsplit(".", 1)[0] in identifiers:
                selected.append(i)
        (source / filename).write_bytes(lines[0] + b"".join(lines[i] for i in selected))
        acquisition["sources"][filename] = dict(
            url=receipt["url"], source_sha256=receipt["sha256"], source_bytes=receipt["bytes"],
            sha256=digest(source / filename), bytes=(source / filename).stat().st_size,
            source_data_rows=selected, selection="Unchanged source lines matching the 63-row pVAC input",
        )
    filenames = ["t2-pvac-regions.bam", "t2-pvac-regions.header.sam"]
    assert digest(cache / "source/t2-pvac-regions.bam") == original["alignment"]["sha256"]
    filenames += [item["file"] for item in original["references"]]
    for filename in filenames:
        shutil.copyfile(cache / "source" / filename, source / filename)
    for filename in ("allele-evidence.tsv", "transcript-evidence.tsv"):
        shutil.copyfile(cache / filename, root / filename)
    acquisition["fixture_note"] = (
        "Regional BAM retained byte-for-byte. Expression tables are unchanged selected original rows. "
        "BAM index is rebuilt in the test temporary directory; the full source index remains in the local cache.")
    (root / "acquisition.json").write_text(json.dumps(acquisition, indent=2) + "\n")
    manifest = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
                if p.is_file() and p.name not in ("manifest.json", "regenerate.py", "README.md")}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
