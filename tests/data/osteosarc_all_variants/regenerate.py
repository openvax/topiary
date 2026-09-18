"""Pin an acquired full-index audit without editing any source read records.

Usage: PYTHONPATH=. python tests/data/osteosarc_all_variants/regenerate.py AUDIT_ROOT
Run the audit acquisition/reference stages first. CI never downloads data.
"""

import argparse
import gzip
import json
from pathlib import Path
import shutil

from scripts.osteosarc_rna_overlay import digest, write_json


def regenerate(root, destination):
    """Verify acquisition receipts, then retain original inputs and outcomes."""
    inventory = json.loads((root / "inventory.json").read_text())
    receipt = json.loads((root / "source/bam.receipt.json").read_text())
    reference = json.loads((root / "reference/manifest.json").read_text())
    originals = {
        "source/t2-all-variant-regions.bam": receipt["sha256"],
        "source/t2-all-variant-regions.bam.bai": receipt["index_sha256"],
        **{"source/" + name: value["sha256"] for name, value in inventory["sources"].items()},
        **{"reference/" + name: value["sha256"] for name, value in reference["files"].items()},
    }
    for name, expected in originals.items():
        if digest(root / name) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    # Gzip text sources losslessly; already compressed original BAM/reference
    # files are copied byte-for-byte, with their original provenance receipts.
    names = [*originals, "inventory.json", "checked-inventory.json", "source/bam.receipt.json",
             "reference/manifest.json", "run.json"]
    for name in names:
        target_name = name + ".gz" if name in ("source/variants.html", "source/vafs.tsv") else name
        target = destination / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target_name != name:
            target.write_bytes(gzip.compress((root / name).read_bytes(), mtime=0))
        else:
            shutil.copyfile(root / name, target)
        files[target_name] = digest(target)
    outcomes = [json.loads((root / "outcomes" / (r["variant_id"] + ".json")).read_text())
                for r in inventory["variants"]]
    write_json(destination / "expected.json", outcomes)
    files["expected.json"] = digest(destination / "expected.json")
    write_json(destination / "manifest.json", dict(files=files, entries=len(outcomes),
               scope="All 182 website entries plus two additional exact indels; T2 STAR RNA only",
               count_unit="sequenced segments and RG/QNAME templates, never independent molecules",
               sample="January 2025 UCLA resection (T2); public collection-label pairing"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    regenerate(args.root, Path(__file__).resolve().parent)
