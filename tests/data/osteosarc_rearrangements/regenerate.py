"""Pin audited RNA windows only after matching actual source-BAM records.

Usage: python regenerate.py /path/to/2026-09-17_05-32-18-330588Z
The source tree is the retained, checksummed Isovar #291 RNA-footprint audit.
"""

import gzip
import hashlib
import json
from pathlib import Path
import sys

import pysam


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(root):
    destination = Path(__file__).resolve().parent
    footprint = root / "rna-footprints"
    evidence_path = footprint / "evidence.json"
    evidence = json.loads(evidence_path.read_text())
    actual = {}
    receipts = {}
    for sample in ("T1", "T2"):
        source = sample + "-ONT-tagged"
        directory = root / "inputs" / source
        receipt = json.loads((directory / "receipt.json").read_text())
        bam_path = directory / "regions.bam"
        assert digest(bam_path.read_bytes()) == receipt["bam_sha256"]
        with pysam.AlignmentFile(bam_path) as bam:
            actual[source] = {r.to_string() for r in bam}
        receipts[source] = receipt
    paths = []
    for event in evidence["fusions"]:
        for product in event["products"]:
            source = product["source"]
            if source not in actual:
                continue
            for path in product["paths"]:
                assert all(record in actual[source] for record in path["original_records"])
                paths.append(dict(
                    event=event["name"], source=source, read_id=path["read_id"],
                    cell_umi=path["cell_umi"],
                    sam_sha256=[digest(r.encode()) for r in path["original_records"]],
                ))
    assert len(paths) == 13
    entries = []
    for event, sample in (("GABBR1--SLC29A1", "T1"), ("GABBR1--SLC29A1", "T2"),
                          ("OTUD7A--FMN1", "T2")):
        source_path = footprint / event / (event + "-" + sample) / "input.json"
        raw = source_path.read_bytes()
        supplied = json.loads(raw)
        for record in supplied["original_records"]:
            assert all(record[key] in actual[sample + "-ONT-tagged"]
                       for key in ("sam", "partner_sam"))
        filename = event + "-" + sample + ".json.gz"
        packed = gzip.compress(raw, mtime=0)
        (destination / filename).write_bytes(packed)
        entries.append(dict(file=filename, sha256=digest(packed), input_sha256=digest(raw),
                            event=event, sample=sample, selected_paths=len(supplied["reads"])))
    manifest = dict(
        upstream_commit="db233f251b4dd965451db7cb3207162465c83c4a",
        evidence_sha256=digest(evidence_path.read_bytes()),
        source_receipts=receipts, observed_paths=paths, inputs=entries,
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(paths)} full paths against original BAMs; pinned {len(entries)} inputs")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
