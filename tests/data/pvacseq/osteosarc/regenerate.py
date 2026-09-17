"""Extract byte-preserving regression rows from the verified public archive.

Usage: python regenerate.py /path/to/osteosarc-pvacseq-2026-09-16
No network access or Topiary code is used to choose expected values.
"""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_source(root, item):
    data = (root / "raw" / item["relative_path"]).read_bytes()
    assert sha256(data) == item["sha256"], item["relative_path"]
    lines = data.splitlines(keepends=True)
    rows = list(csv.DictReader(io.StringIO(data.decode()), delimiter="\t"))
    assert len(lines) == len(rows) + 1  # These reports have no multiline cells.
    return lines, rows


def candidate(row):
    if "ID" in row:
        return (row["ID"].strip(), row["Best Peptide"], row["Allele"],
                row["Best Transcript"])
    return ("-".join(row[c] for c in
                     ("Chromosome", "Start", "Stop", "Reference", "Variant")),
            row["MT Epitope Seq"], row["HLA Allele"], row["Transcript"])


def main(root):
    destination = Path(__file__).resolve().parent
    objects = json.loads((root / "archive-manifest.json").read_text())["objects"]
    reports = [x for x in objects if x["category"] != "supporting"]
    assert len(reports) == 21
    aggregates = {}
    for item in reports:
        if item["category"] == "aggregated":
            _, rows = read_source(root, item)
            aggregates[str(Path(item["relative_path"]).parent)] = {
                candidate(row) for row in rows}
    entries = []
    for item in reports:
        lines, rows = read_source(root, item)
        selected = set()
        if item["category"] != "all_epitopes":
            selected.update(range(len(rows)))
        else:
            best = aggregates[str(Path(item["relative_path"]).parent)]
            found = set()
            seen = set()
            for index, row in enumerate(rows):
                key = candidate(row)
                if key in best:
                    selected.add(index)
                    found.add(key)
                features = [("variant", key[0]), ("allele", key[2]),
                            ("variant_type", row["Variant Type"])]
                # Retain both missing and present measurements, and exact zeros.
                for column, value in row.items():
                    if any(token in column for token in
                           ("Score", "Percentile", "Epitope Seq")):
                        state = "missing" if value in ("NA", "X", "") else (
                            "zero" if value in ("0", "0.0") else "present")
                        features.append((column, state))
                if any(feature not in seen for feature in features):
                    selected.add(index)
                seen.update(features)
            assert found == best, (item["relative_path"], best - found)
        indices = sorted(selected)
        run, view, _ = item["relative_path"].split("/")
        name = ".".join((run[:10] + ("-extended" if "extended" in run else ""),
                         view.lower(), item["category"], "tsv"))
        data = lines[0] + b"".join(lines[i + 1] for i in indices)
        (destination / name).write_bytes(data)
        entries.append(dict(
            file=name, category=item["category"], source_url=item["url"],
            source_sha256=item["sha256"], source_rows=len(rows),
            sha256=sha256(data), source_data_rows=[i + 1 for i in indices],
            row_sha256=[sha256(lines[i + 1]) for i in indices],
            pair=".".join((run, view)),
        ))
    # Pin the input-side annotation gap as well as final prediction reports.
    suffix = "2025.04.27.sg.curated.neoantigen.predictions/MHC_Class_I/SG.WGS_SG.WGS.UCLA.2025.01.tumor.tsv"
    item = next(x for x in objects if x["relative_path"] == suffix)
    lines, rows = read_source(root, item)
    data = b"".join(lines)
    name = "2025.04.27.variant-input.tsv"
    (destination / name).write_bytes(data)
    manifest = dict(
        schema_version=1, reports=entries,
        variant_input=dict(file=name, sha256=sha256(data), rows=len(rows),
                           source_url=item["url"]),
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(entries)} reports; {sum(len(x['source_data_rows']) for x in entries)} rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
