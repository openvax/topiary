"""Pin original locus records and raw Ensembl 87 records; no predictions.

Usage: python regenerate.py INPUTS_DIRECTORY ENSEMBL87_DIRECTORY
INPUTS_DIRECTORY contains checksummed regional BAMs/receipts from Isovar #291.
"""

import gzip
import hashlib
import json
from pathlib import Path
import re
import sys

import pysam


VARIANTS = [
    dict(gene="GLIS3", chrom="chr9", pos=3856149, ref="CTGATGTGG", alt="C"),
    dict(gene="KTN1", chrom="chr14", pos=55627965, ref="G", alt="GTT"),
]
TRANSCRIPTS = {"GLIS3": "ENST00000324333", "KTN1": "ENST00000395308"}
SOURCES = ("T1-ONT-dedup", "T2-ONT-dedup", "T1-short", "T2-short")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main(inputs, ensembl):
    output = Path(__file__).resolve().parent
    save_json(output / "selection.json", dict(variants=VARIANTS))
    datasets = {}
    for source in SOURCES:
        directory = inputs / source
        receipt = json.loads((directory / "receipt.json").read_text())
        bam_path = directory / "regions.bam"
        assert digest(bam_path.read_bytes()) == receipt["bam_sha256"]
        with pysam.AlignmentFile(bam_path) as bam:
            for variant in VARIANTS:
                start = variant["pos"] - 2
                end = variant["pos"] + len(variant["ref"]) + 2
                records = list(bam.fetch(variant["chrom"], start, end))
                # Preserve all returned records, including competing placements.
                # Evidence filtering is exercised by the consumer, not here.
                data = (str(bam.header) + "".join(r.to_string() + "\n" for r in records)).encode()
                name = variant["gene"] + "." + source
                filename = name + ".sam.gz"
                packed = gzip.compress(data, mtime=0)
                (output / filename).write_bytes(packed)
                datasets[name] = dict(file=filename, sam_sha256=digest(data),
                                      sha256=digest(packed), records=len(records),
                                      region=[variant["chrom"], start, end], source=receipt)
    save_json(output / "manifest.json", dict(
        datasets=datasets, assembly="GRCh38",
        count_unit="RG/QNAME templates, not proven independent molecules",
        upstream_commit="db233f251b4dd965451db7cb3207162465c83c4a",
    ))
    reference = output / "protein_reference"
    reference.mkdir(exist_ok=True)
    files = {}
    selected = set(TRANSCRIPTS.values())
    for kind, source_name in (
        ("gtf", "Homo_sapiens.GRCh38.87.gtf.gz"),
        ("cdna", "Homo_sapiens.GRCh38.cdna.all.fa.gz"),
        ("pep", "Homo_sapiens.GRCh38.pep.all.fa.gz"),
    ):
        source = ensembl / source_name
        chunks = []
        with gzip.open(source, "rb") as handle:
            keep = False
            for line in handle:
                if kind == "gtf":
                    match = re.search(rb'transcript_id "([^"]+)"', line)
                    keep = bool(match and match[1].decode() in selected)
                elif line.startswith(b">"):
                    match = (re.match(rb">(ENST\d+)", line) if kind == "cdna" else
                             re.search(rb"transcript:(ENST\d+)", line))
                    keep = bool(match and match[1].decode() in selected)
                if keep:
                    chunks.append(line)
        data = b"".join(chunks)
        assert data
        packed = gzip.compress(data, mtime=0)
        filename = "reference." + kind + (".gz" if kind == "gtf" else ".fa.gz")
        (reference / filename).write_bytes(packed)
        category = "gtf/homo_sapiens" if kind == "gtf" else "fasta/homo_sapiens/" + kind
        files[filename] = dict(
            source_url="https://ftp.ensembl.org/pub/release-87/" + category + "/" + source_name,
            source_sha256=digest(source.read_bytes()), subset_sha256=digest(packed),
            uncompressed_sha256=digest(data))
    save_json(reference / "protein_reference_manifest.json", dict(
        transcripts=TRANSCRIPTS, reference_name="GRCh38-osteosarc-two-indel-subset", files=files))
    print({name: entry["records"] for name, entry in datasets.items()})


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
