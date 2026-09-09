"""Pinned real RNA and independent reference edits for integration tests.

The reference oracle is adapted from Isovar's six-locus fixture oracle. It
reads original GTF/cDNA/protein records, never Isovar/Varcode predictions.
Imports and index creation are deferred until an integration fixture runs.
"""

from collections import defaultdict
import gzip
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import re


def translate(sequence):
    """NCBI standard genetic code; stop at the first complete stop codon."""
    code = dict(zip(
        ("".join(bases) for bases in product("TCAG", repeat=3)),
        "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
    ))
    amino_acids = []
    for i in range(0, len(sequence) - 2, 3):
        amino_acid = code[sequence[i:i + 3]]
        if amino_acid == "*":
            break
        amino_acids.append(amino_acid)
    return "".join(amino_acids)


def transcript_offset(exons, strand, position):
    """Map one genomic base using raw exon coordinates, not the annotator."""
    offset = 0
    for start, end in sorted(exons, reverse=strand == "-"):
        if start <= position <= end:
            return offset + (position - start if strand == "+" else end - position)
        offset += end - start + 1
    raise ValueError(f"Position outside the pinned transcript: {position}")


def reference_expectations(reference, metadata, selection):
    """Apply only the pinned SNVs/deletions to independently verified cDNA."""
    import pysam

    with pysam.FastxFile(str(reference / "reference.cdna.fa.gz")) as records:
        cdnas = {record.name.split(".")[0]: record.sequence for record in records}
    with pysam.FastxFile(str(reference / "reference.pep.fa.gz")) as records:
        proteins = {
            re.search(r"transcript:(ENST\d+)", record.comment)[1]: record.sequence
            for record in records
        }
    features = defaultdict(lambda: defaultdict(list))
    with gzip.open(reference / "reference.gtf.gz", "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            transcript = re.search(r'transcript_id "([^"]+)"', fields[8])
            if transcript:
                features[transcript[1]][fields[2]].append(fields)
    expected = {}
    for variant in selection["variants"]:
        gene = variant["gene"]
        transcript = metadata["transcripts"][gene]
        rows = features[transcript]
        strand = rows["transcript"][0][6]
        assert variant["chrom"].removeprefix("chr") == rows["transcript"][0][0]
        exons = [(int(r[3]), int(r[4])) for r in rows["exon"]]
        cds_start = min(
            transcript_offset(exons, strand, position)
            for row in rows["start_codon"]
            for position in range(int(row[3]), int(row[4]) + 1)
        )
        cdna = cdnas[transcript]
        assert translate(cdna[cds_start:]) == proteins[transcript]
        position, ref, alt = int(variant["pos"]), variant["ref"], variant["alt"]
        if len(ref) != len(alt):
            assert len(alt) == 1 and ref.startswith(alt)
            position, ref, alt = position + 1, ref[1:], ""
        offsets = sorted(transcript_offset(exons, strand, p)
                         for p in range(position, position + len(ref)))
        offset = offsets[0]
        assert offsets == list(range(offset, offset + len(ref)))
        if strand == "-":
            ref = ref.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            alt = alt.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        assert cdna[offset:offset + len(ref)] == ref
        mutant = cdna[:offset] + alt + cdna[offset + len(ref):]
        frameshift = (len(alt) - len(ref)) % 3 != 0
        protein = translate(mutant[cds_start:])
        expected[gene] = {
            "sequence": protein,
            "start": (offset - cds_start) // 3,
            "end": len(protein) if frameshift else (offset + len(alt) - cds_start + 2) // 3,
            "transcript_id": transcript,
        }
    return expected


def load_osteosarc(directory):
    """Verify the original assets, then build private temporary indices."""
    import pysam
    from pyensembl import Genome
    from varcode import Variant

    data = Path(__file__).parent / "data" / "osteosarc"
    selection = json.loads((data / "selection.json").read_text())
    manifest = json.loads((data / "manifest.json").read_text())
    reference = data / "protein_reference"
    metadata = json.loads((reference / "protein_reference_manifest.json").read_text())
    for filename, checksums in metadata["files"].items():
        contents = (reference / filename).read_bytes()
        assert sha256(contents).hexdigest() == checksums["subset_sha256"]
        assert sha256(gzip.decompress(contents)).hexdigest() == checksums["uncompressed_sha256"]
    genome = Genome(
        reference_name="GRCh38-osteosarc-six-transcript-subset",
        annotation_name="osteosarc-ensembl-subset", annotation_version=87,
        gtf_path_or_url=str(reference / "reference.gtf.gz"),
        transcript_fasta_paths_or_urls=[str(reference / "reference.cdna.fa.gz")],
        protein_fasta_paths_or_urls=[str(reference / "reference.pep.fa.gz")],
        copy_local_files_to_cache=True,
        cache_directory_path=str(directory / "reference"),
    )
    genome.index()
    variants = {
        record["gene"]: Variant(
            record["chrom"].removeprefix("chr"), int(record["pos"]),
            record["ref"], record["alt"], ensembl=genome,
        ) for record in selection["variants"]
    }
    bams = {}
    for name, dataset in manifest["datasets"].items():
        sam_data = gzip.decompress((data / dataset["file"]).read_bytes())
        assert sha256(sam_data).hexdigest() == dataset["sam_sha256"]
        sam = directory / (name + ".sam")
        sam.write_bytes(sam_data)
        bam = directory / (name + ".bam")
        pysam.sort("--no-PG", "-o", str(bam), str(sam))
        pysam.index(str(bam))
        bams[name] = str(bam)
    return variants, bams, reference_expectations(reference, metadata, selection)


def assert_expected_fragment(fragment, expected):
    """Assert sequence AND genomic-edit interval, including deletion junctions."""
    offset = expected["sequence"].find(fragment.sequence)
    assert offset >= 0
    assert list(fragment.target_intervals) == [
        (expected["start"] - offset, min(expected["end"] - offset, len(fragment.sequence)))
    ]
    assert expected["transcript_id"] in fragment.annotations["supporting_reference_transcripts"]
